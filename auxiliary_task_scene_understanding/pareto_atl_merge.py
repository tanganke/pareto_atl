# Debug Args: --arch HPS --gpu_id 1 --scheduler step --source_tasks segmentation depth normal --target_tasks segmentation --log_dir logs/nyuv2/HPS/segmentation --epoch_step 30 --seed 0 --round_idx 0
import argparse
import json
import logging
import math
from copy import deepcopy
from typing import Dict, Optional

import torch.nn.functional as F
import torch.utils.data
from rich.traceback import install
import wandb
from torch.utils.data import DataLoader
import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL._record import _PerformanceMeter
from LibMTL.aspp import DeepLabHead
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.create_dataset import Cityscapes, NYUv2
from LibMTL.model import resnet_dilated
from LibMTL.utils import *
from pareto_atl_base import NYUtrainer
from fusionlib.utils.torch.state_dict_arithmetic import state_dict_sub
from fusionlib.merge.wrapper import layer_wise_fusion
import pareto_atl_base
from tqdm import tqdm
from src.min_norm_solvers import MinNormSolver

logger = logging.getLogger(__name__)


class NYU_Merge_trainer(NYUtrainer):
    def __init__(
        self,
        params,
        program: pareto_atl_base.ParetoATL,
        **kwargs,
    ):
        self.params = params
        self.program = program
        super().__init__(**kwargs)

        self.meter = _PerformanceMeter(self.program.task_dict, self.multi_input)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        super()._prepare_model(weighting, architecture, encoder_class, decoders)
        params = self.params
        program = self.program
        model = self.model

        if args.round_idx == 0:
            pretrained_sd = model.state_dict()
        else:
            pretrained_sd = program.load_checkpoint(
                f"round={args.round_idx-1}_merged.pth", map_location=self.device
            )["state_dict"]

        task_vectors = {}
        for task_name in params.source_tasks:
            finetuned_sd = program.load_checkpoint(
                f"round={args.round_idx}_task={task_name}_trained.pth",
                map_location=self.device,
            )["state_dict"]
            task_vectors[task_name] = state_dict_sub(finetuned_sd, pretrained_sd)

        pretrained_model = deepcopy(model)
        pretrained_model.load_state_dict(pretrained_sd)
        remove_keys = [
            n
            for n, p in pretrained_model.state_dict(keep_vars=True).items()
            if not p.requires_grad
        ]
        merged_model = layer_wise_fusion.LayerWiseMergedModel(
            pretrained_model=pretrained_model,
            layer_wise_weights=layer_wise_fusion.get_layer_wise_weights(
                len(params.source_tasks), len(pretrained_sd), init_values=0.2
            ).to(self.device),
            task_vectors=list(task_vectors.values()),
            clamp_weights=False,
        ).to(self.device)
        merged_model.remove_keys = remove_keys
        self.model: layer_wise_fusion.LayerWiseMergedModel = merged_model

    def pareto_merging(
        self,
        train_dataloaders: DataLoader,
        val_dataloaders: Optional[DataLoader],
        epoch_start: int,
        epoch_end: int,
        return_weight=False,
    ):
        """Pareto Merging"""
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        step_idx = 0
        self.program.save_checkpoint(
            self.model,
            f"round={self.params.round_idx}_epoch={0}_merged.pth",
            meta_info={
                "layer_wise_weights": self.model.layer_wise_weights.detach().cpu()
            },
        )
        for epoch in tqdm(
            range(epoch_start, epoch_end),
            "pareto merging epochs",
            leave=False,
            dynamic_ncols=True,
        ):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time("begin")
            for batch_index in tqdm(
                range(train_batch),
                "pareto merging batchs",
                leave=False,
                dynamic_ncols=True,
            ):
                self.model.merge_weights(self.model.remove_keys)
                if not self.multi_input:
                    all_grads = []
                    for task_name in self.params.source_tasks:
                        new_task_dict = deepcopy(self.program.target_task_dict)
                        if task_name not in self.params.target_tasks:
                            new_task_dict[task_name] = self.program.task_dict[task_name]

                        # reset task_dict, task_num and task_name
                        self.task_dict = new_task_dict
                        self.task_num = len(self.task_dict)
                        self.task_name = list(new_task_dict.keys())
                        self.model.model.task_name = self.task_name

                        train_inputs, train_gts = self._process_data(train_loader)
                        train_preds = self.model(train_inputs)
                        train_preds = self.process_preds(train_preds)
                        train_losses = self._compute_loss(train_preds, train_gts)
                        for t in self.task_name:
                            self.meter.update(train_preds[t], train_gts[t], task_name=t)

                        _grads = torch.autograd.grad(
                            train_losses.sum(),
                            self.model.parameters(),
                            create_graph=False,
                            retain_graph=True,
                        )
                        assert len(_grads) == 1
                        all_grads.append(_grads[0].flatten())

                    sol, min_norm = MinNormSolver.find_min_norm_element(all_grads)
                    if not isinstance(sol, torch.Tensor):
                        sol = torch.from_numpy(sol)
                    sol = sol.to(
                        device=self.model.layer_wise_weights.device,
                        dtype=self.model.layer_wise_weights.dtype,
                    )

                    self.optimizer.zero_grad()
                    grad = torch.stack(all_grads) * sol.view(-1, 1)
                    self.model.layer_wise_weights.grad = grad.sum(dim=0).view_as(
                        self.model.layer_wise_weights
                    )
                    self.optimizer.step()

                    wandb.log({"min_norm": min_norm.item()}, step=step_idx)
                else:
                    raise NotImplementedError(
                        "multi_input not implemented yet"
                    )  # Not Checked
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(
                            train_pred, train_gt, task
                        )
                        self.meter.update(train_pred, train_gt, task)

                step_idx = step_idx + 1
            self.meter.record_time("end")
            self.meter.get_score()
            self.meter.display(epoch=epoch, mode="train")

            # record logs to wandb
            log_dict = {}
            for task_name in self.meter.task_name:
                for metric_name, value in zip(
                    self.meter.task_dict[task_name]["metrics"],
                    self.meter.results[task_name],
                ):
                    log_dict[f"{task_name}_{metric_name}"] = value
            for task_name, loss_value in zip(
                self.meter.task_name, self.meter.loss_item
            ):
                log_dict[f"{task_name}_loss"] = loss_value
            wandb.log(log_dict, step=step_idx)

            self.meter.reinit()

            if val_dataloaders is not None:
                self.val(val_dataloaders, epoch)
            print()
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.params.save_interval == 0:
                self.program.save_checkpoint(
                    self.model,
                    f"round={self.params.round_idx}_epoch={epoch+1}_merged.pth",
                    meta_info={
                        "layer_wise_weights": self.model.layer_wise_weights.detach().cpu(),
                        "beta": sol.detach().cpu(),
                    },
                )
        self.display_best_result()


class ParetoATL_Merging(pareto_atl_base.ParetoATL):
    def __init__(self, params):
        super().__init__(params)
        if os.path.exists(self.get_checkpoint_path(f'round={params.round_idx}_merged.pth')):
            logger.info(f"round={params.round_idx}_merged.pth exists, skip")
            exit(0)
            
        wandb.init(
            project="pareto_atl",
            name=params.run_name,
            group=f"nyuv2-{params.arch}-{params.target_tasks[0]}",
            job_type="pareto_merge",
            config={
                "architecture": params.arch,
                "dataset": "NYUv2",
                "learning_lr": self.optim_param["lr"],
                "optim": self.optim_param["optim"],
                "weight_decay": self.optim_param["weight_decay"],
            },
        )

    def run(self):
        params = self.params

        self.load_data()
        self.load_model()

        self.pareto_merging()

        self.cleanup()

    def cleanup(self):
        wandb.finish()

    def load_model(self):
        super().load_model()
        params = self.params
        # the configuration of hyperparameters, optimizier, and learning rate scheduler.
        kwargs, optim_param, scheduler_param = (
            self.kwargs,
            self.optim_param,
            self.scheduler_param,
        )

        base_result = self.base_result

        encoder_class = self.encoder_class
        decoders = self.decoders

        aux_task_name = params.task_name
        new_task_dict = deepcopy(self.target_task_dict)
        if aux_task_name not in params.target_tasks:
            new_task_dict[aux_task_name] = self.task_dict[aux_task_name]
        logger.info(f"new_task_dict: {new_task_dict}")
        kwargs["weight_args"]["weights"] = len(new_task_dict) * [1]
        logger.info(f'{kwargs["weight_args"]["weights"] = }')

        trainer = NYU_Merge_trainer(
            params=params,
            program=self,
            task_dict=new_task_dict,
            weighting=params.weighting,
            architecture=params.arch,
            encoder_class=encoder_class,
            decoders=deepcopy(decoders),
            rep_grad=params.rep_grad,
            multi_input=params.multi_input,
            optim_param=optim_param,
            scheduler_param=scheduler_param,
            base_result=base_result,
            save_path=self.get_checkpoint_path(
                f"round={params.round_idx}_task={params.task_name}_trained.pth"
            ),
            img_size=self.nyuv2_train_set.image_size,
            **deepcopy(kwargs),
        )
        trainer.params = params

        self.trainer = trainer

    @property
    def model(self) -> nn.Module:
        return self.trainer.model

    def pareto_merging(self):
        params = self.params
        trainer = self.trainer

        trainer.pareto_merging(self.nyuv2_train_loader, None, 0, params.epoch_step)

        self.save_checkpoint(
            self.model,
            f"round={params.round_idx}_merged.pth",
            meta_info={
                "layer_wise_weights": self.model.layer_wise_weights.detach().cpu()
            },
        )


def parser_args():
    parser = pareto_atl_base.parser
    parser.add_argument(
        "--run_name", type=str, default="pareto_atl", help="name of the run (wandb)"
    )
    parser.add_argument(
        "--round_idx", type=int, default=0, help="index of current round"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="segmentation",
        help="name of current training task",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="interval for saving the merged model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    pareto_atl = ParetoATL_Merging(args)
    pareto_atl.run()
