import abc
import argparse
import os
import random
import time
import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, cast

import lightning as L
import numpy as np
import pareto_atl_base
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from forkmerge import SpecifiedProbMultiTaskSampler
from fusionlib.merge.wrapper import layer_wise_fusion
from fusionlib.utils.torch.state_dict_arithmetic import state_dict_sub, to_device
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
import wandb
from src.min_norm_solvers import MinNormSolver
from utils.classifier import MultiOutputClassifier
from utils.data import ForeverDataIterator
from utils.logger import CompleteLogger
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_grad(
    model: nn.Module,
    task_weights,
    train_loaders,
    average_steps: int,
    device,
):
    dataset_sampler = SpecifiedProbMultiTaskSampler(train_loaders, 0, task_weights)
    model.train()
    grads = None

    losses = 0
    accs = 0
    for _ in tqdm(range(average_steps), "computing grad", leave=False):
        dataset_name, dataloader = dataset_sampler.pop()
        x, labels = next(dataloader)[:2]
        x = cast(Tensor, x).to(device)
        labels = cast(Tensor, labels).to(device)

        logits: Tensor = model(x, dataset_name)
        loss = F.cross_entropy(logits, labels)

        _grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=False,
            retain_graph=True,
        )
        if grads is None:
            grads = _grads
        else:
            grads = [g1 + g2 for g1, g2 in zip(grads, _grads)]

        losses += loss.item()
        accs += logits.argmax(1).eq(labels).float().mean().item()

    grads = [g / average_steps for g in grads]
    losses /= average_steps
    accs /= average_steps

    return {
        "grads": grads,
        "loss": losses,
        "acc": accs,
    }


class ParetoATL_Merging(pareto_atl_base.ParetoATL):
    def __init__(self, args):
        super().__init__(args)

        wandb.init(
            project="pareto_atl",
            name=args.run_name,
            config={
                "learning_rate": args.lr,
                "architecture": args.arch,
                "dataset": "DomainNet",
            },
        )
        print(args)

        if args.seed is not None:
            L.seed_everything(args.seed)

    def cleanup(self):
        wandb.finish()

    def run(self):
        args = self.args
        self.load_data()
        self.load_model()

        task_weights = {name: 0 for name in args.source}
        for name in args.target:
            task_weights[name] = 1 / len(args.target)
        self.task_weights = task_weights

        self.pareto_merge()
        self.cleanup()

    def load_model(self):
        args = self.args
        super().load_model()

        if args.round_idx == 0:
            pretrained_sd = self.model.state_dict()
        else:
            pretrained_sd = self.load_checkpoint(
                f"round={args.round_idx-1}_merged.pth"
            )["state_dict"]

        task_vectors = {}
        for task_name in zip(args.source):
            finetuned_sd = self.load_checkpoint(
                f"round={args.round_idx}_task={task_name}.pth"
            )["state_dict"]
            task_vectors[task_name] = state_dict_sub(finetuned_sd, pretrained_sd)

        self.pretrained_sd = pretrained_sd
        self.task_vectors = task_vectors

    def pareto_merge(self):
        model = self.model
        pretrained_sd = to_device(self.pretrained_sd, device, copy=True)
        task_vectors = {k: to_device(v, device) for k, v in self.task_vectors.items()}
        task_weights = self.task_weights

        # 2. Pareto optimization (MGDA)
        layer_wise_weights = layer_wise_fusion.get_layer_wise_weights(
            num_models=len(args.source), num_layers=len(pretrained_sd), init_values=0.2
        ).to(device)
        pretrained_model = deepcopy(model).to(device)
        pretrained_model.load_state_dict(pretrained_sd)
        remove_keys = [
            n
            for n, p in pretrained_model.state_dict(keep_vars=True).items()
            if not p.requires_grad
        ]
        merged_model = layer_wise_fusion.LayerWiseMergedModel(
            pretrained_model=pretrained_model,
            layer_wise_weights=layer_wise_weights,
            task_vectors=list(task_vectors.values()),
            clamp_weights=False,
        )
        optimizer = torch.optim.Adam(merged_model.parameters(), lr=args.lr)
        merged_model.merge_weights(remove_keys=remove_keys)
        # save the initial merged model
        self.save_checkpoint(
            merged_model,
            f"round={args.round_idx}_step=0_merged.pth",
            meta_info={
                "layer_wise_weights": layer_wise_weights.detach().cpu(),
                "beta": sol.detach().cpu(),
            },
        )
        for step_idx in tqdm(range(args.iters_per_epoch), "Pareto optimization"):
            all_grads = []
            for aux_task_name in tqdm(
                args.source, "computing grads on tasks", leave=False
            ):
                # if target task, minimize the loss on the target distribution
                # else if auxiliary task, minimize the loss on the joint distribution
                new_task_weights = deepcopy(task_weights)
                if aux_task_name not in args.target:
                    new_task_weights[aux_task_name] = 1

                outputs = compute_grad(
                    merged_model,
                    task_weights=new_task_weights,
                    train_loaders=self.train_loaders,
                    average_steps=5,
                    device=device,
                )
                grads = outputs["grads"]
                assert (
                    len(grads) == 1
                ), f"len(grads)={len(grads)}, type(grads)={type(grads)}"
                grads = grads[0].flatten()
                all_grads.append(grads)

                if aux_task_name in args.target:
                    wandb.log(
                        {
                            f"pareto_loss": outputs["loss"],
                            f"pareto_acc": outputs["acc"],
                        },
                        step=step_idx,
                    )

            sol, min_norm = MinNormSolver.find_min_norm_element(all_grads)
            if not isinstance(sol, torch.Tensor):
                sol = torch.from_numpy(sol)
            sol = sol.to(
                device=layer_wise_weights.device, dtype=layer_wise_weights.dtype
            )

            optimizer.zero_grad()
            grad = torch.stack(all_grads) * sol.view(-1, 1)
            merged_model.layer_wise_weights.grad = grad.sum(dim=0).view_as(
                layer_wise_weights
            )
            optimizer.step()

            wandb.log({f"min_norm": min_norm.item()}, step=step_idx)
            merged_model.merge_weights(remove_keys=remove_keys)

            if (step_idx + 1) % args.save_interval == 0:
                self.save_checkpoint(
                    merged_model,
                    f"round={args.round_idx}_step={step_idx+1}_merged.pth",
                    meta_info={
                        "layer_wise_weights": layer_wise_weights.detach().cpu(),
                        "beta": sol.detach().cpu(),
                    },
                )
        self.save_checkpoint(
            merged_model,
            f"round={args.round_idx}_merged.pth",
            meta_info={
                "layer_wise_weights": layer_wise_weights.detach().cpu(),
                "beta": sol.detach().cpu(),
            },
        )


def parse_args():
    parser = pareto_atl_base.parser
    # dataset parameters
    parser.add_argument(
        "--run_name",
        type=str,
        default="pareto_atl_merge",
        help="name of the run (wandb)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="interval for saving the merged model",
    )
    parser.add_argument(
        "--round_idx",
        type=int,
        default=0,
        help="index of the round",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    program = ParetoATL_Merging(args)
    program.run()
