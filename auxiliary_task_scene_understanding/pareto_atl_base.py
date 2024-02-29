import argparse
import json
import logging
import math
from copy import deepcopy
from typing import Dict, Optional, Callable, Tuple, List

import torch.nn.functional as F
import torch.utils.data
from rich.traceback import install
import wandb
from tqdm import tqdm

import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL._record import _PerformanceMeter
from LibMTL.aspp import DeepLabHead
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.create_dataset import Cityscapes, NYUv2
from LibMTL.model import resnet_dilated
from LibMTL.utils import *
from rich.logging import RichHandler
from torch.utils.data import DataLoader

install()


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


parser = LibMTL_args
parser.add_argument(
    "--aug", action="store_true", default=False, help="data augmentation"
)
parser.add_argument("--train_mode", default="train", type=str, help="train")
parser.add_argument("--train_bs", default=8, type=int, help="batch size for training")
parser.add_argument("--test_bs", default=8, type=int, help="batch size for test")
parser.add_argument("--dataset", default="NYUv2", choices=["NYUv2", "Cityscapes"])
parser.add_argument(
    "--dataset_path", default="data/nyuv2", type=str, help="dataset path"
)
parser.add_argument("--n_epochs", default=200, type=int)
parser.add_argument(
    "--log_dir",
    type=str,
    default="debug",
    help="Where to save logs, checkpoints and debugging images.",
)
parser.add_argument(
    "--phase",
    type=str,
    default="train",
    choices=["train", "test", "analysis"],
    help="When phase is 'test', only test the model."
    "When phase is 'analysis', only analysis the model.",
)
parser.add_argument(
    "--source_tasks", default=["segmentation", "depth", "normal"], nargs="+"
)
parser.add_argument("--target_tasks", default=None, nargs="+")
parser.add_argument(
    "--alphas", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0], nargs="+"
)
parser.add_argument("--epoch_step", type=int, default=1)
parser.add_argument("--pruning_epochs", type=int, default=10)
parser.add_argument("--topk", type=int, default=[0, 1, 2], nargs="+")
parser.add_argument(
    "--task_weights",
    default=[1, 1, 1],
    type=float,
    nargs="+",
    help="weight specific for EW",
)
parser.add_argument("--pretrained", default=None)


class NYUtrainer(Trainer):
    def __init__(
        self,
        task_dict,
        weighting,
        architecture: str,
        encoder_class: Callable[[], nn.Module],
        decoders,
        rep_grad,
        multi_input,
        optim_param,
        scheduler_param,
        save_path: str,
        base_result,
        img_size,
        **kwargs,
    ):
        super(NYUtrainer, self).__init__(
            task_dict=task_dict,
            weighting=weighting_method.__dict__[weighting],
            architecture=architecture_method.__dict__[architecture],
            encoder_class=encoder_class,
            decoders=decoders,
            rep_grad=rep_grad,
            multi_input=multi_input,
            optim_param=optim_param,
            scheduler_param=scheduler_param,
            **kwargs,
        )
        self.img_size = img_size
        self.base_result = base_result
        self.best_result = None
        self.best_epoch = None
        self.weight = {
            "segmentation": [1.0, 1.0],
            "depth": [0.0, 0.0],
            "normal": [0.0, 0.0, 1.0, 1.0, 1.0],
        }
        self.best_improvement = -math.inf
        self.meter = _PerformanceMeter(
            self.task_dict, self.multi_input, base_result=None
        )

        self.save_path = save_path

    def process_preds(self, preds):
        for task in self.task_name:
            preds[task] = F.interpolate(
                preds[task], self.img_size, mode="bilinear", align_corners=True
            )
        return preds

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "adamw": torch.optim.AdamW,
        }
        scheduler_dict = {
            "exp": torch.optim.lr_scheduler.ExponentialLR,
            "step": torch.optim.lr_scheduler.StepLR,
            "cos": torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        optim_arg = {k: v for k, v in optim_param.items() if k != "optim"}
        self.optimizer: torch.optim.Optimizer = optim_dict[optim_param["optim"]](
            self.model.parameters(), **optim_arg
        )
        if scheduler_param is not None:
            scheduler_arg = {
                k: v for k, v in scheduler_param.items() if k != "scheduler"
            }
            self.scheduler = scheduler_dict[scheduler_param["scheduler"]](
                self.optimizer, **scheduler_arg
            )
        else:
            self.scheduler = None

    def train(
        self,
        train_dataloaders: DataLoader,
        val_dataloaders: Optional[DataLoader],
        epoch_start: int,
        epoch_end: int,
        return_weight=False,
    ):
        r"""The training process of multi-task learning.

            Args:
                train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                                If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                                Otherwise, it is a single dataloader which returns data and a dictionary \
                                of name-label pairs in each iteration.

                test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                                The same structure with ``train_dataloaders``.
                epochs (int): The total training epochs.
                return_weight (bool): if ``True``, the loss weights will be returned.
            """
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        step_idx = 0
        for epoch in tqdm(
            range(epoch_start, epoch_end),
            "training epochs",
            leave=False,
            dynamic_ncols=True,
        ):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time("begin")
            for batch_index in tqdm(
                range(train_batch), "training batchs", leave=False, dynamic_ncols=True
            ):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
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

                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs["weight_args"])
                self.optimizer.step()

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
        self.display_best_result()

    def display_best_result(self):
        print("=" * 40)
        print(
            "Best Result: Epoch {}, result {} improvement: {}".format(
                self.best_epoch, self.best_result, self.best_improvement
            )
        )
        print("=" * 40)

    def val(self, val_dataloaders, epoch=None):
        self.meter.has_val = True
        new_result = self.test(val_dataloaders, epoch, mode="val")

        from LibMTL.utils import count_improvement

        improvement = count_improvement(self.base_result, new_result, self.weight)
        print("improvement", improvement)
        if improvement > self.best_improvement:
            self.save()
            self.best_result = new_result
            self.best_epoch = epoch
        self.best_improvement = max(improvement, self.best_improvement)

    def test(self, test_dataloaders, epoch=None, mode="test"):
        r"""The test process of multi-task learning.

            Args:
                test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                                it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                                dataloader which returns data and a dictionary of name-label pairs in each iteration.
                epoch (int, default=None): The current epoch.
            """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time("begin")
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in tqdm(range(test_batch), 'testing'):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time("end")
        self.meter.get_score()
        results = deepcopy(self.meter.results)
        # self.meter.display(epoch=epoch, mode=mode)
        self.meter.reinit()
        results = {
            task_name: results[task_name] for task_name in self.params.target_tasks
        }
        return results

    def load_best(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location="cpu"))

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)


class ParetoATL:
    def __init__(self, params):
        # the command-line arguments.
        self.params = params
        self.params.tasks = params.source_tasks
        self.log_dir = params.log_dir

        # the configuration of hyperparameters, optimizier, and learning rate scheduler.
        self.kwargs, self.optim_param, self.scheduler_param = prepare_args(params)

    def get_checkpoint_path(self, filename: Optional[str] = None):
        dirname = os.path.join(self.log_dir, "checkpoints")
        if filename is not None:
            return os.path.join(dirname, filename)
        return dirname

    def save_checkpoint(
        self,
        model: nn.Module,
        filename: str,
        meta_info: Optional[Dict] = None,
    ):
        """save state dict to checkpoint at `self.checkpoint_path(filename)`"""
        state_dict = model.state_dict()
        for k in state_dict:
            state_dict[k] = state_dict[k].cpu()
        if not os.path.exists(self.get_checkpoint_path()):
            os.makedirs(self.get_checkpoint_path())

        data = {"state_dict": state_dict}
        if meta_info is not None:
            data.update(meta_info)
        logger.info(f"Saving model to {self.get_checkpoint_path(filename)}")
        torch.save(data, self.get_checkpoint_path(filename))

    def load_checkpoint(self, filename: str, map_location="cpu"):
        logger.info(f"Loading model from {self.get_checkpoint_path(filename)}")
        checkpoint = torch.load(
            self.get_checkpoint_path(filename), map_location=map_location
        )
        return checkpoint

    def load_data(self):
        params = self.params

        # prepare dataloaders
        if params.dataset == "NYUv2":
            dataset = NYUv2
            base_result = {
                "segmentation": [0.5251, 0.7478],
                "depth": [0.4047, 0.1719],
                "normal": [22.6744, 15.9096, 0.3717, 0.6353, 0.7418],
            }
        else:
            dataset = Cityscapes
            base_result = {"segmentation": [0.7401, 0.9316], "depth": [0.0125, 27.77]}
        nyuv2_train_set = dataset(
            root=params.dataset_path, mode="train", augmentation=params.aug
        )
        nyuv2_val_set = dataset(
            root=params.dataset_path, mode="val", augmentation=False
        )
        nyuv2_test_set = dataset(
            root=params.dataset_path, mode="test", augmentation=False
        )
        logger.info(
            "train: {} val: {} test: {}".format(
                len(nyuv2_train_set), len(nyuv2_val_set), len(nyuv2_test_set)
            )
        )

        nyuv2_train_loader = torch.utils.data.DataLoader(
            dataset=nyuv2_train_set,
            batch_size=params.train_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        nyuv2_val_loader = torch.utils.data.DataLoader(
            dataset=nyuv2_val_set,
            batch_size=params.test_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        nyuv2_test_loader = torch.utils.data.DataLoader(
            dataset=nyuv2_test_set,
            batch_size=params.test_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # define tasks
        task_dict = {
            "segmentation": {
                "metrics": ["mIoU", "pixAcc"],
                "metrics_fn": SegMetric(num_classes=nyuv2_train_set.num_classes),
                "loss_fn": SegLoss(),
                "weight": [1, 1],
            },
            "depth": {
                "metrics": ["abs_err", "rel_err"],
                "metrics_fn": DepthMetric(),
                "loss_fn": DepthLoss(),
                "weight": [0, 0],
            },
            "normal": {
                "metrics": ["mean", "median", "<11.25", "<22.5", "<30"],
                "metrics_fn": NormalMetric(),
                "loss_fn": NormalLoss(),
                "weight": [0, 0, 1, 1, 1],
            },
            "noise": {
                "metrics": ["dummy metric"],
                "metrics_fn": NoiseMetric(),
                "loss_fn": NoiseLoss(),
                "weight": [1],
            },
        }
        source_task_dict = {
            task_name: task_dict[task_name] for task_name in params.source_tasks
        }
        target_task_dict = {
            task_name: task_dict[task_name] for task_name in params.target_tasks
        }
        base_result = {
            task_name: base_result[task_name] for task_name in params.target_tasks
        }

        self.nyuv2_train_set = nyuv2_train_set
        self.nyuv2_val_set = nyuv2_val_set
        self.nyuv2_test_set = nyuv2_test_set

        self.nyuv2_train_loader = nyuv2_train_loader
        self.nyuv2_val_loader = nyuv2_val_loader
        self.nyuv2_test_loader = nyuv2_test_loader

        self.task_dict = task_dict
        self.source_task_dict = source_task_dict
        self.target_task_dict = target_task_dict
        self.base_result = base_result

    def load_model(self):
        params = self.params
        nyuv2_train_set = self.nyuv2_train_set
        source_task_dict = self.source_task_dict

        # define encoder and decoders
        def encoder_class():
            return resnet_dilated("resnet50")

        num_out_channels = nyuv2_train_set.num_out_channels
        decoders = nn.ModuleDict(
            {
                task: DeepLabHead(2048, num_out_channels[task])
                for task in list(source_task_dict.keys())
            }
        )
        print(f"{type(decoders)=}")

        self.encoder_class = encoder_class
        self.decoders = decoders
