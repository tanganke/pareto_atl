import abc
import argparse
import os
import random
import time
import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, cast
import logging

import lightning as L
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from forkmerge import SpecifiedProbMultiTaskSampler
from fusionlib.merge.wrapper import layer_wise_fusion
from fusionlib.utils.torch.state_dict_arithmetic import state_dict_sub
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
import pareto_atl_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        name: str,
        model: MultiOutputClassifier,
        save_path: str,
        optimizer_name: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        epochs: int,  # default: 20 --epochs
        task_to_unweighted_probs: Dict[str, float],
    ):
        self.name = name
        self.model = model
        self.save_path = save_path
        self.epochs = epochs
        self.optimizer = utils.get_optimizer(
            model, optimizer_name, lr, momentum=momentum, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.best_acc1 = {}
        self.task_to_unweighted_probs = task_to_unweighted_probs

    def train(
        self,
        train_loaders: Dict[str, DataLoader],
        epoch_start: int,
        epoch_end: int,  # default: 1 --pruning-epochs
        iters_per_epoch: int,  # default: 2500 -i --iters-per-epoch
        val_loaders: Dict[str, DataLoader],
        args: argparse.Namespace,
    ):
        dataset_sampler = SpecifiedProbMultiTaskSampler(
            train_loaders, 0, self.task_to_unweighted_probs
        )
        for epoch in range(epoch_start, epoch_end):
            print(self.scheduler.get_lr())
            # train for one epoch
            self.train_one_epoch(
                dataset_sampler,
                self.model,
                self.optimizer,
                epoch,
                iters_per_epoch,
                args,
                device,
            )
            self.scheduler.step()

            # evaluate on validation set
            acc1 = utils.validate_all(val_loaders, self.model, args, device)

            # remember best acc@1 and save checkpoint
            if sum(acc1.values()) > sum(self.best_acc1.values()):
                self.save()
                self.best_acc1 = acc1
            print(
                self.name,
                "Epoch:",
                epoch,
                "lr:",
                self.scheduler.get_lr()[0],
                "val_criteria:",
                round(sum(acc1.values()) / len(acc1), 3),
                "best_val_criteria:",
                round(sum(self.best_acc1.values()) / len(self.best_acc1), 3),
            )

    def train_one_epoch(
        self,
        dataset_sampler: SpecifiedProbMultiTaskSampler,
        model: MultiOutputClassifier,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        iters_per_epoch: int,
        args,
        device,
    ):
        # switch to train mode
        model.train()

        for step_idx in (
            pbar := tqdm(
                range(iters_per_epoch),
                f"training: {self.name}",
                dynamic_ncols=True,
                leave=False,
            )
        ):
            dataset_name, dataloader = dataset_sampler.pop()
            x, labels = next(dataloader)[:2]
            x = cast(Tensor, x).to(device)
            labels = cast(Tensor, labels).to(device)

            logits: Tensor = model(x, dataset_name)
            loss = F.cross_entropy(logits, labels)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_info = {
                "loss": loss.item(),
                "acc": logits.argmax(1).eq(labels).float().mean().item(),
            }
            pbar.set_postfix(log_info)
            wandb.log(
                {f"{self.name}/{k}": v for k, v in log_info.items()},
            )

    def test(self, val_loaders):
        acc1 = utils.validate_all(val_loaders, self.model, args, device)
        return sum(acc1.values()) / len(acc1), acc1

    def load_best(self):
        """
        Load the best model. load the model from `self.save_path`.
        """
        print(f"loading model weights from {self.save_path}")
        self.model.load_state_dict(torch.load(self.save_path, map_location="cpu"))

    def save(self):
        """
        Save the model to `self.save_path`.
        """
        print(f"saving model weights to {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)


def parse_args():
    parser = pareto_atl_base.parser
    parser.add_argument("--run_name", type=str, help="run name (for wandb)")
    parser.add_argument("--task_name", type=str, default=None, help="current task name")
    parser.add_argument("--round_idx", type=int, default=0, help="round index")
    args = parser.parse_args()
    return args


class ParetoATL_Training(pareto_atl_base.ParetoATL):
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
        if os.path.exists(
            self.get_checkpoint_path(
                f"round={args.round_idx}_task={args.task_name}.pth"
            )
        ):
            logger.info(f"task {args.task_name} already trained")
            return
        self.load_data()
        self.load_model()

        task_weights = {name: 0 for name in args.source}
        for name in args.target:
            task_weights[name] = 1 / len(args.target)
        self.task_weights = task_weights

        self.local_training()

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

        self.pretrained_sd = pretrained_sd

    def local_training(self):
        # 1. local training
        # a) load model from checkpoint `{log_dir}/checkpoints/round={round_idx-1}_merged.pth`
        #    if round_idx=0, then load the pre-trained model from torchvision.
        # b) train the K+1 models for each task and save the models to `{log_dir}/checkpoints/round={round_idx}_task={task_name}_trained.pth`

        model = self.model
        task_weights = self.task_weights
        pretrained_sd = self.pretrained_sd
        model.load_state_dict(pretrained_sd)
        model = model.to(device)

        # 1. local training
        # copy models K+1 times and train them independently
        aux_task_name = args.task_name

        # if target task, minimize the loss on the target distribution
        # else if auxiliary task, minimize the loss on the joint distribution
        new_task_weights = deepcopy(task_weights)
        if aux_task_name not in args.target:
            new_task_weights[aux_task_name] = 1

        logger.info("local training:", new_task_weights)
        trainer = Trainer(
            f"target={args.target[0]}_round={args.round_idx}_task={aux_task_name}",
            model,
            self.get_checkpoint_path(aux_task_name),
            args.optimizer,
            args.lr,
            args.momentum,
            args.wd,
            args.epochs,
            new_task_weights,
        )
        dataset_sampler = SpecifiedProbMultiTaskSampler(
            self.train_loaders, 0, new_task_weights
        )
        trainer.train_one_epoch(
            dataset_sampler,
            model,
            trainer.optimizer,
            epoch=args.round_idx,
            iters_per_epoch=args.iters_per_epoch,
            args=args,
            device=device,
        )
        self.save_checkpoint(
            model,
            f"round={args.round_idx}_task={aux_task_name}.pth",
            meta_info={"task_weights": new_task_weights},
        )


if __name__ == "__main__":
    args = parse_args()
    pareto_atl = ParetoATL_Training(args)
    pareto_atl.run()
