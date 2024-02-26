import abc
import argparse
import os
import random
import time
import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, cast

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ParetoForkMerge:
    def __init__(self, args):
        self.args = args

        wandb.init(
            project="pareto_atl",
            name="+".join(args.target),
            config={
                "training_lr": args.lr,
                "architecture": args.arch,
                "dataset": "DomainNet",
            },
        )
        print(args)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )

        cudnn.benchmark = True

    def cleanup(self):
        wandb.finish()

    def get_checkpoint_path(self, filename: Optional[str] = None):
        dirname = os.path.join(self.args.log, "checkpoints")
        if filename is not None:
            return os.path.join(dirname, filename)
        return dirname

    def save_model(
        self, model: nn.Module, filename: str, meta_info: Optional[Dict] = None
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
        torch.save(data, self.get_checkpoint_path(filename))

    def run(self):
        args = self.args
        self.load_data()
        self.load_model()

        task_weights = {name: 0 for name in args.source}
        for name in args.target:
            task_weights[name] = 1 / len(args.target)
        self.task_weights = task_weights

        for round_idx in tqdm(range(args.epochs)):
            self.round_idx = round_idx
            self.local_training()
            self.pareto_merge()
            self.validate()

        self.cleanup()

    def load_data(self):
        args = self.args

        # Data loading code
        train_transform = utils.get_train_transform(
            args.train_resizing,
            random_horizontal_flip=not args.no_hflip,
            random_color_jitter=False,
            resize_size=args.resize_size,
            norm_mean=args.norm_mean,
            norm_std=args.norm_std,
        )
        val_transform = utils.get_val_transform(
            args.val_resizing,
            resize_size=args.resize_size,
            norm_mean=args.norm_mean,
            norm_std=args.norm_std,
        )
        print("train_transform: ", train_transform)
        print("val_transform: ", val_transform)

        (
            train_source_datasets,
            train_target_datasets,
            val_datasets,
            test_datasets,
            self.num_classes,
            args.class_names,
        ) = utils.get_dataset(
            args.data,
            args.root,
            args.source,
            args.target,
            train_transform,
            val_transform,
        )
        val_loaders = {
            name: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            for name, dataset in val_datasets.items()
        }
        test_loaders = {
            name: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            for name, dataset in test_datasets.items()
        }

        train_loaders = {
            name: ForeverDataIterator(
                DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    drop_last=True,
                    pin_memory=True,
                )
            )
            for name, dataset in train_source_datasets.items()
        }

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders

    def load_model(self):
        args = self.args

        # create model
        print("=> using model '{}'".format(args.arch))
        backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        pool_layer = nn.Identity() if args.no_pool else None
        heads = nn.ModuleDict(
            {
                dataset_name: nn.Linear(backbone.out_features, self.num_classes)
                for dataset_name in args.source
            }
        )
        model = MultiOutputClassifier(
            backbone, heads, pool_layer=pool_layer, finetune=not args.scratch
        ).to(device)
        if args.pretrained is not None:
            print("Loading from ", args.pretrained)
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        print(heads)

        # resume from the best checkpoint
        if args.phase != "train":
            checkpoint = torch.load(
                self.get_checkpoint_path("best"), map_location="cpu"
            )
            model.load_state_dict(checkpoint)

        if args.phase == "test":
            acc1 = utils.validate_all(self.test_loaders, model, args, device)
            print(acc1)
            exit(0)

        self.model = model

    def local_training(self):
        model = self.model
        task_weights = self.task_weights

        self.theta_t = theta_t = deepcopy(model.state_dict())
        self.save_model(model, f"round={self.round_idx}_step=0.pth")
        # 1. local training
        tau: Dict[str, Dict[str, Tensor]] = {}  # task vector
        # copy models K+1 times and train them independently
        for aux_task_idx, aux_task_name in enumerate(args.source):
            # if target task, minimize the loss on the target distribution
            # else if auxiliary task, minimize the loss on the joint distribution
            new_task_weights = deepcopy(task_weights)
            if aux_task_name not in args.target:
                new_task_weights[aux_task_name] = 1
            model.load_state_dict(theta_t)
            print("local training:", new_task_weights)
            trainer = Trainer(
                f"{aux_task_name}/round={self.round_idx}",
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
                epoch=self.round_idx,
                iters_per_epoch=args.iters_per_epoch,
                args=args,
                device=device,
            )
            self.save_model(
                model,
                f"task={aux_task_name}_round={self.round_idx}_step={args.iters_per_epoch}.pth",
                meta_info={"task_weights": new_task_weights},
            )
            with torch.no_grad():
                tau[aux_task_name] = state_dict_sub(model.state_dict(), theta_t)
        self.tau = tau

    def compute_grad(
        self,
        model,
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
        for _ in range(average_steps):
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

    def pareto_merge(self):
        model = self.model
        theta_t = self.theta_t
        tau = self.tau
        task_weights = self.task_weights

        # 2. Pareto optimization (MGDA)
        layer_wise_weights = layer_wise_fusion.get_layer_wise_weights(
            num_models=len(args.source), num_layers=len(theta_t), init_values=0.2
        ).to(device)
        pretrained_model = deepcopy(model)
        pretrained_model.load_state_dict(theta_t)
        remove_keys = [
            n
            for n, p in pretrained_model.state_dict(keep_vars=True).items()
            if not p.requires_grad
        ]
        merged_model = layer_wise_fusion.LayerWiseMergedModel(
            pretrained_model=pretrained_model,
            layer_wise_weights=layer_wise_weights,
            task_vectors=list(tau.values()),
            clamp_weights=False,
        )
        optimizer = torch.optim.Adam(merged_model.parameters(), lr=args.lr)
        merged_model.merge_weights(remove_keys=remove_keys)
        for step_idx in tqdm(range(args.pareto_steps), "Pareto optimization"):
            all_grads = []
            for aux_task_name in args.source:
                # if target task, minimize the loss on the target distribution
                # else if auxiliary task, minimize the loss on the joint distribution
                new_task_weights = deepcopy(task_weights)
                if aux_task_name not in args.target:
                    new_task_weights[aux_task_name] = 1

                outputs = self.compute_grad(
                    merged_model,
                    task_weights=new_task_weights,
                    train_loaders=self.train_loaders,
                    average_steps=20,
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
                            f"{aux_task_name}/round={self.round_idx}/pareto_loss": outputs[
                                "loss"
                            ],
                            f"{aux_task_name}/round={self.round_idx}/pareto_acc": outputs[
                                "acc"
                            ],
                        },
                    )

            sol, min_norm = MinNormSolver.find_min_norm_element(all_grads)
            if not isinstance(sol, torch.Tensor):
                sol = torch.from_numpy(sol)
            sol = sol.to(
                device=layer_wise_weights.device, dtype=layer_wise_weights.dtype
            )

            optimizer.zero_grad()
            grad = torch.stack(all_grads) * sol.view(-1, 1)
            layer_wise_weights.grad = grad.sum(dim=0).view_as(layer_wise_weights)
            optimizer.step()

            wandb.log({"round={self.round_idx}/min_norm": min_norm.item()})
            merged_model.merge_weights(remove_keys=remove_keys)

        self.merged_state_dict = merged_model.merged_state_dict

    def validate(self):
        theta_t = self.theta_t
        model = self.model

        # 3. evaluate on validation data
        # update the model with the merged weights
        for k in self.merged_state_dict:
            theta_t[k] = self.merged_state_dict[k].detach()
        model.load_state_dict(theta_t)
        accs = utils.validate_all(self.val_loaders, model, args, device)
        wandb.log(
            {
                f"{task}/round={self.round_idx}/val_acc": acc
                for task, acc in accs.items()
            },
        )
        self.save_model(model, f"round={self.round_idx}_merged.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="forkmerge algorithm")
    # dataset parameters
    parser.add_argument("root", metavar="DIR", help="root path of dataset")
    parser.add_argument(
        "-d",
        "--data",
        metavar="DATA",
        default="DomainNet",
        choices=utils.get_dataset_names(),
        help="dataset: "
        + " | ".join(utils.get_dataset_names())
        + " (default: DomainNet)",
    )
    parser.add_argument("-s", "--source", help="source domain(s)", nargs="+")
    parser.add_argument("-t", "--target", help="target domain(s)", nargs="+")
    parser.add_argument("--train-resizing", type=str, default="default")
    parser.add_argument("--val-resizing", type=str, default="default")
    parser.add_argument(
        "--resize-size", type=int, default=224, help="the image size after resizing"
    )
    parser.add_argument(
        "--no-hflip",
        action="store_true",
        help="no random horizontal flipping during training",
    )
    parser.add_argument(
        "--norm-mean",
        type=float,
        nargs="+",
        default=(0.485, 0.456, 0.406),
        help="normalization mean",
    )
    parser.add_argument(
        "--norm-std",
        type=float,
        nargs="+",
        default=(0.229, 0.224, 0.225),
        help="normalization std",
    )
    # model parameters
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=utils.get_model_names(),
        help="backbone architecture: "
        + " | ".join(utils.get_model_names())
        + " (default: resnet18)",
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="no pool layer after the feature extractor.",
    )
    parser.add_argument(
        "--scratch", action="store_true", help="whether train from scratch."
    )
    # training parameters
    parser.add_argument(
        "-b",
        "--batch-size",
        default=48,
        type=int,
        metavar="N",
        help="mini-batch size (default: 48)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.01,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0005,
        type=float,
        metavar="W",
        help="weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-i",
        "--iters-per-epoch",
        default=2500,
        type=int,
        help="Number of iterations per epoch",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--per-class-eval",
        action="store_true",
        help="whether output per-class accuracy during evaluation",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="src_only",
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
        "--alphas", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0], nargs="+"
    )
    parser.add_argument("--epoch_step", type=int, default=5)
    parser.add_argument("--topk", type=int, default=[0, 3, 5], nargs="+")
    parser.add_argument("--pruning_epochs", type=int, default=1)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--pareto_steps", type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    program = ParetoForkMerge(args)
    program.run()
