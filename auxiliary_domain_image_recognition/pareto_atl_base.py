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
from rich.traceback import install

install()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# dataset parameters
parser.add_argument("root", metavar="DIR", help="root path of dataset")
parser.add_argument(
    "-d",
    "--data",
    metavar="DATA",
    default="DomainNet",
    choices=utils.get_dataset_names(),
    help="dataset: " + " | ".join(utils.get_dataset_names()) + " (default: DomainNet)",
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
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
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
    "--alphas", type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0], nargs="+"
)
parser.add_argument("--epoch_step", type=int, default=5)
parser.add_argument("--topk", type=int, default=[0, 3, 5], nargs="+")
parser.add_argument("--pruning_epochs", type=int, default=1)
parser.add_argument("--pretrained", default=None)


def parse_args():
    args = parser.parse_args()
    return args


class ParetoATL:
    def __init__(self, args):
        self.args = args
        self.log_dir = args.log_dir

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

    def load_checkpoint(self, filename: str):
        logger.info(f"Loading model from {self.get_checkpoint_path(filename)}")
        checkpoint = torch.load(filename, map_location="cpu")
        return checkpoint

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
        )
        if args.pretrained is not None:
            print("Loading from ", args.pretrained)
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        print(heads)

        self.model = model

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


if __name__ == "__main__":
    args = parse_args()
    program = ParetoATL(args)
    program.run()
