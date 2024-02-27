import argparse
import json
import logging
import math
from copy import deepcopy
from typing import Dict, Optional

import torch.nn.functional as F
import torch.utils.data
from rich.traceback import install

import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL._record import _PerformanceMeter
from LibMTL.aspp import DeepLabHead
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.create_dataset import Cityscapes, NYUv2
from LibMTL.model import resnet_dilated
from LibMTL.utils import *

install()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
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


class ParetoATL:
    def __init__(self, params):
        # the command-line arguments.
        self.params = params
        self.params.tasks = params.source_tasks

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

    def load_checkpoint(self, filename: str):
        logger.info(f"Loading model from {self.get_checkpoint_path(filename)}")
        checkpoint = torch.load(self.get_checkpoint_path(filename), map_location="cpu")
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
        print(
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

        self.base_result = base_result
        self.nyuv2_train_loader = nyuv2_train_loader
        self.nyuv2_val_loader = nyuv2_val_loader
        self.nyuv2_test_loader = nyuv2_test_loader
