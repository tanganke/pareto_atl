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
import pandas as pd
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
logger = logging.getLogger(__name__)

class ParetoATL_Testing(pareto_atl_base.ParetoATL):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def run(self):
        args = self.args
        self.load_data()
        self.load_model()

        model = deepcopy(self.model).to(device)
        logger.info(f"Loading checkpoint from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)["state_dict"]
        model.load_state_dict(ckpt)

        acc_dict = utils.validate_all(
            self.test_loaders, model, args=args, device=device
        )
        # save the results as csv, to the same directory as the checkpoint, but with a different ext name
        if args.output_path is not None:
            csv_path = args.output_path
        else:
            csv_path = args.ckpt_path.replace(".pth", ".csv")
        acc_df = pd.DataFrame({k: [v] for k, v in acc_dict.items()})
        acc_df.to_csv(csv_path, index=False)
        print(f"Validation results saved to {csv_path}")
        print(acc_df)


def parse_args():
    parser = pareto_atl_base.parser
    # dataset parameters
    parser.add_argument("--ckpt_path", type=str, help="path to the checkpoint")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output path of results (csv file). If None, save to the same directory as the checkpoint with a different ext name.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    pareto_atl = ParetoATL_Testing(args)
    pareto_atl.run()
