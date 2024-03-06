# Debug Args: --arch HPS --gpu_id 1 --scheduler step --source_tasks segmentation depth normal --target_tasks segmentation --log logs/nyuv2/forkmerge/segementation --epoch_step 30 --seed 0 --task_name depth --round_idx 0
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

import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL._record import _PerformanceMeter
from LibMTL.aspp import DeepLabHead
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.create_dataset import Cityscapes, NYUv2
from LibMTL.model import resnet_dilated
from LibMTL.utils import *
import pandas as pd

import pareto_atl_base

logger = logging.getLogger(__name__)


class ParetoATL_Testing(pareto_atl_base.ParetoATL):
    def run(self):
        params = self.params
        assert os.path.exists(params.ckpt), f"{params.ckpt} does not exist"

        self.load_data()
        self.load_model()

        self.test()

    def load_model(self):
        from pareto_atl_base import NYUtrainer

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

        new_task_dict = deepcopy(self.target_task_dict)
        logger.info(f"new_task_dict: {new_task_dict}")
        kwargs["weight_args"]["weights"] = len(new_task_dict) * [1]
        logger.info(f'{kwargs["weight_args"]["weights"] = }')

        trainer = NYUtrainer(
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
            save_path=None,
            img_size=self.nyuv2_train_set.image_size,
            **deepcopy(kwargs),
        )
        trainer.params = params
        self.trainer = trainer

    @property
    def model(self) -> nn.Module:
        return self.trainer.model

    def test(self):
        params = self.params
        trainer = self.trainer

        ckpt = torch.load(params.ckpt, map_location=trainer.device)
        state_dict = ckpt["state_dict"]
        self.model.load_state_dict(state_dict)

        final_results = trainer.test(self.nyuv2_test_loader, None, mode="test")
        logger.info(final_results)

        if params.output_path is not None:
            save_path = params.output_path
        else:
            save_path = params.ckpt.replace(".pth", ".csv")
        pd.DataFrame(final_results).to_csv(save_path)
        print(f"Results saved to {save_path}")
        print(f"Results: {pd.DataFrame(final_results)}")


def parser_args():
    parser = pareto_atl_base.parser
    parser.add_argument(
        "--ckpt",
        type=str,
        help="ckpt_path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output path for the results. If not specified, the results will be saved in the same directory as the ckpt file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    pareto_atl = ParetoATL_Testing(args)
    pareto_atl.run()
