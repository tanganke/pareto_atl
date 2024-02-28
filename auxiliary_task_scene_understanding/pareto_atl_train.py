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

import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL._record import _PerformanceMeter
from LibMTL.aspp import DeepLabHead
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.create_dataset import Cityscapes, NYUv2
from LibMTL.model import resnet_dilated
from LibMTL.utils import *


import pareto_atl_base

logger = logging.getLogger(__name__)


class ParetoATL_training(pareto_atl_base.ParetoATL):
    def __init__(self, params):
        super().__init__(params)

    def run(self):
        params = self.params

        self.load_data()
        self.load_model()

        self.local_trainging()

        self.cleanup()

    def cleanup(self):
        pass

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

        aux_task_name = params.task_name
        new_task_dict = deepcopy(self.target_task_dict)
        if aux_task_name not in params.target_tasks:
            new_task_dict[aux_task_name] = self.task_dict[aux_task_name]
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

    def local_trainging(self):
        params = self.params
        trainer = self.trainer

        if params.round_idx != 0:
            ckpt = self.load_checkpoint(
                f"round={params.round_idx-1}_merged.pth", map_location=trainer.device
            )
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict)

        trainer.train(self.nyuv2_train_loader, None, 0, params.epoch_step)

        self.save_checkpoint(
            self.model, f"round={params.round_idx}_task={params.task_name}_trained.pth"
        )


def parser_args():
    parser = pareto_atl_base.parser
    parser.add_argument(
        "--round_idx", type=int, default=0, help="index of current round"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="segmentation",
        help="name of current training task",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    pareto_atl = ParetoATL_training(args)
    pareto_atl.run()
