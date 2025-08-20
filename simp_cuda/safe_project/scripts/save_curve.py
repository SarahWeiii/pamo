import os
import argparse
import numpy as np
import trimesh
import time

import stage3
from stage3.metrics import *
from stage3.utils import stage3_logger as logger
from stage3.energy import *
from stage3.processing import get_normalization_transform

from test_stage2 import get_ids_and_paths


def get_args():
    parser = argparse.ArgumentParser(description="Run Stage 3")
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default="data/wostage3_0511",
        help="Directory containing Stage 2 outputs",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="data/dataset_baseline_ply",
        help="Directory containing ground truth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/stage3_0512",
        help="Directory of output meshes",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Save the output meshes",
    )
    parser.add_argument(
        "--resume",  # don't use, behavior changed
        action="store_true",
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation",
    )
    return parser.parse_args()



