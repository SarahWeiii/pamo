import stage3
from stage3.utils import stage3_logger as logger
from stage3.energy import *
import run_stage3
import time
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Run Stage 3")
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default="data/wostage3_0515",
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
        default="output/stage3_0515_ablation",
        help="Root directory of output",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Save the output meshes",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="mesh ids to skip",
    )
    return parser.parse_args()


arg = get_args()
output_root_dir = arg.output_dir

config = stage3.Stage3Config()
# config.debug = True
# config.cg_precond = "none"
# config.use_cuda_graph = False
logger.setLevel("DEBUG")
system = stage3.Stage3System(config, "cuda:0")

all_energies = list(config.energy_calcs)

# exp_name = "all"
# arg.output_dir = os.path.join(output_root_dir, exp_name)
# CD_mean, HD_mean = run_stage3.run_all(arg, config)

for ec_off in all_energies:
# ec_off = Mesh2GTDistanceEnergyCalculator
# for ec_off in [HingeEnergyCalculator, GT2MeshDistanceEnergyCalculator, Mesh2GTDistanceEnergyCalculator, ElasticEnergyCalculator, CollisionEnergyCalculator]:

    if ec_off == CollisionEnergyCalculator:
        continue

    exp_name = "wo_" + ec_off.name
    if ec_off == Mesh2GTDistanceEnergyCalculator:
        config.cg_precond = "none"
    else:
        config.cg_precond = "jacobi"
    arg.output_dir = os.path.join(output_root_dir, exp_name)
    config.energy_calcs = [ec for ec in all_energies if ec != ec_off]
    CD_wo, HD_wo = run_stage3.run_all(arg, config, system)
