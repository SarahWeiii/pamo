import stage3
from stage3.utils import stage3_logger as logger
import run_stage3
import time
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Run Stage 3")
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default="data/wostage3_0510",
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
        default="output/stage3_0510_grid_search",
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
    return parser.parse_args()


arg = get_args()

config = stage3.Stage3Config()
system = stage3.Stage3System(config, "cuda:0")

# to_search = {
#     "elas_young_modulus": [1.0, 1e-1, 1e-2],
#     "hinge_stiffness": [1e0, 1e-1, 1e-2, 1e-3, 1e-4],
#     "mesh2gt_dist_stiffness": [1e3],
#     "gt2mesh_dist_stiffness": [1e3],
# }

to_search = {
    "elas_young_modulus": [3e-1, 1e-1, 3e-2],
    "hinge_stiffness": [3e-2, 1e-2, 3e-3],
    "mesh2gt_dist_stiffness": [1e3],
    "gt2mesh_dist_stiffness": [1e3],
}

best_CD = 1e9
best_config = None

output_root_dir = arg.output_dir

def dfs(config, to_search, i):
    global arg
    global best_CD
    global best_config
    global system
    
    if i == len(to_search):
        arg.output_dir = os.path.join(
            output_root_dir,
            f"elas{config.elas_young_modulus}_hinge{config.hinge_stiffness}_m2g{config.mesh2gt_dist_stiffness}_g2m{config.gt2mesh_dist_stiffness}",
        )
        CD_mean, HD_mean = run_stage3.run_all(arg, config, system)
        print("\n" + "=" * 30 + "\n" + f"CD: {CD_mean}, HD: {HD_mean}")
        if CD_mean < best_CD:
            best_CD = CD_mean
            best_config = dict(config.__dict__)
            print(f"New best CD: {best_CD}")
        else:
            print(f"Current best CD: {best_CD}")
        print(f"Best config: {best_config}")
        print("=" * 30 + "\n")
        return
    key = list(to_search.keys())[i]
    for value in to_search[key]:
        setattr(config, key, value)
        dfs(config, to_search, i + 1)


dfs(config, to_search, 0)
