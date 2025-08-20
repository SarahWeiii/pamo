import os
import argparse
import numpy as np
import trimesh
import time

import stage3
from stage3.utils import stage3_logger as logger


def get_args():
    parser = argparse.ArgumentParser(description="Test Stage 2")
    parser.add_argument("--id", type=str, help="ID of the mesh", required=True)
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
        default="output/mesh",
        help="Directory of output meshes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arg = get_args()
    logger.setLevel("DEBUG")
    
    time_start = time.time()

    stage2_path = os.path.join(arg.stage2_dir, f"{arg.id}_processed.obj")
    gt_path = os.path.join(arg.gt_dir, f"{arg.id}.ply")
    if not os.path.exists(gt_path):
        for ext in [".obj", ".stl", ".off"]:
            gt_path = os.path.join(arg.gt_dir, f"{arg.id}{ext}")
            if os.path.exists(gt_path):
                break

    stage2_mesh = trimesh.load(stage2_path)
    gt_mesh = trimesh.load(gt_path)
    
    trimesh_load_time = time.time() - time_start
    logger.info(f"Trimesh loaded in {trimesh_load_time:.3f}s")

    config = stage3.config.Stage3Config()
    # config.debug = True
    # config.use_cuda_graph = False
    
    stage3_V, stage3_F = stage3.process(
        gt_mesh.vertices,
        gt_mesh.faces,
        stage2_mesh.vertices,
        stage2_mesh.faces,
        5,
        config=config,
        # eval=True,
        # eval=False,
    )
    
    stage3_mesh = trimesh.Trimesh(stage3_V, stage3_F, process=False)
    os.makedirs(arg.output_dir, exist_ok=True)
    output_path = os.path.join(arg.output_dir, f"{arg.id}_stage3.obj")
    stage3_mesh.export(output_path)
    print(f"Output mesh saved to {output_path}")
    
    
    
    
