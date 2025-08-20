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
        default="data/wostage3_0514",
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
        default="output/stage3_0513_0514",
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
    parser.add_argument(
        "--save_curve",
        action="store_true",
        help="Save the CD and HD curves",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="mesh ids to skip",
    )
    return parser.parse_args()


def run_stage3(ids, stage2_paths, gt_paths, arg, config=None, system=None):
    skip_ids = arg.skip.split(",") if arg.skip else []
    
    if hasattr(arg, "resume") and arg.resume:
        if not arg.resume:
            raise ValueError(f"Output directory {arg.output_dir} already exists")
    
    os.makedirs(arg.output_dir, exist_ok=True)
    mesh_dir = os.path.join(arg.output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    csv_path = f"{arg.output_dir}/stage3_eval.csv"
    
    save_curve = arg.save_curve if hasattr(arg, "save_curve") else False
    if save_curve:
        cd_curve_path = f"{arg.output_dir}/cd_curve.csv"
        hd_curve_path = f"{arg.output_dir}/hd_curve.csv"
    
    last_id = None
    if hasattr(arg, "resume") and arg.resume:
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1]
                last_id_repeat = last_line.split(",")[0]
                last_id = last_line.split(",")[1]
        else:
            raise ValueError(f"Tried to resume but CSV file {csv_path} does not exist")
    else:
        with open(csv_path, "w") as f:
            f.write("Repeat,ID,Stage 2 CD,Stage 2 HD,Stage 2 MA,Stage 3 CD,Stage 3 HD,Stage 3 MA,Vertices,Faces,Time(s)\n")
        if save_curve:
            with open(cd_curve_path, "w") as f:
                f.write("Repeat,ID,Stage2 CD, Stage3 CDs\n")
            with open(hd_curve_path, "w") as f:
                f.write("Repeat,ID,Stage2 HD, Stage3 HDs\n")

    if config is None:
        config = stage3.config.Stage3Config()
        
    # if "0.2" in arg.stage2_dir or "0.1" in arg.stage2_dir:
    #     logger.info("Using 0.2/0.1 config: setting detect_contact_every='step'")
    #     config.detect_contact_every = "step"
        
    if system is None:
        system = stage3.system.Stage3System(config, "cuda:0")
    else:
        system.clear()
        system.config = config
    # system = None
    
    config_path = os.path.join(arg.output_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(str(config.__dict__))

    mean_CDs_stage2 = []
    mean_HDs_stage2 = []
    mean_MAs_stage2 = []
    mean_CDs = []
    mean_HDs = []
    mean_MAs = []

    for R in range(1, arg.repeat + 1):
        mesh_dir_R = os.path.join(mesh_dir, f"repeat_{R}")
        if arg.save_mesh:
            os.makedirs(mesh_dir_R, exist_ok=True)
        
        CDs_stage2 = []
        HDs_stage2 = []
        MAs_stage2 = []
        CDs = []
        HDs = []
        MAs = []
    
        logger.info(f"Repeat {R}/{arg.repeat}")
        for id, stage2_path, gt_path in zip(ids, stage2_paths, gt_paths):
            if id in skip_ids:
                continue
            
            if last_id is not None and arg.resume and id <= last_id:
                continue
            
            logger.info(f"[ID: {id}] Processing...")
            
            time_start = time.time()

            stage2_mesh = trimesh.load(stage2_path)
            gt_mesh = trimesh.load(gt_path, force="mesh")

            trimesh_load_time = time.time() - time_start
            logger.debug(f"Trimesh loaded in {trimesh_load_time:.3f}s")

            # scale = 1.0 / np.max(gt_mesh.bounding_box.extents)
            scale_norm, t_norm = get_normalization_transform(gt_mesh.vertices)

            CD_stage2, HD_stage2 = compute_igl_CD_HD(
                gt_mesh.vertices * scale_norm + t_norm,
                gt_mesh.faces,
                stage2_mesh.vertices * scale_norm + t_norm,
                stage2_mesh.faces,
            )
            MA_stage2 = compute_min_angle(trimesh.Trimesh(stage2_mesh.vertices, stage2_mesh.faces, process=False)) * 180 / np.pi
            CDs_stage2.append(CD_stage2)
            HDs_stage2.append(HD_stage2)
            MAs_stage2.append(MA_stage2)
            
            time_start = time.time()
            
            if save_curve:
                stage3_V, stage3_F, CDs_stage3, HDs_stage3 = stage3.process(
                    gt_mesh.vertices,
                    gt_mesh.faces,
                    stage2_mesh.vertices,
                    stage2_mesh.faces,
                    5,
                    config=config,
                    system=system,
                    eval=False,
                    return_curve=True,
                )
                with open(cd_curve_path, "a") as f:
                    f.write(f"{R},{id},{CD_stage2},{','.join(map(str, CDs_stage3))}\n")
                with open(hd_curve_path, "a") as f:
                    f.write(f"{R},{id},{HD_stage2},{','.join(map(str, HDs_stage3))}\n")
            else:
                stage3_V, stage3_F = stage3.process(
                    gt_mesh.vertices,
                    gt_mesh.faces,
                    stage2_mesh.vertices,
                    stage2_mesh.faces,
                    5,
                    config=config,
                    system=system,
                    eval=False,
                    return_curve=False,
                )
            
            stage3_time = time.time() - time_start

            CD_stage3, HD_stage3 = compute_igl_CD_HD(
                gt_mesh.vertices * scale_norm + t_norm, 
                gt_mesh.faces, 
                stage3_V * scale_norm + t_norm, 
                stage3_F,
            )
            MA_stage3 = compute_min_angle(trimesh.Trimesh(stage3_V, stage3_F, process=False)) * 180 / np.pi
            CDs.append(CD_stage3)
            HDs.append(HD_stage3)
            MAs.append(MA_stage3)

            stage3_mesh = trimesh.Trimesh(stage3_V, stage3_F, process=False)
            if arg.save_mesh:
                stage3_mesh.export(f"{mesh_dir_R}/{id}_stage3.obj")

            logger.info(
                f"[ID: {id}] Stage 2 CD: {CD_stage2:.3e}, HD: {HD_stage2:.3e}, MA: {MA_stage2:.3e}; Stage 3 CD: {CD_stage3:.3e}, HD: {HD_stage3:.3e}, MA: {MA_stage3:.3e}, V: {stage3_V.shape[0]}, F: {stage3_F.shape[0]}, Time(s): {stage3_time:.3f}"
            )
            with open(csv_path, "a") as f:
                f.write(f"{R},{id},{CD_stage2},{HD_stage2},{MA_stage2},{CD_stage3},{HD_stage3},{MA_stage3},{stage3_V.shape[0]},{stage3_F.shape[0]},{stage3_time}\n")

        mean_CD_stage2 = np.mean(CDs_stage2)
        mean_HD_stage2 = np.mean(HDs_stage2)
        mean_MA_stage2 = np.mean(MAs_stage2)
        mean_CD = np.mean(CDs)
        mean_HD = np.mean(HDs)
        mean_MA = np.mean(MAs)
        logger.info(f"Mean CD: {mean_CD:.3e}, HD: {mean_HD:.3e}, MA: {mean_MA:.3e}")
        with open(csv_path, "a") as f:
            f.write(f"{R},Mean,{mean_CD_stage2},{mean_HD_stage2},{mean_MA_stage2},{mean_CD},{mean_HD},{mean_MA}\n")
        
        mean_CDs_stage2.append(mean_CD_stage2)
        mean_HDs_stage2.append(mean_HD_stage2)
        mean_MAs_stage2.append(mean_MA_stage2)
        mean_CDs.append(mean_CD)
        mean_HDs.append(mean_HD)
        mean_MAs.append(mean_MA)
        
    R_mean_CD_stage2 = np.mean(mean_CDs_stage2)
    R_mean_HD_stage2 = np.mean(mean_HDs_stage2)
    R_mean_MA_stage2 = np.mean(mean_MAs_stage2)
    R_mean_CD = np.mean(mean_CDs)
    R_mean_HD = np.mean(mean_HDs)
    R_mean_MA = np.mean(mean_MAs)
    logger.info(f"Repeat Mean CD: {R_mean_CD:.3e}, HD: {R_mean_HD:.3e}, MA: {R_mean_MA:.3e}")
    with open(csv_path, "a") as f:
        f.write(f"Mean,Mean,{R_mean_CD_stage2},{R_mean_HD_stage2},{R_mean_MA_stage2},{R_mean_CD},{R_mean_HD},{R_mean_MA}\n")
    
    return mean_CD, mean_HD


def run_all(arg, config=None, system=None):
    ids, stage2_paths, gt_paths = get_ids_and_paths(arg.stage2_dir, arg.gt_dir)
    return run_stage3(ids, stage2_paths, gt_paths, arg, config=config, system=system)


if __name__ == "__main__":
    arg = get_args()
    logger.setLevel("DEBUG")
    
    run_all(arg)
