"""
Test the Hausdorff distance and Chamfer distance between
Stage 2 outputs and ground truth.  

Motivation: Get familiar with the evaluation metrics and
data formats. 

Author: Xiaodi

Date: 05/02/2024
"""

import argparse
import os

import numpy as np
import trimesh

from stage3.metrics import *
from stage3.processing import get_normalization_transform


def get_args():
    parser = argparse.ArgumentParser(description="Test Stage 2")
    parser.add_argument(
        "--stage2_dir",
        type=str,
        default="data/wostage3_0506_avoid_sk",
        help="Directory containing Stage 2 outputs",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="data/dataset_baseline_ply",
        help="Directory containing ground truth",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/test_stage2.csv",
        help="Path to evaluation results (CSV file)",
    )
    parser.add_argument(
        "--rolopm",
        action="store_true",
    )
    return parser.parse_args()


def get_ids_and_paths(stage2_dir, gt_dir):
    """
    input: directories of stage 2 outputs and ground truth
        which contain files like {id}_{description}.{obj or ply}
    output: 3 lists of strings that contain the id, the paths to
        stage 2 outputs, the paths to ground truth, all sorted
        according to the id (as strings)
    """
    stage2_ids, stage2_paths = [], []
    for filename in os.listdir(stage2_dir):
        fileid, ext = os.path.splitext(filename)
        if ext in [".ply", ".obj", ".stl", ".off"]:
            fileid = fileid.split("_")[0]
            stage2_ids.append(fileid)
            stage2_paths.append(os.path.join(stage2_dir, filename))
    stage2_joined = sorted(list(zip(stage2_ids, stage2_paths)), key=lambda x: x[0])
    stage2_ids, stage2_paths = zip(*stage2_joined)

    gt_ids, gt_paths = [], []
    for filename in os.listdir(gt_dir):
        fileid, ext = os.path.splitext(filename)
        if ext in [".ply", ".obj", ".stl", ".off"]:
            fileid = fileid.split("_")[0]
            if not fileid in stage2_ids:
                continue
            gt_ids.append(fileid)
            gt_paths.append(os.path.join(gt_dir, filename))
    gt_joined = sorted(list(zip(gt_ids, gt_paths)), key=lambda x: x[0])
    gt_ids, gt_paths = zip(*gt_joined)

    if len(stage2_ids) != len(gt_ids):
        for id in stage2_ids:
            if not id in gt_ids:
                print(f"ID {id} in stage 2 but not in ground truth")

    assert len(stage2_ids) == len(
        gt_ids
    ), "Number of files do not match: stage 2 {} files vs gt {} files".format(
        len(stage2_ids), len(gt_ids)
    )
    assert stage2_ids == gt_ids, "IDs do not match"

    return stage2_ids, stage2_paths, gt_paths


def evaluate_stage2(ids, stage2_paths, gt_paths, output_path, rolopm=False):
    with open(output_path, "w") as f:
        f.write("ID,CD,HD,MA,BT_1,BT_5,BT_10,V,F,Time\n")

    CDs = []
    HDs = []
    min_angles = []
    BTs = []
    nVs = []
    nFs = []
    times = []

    for id, stage2_path, gt_path in zip(ids, stage2_paths, gt_paths):
        stage2_mesh = trimesh.load(stage2_path)
        gt_mesh = trimesh.load(gt_path)

        # scale = 1.0 / np.max(gt_mesh.bounding_box.extents)
        scale_norm, t_norm = get_normalization_transform(gt_mesh.vertices)
        
        # stage2_mesh.apply_scale(scale)
        # gt_mesh.apply_scale(scale)

        # chamfer_distance, hausdorff_distance = compute_trimesh_CD_HD(gt_mesh, stage2_mesh)
        chamfer_distance, hausdorff_distance = compute_igl_CD_HD(
            gt_mesh.vertices * scale_norm + t_norm, 
            gt_mesh.faces, 
            stage2_mesh.vertices * scale_norm + t_norm, 
            stage2_mesh.faces,
        )
        min_angle = compute_min_angle(stage2_mesh) * 180 / np.pi
        bt = compute_bad_tri_ratio(stage2_mesh.vertices, stage2_mesh.faces)
        nV = stage2_mesh.vertices.shape[0]
        nF = stage2_mesh.triangles.shape[0]
        
        if rolopm:
            time_txt_path = os.path.join(os.path.dirname(stage2_path), f"{id}_ours_time.txt")
            with open(time_txt_path, "r") as f:
                time = float(f.readlines()[-1].strip().split(' ')[-1])
        else:
            time = -1
            
        CDs.append(chamfer_distance)
        HDs.append(hausdorff_distance)
        min_angles.append(min_angle)
        BTs.append(bt)
        nVs.append(nV)
        nFs.append(nF)
        times.append(time)

        print(
            f"ID: {id}, CD: {chamfer_distance: .3e}, HD: {hausdorff_distance: .3e}, MA: {min_angle:.3e}, BTs: {bt}, V: {nV}, F: {nF}"
        )
        with open(output_path, "a") as f:
            f.write(
                f"{id},{chamfer_distance},{hausdorff_distance},{min_angle},{bt[0]},{bt[1]},{bt[2]},{nV},{nF},{time}\n"
            )

    min_bt = np.mean(np.array(BTs), axis=0)

    with open(output_path, "a") as f:
        f.write(f"Mean,{np.mean(CDs)},{np.mean(HDs)},{np.mean(min_angles)},{min_bt[0]},{min_bt[1]},{min_bt[2]},{np.mean(nVs)},{np.mean(nFs)},{np.mean(times)}\n")


if __name__ == "__main__":
    arg = get_args()
    ids, stage2_paths, gt_paths = get_ids_and_paths(arg.stage2_dir, arg.gt_dir)
    output_dir = arg.output_path
    evaluate_stage2(ids, stage2_paths, gt_paths, arg.output_path, rolopm=arg.rolopm)
