import warp as wp
import numpy as np
import trimesh
import igl
import scipy.sparse as sp
import time
import gc

from .system import Stage3System
from .config import Stage3Config
from .utils import stage3_logger as logger
from .metrics import *


def get_normalization_transform(V):
    center = np.mean(V, axis=0)
    extents = np.max(V, axis=0) - np.min(V, axis=0)
    scale = 1.0 / np.max(extents)
    t = -center * scale
    return scale, t


def process(
    gt_V,
    gt_F,
    stage2_V,
    stage2_F,
    n_iters,
    system: Stage3System = None,
    config: Stage3Config = None,
    eval=False,
    return_curve=False,
):      
    """
    Reuse system if provided, otherwise create a new one with the given
    config (if provided) or a default one (Stage3Config()).
    """
    # ------------------------ Initialize System ------------------------ #
    time_start = time.time()

    if system is None:
    # if True:
        gc.collect()
        if config is None:
            config = Stage3Config()
        system = Stage3System(config, "cuda:0")
    else:
        system.clear()

    system_init_time = time.time() - time_start
    logger.debug(f"System initialized in {system_init_time:.3f}s")

    # ------------------------ Mesh Preprocess ------------------------ #
    time_start = time.time()

    # Normalize the geometry
    scale_norm, t_norm = get_normalization_transform(gt_V)
    gt_V = gt_V * scale_norm + t_norm
    stage2_V = stage2_V * scale_norm + t_norm

    system.register_mesh(gt_V, gt_F, stage2_V, stage2_F)
    
    geometry_preprocess_time = time.time() - time_start
    logger.debug(f"Preprocessed mesh in {geometry_preprocess_time:.3f}s")
    
    # ------------------------ Register Mesh in System ------------------------ #

    if eval:
        # gt_mesh = trimesh.Trimesh(gt_V, gt_F, process=False)
        # stage2_mesh = trimesh.Trimesh(stage2_V, stage2_F, process=False)
        # cd, hd = compute_trimesh_CD_HD(gt_mesh, stage2_mesh)
        cd, hd = compute_igl_CD_HD(gt_V, gt_F, stage2_V, stage2_F)
        logger.info(f"Initial CD: {cd:.3e}, HD: {hd:.3e}")

    step_time = 0.0
    
    if return_curve:
        CDs = []
        HDs = []

    for i in range(n_iters):
        time_start = time.time()
        
        if return_curve:
            system.step(eval=True, eval_CDs=CDs, eval_HDs=HDs, gt_V=gt_V, gt_F=gt_F, eval_F=stage2_F)
        else:
            system.step()
        
        step_time += time.time() - time_start
        
        if eval:
            stage3_V = system.get_vertices().astype(np.float64)
            # stage2_mesh = trimesh.Trimesh(stage2_V, stage2_F, process=False)
            # cd, hd = compute_trimesh_CD_HD(gt_mesh, stage2_mesh)
            cd, hd = compute_igl_CD_HD(gt_V, gt_F, stage3_V, stage2_F)
            logger.info(f"Iter {i}: CD: {cd:.3e}, HD: {hd:.3e}")

    logger.debug(f"System stepped {n_iters} steps in {step_time:.3f}s")
    
    # ------------------------ Mesh Postprocess ------------------------ #
    stage3_V = system.get_vertices()
    stage3_F = stage2_F
    stage3_V = (stage3_V - t_norm) / scale_norm

    if return_curve:
        return stage3_V, stage3_F, CDs, HDs

    return stage3_V, stage3_F
