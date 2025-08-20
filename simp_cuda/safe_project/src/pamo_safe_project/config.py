import numpy as np
from typing import List
from .energy import *


class Stage3Config:
    def __init__(self):
        self.debug = False  # enable debug mode, may slow down the simulation
        self.seed = 0  # random seed for trimesh sampling
        
        # Memory parameters
        self.max_particles = 1 << 20
        self.max_blocks = 1 << 25
        self.max_gt_particles = 1 << 24
        self.max_gt_samples = 1 << 14

        # Solver parameters
        self.n_newton_iters = 10  # Number of Newton iterations per Step
        self.n_cg_iters = 40  # Number of CG iterations per Newton iteration
        self.cg_precond = "jacobi"  # ["none", "jacobi"]
        self.use_cuda_graph = True
        self.n_line_search_iters = 10
        
        self.detect_contact_every = "newton"  # ["step", "newton"]
        # self.detect_contact_every = "step"

        self.energy_calcs: List[EnergyCalculator] = [
            # Mesh2GTDistanceEnergyCalculator,
            Mesh2GTDistanceBvhEnergyCalculator,
            # GT2MeshDistanceEnergyCalculator,
            # GT2MeshDistanceBvhEnergyCalculator,
            ElasticEnergyCalculator,
            # LBCurvatureEnergyCalculator,
            HingeEnergyCalculator, 
            # CollisionEnergyCalculator,
            # CollisionWoBufferEnergyCalculator
            CollisionBvhEnergyCalculator,
        ]
        
        self.system_scale = 1.0  # scale the vertices in the system by this factor

        # Collision energy parameters
        self.d_hat = 1e-3
        self.coll_stiffness = 1e2
        self.contact_detection_radius = 1e-2
        self.ee_classify_thres = 1e-3
        # self.ee_classify_thres = 1e-4  # In distance classification, ban EE class if sin(theta) < thres, only keep PE and PP classes
        # self.ee_mollifier_thres = 3e-2  # In distance energy calculation, use mollifier if sin(theta) ≈ |e1 x e2| / (|e1||e2|) < thres
        
        # CCD parameters
        self.ccd_slackness = 0.7
        self.ccd_thickness = 1e-6
        self.ccd_max_iters = 100
        
        # SPD projection parameters
        self.spd_max_iters = 3
        
        # Elastic energy parameters
        self.elas_young_modulus = 1e-1
        self.elas_poisson_ratio = 0.0

        # Curvature energy parameters
        self.curv_stiffness = 1e-8
        
        # Hinge energy parameters
        self.hinge_stiffness = 1e-2

        # Distance (to target) energy parameters
        self.mesh2gt_dist_stiffness = 1e3
        self.gt2mesh_dist_stiffness = 1e3
        
        # self.min_angle_thres = 1e-2  # in degree
        self.min_angle_thres = 0.0
