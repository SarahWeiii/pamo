"""
    References:
    - "IPC paper": Li, Minchen, Zachary Ferguson, Teseo Schneider, Timothy R. Langlois, Denis Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. "Incremental potential contact: intersection-and inversion-free, large-deformation dynamics." ACM Trans. Graph. 39, no. 4 (2020): 49.
    - "CG slides": Xu, Zhiliang. "ACMS 40212/60212: Advanced Scientific Computing, Lecture 8: Fast Linear Solvers (Part 5)." (https://www3.nd.edu/~zxu2/acms60212-40212/Lec-09-5.pdf)
    - "FEM tutorial": Sifakis, Eftychios. "FEM simulation of 3D deformable solids: a practitioner's guide to theory, discretization and model reduction. Part One: The classical FEM method and discretization methodology." In Acm siggraph 2012 courses, pp. 1-50. 2012.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import Stage3System

import numpy as np
import warp as wp

from .kernels.solver_kernels import *
from .kernels.utils_kernels import *
from .utils import stage3_logger as logger
from .utils import wp_slice


class CGSolver:
    def __init__(self, system: Stage3System):
        self.system = system
        config = system.config

        self.graph = None

        MP = config.max_particles
        MB = config.max_blocks

        with wp.ScopedDevice(self.system.device):
            self.r = wp.zeros(MP, dtype=wp.vec3)
            self.v = wp.zeros(MP, dtype=wp.vec3)
            self.A_v = wp.zeros(MP, dtype=wp.vec3)
            self.v_A_v = wp.zeros(1, dtype=wp.float32)

            self.z = wp.zeros(MP, dtype=wp.vec3)
            self.z_r_last = wp.zeros(1, dtype=wp.float32)
            self.z_r = wp.zeros(1, dtype=wp.float32)
            
    def clear(self):
        self.graph = None

    def _launch_main_loop(self):
        s = self.system
        c = s.config
        NP = s.n_particles

        for cg_iter in range(c.n_cg_iters):
            self.A_v.zero_()
            s._compute_hess_dx(self.v, self.A_v)
            
            self.v_A_v.zero_()
            wp.launch(
                kernel=compute_dot_kernel,
                dim=NP,
                inputs=[self.v, self.A_v],
                outputs=[self.v_A_v],
            )
            
            if c.debug and not c.use_cuda_graph:
                v_A_v_val = self.v_A_v.numpy()[0]
                # logger.debug(f"v^T A v = {v_A_v_val}")
                if v_A_v_val < 0:
                    logger.warning(f"v^T A v = {v_A_v_val} < 0, Hessian is not positive definite")
            
            wp.copy(self.z_r_last, self.z_r, count=1)
            self.z_r.zero_()
            wp.launch(
                kernel=update_p_r_z_compute_zr_kernel,
                dim=NP,
                inputs=[
                    self.v,
                    self.A_v,
                    self.v_A_v,
                    self.z_r_last,
                    s.hess_diag,
                ],
                outputs=[
                    s.p,
                    self.r,
                    self.z,
                    self.z_r,
                ],
            )
            
            wp.launch(
                kernel=update_v_kernel,
                dim=NP,
                inputs=[self.z, self.z_r_last, self.z_r, self.v],
            )
            

    def solve(self):
        """
        Copy from self.system:
            - Gradient: self.system.grad => self.r
            - Diagonal of Hessian: self.system.hess_diag
        Call energy caculators to compute:
            - Hessian-vector product: => self.A_v
        Save the solution of A p = b into self.system.p
        """

        s = self.system
        c = s.config
        NP = s.n_particles
        
        if c.debug and not c.use_cuda_graph:
            hess_diag_np = s.hess_diag.numpy()
            if np.any(np.isnan(hess_diag_np)):
                nan_indices = np.where(np.isnan(hess_diag_np))[0]
                logger.error(f"NaN found in Hessian diagonal at indices: {nan_indices}")
            
            assert not np.isnan(s.grad.numpy()).any(), "Initial gradient is NaN"
            assert not np.isnan(s.hess_diag.numpy()).any(), "Initial Hessian diagonal is NaN"

        with wp.ScopedDevice(self.system.device):
            s.p.zero_()
            wp.copy(self.r, s.grad, count=NP)

            # ------------------------- Preconditioning ------------------------- #
            if c.cg_precond == "jacobi":
                wp.launch(
                    kernel=compute_block_diag_inv_kernel,
                    dim=NP,
                    inputs=[s.hess_diag, self.r],
                    outputs=[self.z],
                )
            else:
                wp.copy(self.z, self.r, count=NP)

            wp.copy(self.v, self.z, count=NP)

            self.z_r.zero_()
            wp.launch(
                kernel=compute_dot_kernel,
                dim=NP,
                inputs=[self.z, self.r],
                outputs=[self.z_r],
            )
            z_r_start = self.z_r.numpy()[0]
            
            if c.debug and not c.use_cuda_graph:
                assert not np.isnan(self.r.numpy()).any(), "Initial residual is NaN"
                assert not np.isnan(z_r_start), "Initial residual norm z_r is NaN"

            # ------------------------- Main loop ------------------------- #
            if c.use_cuda_graph and self.graph is None:
                wp.capture_begin()
                self._launch_main_loop()
                self.graph = wp.capture_end()
            
            if c.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._launch_main_loop()
            
            if c.debug:
                z_r_end = self.z_r.numpy()[0]
                logger.debug(f"CG error from {z_r_start:.3e} to {z_r_end:.3e}, ratio: {z_r_end / z_r_start:.3e}")
            
            
            
