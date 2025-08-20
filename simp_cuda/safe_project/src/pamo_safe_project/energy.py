from __future__ import annotations
import numpy as np
import warp as wp
import igl
import scipy.sparse as sp
from typing import Union, List

from .kernels.energy_kernels.collision_energy import collision_hess_dx_kernel

from .kernels.ccd_kernels import *
from .kernels.energy_kernels.distance_energy import *
from .kernels.energy_kernels.elastic_energy import *
from .kernels.energy_kernels.lb_curvature_energy import *
from .kernels.energy_kernels.hinge_energy import *
from .kernels.energy_kernels.collision_energy import *
from .kernels.energy_kernels.contact_detection import *
from .kernels.geometry_kernels import *
from .kernels.utils_kernels import *

# from .kernels.utils_kernels import block_spd_project_kernel
from .utils import wp_slice
from .utils import stage3_logger as logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import Stage3System


mat23 = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


class EnergyCalculator:
    name = None
    
    def __init__(self, system: Stage3System):
        self.system = system

    def preprocess(self, V, F):
        pass

    def compute_energy(self, x: wp.array, energy: wp.array):
        """Compute the energy and add onto the energy array"""
        raise NotImplementedError

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        """
        ALWAYS compute the gradient and add onto the grad array;
        ALWAYS compute the hess_diag and add onto the hess_diag array;
        MAY also compute the hessian (blocks) and other needed buffers
            and store them into the member arrays of the energy calculator
        """
        raise NotImplementedError

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        """Compute hessian * dx and add onto the hess_dx array"""
        raise NotImplementedError


class DummyEnergyCalculator(EnergyCalculator):
    """
    Dummy energy calculator for testing.
    Always return energy E = 0.5 * x^T * A * x + b^T * x + c,
        where A is a random symmetric positive semi-definite matrix,
        b, c are random vectors.
    """

    name = "Dummy"

    def __init__(self, system: Stage3System):
        super().__init__(system)
        
        s = self.system
        c = s.config

        MP = c.max_particles
        assert (
            MP <= 1024
        ), f"Dummy energy calculator only supports max_particles <= 1024, got max_particles = {MP}"

        self.A_np = np.random.randn(MP * 3, MP * 3)
        self.A_np = self.A_np.T @ self.A_np
        self.b_np = np.random.randn(MP * 3)
        self.c_np = np.random.randn(1)

    def compute_energy(self, x: wp.array, energy: wp.array):
        x_np = x.numpy().reshape(-1)
        energy_np = (
            0.5 * np.dot(x_np, np.dot(self.A_np, x_np))
            + np.dot(self.b_np, x_np)
            + self.c_np
        )
        energy.assign(energy_np)

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        x_np = x.numpy().reshape(-1)
        grad_np = self.A_np @ x_np + self.b_np
        hess_diag_np = np.diag(self.A_np)

        grad.assign(grad_np.reshape(-1, 3) * grad_coeff)
        hess_diag.assign(hess_diag_np.reshape(-1, 3))

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        dx_np = dx.numpy().reshape(-1)
        hess_dx_np = self.A_np @ dx_np

        hess_dx.assign(hess_dx_np.reshape(-1, 3))


class Mesh2GTDistanceEnergyCalculator(EnergyCalculator):
    name = "M2GT"
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config

        MP = c.max_particles

        # Distance energy
        with wp.ScopedDevice(s.device):
            self.target = wp.zeros(MP, dtype=wp.vec3)  # L2 distance energy target
            self.target_distance = wp.zeros(MP, dtype=wp.float32)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.target_distance.fill_(1e9)
        self.target.fill_(1e9)
        wp.launch(
            kernel=update_min_pt_distance_kernel,
            dim=(s.n_particles, s.n_gt_triangles),
            inputs=[x, s.gt_vertices, s.gt_triangles],
            outputs=[self.target_distance],
            device=s.device,
        )
        wp.launch(
            kernel=update_closest_point_on_target_kernel,
            dim=(s.n_particles, s.n_gt_triangles),
            inputs=[x, s.gt_vertices, s.gt_triangles, self.target_distance],
            outputs=[self.target],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_kernel,
            dim=s.n_particles,
            inputs=[x, self.target, s.voronoi_areas, c.mesh2gt_dist_stiffness],
            outputs=[energy],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.target,
                s.voronoi_areas,
                c.mesh2gt_dist_stiffness,
                grad_coeff,
            ],
            outputs=[grad, hess_diag],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_hess_dx_kernel,
            dim=s.n_particles,
            inputs=[s.voronoi_areas, c.mesh2gt_dist_stiffness, dx],
            outputs=[hess_dx],
            device=s.device,
        )
        
        
class Mesh2GTDistanceBvhEnergyCalculator(Mesh2GTDistanceEnergyCalculator):
    name = "M2GT"
    def __init__(self, system: Stage3System):
        super().__init__(system)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.target.fill_(1e9)
        wp.launch(
            kernel=update_closest_point_on_target_bvh_kernel,
            dim=s.n_particles,
            inputs=[x, s.gt_mesh_bvh.id],
            outputs=[self.target],
            device=s.device,
        )


class GT2MeshDistanceEnergyCalculator(EnergyCalculator):
    """
    Distance between remeshed triangles and ground truth points.
    The ground truth points are uniformly sampled on the ground truth mesh.
    """
    name = "GT2M"

    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config

        MS_GT = c.max_gt_samples

        with wp.ScopedDevice(s.device):
            self.closest_tids = wp.zeros(MS_GT, dtype=wp.int32)
            self.pt_types = wp.zeros(MS_GT, dtype=wp.int32)
            self.d = wp.zeros(MS_GT, dtype=wp.float32)
            self.dd_dx = wp.zeros((MS_GT, 4), dtype=wp.vec3)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.d.fill_(1e9)
        wp.launch(
            kernel=update_min_pt_distance_kernel,
            dim=(s.n_gt_samples, s.n_triangles),
            inputs=[s.gt_samples, x, s.triangles],
            outputs=[self.d],
            device=s.device,
        )
        wp.launch(
            kernel=update_closest_triangle_on_target_kernel,
            dim=(s.n_gt_samples, s.n_triangles),
            inputs=[s.gt_samples, x, s.triangles, self.d],
            outputs=[self.closest_tids],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_kernel,
            dim=s.n_gt_samples,
            inputs=[
                x,
                s.gt_samples,
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
            ],
            outputs=[
                self.pt_types,
                self.d,
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_diff_kernel,
            dim=s.n_gt_samples,
            inputs=[
                x,
                s.gt_samples,
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
                self.pt_types,
                self.d,
                grad_coeff,
            ],
            outputs=[self.dd_dx, grad, hess_diag],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_hess_dx_kernel,
            dim=s.n_gt_samples,
            inputs=[
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
                self.dd_dx,
                dx,
            ],
            outputs=[hess_dx],
            device=s.device,
        )


class GT2MeshDistanceBvhEnergyCalculator(GT2MeshDistanceEnergyCalculator):
    """
    Distance between remeshed triangles and ground truth points.
    The ground truth points are uniformly sampled on the ground truth mesh.
    """
    name = "GT2M"

    def __init__(self, system: Stage3System):
        super().__init__(system)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        s.mesh_bvh.refit()

        wp.launch(
            kernel=update_closest_triangle_on_target_bvh_kernel,
            dim=s.n_gt_samples,
            inputs=[s.gt_samples, s.mesh_bvh.id],
            outputs=[self.closest_tids],
            device=s.device,
        )

class ElasticEnergyCalculator(EnergyCalculator):
    name = "Elas"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config

        MP = c.max_particles
        MT = 2 * MP + 1024

        self.mu = c.elas_young_modulus / (2 * (1 + c.elas_poisson_ratio))
        self.la = (
            c.elas_young_modulus
            * c.elas_poisson_ratio
            / ((1 + c.elas_poisson_ratio) * (1 - 2 * c.elas_poisson_ratio))
        )

        with wp.ScopedDevice(s.device):
            self.areas = wp.zeros(MT, dtype=wp.float32)
            self.inv_Dm = wp.zeros(MT, dtype=wp.mat22)
            """ xm_local = Bm * x
                Dm = Bm * [xm1 - xm0, xm2 - xm0]
                Ds = Bs * [xs1 - xs0, xs2 - xs0]
                W = 0.5 * det(Dm)
                F = Ds * inv(Dm)  # local deformation gradient, \partial (Bs * xs) / \partial (Bm * xm)
                [f1, f2] = H = -W * P(F) * inv(Dm).T  # local force, \partial E / \partial (Bs * xs)
                # Note that Psi(RF) = Psi(F), d Psi(F) = d Psi(RF) = P(PF) : (R * dF) = R^T * P(RF) : dF = P(F) : dF
                # So P(RF) = R * P(F)
            """

    def preprocess(self, V, F):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_preprocess_kernel,
            dim=s.n_triangles,
            inputs=[
                s.q_rest,
                s.triangles,
            ],
            outputs=[
                self.areas,
                self.inv_Dm,
            ],
            device=s.device,
        )

        if c.debug:
            areas_np = self.areas.numpy()[: s.n_triangles]
            logger.debug(
                f"[ElasticEnergyCalculator] min area = {np.min(areas_np)}, max area = {np.max(areas_np)}"
            )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_energy_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_diff_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_hess_dx_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )


class LBCurvatureEnergyCalculator(EnergyCalculator):
    name = "Curv"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        
        s = self.system
        c = s.config

        MP = c.max_particles
        MT = 2 * MP + 1024
        ME = MT // 2 * 3
        M_LB_nnz = 2 * ME + MP

        with wp.ScopedDevice(s.device):
            self.LB_nnz = 0

            self.LB_indices = wp.zeros(M_LB_nnz, dtype=wp.int32)
            self.LB_indptr = wp.zeros(MP + 1, dtype=wp.int32)
            self.LB_data = wp.zeros(
                M_LB_nnz, dtype=wp.float32
            )  # Laplace-Beltrami operator
            self.curv_rest = wp.zeros(MP, dtype=wp.float32)
            self.curv = wp.zeros(MP, dtype=wp.vec3)

    def preprocess(self, V, F):
        s = self.system
        c = s.config

        cotmatrix = igl.cotmatrix(V, F)
        massmatrix = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        voronoi_areas = massmatrix.diagonal()
        LB_sp: sp.csr_matrix = -(cotmatrix / voronoi_areas[:, None]).tocsr()  # [N, N]
        curv_rest = np.linalg.norm(LB_sp.dot(V), axis=1)

        NP = s.n_particles
        assert LB_sp.shape == (NP, NP), f"LB_sp shape {LB_sp.shape} != ({NP}, {NP})"
        self.LB_nnz = LB_sp.nnz

        wp_slice(self.LB_indices, 0, self.LB_nnz).assign(LB_sp.indices)
        wp_slice(self.LB_indptr, 0, NP + 1).assign(LB_sp.indptr)
        wp_slice(self.LB_data, 0, self.LB_nnz).assign(LB_sp.data)
        wp_slice(self.curv_rest, 0, NP).assign(curv_rest)

        # logger.info(f"[BendingEnergyCalculator] Mean curvature: \n{curv_rest}")
        # for i in range(100):
        #     logger.info(f"[BendingEnergyCalculator] Mean curvature [{i}]: {curv_rest[i]}")

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_energy_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
                grad_coeff,
            ],
            outputs=[
                self.curv,
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_hess_dx_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
                self.curv,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )


class HingeEnergyCalculator(EnergyCalculator):
    name = "Bend"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config

        MP = c.max_particles
        MT = 2 * MP + 1024
        ME = MT // 2 * 3

        with wp.ScopedDevice(s.device):
            self.rest_angles = wp.zeros(ME, dtype=wp.float32)
            self.rest_elens = wp.zeros(ME, dtype=wp.float32)
            self.blocks = wp.zeros((ME, 4, 4), dtype=wp.mat33)
            self.block_indices = wp.zeros((ME, 4), dtype=wp.int32)

    def preprocess(self, V, F):
        s = self.system
        c = s.config

        hinge_counter = wp.zeros(1, dtype=wp.int32)

        wp.launch(
            kernel=hinge_preprocess_slow_kernel,
            dim=(s.n_triangles, s.n_triangles),
            inputs=[
                s.q_rest,
                s.triangles,
            ],
            outputs=[
                hinge_counter,
                self.block_indices,
                self.rest_angles,
                self.rest_elens,
            ],
            device=s.device,
        )

        n_hinges = hinge_counter.numpy()[0]
        assert (
            n_hinges == s.n_edges
        ), f"Number of hinges {n_hinges} != number of edges {s.n_edges}"

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=hinge_energy_kernel,
            dim=s.n_edges,
            inputs=[
                x,
                self.block_indices,
                self.rest_angles,
                self.rest_elens,
                c.hinge_stiffness,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config
        
        wp.launch(
            kernel=hinge_diff_kernel,
            dim=s.n_edges,
            inputs=[
                x,
                self.block_indices,
                self.rest_angles,
                self.rest_elens,
                c.hinge_stiffness,
                grad_coeff,
            ],
            outputs=[
                self.blocks,
                grad,
                hess_diag,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=block_spd_project_kernel,
            dim=s.n_edges,
            inputs=[
                self.blocks,
                c.spd_max_iters,
            ],
            device=s.device,
        )
        
    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config
        
        wp.launch(
            kernel=hinge_hess_dx_kernel,
            dim=s.n_edges,
            inputs=[
                self.block_indices,
                self.blocks,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )

class CollisionEnergyCalculator(EnergyCalculator):
    name = "Coll"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        
        s = self.system
        c = s.config

        MP = c.max_particles
        MB = c.max_blocks

        # Collision energy
        with wp.ScopedDevice(s.device):
            self.contact_counter = wp.zeros(1, dtype=wp.int32)
            self.block_indices = wp.zeros((MB, 4), dtype=wp.int32)
            self.block_types = wp.zeros((MB, 2), dtype=wp.int32)  # 2 hierarchy levels
            self.d = wp.zeros(MB, dtype=wp.float32)  # distance
            self.dd_dx = wp.zeros((MB, 4), dtype=wp.vec3)
            
    def preprocess(self, V, F):
        c = self.system.config
        self.radius = c.contact_detection_radius

    def detect_contact(self, x: wp.array):
        s = self.system
        c = s.config

        self.contact_counter.zero_()
        wp.launch(
            kernel=detect_pt_contact_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                self.contact_counter,
                c.max_blocks,
                x,
                s.triangles,
                c.d_hat,
                self.radius,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=detect_ee_contact_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                self.contact_counter,
                c.max_blocks,
                x,
                s.edges,
                c.d_hat,
                self.radius,
                c.ee_classify_thres,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )

        # if c.debug:
        n_contacts = self.contact_counter.numpy()[0]
        logger.debug(f"Detected {n_contacts} contact pairs")
        if n_contacts > c.max_blocks:
            logger.warning(
                f"Number of contacts ({n_contacts}) exceeds max_blocks ({c.max_blocks})"
            )
            self.radius /= 2.0
            logger.info(f"Retrying with smaller detection radius: {self.radius}")
            self.detect_contact(x)
        # assert (
        #     n_contacts <= c.max_blocks
        # ), f"Number of contacts ({n_contacts}) exceeds max_blocks ({c.max_blocks})"
        # logger.debug(f"Detected {n_contacts} contact pairs")

    def ccd(self, x: wp.array, v: wp.array, ccd_step: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=accd_kernel,
            dim=c.max_blocks,
            inputs=[
                self.contact_counter,
                x,
                v,
                self.block_types,
                self.block_indices,
                c.ccd_slackness,
                c.ccd_thickness,
                c.ccd_max_iters,
                c.ee_classify_thres,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_energy_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                self.contact_counter,
                self.block_types,
                self.block_indices,
                c.coll_stiffness,
                c.d_hat,
                c.ee_classify_thres,
            ],
            outputs=[
                self.d,
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_diff_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                self.contact_counter,
                self.block_types,
                self.block_indices,
                c.coll_stiffness,
                c.d_hat,
                self.d,
                grad_coeff,
            ],
            outputs=[
                self.dd_dx,
                grad,
                hess_diag,
            ],
            device=s.device,
        )

        # if c.debug:
        #     blocks_np = self.blocks.numpy()[: self.contact_counter.numpy()[0]]
        #     assert not np.isnan(blocks_np).any(), "NaN in Hessian blocks"

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_hess_dx_kernel,
            dim=c.max_blocks,
            inputs=[
                self.contact_counter,
                self.block_indices,
                self.d,
                c.coll_stiffness,
                c.d_hat,
                self.dd_dx,
                dx,
            ],
            outputs=[hess_dx],
            device=s.device,
        )


class CollisionBvhEnergyCalculator(CollisionEnergyCalculator):
    name = "Coll"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        
    def preprocess(self, V, F):
        super().preprocess(V, F)

    def detect_contact(self, x: wp.array):
        s = self.system
        c = s.config

        self.contact_counter.zero_()
        wp.launch(
            kernel=detect_pt_contact_bvh_kernel,
            dim=s.n_particles,
            inputs=[
                s.tri_bvh.id,
                self.contact_counter,
                c.max_blocks,
                x,
                s.triangles,
                c.d_hat,
                self.radius,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=detect_ee_contact_bvh_kernel,
            dim=s.n_edges,
            inputs=[
                s.edge_bvh.id,
                self.contact_counter,
                c.max_blocks,
                x,
                s.edges,
                c.d_hat,
                self.radius,
                c.ee_classify_thres,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )

        # if c.debug:
        n_contacts = self.contact_counter.numpy()[0]
        logger.debug(f"Detected {n_contacts} contact pairs")
        if n_contacts > c.max_blocks:
            logger.warning(
                f"Number of contacts ({n_contacts}) exceeds max_blocks ({c.max_blocks})"
            )
            self.radius /= 2.0
            logger.info(f"Retrying with smaller detection radius: {self.radius}")
            self.detect_contact(x)
            

class CollisionWoBufferEnergyCalculator(EnergyCalculator):
    name = "collision (wo buffer)"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config
        
    def ccd(self, x: wp.array, v: wp.array, ccd_step: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=accd_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                v,
                s.triangles,
                c.ccd_slackness,
                c.ccd_thickness,
                c.contact_detection_radius,
                c.ccd_max_iters,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=accd_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                v,
                s.edges,
                c.ccd_slackness,
                c.ccd_thickness,
                c.contact_detection_radius,
                c.ee_classify_thres,
                c.ccd_max_iters,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_energy_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.coll_stiffness,
                c.d_hat,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_energy_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.coll_stiffness,
                c.ee_classify_thres,
                c.d_hat,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_diff_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.coll_stiffness,
                c.d_hat,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_diff_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.coll_stiffness,
                c.d_hat,
                c.ee_classify_thres,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_hess_dx_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.d_hat,
                c.coll_stiffness,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_hess_dx_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.d_hat,
                c.ee_classify_thres,
                c.coll_stiffness,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )

