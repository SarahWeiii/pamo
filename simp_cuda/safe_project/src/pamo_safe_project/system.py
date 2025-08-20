import numpy as np
import warp as wp
import trimesh
import igl
import scipy.sparse as sp
import logging

from .config import Stage3Config
from .energy import *
from .cg_solver import CGSolver
from .kernels.solver_kernels import line_search_kernel, clamp_p_kernel
from .utils import stage3_logger as logger
from .utils import wp_slice
from .metrics import compute_igl_CD_HD
from .geometry import *


class Stage3System:
    def __init__(self, config: Stage3Config = None, device="cuda:0"):
        wp.init()

        if config is None:
            config = Stage3Config()
        self.config = config
        self.device = device

        if config.debug:
            if config.use_cuda_graph:
                logger.warning(
                    "Debug mode is enabled but CUDA graph is also enabled; some debug information will not be generated"
                )
            if logger.level > logging.DEBUG:
                logger.warning(
                    "Debug mode is enabled but logger level is higher than DEBUG; some debug information will not be displayed"
                )

        self._init_counters()
        self._init_arrays()
        self._init_energy_calcs()

        self.cg_solver = CGSolver(self)
        self.line_search_graph = None

        self.do_nothing = False

    def _init_counters(self):
        self.n_particles = 0
        self.n_triangles = 0
        self.n_edges = 0
        self.n_gt_particles = 0
        self.n_gt_triangles = 0
        self.n_gt_samples = 0

    def _init_arrays(self):
        MP = self.config.max_particles
        MP_GT = self.config.max_gt_particles
        MS_GT = self.config.max_gt_samples
        MF = (
            self.config.max_particles * 2 + 1024
        )  # max number of faces, assume < 256 genus
        MF_GT = self.config.max_gt_particles * 2 + 1024
        ME = MF // 2 * 3  # max number of edges
        MB = self.config.max_blocks

        with wp.ScopedDevice(self.device):
            self.q = wp.zeros(MP, dtype=wp.vec3)
            self.q_prev_newton = wp.zeros(MP, dtype=wp.vec3)
            self.q_prev_detection = wp.zeros(MP, dtype=wp.vec3)
            self.q_rest = wp.zeros(MP, dtype=wp.vec3)
            self.p = wp.zeros(MP, dtype=wp.vec3)  # update on q in each Newton iteration

            # GT mesh data
            self.gt_vertices = wp.zeros(MP_GT, dtype=wp.vec3)
            self.gt_triangles = wp.zeros((MF_GT, 3), dtype=wp.int32)
            self.gt_samples = wp.zeros(MS_GT, dtype=wp.vec3)
            self.gt_sample_weights = wp.zeros(MS_GT, dtype=wp.float32)

            # Remeshed mesh data
            self.triangles = wp.zeros((MF, 3), dtype=wp.int32)
            self.edges = wp.zeros((ME, 2), dtype=wp.int32)
            self.voronoi_areas = wp.zeros(MP, dtype=wp.float32)

            # System-wise energy and its differentials
            self.energy = wp.zeros(1, dtype=wp.float32)
            self.energy_prev = wp.zeros(1, dtype=wp.float32)
            self.grad = wp.zeros(MP, dtype=wp.vec3)
            self.hess_diag = wp.zeros(MP, dtype=wp.vec3)

            # CCD
            self.ccd_step = wp.zeros(1, dtype=wp.float32)

            # # Curvature energy
            # self.curv_rest = wp.zeros(MP, dtype=wp.vec3)  # mean curvature at rest
            # self.curv_pids = wp.zeros(2 * ME + MP, dtype=wp.int32)
            # self.curv_indptr = wp.zeros(MP + 1, dtype=wp.int32)
            # self.curv_data = wp.zeros(2 * ME + MP, dtype=wp.float32)

            # BVH
            self.edge_lowers = wp.zeros(ME, dtype=wp.vec3)
            self.edge_uppers = wp.zeros(ME, dtype=wp.vec3)
            self.tri_lowers = wp.zeros(MF, dtype=wp.vec3)
            self.tri_uppers = wp.zeros(MF, dtype=wp.vec3)
            # self.gt_tri_lowers = wp.zeros(MF_GT, dtype=wp.vec3)
            # self.gt_tri_uppers = wp.zeros(MF_GT, dtype=wp.vec3)

    def clear(self):
        self._init_counters()
        self.cg_solver.clear()
        self.line_search_graph = None
        for k in self.config.energy_calcs:
            if not k in self.energy_calcs:
                self.energy_calcs[k] = k(self)
        self.do_nothing = False

    def _get_energy_calculator(self, cls: type):
        if cls in self.config.energy_calcs:
            return self.energy_calcs[cls]
        else:
            return None

    def _refit_edge_bvh(self, x):
        with wp.ScopedDevice(self.device):
            compute_edge_bounds(
                self.n_edges,
                x,
                self.edges,
                self.edge_lowers,
                self.edge_uppers,
                0.0,
            )
            self.edge_bvh.refit()
        # self.edge_bvh = wp.Bvh(
        #     wp_slice(self.edge_lowers, 0, self.n_edges),
        #     wp_slice(self.edge_uppers, 0, self.n_edges),
        # )
    
    def _refit_tri_bvh(self, x):
        with wp.ScopedDevice(self.device):
            compute_tri_bounds(
                self.n_triangles,
                x,
                self.triangles,
                self.tri_lowers,
                self.tri_uppers,
                0.0,
            )
            # logger.debug(f"(tri_lowers, tri_uppers): {np.hstack((self.tri_lowers.numpy()[:10], self.tri_uppers.numpy()[:10]))}")
            self.tri_bvh.refit()
        # self.tri_bvh = wp.Bvh(
        #     wp_slice(self.tri_lowers, 0, self.n_triangles),
        #     wp_slice(self.tri_uppers, 0, self.n_triangles),
        # )
        
    # def _refit_gt_tri_bvh(self):
    #     compute_tri_bounds(
    #         self.n_gt_triangles,
    #         self.gt_vertices,
    #         self.gt_triangles,
    #         self.gt_tri_lowers,
    #         self.gt_tri_uppers,
    #     )
    #     self.gt_tri_bvh.refit()

    def get_vertices(self):
        return self.q.numpy()[: self.n_particles] / self.config.system_scale

    def register_mesh(
        self, V_gt: np.ndarray, F_gt: np.ndarray, V: np.ndarray, F: np.ndarray
    ):
        c = self.config

        V_gt = V_gt * c.system_scale
        V = V * c.system_scale

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        gt_mesh = trimesh.Trimesh(vertices=V_gt, faces=F_gt, process=False)
        E: np.ndarray = mesh.edges_unique

        self.n_particles = V.shape[0]
        self.n_triangles = F.shape[0]
        self.n_edges = E.shape[0]
        self.n_gt_particles = V_gt.shape[0]
        self.n_gt_triangles = F_gt.shape[0]

        NP, NT, NE = self.n_particles, self.n_triangles, self.n_edges
        NP_GT, NT_GT = self.n_gt_particles, self.n_gt_triangles

        wp_slice(self.q, 0, NP).assign(V)
        wp_slice(self.q_rest, 0, NP).assign(V)
        wp_slice(self.triangles, 0, NT).assign(F)
        wp_slice(self.edges, 0, NE).assign(E)

        # gt_samples = trimesh.sample.sample_surface(gt_mesh, c.max_gt_samples, seed=c.seed)[0]
        gt_samples = trimesh.sample.sample_surface_even(
            gt_mesh, c.max_gt_samples, seed=c.seed
        )[0]
        self.n_gt_samples = gt_samples.shape[0]
        NS_GT = self.n_gt_samples
        areas_gt = igl.doublearea(V_gt, F_gt) / 2

        wp_slice(self.gt_vertices, 0, NP_GT).assign(V_gt)
        wp_slice(self.gt_triangles, 0, NT_GT).assign(F_gt)
        wp_slice(self.gt_samples, 0, NS_GT).assign(gt_samples)
        wp_slice(self.gt_sample_weights, 0, NS_GT).fill_(areas_gt.sum() / NS_GT)

        areas = igl.doublearea(V, F) / 2
        logger.debug(f"min area = {np.min(areas)}, max area = {np.max(areas)}")
        if np.min(areas) < 1e-8:
            f_ids = np.where(areas < 1e-8)[0]
            # logger.warning(
            #     f"Triangle area is too small (id={f_ids}, area={areas[f_ids]}, indices={F[f_ids]}), may cause numerical instability"
            # )
            logger.warning(
                f"Triangle area is too small (min_area={np.min(areas)}), may cause numerical instability"
            )

        angles = igl.internal_angles(V, F)
        min_angle = np.min(angles) * 180 / np.pi
        logger.debug(f"min angle = {min_angle}")
        if min_angle <= c.min_angle_thres:
            self.do_nothing = True

        massmatrix = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        voronoi_areas = massmatrix.diagonal()
        wp_slice(self.voronoi_areas, 0, NP).assign(voronoi_areas)
        
        compute_edge_bounds(
            NE,
            self.q,
            self.edges,
            self.edge_lowers,
            self.edge_uppers,
            0.0,
        )
        self.edge_bvh = wp.Bvh(
            wp_slice(self.edge_lowers, 0, NE),
            wp_slice(self.edge_uppers, 0, NE),
        )
        compute_tri_bounds(
            NT,
            self.q,
            self.triangles,
            self.tri_lowers,
            self.tri_uppers,
            0.0,
        )
        self.tri_bvh = wp.Bvh(
            wp_slice(self.tri_lowers, 0, NT),
            wp_slice(self.tri_uppers, 0, NT),
        )
        # compute_tri_bounds(
        #     NT_GT,
        #     self.gt_vertices,
        #     self.gt_triangles,
        #     self.gt_tri_lowers,
        #     self.gt_tri_uppers,
        # )
        # self.gt_tri_bvh = wp.Bvh(
        #     wp_slice(self.gt_tri_lowers, 0, NT_GT),
        #     wp_slice(self.gt_tri_uppers, 0, NT_GT),
        # )
        
        self.mesh_bvh = wp.Mesh(
            wp_slice(self.q, 0, NP),
            wp_slice(self.triangles.reshape(-1), 0, NT * 3),
        )
        self.gt_mesh_bvh = wp.Mesh(
            wp_slice(self.gt_vertices, 0, NP_GT),
            wp_slice(self.gt_triangles.reshape(-1), 0, NT_GT * 3),
        )

        self._energy_preprocess(V, F)

    def _energy_preprocess(self, V, F):
        for k in self.config.energy_calcs:
            ec: EnergyCalculator = self.energy_calcs.get(k, None)
            ec.preprocess(V, F)

    def _update_vertex_target(self):
        ec_mesh2gt: Mesh2GTDistanceEnergyCalculator = self._get_energy_calculator(
            Mesh2GTDistanceEnergyCalculator
        )
        if ec_mesh2gt is not None:
            ec_mesh2gt.update_target(self.q)
            
        ec_mesh2gt: Mesh2GTDistanceBvhEnergyCalculator = self._get_energy_calculator(
            Mesh2GTDistanceBvhEnergyCalculator
        )
        if ec_mesh2gt is not None:
            ec_mesh2gt.update_target(self.q)

        ec_gt2mesh: GT2MeshDistanceEnergyCalculator = self._get_energy_calculator(
            GT2MeshDistanceEnergyCalculator
        )
        if ec_gt2mesh is not None:
            ec_gt2mesh.update_target(self.q)
            
        ec_gt2mesh: GT2MeshDistanceBvhEnergyCalculator = self._get_energy_calculator(
            GT2MeshDistanceBvhEnergyCalculator
        )
        if ec_gt2mesh is not None:
            ec_gt2mesh.update_target(self.q)

    def _detect_contact(self):
        ec: CollisionEnergyCalculator = self._get_energy_calculator(
            CollisionEnergyCalculator
        )
        if ec is not None:
            ec.detect_contact(self.q)
        ec: CollisionBvhEnergyCalculator = self._get_energy_calculator(
            CollisionBvhEnergyCalculator
        )
        if ec is not None:
            self._refit_edge_bvh(self.q)
            self._refit_tri_bvh(self.q)
            ec.detect_contact(self.q)
            
        wp.copy(self.q_prev_detection, self.q, count=self.n_particles)

    def _init_energy_calcs(self):
        self.energy_calcs = {}
        for ec in self.config.energy_calcs:
            self.energy_calcs[ec] = ec(self)

    def _compute_energy(self, verbose=False):
        self.energy.zero_()
        last_energy = 0.0
        energy_vals = {}
        for ec in self.energy_calcs.values():
            ec: EnergyCalculator
            ec.compute_energy(self.q, self.energy)
            if verbose:
                new_energy = self.energy.numpy()[0]
                energy_vals[ec.name] = new_energy - last_energy
                last_energy = new_energy
        if verbose:
            logger.debug(
                f"Energy values: {', '.join(f'{k}: {v:.3e}' for k, v in energy_vals.items())}"
            )

    def _compute_diff(self):
        self.grad.zero_()
        self.hess_diag.zero_()
        for k in self.config.energy_calcs:
            ec: EnergyCalculator = self.energy_calcs.get(k, None)
            ec.compute_diff(self.q, -1.0, self.grad, self.hess_diag)

    def _compute_hess_dx(self, dx: wp.array, hess_dx: wp.array):
        hess_dx.zero_()
        for k in self.config.energy_calcs:
            ec: EnergyCalculator = self.energy_calcs.get(k, None)
            ec.compute_hess_dx(self.q, dx, hess_dx)

    def _clamp_p(self):
        c = self.config
        wp.launch(
            kernel=clamp_p_kernel,
            dim=self.n_particles,
            inputs=[
                self.p,
                self.q_prev_newton,
                self.q_prev_detection,
                c.contact_detection_radius,
            ],
            device=self.device,
        )

    def _ccd(self):
        self.ccd_step.fill_(1.0)

        ec = self._get_energy_calculator(CollisionEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)

        ec = self._get_energy_calculator(CollisionWoBufferEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionWoBufferEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)
            
        ec = self._get_energy_calculator(CollisionBvhEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionBvhEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)

        if self.config.debug:
            ccd_step_val = self.ccd_step.numpy()[0]
            logger.debug(f"CCD step: {ccd_step_val:.6f}")

    def _line_search(self):
        wp.copy(self.energy_prev, self.energy, count=1)
        for n_halves in range(self.config.n_line_search_iters):
            wp.launch(
                kernel=line_search_kernel,
                dim=self.n_particles,
                inputs=[
                    self.q_prev_newton,
                    self.p,
                    self.ccd_step,
                    n_halves,
                    self.energy_prev,
                    self.energy,
                ],
                outputs=[self.q],
            )
            self._compute_energy()

    def step(
        self,
        eval=False,
        eval_CDs=None,
        eval_HDs=None,
        gt_V=None,
        gt_F=None,
        eval_F=None,
    ):
        if self.do_nothing:
            logger.warning("Doing nothing in step() because of small internal angles")
            return

        """One Newton's iteration"""
        c = self.config
        
        self._update_vertex_target()
        if c.detect_contact_every == "step":
            self._detect_contact()

        for newton_iter in range(c.n_newton_iters):
            wp.copy(self.q_prev_newton, self.q, count=self.n_particles)

            # --------------------- Compute linear system --------------------- #
            # self._update_vertex_target()
            if c.detect_contact_every == "newton":
                self._detect_contact()
            self._compute_energy()
            self._compute_diff()

            # --------------------- CG linear solve --------------------- #
            self.cg_solver.solve()

            # --------------------- CCD --------------------- #
            self._clamp_p()
            self._ccd()

            # --------------------- Line search --------------------- #
            if c.use_cuda_graph:
                if not self.line_search_graph:
                    wp.capture_begin()
                    self._line_search()
                    self.line_search_graph = wp.capture_end()
                wp.capture_launch(self.line_search_graph)
            else:
                self._line_search()

            if c.debug:
                self._compute_energy(verbose=True)
                energy_val = self.energy.numpy()[0]
                energy_diff = energy_val - self.energy_prev.numpy()[0]
                logger.debug(
                    f"Newton iter {newton_iter}: energy = {energy_val:.3e}, energy_diff = {energy_diff:.3e}"
                )

            if eval:
                eval_V = self.get_vertices().astype(np.float64)
                cd, hd = compute_igl_CD_HD(gt_V, gt_F, eval_V, eval_F)
                eval_CDs.append(cd)
                eval_HDs.append(hd)
