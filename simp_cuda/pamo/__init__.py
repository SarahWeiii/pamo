"""PaMO's three-stage GPU mesh optimization pipeline."""

from __future__ import annotations

import copy
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pamo_safe_project
import torch
import torchcumesh2sdf
import trimesh
from pdmc import DMC
from torch import nn
from torch.autograd import Function

from . import _C

try:
    from ._cuda_abi import MIN_CUDA_CAPABILITY
except ImportError:
    MIN_CUDA_CAPABILITY = (8, 0)

try:
    __version__ = version("pamo")
except PackageNotFoundError:
    __version__ = "0.1.0"


def _resolve_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"PaMO requires a CUDA device, got {resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("PaMO requires an available NVIDIA CUDA device")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {resolved.index} is unavailable; "
            f"only {torch.cuda.device_count()} device(s) are visible"
        )
    capability = torch.cuda.get_device_capability(resolved)
    if capability < MIN_CUDA_CAPABILITY:
        raise RuntimeError(
            "This internal PaMO wheel requires CUDA compute capability "
            f"{MIN_CUDA_CAPABILITY[0]}.{MIN_CUDA_CAPABILITY[1]} or newer; "
            f"device {resolved} reports {capability[0]}.{capability[1]}"
        )
    return resolved


class PaMO(nn.Module):
    """Run PaMO remeshing, simplification, and safe projection on one GPU."""

    def __init__(
        self,
        input_mesh: trimesh.Trimesh,
        use_stage1: bool = True,
        use_stage3: bool = True,
        device: str | torch.device = "cuda",
        stage3_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.device = _resolve_cuda_device(device)
        self.use_stage1 = use_stage1
        self.use_stage3 = use_stage3
        self._pamo = _C.CUDSP_Free()

        bounds = np.asarray(input_mesh.bounding_box.bounds)
        diameter = float(np.abs(bounds[1] - bounds[0]).max())
        if not np.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("input_mesh must have a finite, non-zero bounding box")

        self.bbox = bounds
        self.gt_mesh = copy.deepcopy(input_mesh)
        self.config = stage3_config
        self.system = None

        self.vol2mesh = DMC(dtype=torch.float32) if self.use_stage1 else None
        self.R = 256
        self.band = 3 / self.R
        self.margin = self.band * 2 + 1
        self.target_faces: int | None = None

        pamo_extension = self._pamo

        class DSPFunction(Function):
            @staticmethod
            def forward(
                ctx,
                points,
                triangles,
                vertices_undo,
                num_vertices_undo,
                scale,
                threshold,
                is_stuck,
                init,
            ):
                outputs = pamo_extension.forward(
                    points,
                    triangles,
                    vertices_undo,
                    num_vertices_undo,
                    scale,
                    threshold,
                    is_stuck,
                    init,
                )
                ctx.set_materialize_grads(False)
                return outputs

        self.func = DSPFunction

    @staticmethod
    def _stage3_config_for_mesh(
        gt_vertices: int,
        gt_faces: int,
        stage2_vertices: int,
        stage2_faces: int,
    ):
        """Size Stage 3 buffers for the mesh instead of upstream worst cases."""

        def power_of_two_capacity(value: int, minimum: int) -> int:
            return max(minimum, 1 << max(0, value - 1).bit_length())

        config = pamo_safe_project.Stage3Config()
        stage2_face_capacity = max(1, (stage2_faces - 1_024 + 1) // 2)
        gt_face_capacity = max(1, (gt_faces - 1_024 + 1) // 2)
        config.max_particles = power_of_two_capacity(
            max(stage2_vertices, stage2_face_capacity), 1 << 10
        )
        config.max_gt_particles = power_of_two_capacity(
            max(gt_vertices, gt_face_capacity), 1 << 10
        )
        config.max_gt_samples = min(config.max_gt_samples, 1 << 13)

        # Collision buffers dominate Stage 3 memory. A one-million-block floor
        # is ample for ordinary low-poly outputs while remaining practical on
        # a 16 GiB RTX 4080. The existing overflow check still fails closed.
        estimated_blocks = max(stage2_faces, stage2_vertices) * 256
        config.max_blocks = power_of_two_capacity(estimated_blocks, 1 << 20)
        config.max_blocks = min(config.max_blocks, 1 << 22)

        return config

    @staticmethod
    def tri_area(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        cross_prod = torch.cross(v1 - v0, v2 - v0, dim=1)
        return 0.5 * torch.norm(cross_prod, dim=1)

    @staticmethod
    def preprocess_mesh(
        points: torch.Tensor,
        triangles: torch.Tensor,
        band: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        tris = points[triangles.long()].detach().cpu().numpy()
        tris_mean = tris.mean(axis=1).mean(axis=0)
        tris = tris - tris_mean
        tris_min = tris.min(axis=0).min(axis=0)
        tris = tris - tris_min
        tris_max = float(tris.max())
        if not np.isfinite(tris_max) or tris_max <= 0.0:
            raise ValueError("input tensors must describe a finite, non-degenerate mesh")
        tris = (tris / tris_max + band) / margin
        return tris, tris_min, tris_max, tris_mean

    def remesh(
        self,
        tris: torch.Tensor,
        tris_min: np.ndarray,
        tris_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vol2mesh is None:
            raise RuntimeError("Stage 1 is disabled")

        with torch.cuda.device(self.device):
            distances = torchcumesh2sdf.get_sdf(tris, self.R, self.band)
            distances = distances - 0.9 / self.R
            vertices, faces = self.vol2mesh(distances, return_quads=False)

        vertices_np = vertices.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        vertices_np = (
            (vertices_np * self.R + 0.5) / (self.R + 1) * self.margin - self.band
        ) * tris_max + tris_min
        return (
            torch.as_tensor(vertices_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(faces_np, dtype=torch.int32, device=self.device),
        )

    def run(
        self,
        points: torch.Tensor,
        triangles: torch.Tensor,
        ratio: float,
        tolerance: int = 4,
        threshold: float = 1e-3,
        iter: int = 1_000_000,
        min_verts: int = 0,
        stage3_iters: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimize a mesh and return CPU NumPy vertex and face arrays.

        ``min_verts`` is retained for upstream API compatibility. Upstream uses
        it as a lower bound on the target face count.
        """
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        if iter < 0 or tolerance < 0 or stage3_iters < 0 or min_verts < 0:
            raise ValueError(
                "iteration counts, tolerance, and min_verts must be non-negative"
            )

        points = points.to(device=self.device, dtype=torch.float32).contiguous()
        triangles = triangles.to(device=self.device, dtype=torch.int32).contiguous()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
            raise ValueError(
                f"triangles must have non-empty shape (M, 3), got {tuple(triangles.shape)}"
            )

        self.target_faces = max(int(ratio * triangles.shape[0]), min_verts, 10)
        if self.target_faces <= 50:
            self.R = 64
        elif self.target_faces <= 1_000:
            self.R = 128
        else:
            self.R = 256
        self.band = 3 / self.R
        self.margin = self.band * 2 + 1

        tris_np, tris_min, tris_max, tris_mean = self.preprocess_mesh(
            points,
            triangles,
            self.band,
            self.margin,
        )
        tris = torch.as_tensor(
            tris_np, dtype=torch.float32, device=self.device
        ).contiguous()

        if self.use_stage1:
            stage1_start = time.perf_counter()
            vertices, faces = self.remesh(tris, tris_min, tris_max)
            print(f"Time for Remeshing: {time.perf_counter() - stage1_start:.3f} sec")
        else:
            center = torch.as_tensor(tris_mean, dtype=torch.float32, device=self.device)
            vertices = (points - center).contiguous()
            faces = triangles.contiguous()

        stage2_start = time.perf_counter()
        vertices_undo = torch.empty(0, dtype=torch.int32, device=self.device)
        n_vertices_undo = 0
        unchanged_count = 0
        is_stuck = False
        init = True
        scale = float(
            torch.max(
                torch.max(vertices, dim=0).values - torch.min(vertices, dim=0).values
            )
        )

        with torch.cuda.device(self.device):
            for _ in range(iter):
                if faces.shape[0] <= self.target_faces or faces.shape[0] <= 10:
                    break
                previous_face_count = faces.shape[0]
                vertices, faces, vertices_occ, vertices_map, vertices_undo = (
                    self.func.apply(
                        vertices,
                        faces,
                        vertices_undo,
                        n_vertices_undo,
                        scale,
                        threshold,
                        is_stuck,
                        init,
                    )
                )
                init = False
                n_vertices_undo = vertices_undo.shape[0]

                vertices = vertices[vertices_occ.view(-1).bool()].contiguous()
                faces = faces[faces[:, 0] >= 0]
                faces[:, 0] = vertices_map[faces[:, 0].long()].view(-1)
                faces[:, 1] = vertices_map[faces[:, 1].long()].view(-1)
                faces[:, 2] = vertices_map[faces[:, 2].long()].view(-1)
                faces = faces.contiguous()

                if faces.shape[0] == previous_face_count:
                    unchanged_count += 1
                else:
                    unchanged_count = 0
                    is_stuck = False
                if unchanged_count >= 2:
                    is_stuck = True
                if unchanged_count >= tolerance:
                    print("Not enough edges available to be collapsed.")
                    break

        vertices_np = vertices.detach().cpu().numpy() + tris_mean
        faces_np = faces.detach().cpu().numpy()
        print(f"Time for Simplification: {time.perf_counter() - stage2_start:.3f} sec")

        if self.use_stage3:
            stage2_mesh = trimesh.Trimesh(
                vertices=vertices_np, faces=faces_np, process=False
            )
            if self.config is None:
                self.config = self._stage3_config_for_mesh(
                    len(self.gt_mesh.vertices),
                    len(self.gt_mesh.faces),
                    len(stage2_mesh.vertices),
                    len(stage2_mesh.faces),
                )
            if self.system is None:
                self.system = pamo_safe_project.Stage3System(
                    self.config,
                    str(self.device),
                )
            vertices_np, faces_np = pamo_safe_project.process(
                self.gt_mesh.vertices,
                self.gt_mesh.faces,
                stage2_mesh.vertices,
                stage2_mesh.faces,
                stage3_iters,
                system=self.system,
                config=self.config,
                device=str(self.device),
            )

        return np.asarray(vertices_np), np.asarray(faces_np)


class PaSP(nn.Module):
    """Run only PaMO's parallel simplification stage."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.device = _resolve_cuda_device(device)
        self._simplifier = _C.CUDSP()

    def run(
        self,
        points: torch.Tensor,
        triangles: torch.Tensor,
        threshold: float = 0.001,
        iter: int = 1_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vertices = points.to(self.device, dtype=torch.float32).contiguous()
        faces = triangles.to(self.device, dtype=torch.int32).contiguous()
        scale = float(
            torch.max(
                torch.max(vertices, dim=0).values - torch.min(vertices, dim=0).values
            )
        )
        init = True

        with torch.cuda.device(self.device):
            for iteration in range(iter):
                previous_face_count = faces.shape[0]
                vertices, faces, vertices_occ, vertices_map = self._simplifier.forward(
                    vertices,
                    faces,
                    scale,
                    threshold,
                    init,
                )
                vertices = vertices[vertices_occ.view(-1).bool()].contiguous()
                faces = faces[faces[:, 0] >= 0]
                faces[:, 0] = vertices_map[faces[:, 0].long()].view(-1)
                faces[:, 1] = vertices_map[faces[:, 1].long()].view(-1)
                faces[:, 2] = vertices_map[faces[:, 2].long()].view(-1)
                faces = faces.contiguous()
                init = False
                if faces.shape[0] == previous_face_count:
                    print(f"Converged at iteration {iteration}")
                    break
        return vertices, faces
