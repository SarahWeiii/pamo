import trimesh
import igl
import numpy as np
import torch


def compute_trimesh_CD_HD(gt_mesh, gen_mesh, num_mesh_samples=5000, seed=42):
    gen_points_sampled = trimesh.sample.sample_surface(
        gen_mesh, num_mesh_samples, seed=seed
    )[0]
    gt_points_sampled = trimesh.sample.sample_surface(
        gt_mesh, num_mesh_samples, seed=seed
    )[0]

    gt_to_gen_distances = trimesh.proximity.closest_point(gt_mesh, gen_points_sampled)[
        1
    ]
    gen_to_gt_distances = trimesh.proximity.closest_point(gen_mesh, gt_points_sampled)[
        1
    ]

    gt_to_gen_chamfer = np.mean(np.square(gt_to_gen_distances))
    gen_to_gt_chamfer = np.mean(np.square(gen_to_gt_distances))
    gt_to_gen_hausdorff = np.amax(gt_to_gen_distances)
    gen_to_gt_hausdorff = np.amax(gen_to_gt_distances)
    hausdorff_distance = max(gt_to_gen_hausdorff, gen_to_gt_hausdorff)

    return gt_to_gen_chamfer + gen_to_gt_chamfer, hausdorff_distance


def compute_igl_CD_HD(gt_V, gt_F, gen_V, gen_F, num_mesh_samples=5000, seed=42):
    gt_mesh = trimesh.Trimesh(gt_V, gt_F, process=False)
    gen_mesh = trimesh.Trimesh(gen_V, gen_F, process=False)

    # _, _, gen_samples = igl.random_points_on_mesh(num_mesh_samples, gen_V, gen_F)
    # _, _, gt_samples = igl.random_points_on_mesh(num_mesh_samples, gt_V, gt_F)

    gt_samples = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples, seed=seed)[0]
    gen_samples = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples, seed=seed)[
        0
    ]

    gt_to_gen_dist_sq, _, _ = igl.point_mesh_squared_distance(gt_samples, gen_V, gen_F)
    gen_to_gt_dist_sq, _, _ = igl.point_mesh_squared_distance(gen_samples, gt_V, gt_F)

    gt_to_gen_chamfer = np.mean(gt_to_gen_dist_sq)
    gen_to_gt_chamfer = np.mean(gen_to_gt_dist_sq)

    gt_to_gen_hausdorff = igl.hausdorff(gt_V, gt_F, gen_V, gen_F)

    return gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_hausdorff


def compute_min_angle(mesh: trimesh.Trimesh):
    # Compute all angles on the mesh
    V = mesh.vertices
    F = mesh.faces

    F_mask = np.logical_and(
        F[:, 0] != F[:, 1], np.logical_and(F[:, 1] != F[:, 2], F[:, 2] != F[:, 0])
    )  # for RoLoPM which has degenerate faces
    F = F[F_mask]

    edges = np.concatenate(
        [F[:, None, [0, 1]], F[:, None, [1, 2]], F[:, None, [2, 0]]], axis=1
    )  # [|F|, 3, 2]
    edge_vectors = V[edges[:, :, 0]] - V[edges[:, :, 1]]  # [|F|, 3, 3]
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)  # [|F|, 3]
    angle_cosines = -np.sum(
        edge_vectors * np.roll(edge_vectors, 1, axis=1), axis=-1
    ) / (
        edge_lengths * np.roll(edge_lengths, 1, axis=1)
    )  # [|F|, 3]
    angles = np.arccos(np.clip(angle_cosines, -1.0, 1.0))  # [|F|, 3]
    # print(angles * 180 / np.pi)

    return np.min(angles)


def compute_min_angle_VF(V, F):
    F_mask = np.logical_and(
        F[:, 0] != F[:, 1], np.logical_and(F[:, 1] != F[:, 2], F[:, 2] != F[:, 0])
    )  # for RoLoPM which has degenerate faces
    F = F[F_mask]

    edges = np.concatenate(
        [F[:, None, [0, 1]], F[:, None, [1, 2]], F[:, None, [2, 0]]], axis=1
    )  # [|F|, 3, 2]
    edge_vectors = V[edges[:, :, 0]] - V[edges[:, :, 1]]  # [|F|, 3, 3]
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)  # [|F|, 3]
    angle_cosines = -np.sum(
        edge_vectors * np.roll(edge_vectors, 1, axis=1), axis=-1
    ) / (
        edge_lengths * np.roll(edge_lengths, 1, axis=1)
    )  # [|F|, 3]
    angles = np.arccos(np.clip(angle_cosines, -1.0, 1.0))  # [|F|, 3]
    # print(angles * 180 / np.pi)

    return np.min(angles)


def compute_bad_tri_ratio(V, F, thresholds=[1, 5, 10]):
    thresholds = np.array(thresholds)
    angles = igl.internal_angles(V, F) * 180 / np.pi
    min_angles = angles.min(axis=1)
    bad_tri_num = np.sum(min_angles[:, None] < thresholds[None, :], axis=0)
    return bad_tri_num / F.shape[0]
