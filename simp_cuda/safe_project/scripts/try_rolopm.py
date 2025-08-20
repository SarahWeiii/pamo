import trimesh
import igl
import numpy as np
import os


mesh = trimesh.load(
    "/home/xiaodi/Desktop/Repos/stage3/data/RoLoPM_0.01/372112_ours_final.obj"
)
V = mesh.vertices
F = mesh.faces
E = mesh.edges_unique
E_len = mesh.edges_unique_length

F_remove_mask = np.logical_or(
    np.logical_or(F[:, 0] == F[:, 1], F[:, 1] == F[:, 2]), F[:, 2] == F[:, 0]
)
new_F = F[~F_remove_mask]

new_mesh = trimesh.Trimesh(vertices=V, faces=new_F, process=False)
new_E = new_mesh.edges_unique
new_E_len = new_mesh.edges_unique_length

print(new_E_len.min())
