import trimesh
import igl
import random


mesh_path = "/home/xiaodi/Repos/stage3/data/examples/gt/cube.ply"
mesh = trimesh.load(mesh_path)
V = mesh.vertices
F = mesh.faces

# print(f"V:\n{V}")
# print(f"F:\n{F}")

# print(igl.random_points_on_mesh(4, V, F))

gen_samples = trimesh.sample.sample_surface(mesh, 4, seed=42)[0]
print(gen_samples)

_, _, gen_samples = igl.random_points_on_mesh(4, V, F)
print(gen_samples)

