import warp as wp
import numpy as np
from stage3.kernels.geometry_kernels import *

wp.init()


@wp.kernel
def query_intersect_kernel(
    bvh: wp.uint64,
    vertices: wp.array(dtype=wp.vec3),
    d_thres: float,
    is_intersect: wp.array(dtype=int),
):
    i0 = wp.tid()
    x0 = vertices[i0]
    query = wp.bvh_query_aabb(bvh, x0 - wp.vec3(d_thres), x0 + wp.vec3(d_thres))
    aabb_id = wp.int32(0)
    
    while wp.bvh_query_next(query, aabb_id):
        is_intersect[aabb_id] = 1
    

lowers_np = np.array([
    [0, -2, -2],
    [1, -2, -2],
    [2, -2, -2],
], dtype=np.float32)
uppers_np = np.array([
    [1, 3, 3],
    [2, 3, 3],
    [3, 3, 3],
], dtype=np.float32)
vertices_np = np.array([
    [0.5, 0.5, 0.5],
], dtype=np.float32)

device = wp.get_preferred_device()
with wp.ScopedDevice(device):
    lowers = wp.array(lowers_np, dtype=wp.vec3)
    uppers = wp.array(uppers_np, dtype=wp.vec3)
    vertices = wp.array(vertices_np, dtype=wp.vec3)
    bvh = wp.Bvh(lowers, uppers)
    
    is_intersect = wp.zeros((len(lowers)), dtype=int)
    
    wp.launch(
        kernel=query_intersect_kernel,
        dim=1,
        inputs=[bvh.id, vertices, 0.0],
        outputs=[is_intersect],
    )
    
print(is_intersect.numpy())
    
