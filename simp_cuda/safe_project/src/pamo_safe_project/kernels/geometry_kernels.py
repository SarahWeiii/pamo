import warp as wp
from .distance_kernels.distance_kernels import *


@wp.kernel
def update_min_pt_distance_kernel(
    x: wp.array(dtype=wp.vec3),
    target_vertices: wp.array(dtype=wp.vec3),
    target_triangles: wp.array(dtype=wp.int32, ndim=2),
    min_distance: wp.array(dtype=wp.float32),
):
    i0, tri_id = wp.tid()

    i1 = target_triangles[tri_id, 0]
    i2 = target_triangles[tri_id, 1]
    i3 = target_triangles[tri_id, 2]

    x0 = x[i0]
    x1 = target_vertices[i1]
    x2 = target_vertices[i2]
    x3 = target_vertices[i3]

    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, pt_type)

    wp.atomic_min(min_distance, i0, d)
        

@wp.kernel
def update_closest_triangle_on_target_kernel(
    x: wp.array(dtype=wp.vec3),
    target_vertices: wp.array(dtype=wp.vec3),
    target_triangles: wp.array(dtype=wp.int32, ndim=2),
    min_distance: wp.array(dtype=wp.float32),
    closest_triangle_on_target: wp.array(dtype=wp.int32),
):
    i0, tri_id = wp.tid()

    i1 = target_triangles[tri_id, 0]
    i2 = target_triangles[tri_id, 1]
    i3 = target_triangles[tri_id, 2]

    x0 = x[i0]
    x1 = target_vertices[i1]
    x2 = target_vertices[i2]
    x3 = target_vertices[i3]

    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, pt_type)

    if d <= min_distance[i0] + 1e-6:
        closest_triangle_on_target[i0] = tri_id


@wp.kernel
def update_closest_point_on_target_kernel(
    x: wp.array(dtype=wp.vec3),
    target_vertices: wp.array(dtype=wp.vec3),
    target_triangles: wp.array(dtype=wp.int32, ndim=2),
    min_distance: wp.array(dtype=wp.float32),
    closest_point_on_target: wp.array(dtype=wp.vec3),
):
    i0, tri_id = wp.tid()

    i1 = target_triangles[tri_id, 0]
    i2 = target_triangles[tri_id, 1]
    i3 = target_triangles[tri_id, 2]

    x0 = x[i0]
    x1 = target_vertices[i1]
    x2 = target_vertices[i2]
    x3 = target_vertices[i3]

    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, pt_type)

    if d <= min_distance[i0] + 1e-6:
        closest_point_on_target[i0] = pt_pair_closest_point(x0, x1, x2, x3, pt_type)
        

@wp.kernel
def update_closest_point_on_target_bvh_kernel(
    x: wp.array(dtype=wp.vec3),
    target_mesh: wp.uint64,
    closest_point_on_target: wp.array(dtype=wp.vec3),
):
    i0 = wp.tid()
    face_index = wp.int32(0)
    face_u = wp.float32(0)
    face_v = wp.float32(0)
    wp.mesh_query_point_no_sign(target_mesh, x[i0], 1e9, face_index, face_u, face_v)
    target_x = wp.mesh_eval_position(target_mesh, face_index, face_u, face_v)
    closest_point_on_target[i0] = target_x
    

@wp.kernel
def update_closest_triangle_on_target_bvh_kernel(
    x: wp.array(dtype=wp.vec3),
    target_mesh: wp.uint64,
    closest_triangle_on_target: wp.array(dtype=wp.int32),
):
    i0 = wp.tid()
    face_index = wp.int32(0)
    face_u = wp.float32(0)
    face_v = wp.float32(0)
    wp.mesh_query_point_no_sign(target_mesh, x[i0], 1e9, face_index, face_u, face_v)
    closest_triangle_on_target[i0] = face_index


@wp.kernel
def compute_tri_bounds_kernel(
    V: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.int32, ndim=2),
    radius: float,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i0 = F[tid, 0]
    i1 = F[tid, 1]
    i2 = F[tid, 2]
 
    lowers[tid] = wp.min(V[i0], wp.min(V[i1], V[i2])) - wp.vec3(radius)
    uppers[tid] = wp.max(V[i0], wp.max(V[i1], V[i2])) + wp.vec3(radius)


@wp.kernel
def compute_edge_bounds_kernel(
    V: wp.array(dtype=wp.vec3),
    E: wp.array(dtype=wp.int32, ndim=2),
    radius: float,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i0 = E[tid, 0]
    i1 = E[tid, 1]

    lowers[tid] = wp.min(V[i0], V[i1]) - wp.vec3(radius)
    uppers[tid] = wp.max(V[i0], V[i1]) + wp.vec3(radius)

