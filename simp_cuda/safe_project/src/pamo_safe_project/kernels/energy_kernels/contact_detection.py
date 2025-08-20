import warp as wp

from ..distance_kernels.distance_kernels import *
from ...defs import *


@wp.kernel
def detect_pt_contact_kernel(
    contact_counter: wp.array(dtype=int),
    max_blocks: int,
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    d_hat: float,
    radius: float,  # detect contact if distance < 2 * radius + d_hat
    b_types: wp.array(dtype=int, ndim=2),
    b_indices: wp.array(dtype=int, ndim=2),
):
    i0, tri_id = wp.tid()

    i1 = triangles[tri_id, 0]
    i2 = triangles[tri_id, 1]
    i3 = triangles[tri_id, 2]

    if i0 == i1 or i0 == i2 or i0 == i3:
        return

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    d_thres = 2.0 * radius + d_hat
    p_plane_d = pt_distance(x0, x1, x2, x3)
    if p_plane_d > d_thres:
        return

    sphere_c = (x1 + x2 + x3) / 3.0
    sphere_r = wp.max(
        wp.max(wp.length(x1 - sphere_c), wp.length(x2 - sphere_c)),
        wp.length(x3 - sphere_c),
    )
    p_sphere_d = wp.length(x0 - sphere_c) - sphere_r
    if p_sphere_d > d_thres:
        return
    
    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, pt_type)
    if d > d_thres:
        return
    
    bid = wp.atomic_add(contact_counter, 0, 1)
    if bid >= max_blocks:
        return
    
    b_indices[bid, 0] = i0
    b_indices[bid, 1] = i1
    b_indices[bid, 2] = i2
    b_indices[bid, 3] = i3
    
    b_types[bid, 0] = BlockTypes.PT_CONTACT
    

@wp.kernel
def detect_ee_contact_kernel(
    contact_counter: wp.array(dtype=int),
    max_blocks: int,
    x: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    d_hat: float,
    radius: float,  # detect contact if distance < 2 * radius + d_hat
    ee_classify_thres: float,
    b_types: wp.array(dtype=int, ndim=2),
    b_indices: wp.array(dtype=int, ndim=2),
):
    e0, e1 = wp.tid()
    if e0 >= e1:
        return
    
    i0 = edges[e0, 0]
    i1 = edges[e0, 1]
    i2 = edges[e1, 0]
    i3 = edges[e1, 1]
    
    if i0 == i2 or i0 == i3 or i1 == i2 or i1 == i3:
        return
    
    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]
    
    d_thres = 2.0 * radius + d_hat
    
    sphere_c_0 = (x0 + x1) / 2.0
    sphere_r_0 = wp.length(x0 - x1) / 2.0
    sphere_c_1 = (x2 + x3) / 2.0
    sphere_r_1 = wp.length(x2 - x3) / 2.0
    spheres_d = wp.length(sphere_c_0 - sphere_c_1) - sphere_r_0 - sphere_r_1
    if spheres_d > d_thres:
        return
    
    ee_type = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d = ee_pair_distance(x0, x1, x2, x3, ee_type)
    if d > d_thres:
        return
    
    bid = wp.atomic_add(contact_counter, 0, 1)
    if bid >= max_blocks:
        return
    
    b_indices[bid, 0] = i0
    b_indices[bid, 1] = i1
    b_indices[bid, 2] = i2
    b_indices[bid, 3] = i3
    
    b_types[bid, 0] = BlockTypes.EE_CONTACT
    
    
@wp.kernel
def detect_pt_contact_bvh_kernel(
    bvh: wp.uint64,
    contact_counter: wp.array(dtype=int),
    max_blocks: int,
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    d_hat: float,
    radius: float,  # detect contact if distance < 2 * radius + d_hat
    b_types: wp.array(dtype=int, ndim=2),
    b_indices: wp.array(dtype=int, ndim=2),
):
    i0 = wp.tid()
    x0 = x[i0]
    
    q_d_thres = 2.0 * radius + d_hat
    q_lower = x0 - wp.vec3(q_d_thres)
    q_upper = x0 + wp.vec3(q_d_thres)
    # wp.printf("d_thres: %f\nquery aabb = (%f %f %f)-(%f %f %f)\n", d_thres, q_lower[0], q_lower[1], q_lower[2], q_upper[0], q_upper[1], q_upper[2])
    query = wp.bvh_query_aabb(bvh, q_lower, q_upper)
    tri_id = wp.int32(0)
    
    while wp.bvh_query_next(query, tri_id):
        i1 = triangles[tri_id, 0]
        i2 = triangles[tri_id, 1]
        i3 = triangles[tri_id, 2]

        if not (i0 == i1 or i0 == i2 or i0 == i3):
            x1 = x[i1]
            x2 = x[i2]
            x3 = x[i3]
            
            d_thres = 2.0 * radius + d_hat
            pt_type = pt_pair_classify(x0, x1, x2, x3)
            d = pt_pair_distance(x0, x1, x2, x3, pt_type)
            if d <= d_thres:
                bid = wp.atomic_add(contact_counter, 0, 1)
                if bid >= max_blocks:
                    return
                
                b_indices[bid, 0] = i0
                b_indices[bid, 1] = i1
                b_indices[bid, 2] = i2
                b_indices[bid, 3] = i3
                
                b_types[bid, 0] = BlockTypes.PT_CONTACT


@wp.kernel
def detect_ee_contact_bvh_kernel(
    bvh: wp.uint64,
    contact_counter: wp.array(dtype=int),
    max_blocks: int,
    x: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    d_hat: float,
    radius: float,  # detect contact if distance < 2 * radius + d_hat
    ee_classify_thres: float,
    b_types: wp.array(dtype=int, ndim=2),
    b_indices: wp.array(dtype=int, ndim=2),
):
    e0 = wp.tid()
    
    i0 = edges[e0, 0]
    i1 = edges[e0, 1]
    x0 = x[i0]
    x1 = x[i1]
    
    q_d_thres = 2.0 * radius + d_hat
    query = wp.bvh_query_aabb(
        bvh, 
        wp.min(x0, x1) - wp.vec3(q_d_thres), 
        wp.max(x0, x1) + wp.vec3(q_d_thres),
    )
    e1 = wp.int32(0)
    
    while wp.bvh_query_next(query, e1):
        i2 = edges[e1, 0]
        i3 = edges[e1, 1]
        
        if e0 < e1 and not (i0 == i2 or i0 == i3 or i1 == i2 or i1 == i3):
            x2 = x[i2]
            x3 = x[i3]
            
            d_thres = 2.0 * radius + d_hat
            ee_type = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
            d = ee_pair_distance(x0, x1, x2, x3, ee_type)
            if d <= d_thres:
                bid = wp.atomic_add(contact_counter, 0, 1)
                if bid >= max_blocks:
                    return
                
                b_indices[bid, 0] = i0
                b_indices[bid, 1] = i1
                b_indices[bid, 2] = i2
                b_indices[bid, 3] = i3
                
                b_types[bid, 0] = BlockTypes.EE_CONTACT
