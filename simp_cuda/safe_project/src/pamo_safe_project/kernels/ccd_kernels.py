import warp as wp


from ..defs import *
from .distance_kernels.distance_kernels import *


@wp.kernel
def accd_kernel(
    contact_counter: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    b_types: wp.array(dtype=int, ndim=2),
    b_indices: wp.array(dtype=int, ndim=2),
    slackness: float,
    thickness: float,
    max_iters: int,
    ee_classify_thres: float,
    ccd_step: wp.array(dtype=float),
):
    bid = wp.tid()
    if bid >= contact_counter[0]:
        return

    i0 = b_indices[bid, 0]
    i1 = b_indices[bid, 1]
    i2 = b_indices[bid, 2]
    i3 = b_indices[bid, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    v0 = v[i0]
    v1 = v[i1]
    v2 = v[i2]
    v3 = v[i3]

    type1 = b_types[bid, 0]
    type2 = b_types[bid, 1]

    if type1 == BlockTypes.PT_CONTACT or type1 == BlockTypes.EE_CONTACT:
        v_avg = (v0 + v1 + v2 + v3) / 4.0
        v0 -= v_avg
        v1 -= v_avg
        v2 -= v_avg
        v3 -= v_avg

        if type1 == BlockTypes.PT_CONTACT:
            l_v = wp.length(v0) + wp.max(
                wp.length(v1), wp.max(wp.length(v2), wp.length(v3))
            )
        else:
            l_v = wp.max(wp.length(v0), wp.length(v1)) + wp.max(
                wp.length(v2), wp.length(v3)
            )
        if l_v < 1e-12:
            return 
        
        if type1 == BlockTypes.PT_CONTACT:
            type2 = pt_pair_classify(x0, x1, x2, x3)
            d = pt_pair_distance(x0, x1, x2, x3, type2)
        else:
            type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
            d = ee_pair_distance(x0, x1, x2, x3, type2)
            
        # if d < thickness:
        #     wp.printf("[accd_kernel] type1 = %d, type2 = %d, x0 = (%f, %f, %f), x1 = (%f, %f, %f), x2 = (%f, %f, %f), x3 = (%f, %f, %f), d = %f, thickness = %f\n", type1, type2, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2], d, thickness)

        d_sqr = d * d
        d_minus_thickness = max(
            (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
        )
        g = (1.0 - slackness) * d_minus_thickness
        t = float(0.0)
        t_l = slackness * d_minus_thickness / l_v

        i = int(0)

        while True:
            x0 += t_l * v0
            x1 += t_l * v1
            x2 += t_l * v2
            x3 += t_l * v3

            if type1 == BlockTypes.PT_CONTACT:
                type2 = pt_pair_classify(x0, x1, x2, x3)
                d = pt_pair_distance(x0, x1, x2, x3, type2)
            else:
                type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
                d = ee_pair_distance(x0, x1, x2, x3, type2)

            d_sqr = d * d
            d_minus_thickness = max(
                (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
            )
            if i > 0 and d_minus_thickness <= g or i >= max_iters:
                wp.atomic_min(ccd_step, 0, t)
                return

            t = t + t_l
            if t > 1.0:
                return

            t_l = 0.8 * d_minus_thickness / l_v
            i += 1

    else:
        wp.printf("[accd_kernel] Unknown block type: %d\n", type1)
        

@wp.kernel
def accd_wo_buffer_pt_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    slackness: float,
    thickness: float,
    radius: float,  # radius for contact detection
    max_iters: int,
    ccd_step: wp.array(dtype=float),
):
    i0, tid = wp.tid()
    
    i1 = triangles[tid, 0]
    i2 = triangles[tid, 1]
    i3 = triangles[tid, 2]
    
    if i0 == i1 or i0 == i2 or i0 == i3:
        return
    
    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]
    
    d_thres = 2.0 * radius + thickness
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

    v0 = v[i0]
    v1 = v[i1]
    v2 = v[i2]
    v3 = v[i3]

    v_avg = (v0 + v1 + v2 + v3) / 4.0
    v0 -= v_avg
    v1 -= v_avg
    v2 -= v_avg
    v3 -= v_avg

    l_v = wp.length(v0) + wp.max(
        wp.length(v1), wp.max(wp.length(v2), wp.length(v3))
    )
    if l_v < 1e-12:
        return 
    
    type2 = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, type2)
    if d > d_thres:
        return
        
    if d < thickness:
        wp.printf("[accd_kernel] PT type2 = %d, x0 = (%f, %f, %f), x1 = (%f, %f, %f), x2 = (%f, %f, %f), x3 = (%f, %f, %f), d = %f, thickness = %f\n", type2, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2], d, thickness)

    d_sqr = d * d
    d_minus_thickness = max(
        (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
    )
    g = (1.0 - slackness) * d_minus_thickness
    t = float(0.0)
    t_l = slackness * d_minus_thickness / l_v

    i = int(0)

    while True:
        x0 += t_l * v0
        x1 += t_l * v1
        x2 += t_l * v2
        x3 += t_l * v3

        type2 = pt_pair_classify(x0, x1, x2, x3)
        d = pt_pair_distance(x0, x1, x2, x3, type2)

        d_sqr = d * d
        d_minus_thickness = max(
            (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
        )
        if ((i > 0 and d_minus_thickness <= g) or i >= max_iters) and t < 1.0:
            wp.atomic_min(ccd_step, 0, t)
            return

        t = t + t_l
        if t > 1.0:
            return

        t_l = 0.8 * d_minus_thickness / l_v
        i += 1


@wp.kernel
def accd_wo_buffer_ee_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    slackness: float,
    thickness: float,
    radius: float,  # radius for contact detection
    ee_classify_thres: float,
    max_iters: int,
    ccd_step: wp.array(dtype=float),
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
    
    d_thres = 2.0 * radius + thickness
    
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
    
    v0 = v[i0]
    v1 = v[i1]
    v2 = v[i2]
    v3 = v[i3]

    v_avg = (v0 + v1 + v2 + v3) / 4.0
    v0 -= v_avg
    v1 -= v_avg
    v2 -= v_avg
    v3 -= v_avg

    l_v = wp.max(wp.length(v0), wp.length(v1)) + wp.max(
        wp.length(v2), wp.length(v3)
    )
    if l_v < 1e-12:
        return 
    
    type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d = ee_pair_distance(x0, x1, x2, x3, type2)
        
    if d < thickness:
        wp.printf("[accd_kernel] EE type2 = %d, x0 = (%f, %f, %f), x1 = (%f, %f, %f), x2 = (%f, %f, %f), x3 = (%f, %f, %f), d = %f, thickness = %f\n", type2, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2], d, thickness)

    d_sqr = d * d
    d_minus_thickness = max(
        (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
    )
    g = (1.0 - slackness) * d_minus_thickness
    t = float(0.0)
    t_l = slackness * d_minus_thickness / l_v

    i = int(0)

    while True:
        x0 += t_l * v0
        x1 += t_l * v1
        x2 += t_l * v2
        x3 += t_l * v3

        type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
        d = ee_pair_distance(x0, x1, x2, x3, type2)

        d_sqr = d * d
        d_minus_thickness = max(
            (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
        )
        if ((i > 0 and d_minus_thickness <= g) or i >= max_iters) and t < 1.0:
            wp.atomic_min(ccd_step, 0, t)
            return

        t = t + t_l
        if t > 1.0:
            return

        t_l = 0.8 * d_minus_thickness / l_v
        i += 1
