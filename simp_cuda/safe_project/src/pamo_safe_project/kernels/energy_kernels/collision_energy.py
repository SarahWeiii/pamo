import warp as wp

from ...defs import *
from ..distance_kernels.distance_kernels import *
from ..distance_kernels.distance_kernels_struct import *


@wp.func
def compute_b(d: float, d_hat: float):
    return wp.select(d >= d_hat, -(d - d_hat) * (d - d_hat) * wp.log(d / d_hat), 0.0)


@wp.func
def compute_db_dd(d: float, d_hat: float):
    t = d - d_hat
    return wp.select(d >= d_hat, -2.0 * t * wp.log(d / d_hat) - t * t / d, 0.0)


@wp.func
def compute_d2b_dd2(d: float, d_hat: float):
    return wp.select(
        d >= d_hat,
        -2.0 * wp.log(d / d_hat)
        - 2.0 * (d - d_hat) / d
        - 1.0
        + (d_hat * d_hat) / (d * d),
        0.0,
    )


@wp.func
def compute_e(c: float, eps_cross: float):
    t = c / eps_cross
    return wp.select(c < eps_cross, 1.0, -1.0 * t * t + 2.0 * t)


@wp.func
def compute_de_dc(c: float, eps_cross: float):
    return wp.select(
        c < eps_cross, 0.0, -2.0 * c / (eps_cross * eps_cross) + 2.0 / eps_cross
    )


@wp.func
def compute_d2e_dc2(c: float, eps_cross: float):
    return wp.select(c < eps_cross, 0.0, -2.0 / (eps_cross * eps_cross))


@wp.kernel
def collision_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    contact_counter: wp.array(dtype=wp.int32),
    b_types: wp.array(dtype=wp.int32, ndim=2),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    kappa: float,  # stiffness
    d_hat: float,  # collision barrier threshold distance
    ee_classify_thres: float,
    d_: wp.array(dtype=float),
    energy: wp.array(dtype=float),
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

    d = 0.0
    e = float(1.0)

    type1 = b_types[bid, 0]  # 1st layer type

    if type1 == BlockTypes.PT_CONTACT:
        type2 = pt_pair_classify(x0, x1, x2, x3)
        b_types[bid, 1] = type2
        d = pt_pair_distance(x0, x1, x2, x3, type2)

    elif type1 == BlockTypes.EE_CONTACT:
        type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
        b_types[bid, 1] = type2
        d = ee_pair_distance(x0, x1, x2, x3, type2)

    else:
        wp.printf("[collision_energy_kernel] Unknown block type: %d\n", type1)

    d_[bid] = d

    if d < d_hat:
        energy_upd = compute_b(d, d_hat)
        wp.atomic_add(
            energy, 0, energy_upd * kappa * e
        )  # E = kappa * e(c(x)) * b(d(x))


@wp.kernel
def collision_energy_wo_buffer_pt_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    kappa: float,  # stiffness
    d_hat: float,  # collision barrier threshold distance
    energy: wp.array(dtype=float),
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

    d_thres = d_hat
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

    type2 = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, type2)

    if d < d_hat:
        energy_upd = compute_b(d, d_hat)
        wp.atomic_add(energy, 0, energy_upd * kappa)  # E = kappa * e(c(x)) * b(d(x))


@wp.kernel
def collision_energy_wo_buffer_ee_kernel(
    x: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    kappa: float,  # stiffness
    ee_classify_thres: float,
    d_hat: float,  # collision barrier threshold distance
    energy: wp.array(dtype=float),
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

    d_thres = d_hat

    sphere_c_0 = (x0 + x1) / 2.0
    sphere_r_0 = wp.length(x0 - x1) / 2.0
    sphere_c_1 = (x2 + x3) / 2.0
    sphere_r_1 = wp.length(x2 - x3) / 2.0
    spheres_d = wp.length(sphere_c_0 - sphere_c_1) - sphere_r_0 - sphere_r_1
    if spheres_d > d_thres:
        return

    type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d = ee_pair_distance(x0, x1, x2, x3, type2)
    if d > d_thres:
        return

    if d < d_hat:
        energy_upd = compute_b(d, d_hat)
        wp.atomic_add(energy, 0, energy_upd * kappa)  # E = kappa * e(c(x)) * b(d(x))


@wp.kernel
def collision_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    contact_counter: wp.array(dtype=wp.int32),
    b_types: wp.array(dtype=wp.int32, ndim=2),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    kappa: float,
    d_hat: float,
    d_: wp.array(dtype=float),
    coeff: float,
    dd_dx_: wp.array(dtype=wp.vec3, ndim=2),
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
):
    bid = wp.tid()
    if bid >= contact_counter[0]:
        return

    d = d_[bid]
    if d >= d_hat:
        return

    i0 = b_indices[bid, 0]
    i1 = b_indices[bid, 1]
    i2 = b_indices[bid, 2]
    i3 = b_indices[bid, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    dd_dx = dd_dx_[bid]

    type1 = b_types[bid, 0]
    type2 = b_types[bid, 1]

    db_dd = compute_db_dd(d, d_hat)
    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    dd_coeff = db_dd * kappa * coeff
    d2d_coeff = d2b_dd2 * kappa

    dd_dx[0] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[1] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[2] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[3] = wp.vec3(0.0, 0.0, 0.0)

    if type1 == BlockTypes.PT_CONTACT:
        pt_pair_distance_grad(x0, x1, x2, x3, dd_dx, type2)
    elif type1 == BlockTypes.EE_CONTACT:
        ee_pair_distance_grad(x0, x1, x2, x3, dd_dx, type2)
    else:
        wp.printf("[collision_energy_kernel] Unknown block type: %d\n", type1)

    wp.atomic_add(grad, i0, dd_dx[0] * dd_coeff)
    wp.atomic_add(grad, i1, dd_dx[1] * dd_coeff)
    wp.atomic_add(grad, i2, dd_dx[2] * dd_coeff)
    wp.atomic_add(grad, i3, dd_dx[3] * dd_coeff)

    wp.atomic_add(hess_diag, i0, d2d_coeff * wp.cw_mul(dd_dx[0], dd_dx[0]))
    wp.atomic_add(hess_diag, i1, d2d_coeff * wp.cw_mul(dd_dx[1], dd_dx[1]))
    wp.atomic_add(hess_diag, i2, d2d_coeff * wp.cw_mul(dd_dx[2], dd_dx[2]))
    wp.atomic_add(hess_diag, i3, d2d_coeff * wp.cw_mul(dd_dx[3], dd_dx[3]))


@wp.kernel
def collision_diff_wo_buffer_pt_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    kappa: float,  # stiffness
    d_hat: float,  # collision barrier threshold distance
    coeff: float,
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
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

    d_thres = d_hat
    p_plane_d = pt_distance(x0, x1, x2, x3)
    if p_plane_d >= d_thres:
        return

    sphere_c = (x1 + x2 + x3) / 3.0
    sphere_r = wp.max(
        wp.max(wp.length(x1 - sphere_c), wp.length(x2 - sphere_c)),
        wp.length(x3 - sphere_c),
    )
    p_sphere_d = wp.length(x0 - sphere_c) - sphere_r
    if p_sphere_d >= d_thres:
        return

    type2 = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, type2)
    if d >= d_thres:
        return

    db_dd = compute_db_dd(d, d_hat)
    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    dd_coeff = db_dd * kappa * coeff
    d2d_coeff = d2b_dd2 * kappa

    dd_dx = pt_pair_distance_grad_struct(x0, x1, x2, x3, type2)

    wp.atomic_add(grad, i0, dd_dx.d0 * dd_coeff)
    wp.atomic_add(grad, i1, dd_dx.d1 * dd_coeff)
    wp.atomic_add(grad, i2, dd_dx.d2 * dd_coeff)
    wp.atomic_add(grad, i3, dd_dx.d3 * dd_coeff)

    wp.atomic_add(hess_diag, i0, d2d_coeff * wp.cw_mul(dd_dx.d0, dd_dx.d0))
    wp.atomic_add(hess_diag, i1, d2d_coeff * wp.cw_mul(dd_dx.d1, dd_dx.d1))
    wp.atomic_add(hess_diag, i2, d2d_coeff * wp.cw_mul(dd_dx.d2, dd_dx.d2))
    wp.atomic_add(hess_diag, i3, d2d_coeff * wp.cw_mul(dd_dx.d3, dd_dx.d3))


@wp.kernel
def collision_diff_wo_buffer_ee_kernel(
    x: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    kappa: float,  # stiffness
    d_hat: float,  # collision barrier threshold distance
    ee_classify_thres: float,
    coeff: float,
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
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

    d_thres = d_hat

    sphere_c_0 = (x0 + x1) / 2.0
    sphere_r_0 = wp.length(x0 - x1) / 2.0
    sphere_c_1 = (x2 + x3) / 2.0
    sphere_r_1 = wp.length(x2 - x3) / 2.0
    spheres_d = wp.length(sphere_c_0 - sphere_c_1) - sphere_r_0 - sphere_r_1
    if spheres_d >= d_thres:
        return

    type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d = ee_pair_distance(x0, x1, x2, x3, type2)
    if d >= d_thres:
        return

    db_dd = compute_db_dd(d, d_hat)
    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    dd_coeff = db_dd * kappa * coeff
    d2d_coeff = d2b_dd2 * kappa

    dd_dx = ee_pair_distance_grad_struct(x0, x1, x2, x3, type2)

    wp.atomic_add(grad, i0, dd_dx.d0 * dd_coeff)
    wp.atomic_add(grad, i1, dd_dx.d1 * dd_coeff)
    wp.atomic_add(grad, i2, dd_dx.d2 * dd_coeff)
    wp.atomic_add(grad, i3, dd_dx.d3 * dd_coeff)

    wp.atomic_add(hess_diag, i0, d2d_coeff * wp.cw_mul(dd_dx.d0, dd_dx.d0))
    wp.atomic_add(hess_diag, i1, d2d_coeff * wp.cw_mul(dd_dx.d1, dd_dx.d1))
    wp.atomic_add(hess_diag, i2, d2d_coeff * wp.cw_mul(dd_dx.d2, dd_dx.d2))
    wp.atomic_add(hess_diag, i3, d2d_coeff * wp.cw_mul(dd_dx.d3, dd_dx.d3))


@wp.kernel
def collision_hess_dx_kernel(
    counter: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    d_: wp.array(dtype=wp.float32),
    kappa: float,
    d_hat: float,
    dd_dx_: wp.array(dtype=wp.vec3, ndim=2),
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
):
    bid = wp.tid()
    if bid >= counter[0]:
        return

    d = d_[bid]
    if d >= d_hat:
        return

    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    d2b_coeff = d2b_dd2 * kappa

    for i in range(4):
        global_i = b_indices[bid, i]
        for j in range(4):
            global_j = b_indices[bid, j]
            wp.atomic_add(
                hess_dx,
                global_i,
                dd_dx_[bid, i] * wp.dot(dd_dx_[bid, j], dx[global_j]) * d2b_coeff,
            )


@wp.kernel
def collision_hess_dx_wo_buffer_pt_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=int, ndim=2),
    d_hat: float,
    kappa: float,
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
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

    d_thres = d_hat
    p_plane_d = pt_distance(x0, x1, x2, x3)
    if p_plane_d >= d_thres:
        return

    sphere_c = (x1 + x2 + x3) / 3.0
    sphere_r = wp.max(
        wp.max(wp.length(x1 - sphere_c), wp.length(x2 - sphere_c)),
        wp.length(x3 - sphere_c),
    )
    p_sphere_d = wp.length(x0 - sphere_c) - sphere_r
    if p_sphere_d >= d_thres:
        return

    type2 = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, type2)
    if d >= d_thres:
        return

    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    d2d_coeff = d2b_dd2 * kappa

    dd_dx = pt_pair_distance_grad_struct(x0, x1, x2, x3, type2)

    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    
@wp.kernel
def collision_hess_dx_wo_buffer_ee_kernel(
    x: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=int, ndim=2),
    d_hat: float,
    ee_classify_thres: float,
    kappa: float,
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
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

    d_thres = d_hat

    sphere_c_0 = (x0 + x1) / 2.0
    sphere_r_0 = wp.length(x0 - x1) / 2.0
    sphere_c_1 = (x2 + x3) / 2.0
    sphere_r_1 = wp.length(x2 - x3) / 2.0
    spheres_d = wp.length(sphere_c_0 - sphere_c_1) - sphere_r_0 - sphere_r_1
    if spheres_d >= d_thres:
        return

    type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d = ee_pair_distance(x0, x1, x2, x3, type2)
    if d >= d_thres:
        return
    
    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    d2d_coeff = d2b_dd2 * kappa

    dd_dx = ee_pair_distance_grad_struct(x0, x1, x2, x3, type2)

    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i0, dd_dx.d0 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i1, dd_dx.d1 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i2, dd_dx.d2 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
    
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d0, dx[i0]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d1, dx[i1]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d2, dx[i2]) * d2d_coeff)
    wp.atomic_add(hess_dx, i3, dd_dx.d3 * wp.dot(dd_dx.d3, dx[i3]) * d2d_coeff)
