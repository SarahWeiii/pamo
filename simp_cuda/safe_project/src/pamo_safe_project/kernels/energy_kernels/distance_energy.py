import warp as wp
from ..distance_kernels.distance_kernels import *
from ..distance_kernels.grad_funcs import *


@wp.kernel
def mesh2gt_pp_distance_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    w = weights[i] * stiffness
    E = 0.5 * wp.length_sq(x[i] - target[i]) * w
    wp.atomic_add(energy, 0, E)


@wp.kernel
def mesh2gt_pp_distance_energy_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    grad_coeff: wp.float32,
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    w = weights[i] * stiffness
    g = (x[i] - target[i]) * w
    wp.atomic_add(grad, i, g * grad_coeff)
    wp.atomic_add(hess_diag, i, wp.vec3(w))


@wp.kernel
def mesh2gt_pp_distance_energy_hess_dx_kernel(
    weights: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    w = weights[i] * stiffness
    wp.atomic_add(hess_dx, i, dx[i] * w)


@wp.kernel
def gt2mesh_pt_distance_energy_kernel(
    x: wp.array(dtype=wp.vec3),  # [NP]
    gt_samples: wp.array(dtype=wp.vec3),  # [NS_GT]
    closest_tids: wp.array(dtype=wp.int32),  # [NS_GT]
    triangles: wp.array(dtype=wp.int32, ndim=2),  # [NF, 3]
    weights: wp.array(dtype=wp.float32),  # [NS_GT]
    stiffness: wp.float32,
    pt_types: wp.array(dtype=wp.int32),  # [NS_GT]
    d_: wp.array(dtype=wp.float32),  # [NS_GT]
    energy: wp.array(dtype=wp.float32),
):
    i0 = wp.tid()
    w = weights[i0] * stiffness

    tid = closest_tids[i0]
    i1 = triangles[tid, 0]
    i2 = triangles[tid, 1]
    i3 = triangles[tid, 2]

    x0 = gt_samples[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d = pt_pair_distance(x0, x1, x2, x3, pt_type)
    pt_types[i0] = pt_type
    d_[i0] = d

    E = 0.5 * d * d * w
    wp.atomic_add(energy, 0, E)


@wp.kernel
def gt2mesh_pt_distance_energy_diff_kernel(
    x: wp.array(dtype=wp.vec3),  # [NP]
    gt_samples: wp.array(dtype=wp.vec3),  # [NS_GT]
    closest_tids: wp.array(dtype=wp.int32),  # [NS_GT]
    triangles: wp.array(dtype=wp.int32, ndim=2),  # [NF, 3]
    weights: wp.array(dtype=wp.float32),  # [NS_GT]
    stiffness: wp.float32,
    pt_types: wp.array(dtype=wp.int32),  # [NS_GT]
    d_: wp.array(dtype=wp.float32),  # [NS_GT]
    coeff: wp.float32,
    dd_dx_: wp.array(dtype=wp.vec3, ndim=2),  # [NS_GT, 2]
    grad: wp.array(dtype=wp.vec3),  # [NP]
    hess_diag: wp.array(dtype=wp.vec3),  # [NP]
):
    i0 = wp.tid()
    w = weights[i0] * stiffness

    tid = closest_tids[i0]
    i1 = triangles[tid, 0]
    i2 = triangles[tid, 1]
    i3 = triangles[tid, 2]

    x0 = gt_samples[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    pt_type = pt_types[i0]
    d = d_[i0]

    dE_dd = d * w
    d2E_dd2 = w
    dd_coeff = dE_dd * coeff

    dd_dx = dd_dx_[i0]

    dd_dx[0] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[1] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[2] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[3] = wp.vec3(0.0, 0.0, 0.0)

    pt_pair_distance_grad(x0, x1, x2, x3, dd_dx, pt_type)

    wp.atomic_add(grad, i1, dd_dx[1] * dd_coeff)
    wp.atomic_add(grad, i2, dd_dx[2] * dd_coeff)
    wp.atomic_add(grad, i3, dd_dx[3] * dd_coeff)

    # dE/dx = dE/dd * dd/dx
    # d2E/dx2 = d2E/dd2 * outer(dd/dx, dd/dx) + dE/dd * d2d/dx2
    # ignore d2d/dx2, then
    # d2E/dx2 ~= d2E/dd2 * outer(dd/dx, dd/dx)

    wp.atomic_add(hess_diag, i1, d2E_dd2 * wp.cw_mul(dd_dx[1], dd_dx[1]))
    wp.atomic_add(hess_diag, i2, d2E_dd2 * wp.cw_mul(dd_dx[2], dd_dx[2]))
    wp.atomic_add(hess_diag, i3, d2E_dd2 * wp.cw_mul(dd_dx[3], dd_dx[3]))


@wp.kernel
def gt2mesh_pt_distance_energy_hess_dx_kernel(
    closest_tids: wp.array(dtype=wp.int32),  # [NS_GT]
    triangles: wp.array(dtype=wp.int32, ndim=2),  # [NF, 3]
    weights: wp.array(dtype=wp.float32),  # [NS_GT]
    stiffness: wp.float32,
    dd_dx_: wp.array(dtype=wp.vec3, ndim=2),  # [NS_GT, 4]
    dx: wp.array(dtype=wp.vec3),  # [NP]
    hess_dx: wp.array(dtype=wp.vec3),  # [NP]
):
    i0 = wp.tid()
    w = weights[i0] * stiffness

    tid = closest_tids[i0]

    d2E_dd2 = w

    for i in range(3):
        global_i = triangles[tid, i]
        for j in range(3):
            global_j = triangles[tid, j]
            wp.atomic_add(
                hess_dx,
                global_i,
                dd_dx_[i0, i] * wp.dot(dd_dx_[i0, j], dx[global_j]) * d2E_dd2,
            )
