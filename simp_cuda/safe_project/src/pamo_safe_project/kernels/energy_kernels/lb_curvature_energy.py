import warp as wp


@wp.kernel
def lb_curvature_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    LB_indices: wp.array(dtype=wp.int32),
    LB_indptr: wp.array(dtype=wp.int32),
    LB_data: wp.array(dtype=wp.float32),
    curv_rest: wp.array(dtype=wp.float32),
    curv_stiffness: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    curv = wp.vec3(0.0)  # curv = LB @ x, where x is [NP, 3]
    for j_idx in range(LB_indptr[i], LB_indptr[i + 1]):
        j = LB_indices[j_idx]
        curv += LB_data[j_idx] * x[j]

    delta_curv = wp.length(curv) - curv_rest[i]
    E = 0.5 * delta_curv * delta_curv * curv_stiffness
    wp.atomic_add(energy, 0, E)


@wp.kernel
def lb_curvature_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    LB_indices: wp.array(dtype=wp.int32),
    LB_indptr: wp.array(dtype=wp.int32),
    LB_data: wp.array(dtype=wp.float32),
    curv_rest: wp.array(dtype=wp.float32),
    curv_stiffness: wp.float32,
    coeff: wp.float32,
    curv_: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    curv = wp.vec3(0.0)  # curv = LB @ x, where x is [NP, 3]
    hess_diag_i = wp.vec3(0.0)
    for j_idx in range(LB_indptr[i], LB_indptr[i + 1]):
        j = LB_indices[j_idx]
        curv += LB_data[j_idx] * x[j]
        if i == j:
            hess_diag_i = wp.vec3(LB_data[j_idx] * LB_data[j_idx])

    curv_[i] = curv

    delta_curv = wp.length(curv) - curv_rest[i]
    dE_dcurv = delta_curv * curv_stiffness
    d2E_dcurv2 = curv_stiffness
    # dcurv_dx[i] = LB[i]
    # dE_dx = dE_dcurv * dcurv_dx
    # d2E_dx2 = d2E_dcurv2 * dcurv_dx * dcurv_dx.T + dE_dcurv * d2curv_dx2
    #     ignore d2curv_dx2

    for j_idx in range(LB_indptr[i], LB_indptr[i + 1]):
        j = LB_indices[j_idx]
        wp.atomic_add(grad, j, wp.vec3(coeff * dE_dcurv * LB_data[j_idx]))

    wp.atomic_add(hess_diag, i, wp.vec3(hess_diag_i * d2E_dcurv2))


@wp.kernel
def lb_curvature_hess_dx_kernel(
    x: wp.array(dtype=wp.vec3),
    LB_indices: wp.array(dtype=wp.int32),
    LB_indptr: wp.array(dtype=wp.int32),
    LB_data: wp.array(dtype=wp.float32),
    curv_rest: wp.array(dtype=wp.float32),
    curv_stiffness: wp.float32,
    curv_: wp.array(dtype=wp.vec3),
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    curv = curv_[i]
    delta_curv = wp.length(curv) - curv_rest[i]
    dE_dcurv = delta_curv * curv_stiffness
    d2E_dcurv2 = curv_stiffness

    for j_idx in range(LB_indptr[i], LB_indptr[i + 1]):
        j = LB_indices[j_idx]
        hess_dx_j_upd = wp.vec3(0.0)
        for k_idx in range(LB_indptr[i], LB_indptr[i + 1]):
            k = LB_indices[k_idx]
            hess_dx_j_upd += d2E_dcurv2 * LB_data[j_idx] * LB_data[k_idx] * dx[k]
        wp.atomic_add(hess_dx, j, hess_dx_j_upd)
