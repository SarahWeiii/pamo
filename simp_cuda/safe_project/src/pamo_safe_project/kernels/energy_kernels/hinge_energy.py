import warp as wp


@wp.func
def compute_hinge_angle(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    #     0
    #   /   \
    #  1 --- 2
    #   \   /
    #     3

    e12 = wp.normalize(x2 - x1)
    n1 = wp.normalize(wp.cross(e12, x0 - x1))
    n2 = wp.normalize(wp.cross(x3 - x1, e12))
    cos_theta = wp.dot(n1, n2)
    sin_theta = wp.dot(wp.cross(n1, n2), e12)
    theta = wp.atan2(sin_theta, cos_theta)
    return theta


@wp.kernel
def hinge_preprocess_slow_kernel(
    x_rest: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.int32, ndim=2),
    hinge_counter: wp.array(dtype=wp.int32),
    hinge_indices: wp.array(dtype=wp.int32, ndim=2),
    rest_angles: wp.array(dtype=wp.float32),
    rest_elens: wp.array(dtype=wp.float32),
):
    t0, t1 = wp.tid()

    for e0 in range(3):
        for e1 in range(3):
            i00 = triangles[t0, e0]
            i01 = triangles[t0, (e0 + 1) % 3]
            i10 = triangles[t1, e1]
            i11 = triangles[t1, (e1 + 1) % 3]

            if i00 == i11 and i01 == i10 and i00 < i01:
                i0 = triangles[t0, (e0 + 2) % 3]
                i1 = i00
                i2 = i01
                i3 = triangles[t1, (e1 + 2) % 3]

                hinge_id = wp.atomic_add(hinge_counter, 0, 1)
                hinge_indices[hinge_id, 0] = i0
                hinge_indices[hinge_id, 1] = i1
                hinge_indices[hinge_id, 2] = i2
                hinge_indices[hinge_id, 3] = i3

                x0 = x_rest[i0]
                x1 = x_rest[i1]
                x2 = x_rest[i2]
                x3 = x_rest[i3]

                rest_angles[hinge_id] = compute_hinge_angle(x0, x1, x2, x3)
                rest_elens[hinge_id] = wp.length(x1 - x2)


@wp.kernel
def hinge_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32, ndim=2),
    rest_angles: wp.array(dtype=wp.float32),
    rest_elens: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    hid = wp.tid()

    i0 = indices[hid, 0]
    i1 = indices[hid, 1]
    i2 = indices[hid, 2]
    i3 = indices[hid, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    rest_angle = rest_angles[hid]
    rest_elen = rest_elens[hid]

    theta = compute_hinge_angle(x0, x1, x2, x3)
    delta_theta = theta - rest_angle
    energy_upd = 0.5 * stiffness * delta_theta * delta_theta * rest_elen

    wp.atomic_add(energy, 0, energy_upd)


@wp.func
def _symmetric(A: wp.mat33):
    return A + wp.transpose(A)


@wp.kernel
def hinge_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32, ndim=2),
    rest_angles: wp.array(dtype=wp.float32),
    rest_elens: wp.array(dtype=wp.float32),
    stiffness: wp.float32,
    coeff: wp.float32,
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
):
    hid = wp.tid()

    block = blocks[hid]

    rest_angle = rest_angles[hid]
    rest_elen = rest_elens[hid]

    i0 = indices[hid, 0]
    i1 = indices[hid, 1]
    i2 = indices[hid, 2]
    i3 = indices[hid, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_ = x3 - x2
    e2_ = x3 - x1

    e0_lensq = wp.length_sq(e0)
    e1_lensq = wp.length_sq(e1)
    e2_lensq = wp.length_sq(e2)
    e1_lensq_ = wp.length_sq(e1_)
    e2_lensq_ = wp.length_sq(e2_)

    e0_len = wp.sqrt(e0_lensq)
    e1_len = wp.sqrt(e1_lensq)
    e2_len = wp.sqrt(e2_lensq)
    e1_len_ = wp.sqrt(e1_lensq_)
    e2_len_ = wp.sqrt(e2_lensq_)

    e0_unit = wp.normalize(e0)
    e1_unit = wp.normalize(e1)
    e2_unit = wp.normalize(e2)
    e1_unit_ = wp.normalize(e1_)
    e2_unit_ = wp.normalize(e2_)

    cos_alpha1 = wp.dot(e2_unit, e0_unit)
    cos_alpha2 = -wp.dot(e1_unit, e0_unit)
    cos_alpha1_ = wp.dot(e2_unit_, e0_unit)
    cos_alpha2_ = -wp.dot(e1_unit_, e0_unit)

    h0 = wp.length(wp.cross(e1, e2)) / e0_len
    h1 = wp.length(wp.cross(e0, e2)) / e1_len
    h2 = wp.length(wp.cross(e0, e1)) / e2_len
    h0_ = wp.length(wp.cross(e1_, e2_)) / e0_len
    h1_ = wp.length(wp.cross(e0, e2_)) / e1_len_
    h2_ = wp.length(wp.cross(e0, e1_)) / e2_len_

    n = wp.normalize(wp.cross(e0, e2))
    n_ = wp.normalize(wp.cross(e2_, e0))

    cos_theta = wp.dot(n, n_)
    sin_theta = wp.dot(wp.cross(n, n_), e0) / e0_len
    theta = wp.atan2(sin_theta, cos_theta)  # range (-pi, pi)

    delta_theta = theta - rest_angle
    dE_dtheta = stiffness * delta_theta * rest_elen
    d2E_dtheta2 = stiffness * rest_elen

    dtheta_dx1 = n * (cos_alpha2 / h1) + n_ * (cos_alpha2_ / h1_)
    dtheta_dx2 = n * (cos_alpha1 / h2) + n_ * (cos_alpha1_ / h2_)
    dtheta_dx0 = n * (-1.0 / h0)
    dtheta_dx3 = n_ * (-1.0 / h0_)

    wp.atomic_add(grad, i0, dE_dtheta * dtheta_dx0 * coeff)
    wp.atomic_add(grad, i1, dE_dtheta * dtheta_dx1 * coeff)
    wp.atomic_add(grad, i2, dE_dtheta * dtheta_dx2 * coeff)
    wp.atomic_add(grad, i3, dE_dtheta * dtheta_dx3 * coeff)

    m0 = wp.normalize(wp.cross(e0, n))
    m1 = wp.normalize(wp.cross(e1, n))
    m2 = wp.normalize(wp.cross(n, e2))
    m0_ = wp.normalize(wp.cross(n_, e0))
    m1_ = wp.normalize(wp.cross(n_, e1_))
    m2_ = wp.normalize(wp.cross(e2_, n_))

    M0 = wp.outer(n, m0)
    M1 = wp.outer(n, m1)
    M2 = wp.outer(n, m2)
    M0_ = wp.outer(n_, m0_)
    M1_ = wp.outer(n_, m1_)
    M2_ = wp.outer(n_, m2_)

    N0 = M0 / e0_lensq
    N0_ = M0_ / e0_lensq

    P10 = cos_alpha1 * wp.transpose(M0) / (h1 * h0)
    P11 = cos_alpha1 * wp.transpose(M1) / (h1 * h1)
    P12 = cos_alpha1 * wp.transpose(M2) / (h1 * h2)
    P20 = cos_alpha2 * wp.transpose(M0) / (h2 * h0)
    P21 = cos_alpha2 * wp.transpose(M1) / (h2 * h1)
    P22 = cos_alpha2 * wp.transpose(M2) / (h2 * h2)
    P10_ = cos_alpha1_ * wp.transpose(M0_) / (h1_ * h0_)
    P11_ = cos_alpha1_ * wp.transpose(M1_) / (h1_ * h1_)
    P12_ = cos_alpha1_ * wp.transpose(M2_) / (h1_ * h2_)
    P20_ = cos_alpha2_ * wp.transpose(M0_) / (h2_ * h0_)
    P21_ = cos_alpha2_ * wp.transpose(M1_) / (h2_ * h1_)
    P22_ = cos_alpha2_ * wp.transpose(M2_) / (h2_ * h2_)

    Q0 = M0 / (h0 * h0)
    Q1 = M1 / (h0 * h1)
    Q2 = M2 / (h0 * h2)
    Q0_ = M0_ / (h0_ * h0_)
    Q1_ = M1_ / (h0_ * h1_)
    Q2_ = M2_ / (h0_ * h2_)

    block[0, 0] = -_symmetric(Q0) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx0, dtheta_dx0
    )
    block[3, 3] = -_symmetric(Q0_) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx3, dtheta_dx3
    )
    block[1, 1] = (
        _symmetric(P11) - N0 + _symmetric(P11_) - N0_
    ) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx1)
    block[2, 2] = (
        _symmetric(P22) - N0 + _symmetric(P22_) - N0_
    ) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx2)
    block[1, 0] = (P10 - Q1) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx1, dtheta_dx0
    )
    block[2, 0] = (P20 - Q2) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx2, dtheta_dx0
    )
    block[1, 3] = (P10_ - Q1_) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx1, dtheta_dx3
    )
    block[2, 3] = (P20_ - Q2_) * dE_dtheta + d2E_dtheta2 * wp.outer(
        dtheta_dx2, dtheta_dx3
    )
    block[1, 2] = (
        P12 + wp.transpose(P21) + N0 + P12_ + wp.transpose(P21_) + N0_
    ) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx2)
    block[0, 3] = d2E_dtheta2 * wp.outer(dtheta_dx0, dtheta_dx3)

    block[0, 1] = wp.transpose(block[1, 0])
    block[0, 2] = wp.transpose(block[2, 0])
    block[3, 1] = wp.transpose(block[1, 3])
    block[3, 2] = wp.transpose(block[2, 3])
    block[2, 1] = wp.transpose(block[1, 2])
    block[3, 0] = wp.transpose(block[0, 3])

    for i in range(4):
        wp.atomic_add(hess_diag, indices[hid, i], wp.get_diag(block[i, i]))


@wp.kernel
def hinge_hess_dx_kernel(
    indices: wp.array(dtype=wp.int32, ndim=2),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
):
    hid = wp.tid()

    for i_local in range(4):
        i_global = indices[hid, i_local]
        for j_local in range(4):
            j_global = indices[hid, j_local]
            wp.atomic_add(
                hess_dx, i_global, blocks[hid, i_local, j_local] * dx[j_global]
            )
