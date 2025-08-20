import warp as wp


mat23 = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.kernel
def elastic_preprocess_kernel(
    x_rest: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.int32, ndim=2),
    areas: wp.array(dtype=wp.float32),
    inv_Dm_: wp.array(dtype=wp.mat22),
):
    tid = wp.tid()

    i0 = triangles[tid, 0]
    i1 = triangles[tid, 1]
    i2 = triangles[tid, 2]

    x0 = x_rest[i0]
    x1 = x_rest[i1]
    x2 = x_rest[i2]

    x01 = x1 - x0
    x02 = x2 - x0

    W = 0.5 * wp.length(wp.cross(x01, x02))

    b1 = wp.normalize(x01)
    b2 = x02
    b2 = b2 - wp.dot(b2, b1) * b1
    b2 = wp.normalize(b2)

    Dm = wp.mat22(wp.dot(b1, x01), wp.dot(b1, x02), wp.dot(b2, x01), wp.dot(b2, x02))
    inv_Dm = wp.inverse(Dm)

    areas[tid] = W
    inv_Dm_[tid] = inv_Dm


@wp.kernel
def elastic_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.int32, ndim=2),
    areas: wp.array(dtype=wp.float32),
    inv_Dm_: wp.array(dtype=wp.mat22),
    mu: wp.float32,
    la: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    i0 = triangles[tid, 0]
    i1 = triangles[tid, 1]
    i2 = triangles[tid, 2]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]

    W = areas[tid]
    inv_Dm = inv_Dm_[tid]

    x01 = x1 - x0
    x02 = x2 - x0

    b1 = wp.normalize(x01)
    b2 = x02
    b2 = b2 - wp.dot(b2, b1) * b1
    b2 = wp.normalize(b2)

    Ds = wp.mat22(
        wp.dot(b1, x01),
        wp.dot(b1, x02),
        wp.dot(b2, x01),
        wp.dot(b2, x02),
    )
    F = Ds * inv_Dm

    I = wp.mat22(1.0, 0.0, 0.0, 1.0)
    E = 0.5 * (wp.transpose(F) * F - I)
    tr_E = wp.trace(E)

    elas_energy = W * (mu * wp.trace(E * wp.transpose(E)) + 0.5 * la * tr_E * tr_E)
    wp.atomic_add(energy, 0, elas_energy)


@wp.kernel
def elastic_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.int32, ndim=2),
    areas: wp.array(dtype=wp.float32),
    inv_Dm_: wp.array(dtype=wp.mat22),
    mu: wp.float32,
    la: wp.float32,
    coeff: wp.float32,
    grad: wp.array(dtype=wp.vec3),
    hess_diag: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i0 = triangles[tid, 0]
    i1 = triangles[tid, 1]
    i2 = triangles[tid, 2]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]

    W = areas[tid]
    inv_Dm = inv_Dm_[tid]
    
    x01 = x1 - x0
    x02 = x2 - x0

    b1 = wp.normalize(x01)
    b2 = x02
    b2 = b2 - wp.dot(b2, b1) * b1
    b2 = wp.normalize(b2)

    Ds = wp.mat22(
        wp.dot(b1, x01),
        wp.dot(b1, x02),
        wp.dot(b2, x01),
        wp.dot(b2, x02),
    )
    F = Ds * inv_Dm
    
    I = wp.mat22(1.0, 0.0, 0.0, 1.0)
    E = 0.5 * (wp.transpose(F) * F - I)
    tr_E = wp.trace(E)
    
    I = wp.mat22(1.0, 0.0, 0.0, 1.0)
    P = F * (2.0 * mu * E + la * tr_E * I)
    H = coeff * W * P * wp.transpose(inv_Dm)  # H = [f1, f2] in 2D local space
    
    f1 = H[0, 0] * b1 + H[1, 0] * b2
    f2 = H[0, 1] * b1 + H[1, 1] * b2
    f0 = -f1 - f2
    
    wp.atomic_add(grad, i0, f0)
    wp.atomic_add(grad, i1, f1)
    wp.atomic_add(grad, i2, f2)
    
    # TODO: compute hess_diag
    
    
@wp.kernel
def elastic_hess_dx_kernel(
    x: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.int32, ndim=2),
    areas: wp.array(dtype=wp.float32),
    inv_Dm_: wp.array(dtype=wp.mat22),
    mu: wp.float32,
    la: wp.float32,
    dx: wp.array(dtype=wp.vec3),
    hess_dx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i0 = triangles[tid, 0]
    i1 = triangles[tid, 1]
    i2 = triangles[tid, 2]
    
    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    
    dx0 = dx[i0]
    dx1 = dx[i1]
    dx2 = dx[i2]
    
    W = areas[tid]
    inv_Dm = inv_Dm_[tid]
    
    x01 = x1 - x0
    x02 = x2 - x0
    
    dx01 = dx1 - dx0
    dx02 = dx2 - dx0

    b1 = wp.normalize(x01)
    b2 = x02
    b2 = b2 - wp.dot(b2, b1) * b1
    b2 = wp.normalize(b2)

    Ds = wp.mat22(
        wp.dot(b1, x01),
        wp.dot(b1, x02),
        wp.dot(b2, x01),
        wp.dot(b2, x02),
    )
    dDs = wp.mat22(
        wp.dot(b1, dx01),
        wp.dot(b1, dx02),
        wp.dot(b2, dx01),
        wp.dot(b2, dx02),
    )
    
    F = Ds * inv_Dm
    dF = dDs * inv_Dm
    
    I = wp.mat22(1.0, 0.0, 0.0, 1.0)
    E = 0.5 * (wp.transpose(F) * F - I)
    tr_E = wp.trace(E)
    dE = 0.5 * (wp.transpose(dF) * F + wp.transpose(F) * dF)
    tr_dE = wp.trace(dE)
    
    dP = dF * (2.0 * mu * E + la * tr_E * I) \
        + F * (2.0 * mu * dE + la * tr_dE * I)

    dH = W * dP * wp.transpose(inv_Dm)
    
    df1 = dH[0, 0] * b1 + dH[1, 0] * b2  # actually -df1
    df2 = dH[0, 1] * b1 + dH[1, 1] * b2  # actually -df2
    df0 = -df1 - df2  # actually -df0
    
    wp.atomic_add(hess_dx, i0, df0)
    wp.atomic_add(hess_dx, i1, df1)
    wp.atomic_add(hess_dx, i2, df2)
    