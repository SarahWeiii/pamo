import warp as wp


@wp.kernel
def update_p_r_z_compute_zr_kernel(
    v: wp.array(dtype=wp.vec3),
    A_v: wp.array(dtype=wp.vec3),
    v_A_v: wp.array(dtype=float),
    zr: wp.array(dtype=float),
    diag: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
    zr_new: wp.array(dtype=float),
):
    tid = wp.tid()

    if v_A_v[0] > 1e-16:
        alpha = zr[0] / v_A_v[0]
        p[tid] = p[tid] + alpha * v[tid]
        r[tid] = r[tid] - alpha * A_v[tid]
    z[tid] = wp.cw_div(r[tid], diag[tid])
    wp.atomic_add(zr_new, 0, wp.dot(z[tid], r[tid]))


@wp.kernel
def update_v_kernel(
    z: wp.array(dtype=wp.vec3),
    zr: wp.array(dtype=float),
    zr_new: wp.array(dtype=float),
    v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    s = zr_new[0] / zr[0]
    v[tid] = z[tid] + s * v[tid]


@wp.kernel
def compute_dot_kernel(
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=float),
):
    tid = wp.tid()

    wp.atomic_add(ret, 0, wp.dot(x[tid], y[tid]))
    

@wp.kernel
def compute_block_diag_inv_kernel(
    diag: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    # z = r / diag
    tid = wp.tid()
    z[tid] = wp.cw_div(r[tid], diag[tid] + wp.vec3(1e-6))


@wp.kernel
def line_search_kernel(
    q: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    n_halves: float,
    energy_prev: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    q_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if n_halves > 0.0 and energy[0] < energy_prev[0]:
        return
    q_new[tid] = q[tid] + alpha[0] * wp.pow(0.5, n_halves) * p[tid]


@wp.kernel
def clamp_p_kernel(
    p: wp.array(dtype=wp.vec3),
    q_prev_newton: wp.array(dtype=wp.vec3),
    q_prev_detection: wp.array(dtype=wp.vec3),
    radius: float,
):
    i = wp.tid()
    p_i = p[i]
    q_n_i = q_prev_newton[i]
    q_d_i = q_prev_detection[i]
    # Wanna go to q = q_n_i + p_i
    # but requires |q - q_d_i| <= radius
    delta_q = q_n_i + p_i - q_d_i
    
    if wp.length(delta_q) > radius:
        p_i_new = wp.normalize(delta_q) * radius + q_d_i - q_n_i
        p[i] = p_i_new
