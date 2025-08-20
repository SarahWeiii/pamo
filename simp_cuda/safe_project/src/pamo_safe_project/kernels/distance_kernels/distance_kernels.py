import warp as wp
from .grad_funcs import *
from .hessian_funcs import *

from ...defs import *


@wp.func
def pt_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    return wp.abs(wp.dot(x0 - x1, wp.normalize(wp.cross(x2 - x1, x3 - x1))))


@wp.func
def ee_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    return wp.abs(wp.dot(x0 - x2, wp.normalize(wp.cross(x1 - x0, x3 - x2))))


@wp.func
def pe_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3):
    return wp.length(wp.cross(x1 - x0, x2 - x0)) / wp.length(x1 - x2)


@wp.func
def pp_distance(x0: wp.vec3, x1: wp.vec3):
    return wp.length(x1 - x0)


@wp.func
def pt_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    # find closest point inside triangle
    x10 = x0 - x1
    x12 = x2 - x1
    x13 = x3 - x1

    n = wp.normalize(wp.cross(x12, x13))
    x0_proj = x0 - wp.dot(n, x10) * n
    x10 = x0_proj - x1
    x20 = x0_proj - x2
    x30 = x0_proj - x3

    # Barycentric coordinate from "A (beta, gamma)^T = b"
    a00 = wp.dot(x12, x12)
    a01 = wp.dot(x12, x13)
    a11 = wp.dot(x13, x13)
    b0 = wp.dot(x10, x12)
    b1 = wp.dot(x10, x13)
    det_A = a00 * a11 - a01 * a01
    if det_A == 0.0:
        wp.printf("[contact] Warning: surface triangle area is too small. \n")
        return x1
    beta = (b0 * a11 - b1 * a01) / det_A
    gamma = (a00 * b1 - b0 * a01) / det_A
    alpha = 1.0 - beta - gamma

    return alpha * x1 + beta * x2 + gamma * x3


@wp.func
def pe_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3):
    x10 = x0 - x1
    x12 = x2 - x1

    if wp.length_sq(x12) == 0.0:
        return x1

    x0_proj = x1 + wp.dot(x10, x12) / wp.dot(x12, x12) * x12
    return x0_proj


@wp.func
def ee_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    return wp.abs(wp.dot(x0 - x2, wp.normalize(wp.cross(x1 - x0, x3 - x2))))


@wp.func
def pt_pair_classify(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    # Project x0 onto the plane
    x10 = x0 - x1
    x12 = x2 - x1
    x13 = x3 - x1

    area_sq = wp.length_sq(wp.cross(x12, x13))

    if area_sq <= 1e-6 * (wp.length_sq(x12) * wp.length_sq(x13)) or area_sq <= 1e-16:
        # wp.printf("[contact] Warning: surface triangle area is too small. \n")
        d1 = wp.length(x10)
        d2 = wp.length(x0 - x2)
        d3 = wp.length(x0 - x3)
        if d1 < d2:
            if d1 < d3:
                return PTContactTypes.PP01
            else:
                return PTContactTypes.PP03
        else:
            if d2 < d3:
                return PTContactTypes.PP02
            else:
                return PTContactTypes.PP03

    n = wp.normalize(wp.cross(x12, x13))
    x0_proj = x0 - wp.dot(n, x10) * n
    x10 = x0_proj - x1
    x20 = x0_proj - x2
    x30 = x0_proj - x3

    # Barycentric coordinate from "A (beta, gamma)^T = b"
    a00 = wp.dot(x12, x12)
    a01 = wp.dot(x12, x13)
    a11 = wp.dot(x13, x13)
    b0 = wp.dot(x10, x12)
    b1 = wp.dot(x10, x13)
    det_A = a00 * a11 - a01 * a01
    beta = b0 * a11 - b1 * a01
    gamma = a00 * b1 - b0 * a01
    alpha = det_A - beta - gamma

    # wp.printf("[contact] alpha = %f, beta = %f, gamma = %f\n", alpha, beta, gamma)
    # wp.printf("[contact] x0_proj = (%f, %f, %f)\n", x0_proj[0], x0_proj[1], x0_proj[2])
    # x0_proj_r = (alpha * x1 + beta * x2 + gamma * x3) / det_A
    # wp.printf("[contact] reconstructed x0_proj = (%f, %f, %f)\n", x0_proj_r[0], x0_proj_r[1], x0_proj_r[2])

    n_negative_bary = wp.step(alpha) + wp.step(beta) + wp.step(gamma)

    if n_negative_bary == 0.0:
        # x0_proj inside triangle, distance type is "point-triangle".
        return PTContactTypes.PT
    elif n_negative_bary == 1.0:
        # Edge region
        # Project x0 onto the corresponding edge,
        #     if on the edge, distance type is "point-edge";
        #     if on the side of one vertex, distance type is "point-point".
        if alpha < 0.0:  # Edge x2x3
            x23 = x3 - x2
            x20 = x0 - x2
            dot_res = wp.dot(x23, x20)
            if dot_res < 0.0:  # On the side of x2
                return PTContactTypes.PP02
            elif dot_res > wp.dot(x23, x23):  # On the side of x3
                return PTContactTypes.PP03
            else:  # On edge x2x3
                return PTContactTypes.PE023
        elif beta < 0.0:  # Edge x3x1
            x31 = x1 - x3
            x30 = x0 - x3
            dot_res = wp.dot(x31, x30)
            if dot_res < 0.0:  # On the side of x3
                return PTContactTypes.PP03
            elif dot_res > wp.dot(x31, x31):  # On the side of x1
                return PTContactTypes.PP01
            else:  # On edge x3x1
                return PTContactTypes.PE031
        elif gamma < 0.0:  # Edge x1x2
            dot_res = wp.dot(x12, x10)
            if dot_res < 0.0:  # On the side of x1
                return PTContactTypes.PP01
            elif dot_res > wp.dot(x12, x12):  # On the side of x2
                return PTContactTypes.PP02
            else:  # On edge x1x2
                return PTContactTypes.PE012
        else:
            wp.printf(
                "[contact] Error! point_triangle_distance_classify_kernel error 1\n"
            )
    # elif n_negative_bary == 2.0:
    else:
        if alpha >= 0.0:
            x12 = x2 - x1
            x13 = x3 - x1
            dot12 = wp.dot(x12, x10)
            dot13 = wp.dot(x13, x10)
            if dot12 > 0.0 and dot13 <= 0.0:  # On edge x1x2
                if wp.dot(x12, x20) > 0.0:
                    return PTContactTypes.PP02
                else:
                    return PTContactTypes.PE012
            elif dot13 > 0.0 and dot12 <= 0.0:  # On edge x1x3
                if wp.dot(x13, x30) > 0.0:
                    return PTContactTypes.PP03
                else:
                    return PTContactTypes.PE031
            else:
                return PTContactTypes.PP01
        elif beta >= 0.0:
            x23 = x3 - x2
            x21 = x1 - x2
            dot23 = wp.dot(x23, x20)
            dot21 = wp.dot(x21, x20)
            if dot23 > 0.0 and dot21 <= 0.0:  # On edge x2x3
                if wp.dot(x23, x30) > 0.0:
                    return PTContactTypes.PP03
                else:
                    return PTContactTypes.PE023
            elif dot21 > 0.0 and dot23 <= 0.0:  # On edge x2x1
                if wp.dot(x21, x10) > 0.0:
                    return PTContactTypes.PP01
                else:
                    return PTContactTypes.PE012
            else:
                return PTContactTypes.PP02
        # elif gamma >= 0.0:
        else:
            x31 = x1 - x3
            x32 = x2 - x3
            dot31 = wp.dot(x31, x30)
            dot32 = wp.dot(x32, x30)
            if dot31 > 0.0 and dot32 <= 0.0:  # On edge x3x1
                if wp.dot(x31, x10) > 0.0:
                    return PTContactTypes.PP01
                else:
                    return PTContactTypes.PE031
            elif dot32 > 0.0 and dot31 <= 0.0:  # On edge x3x2
                if wp.dot(x32, x20) > 0.0:
                    return PTContactTypes.PP02
                else:
                    return PTContactTypes.PE023
            else:
                return PTContactTypes.PP03
        # else:
        #     wp.printf(
        #         "[contact] Error! point_triangle_distance_classify_kernel error 2\n"
        #     )
    # else:
    #     wp.printf("[contact] Error! point_triangle_distance_classify_kernel error 3\n")

    # wp.printf("[point = [%f, %f, %f], tri = [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]\n", x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2])
    return -1


@wp.func
def pt_pair_closest_point(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, pt_type: int
):
    if pt_type == PTContactTypes.PT:
        return pt_closest_point(x0, x1, x2, x3)
    if pt_type == PTContactTypes.PP01:
        return x1
    if pt_type == PTContactTypes.PP02:
        return x2
    if pt_type == PTContactTypes.PE012:
        return pe_closest_point(x0, x1, x2)
    if pt_type == PTContactTypes.PP03:
        return x3
    if pt_type == PTContactTypes.PE031:
        return pe_closest_point(x0, x3, x1)
    if pt_type == PTContactTypes.PE023:
        return pe_closest_point(x0, x2, x3)
    wp.printf("[point_triangle_classified_distance] Error!\n")
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def pt_pair_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, pt_type: int):
    if pt_type == PTContactTypes.PT:
        return pt_distance(x0, x1, x2, x3)
    if pt_type == PTContactTypes.PP01:
        return pp_distance(x0, x1)
    if pt_type == PTContactTypes.PP02:
        return pp_distance(x0, x2)
    if pt_type == PTContactTypes.PE012:
        return pe_distance(x0, x1, x2)
    if pt_type == PTContactTypes.PP03:
        return pp_distance(x0, x3)
    if pt_type == PTContactTypes.PE031:
        return pe_distance(x0, x3, x1)
    if pt_type == PTContactTypes.PE023:
        return pe_distance(x0, x2, x3)
    wp.printf("[point_triangle_classified_distance] Error!\n")
    return 1e9


@wp.func
def pt_pair_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    dd_dx: wp.array(dtype=wp.vec3),
    pt_type: int,
):
    if pt_type == PTContactTypes.PT:
        pt_distance_grad(x0, x1, x2, x3, dd_dx)
    if pt_type == PTContactTypes.PP01:  # Equals x1
        pp_distance_grad(x0, x1, 0, 1, dd_dx)
    if pt_type == PTContactTypes.PP02:  # Equals x2
        pp_distance_grad(x0, x2, 0, 2, dd_dx)
    if pt_type == PTContactTypes.PE012:  # On edge x1x2
        pe_distance_grad(x0, x1, x2, 0, 1, 2, dd_dx)
    if pt_type == PTContactTypes.PP03:  # Equals x3
        pp_distance_grad(x0, x3, 0, 3, dd_dx)
    if pt_type == PTContactTypes.PE031:  # On edge x3x1
        pe_distance_grad(x0, x3, x1, 0, 3, 1, dd_dx)
    if pt_type == PTContactTypes.PE023:  # On edge x2x3
        pe_distance_grad(x0, x2, x3, 0, 2, 3, dd_dx)


@wp.func
def pt_pair_distance_hessian(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    kappa: float,
    db_dd: float,
    d2b_dd2: float,
    dd_dx: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=2),
    pt_type: int,
):
    if pt_type == PTContactTypes.PT:  # Within triangle
        pt_distance_hessian(x0, x1, x2, x3, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PP01:  # Equals x1
        pp_distance_hessian(x0, x1, 0, 1, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PP02:  # Equals x2
        pp_distance_hessian(x0, x2, 0, 2, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PE012:  # On edge x1x2
        pe_distance_hessian(x0, x1, x2, 0, 1, 2, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PP03:  # Equals x3
        pp_distance_hessian(x0, x3, 0, 3, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PE031:  # On edge x3x1
        pe_distance_hessian(x0, x3, x1, 0, 3, 1, kappa, db_dd, d2b_dd2, dd_dx, blocks)
    if pt_type == PTContactTypes.PE023:  # On edge x2x3
        pe_distance_hessian(x0, x2, x3, 0, 2, 3, kappa, db_dd, d2b_dd2, dd_dx, blocks)


@wp.func
def ee_pair_classify(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, sin_thres: float
):
    e0 = wp.vec3(x0 - x1)
    e1 = wp.vec3(x2 - x3)

    e0_sq = wp.dot(e0, e0)
    e1_sq = wp.dot(e1, e1)

    a00 = e0_sq
    a01 = -wp.dot(e0, e1)
    a10 = wp.dot(e0, e1)
    a11 = -e1_sq
    b0 = wp.dot(wp.vec3(x3 - x1), e0)
    b1 = wp.dot(wp.vec3(x3 - x1), e1)

    mu0 = -(b0 * a11 - b1 * a01)  # = -det([[b0, a01], [b1, a11]])
    mu1 = -(a00 * b1 - a10 * b0)  # = -det([[a00, b0], [a10, b1]])
    detA = -(a00 * a11 - a01 * a10)  # = -det([[a00, a01], [a10, a11]]) >= 0

    l0_sq = wp.length_sq(e0)
    l1_sq = wp.length_sq(e1)

    # Detect nearly-parallel case:
    edge_sin = wp.length(wp.cross(wp.normalize(e0), wp.normalize(e1)))

    if (
        mu0 >= 0
        and mu0 <= detA
        and mu1 >= 0
        and mu1 <= detA
        and edge_sin >= sin_thres
        and l0_sq > 1e-15
        and l1_sq > 1e-15
    ):
        return EEContactTypes.EE

    best_d = float(1e9)
    ee_type = int(-1)

    # fix x0
    t0 = wp.dot(x0 - x3, e1)
    if t0 >= e1_sq:
        d = pp_distance(x0, x2)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP02
    elif t0 <= 0:
        d = pp_distance(x0, x3)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP03
    else:
        d = pe_distance(x0, x2, x3)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PE023

    # fix x1
    t1 = wp.dot(x1 - x3, e1)
    if t1 >= e1_sq:
        d = pp_distance(x1, x2)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP12
    elif t1 <= 0:
        d = pp_distance(x1, x3)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP13
    else:
        d = pe_distance(x1, x2, x3)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PE123

    # fix x2
    t2 = wp.dot(x2 - x1, e0)
    if t2 >= e0_sq:
        d = pp_distance(x2, x0)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP02
    elif t2 <= 0:
        d = pp_distance(x2, x1)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP12
    else:
        d = pe_distance(x2, x0, x1)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PE201

    # fix x3
    t3 = wp.dot(x3 - x1, e0)
    if t3 >= e0_sq:
        d = pp_distance(x3, x0)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP03
    elif t3 <= 0:
        d = pp_distance(x3, x1)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PP13
    else:
        d = pe_distance(x3, x0, x1)
        if d < best_d:
            best_d = d
            ee_type = EEContactTypes.PE301

    if ee_type == -1:
        wp.printf("[ee_pair_classify] Error!\n")

    return ee_type


@wp.func
def ee_pair_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, ee_type: int):
    if ee_type == EEContactTypes.PP02:
        return pp_distance(x0, x2)
    if ee_type == EEContactTypes.PP03:
        return pp_distance(x0, x3)
    if ee_type == EEContactTypes.PP12:
        return pp_distance(x1, x2)
    if ee_type == EEContactTypes.PP13:
        return pp_distance(x1, x3)
    if ee_type == EEContactTypes.PE023:
        return pe_distance(x0, x2, x3)
    if ee_type == EEContactTypes.PE123:
        return pe_distance(x1, x2, x3)
    if ee_type == EEContactTypes.PE201:
        return pe_distance(x2, x0, x1)
    if ee_type == EEContactTypes.PE301:
        return pe_distance(x3, x0, x1)
    if ee_type == EEContactTypes.EE:
        return ee_distance(x0, x1, x2, x3)
    # wp.printf("[edge_edge_classified_distance] Error!\n")
    return 1e9


@wp.func
def ee_pair_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    dd_dx: wp.array(dtype=wp.vec3),
    ee_type: int,
):
    if ee_type == EEContactTypes.PP02:  # x0x2
        pp_distance_grad(x0, x2, 0, 2, dd_dx)
    if ee_type == EEContactTypes.PP03:  # x0x3
        pp_distance_grad(x0, x3, 0, 3, dd_dx)
    if ee_type == EEContactTypes.PP12:  # x1x2
        pp_distance_grad(x1, x2, 1, 2, dd_dx)
    if ee_type == EEContactTypes.PP13:  # x1x3
        pp_distance_grad(x1, x3, 1, 3, dd_dx)
    if ee_type == EEContactTypes.PE023:  # x0e1
        pe_distance_grad(x0, x2, x3, 0, 2, 3, dd_dx)
    if ee_type == EEContactTypes.PE123:  # x1e1
        pe_distance_grad(x1, x2, x3, 1, 2, 3, dd_dx)
    if ee_type == EEContactTypes.PE201:  # x2e0
        pe_distance_grad(x2, x0, x1, 2, 0, 1, dd_dx)
    if ee_type == EEContactTypes.PE301:  # x3e0
        pe_distance_grad(x3, x0, x1, 3, 0, 1, dd_dx)
    if ee_type == EEContactTypes.EE:  # e0e1
        ee_distance_grad(x0, x1, x2, x3, dd_dx)


@wp.func
def ee_pair_distance_hessian(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    kappa: float,
    d_hess_coeff: float,
    d_grad_coeff: float,
    dd_dx: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=2),
    ee_type: int,
):
    if ee_type == EEContactTypes.PP02:  # x0x2
        pp_distance_hessian(
            x0, x2, 0, 2, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PP03:  # x0x3
        pp_distance_hessian(
            x0, x3, 0, 3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PP12:  # x1x2
        pp_distance_hessian(
            x1, x2, 1, 2, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PP13:  # x1x3
        pp_distance_hessian(
            x1, x3, 1, 3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PE023:  # x0e1
        pe_distance_hessian(
            x0, x2, x3, 0, 2, 3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PE123:  # x1e1
        pe_distance_hessian(
            x1, x2, x3, 1, 2, 3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PE201:  # x2e0
        pe_distance_hessian(
            x2, x0, x1, 2, 0, 1, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.PE301:  # x3e0
        pe_distance_hessian(
            x3, x0, x1, 3, 0, 1, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
    if ee_type == EEContactTypes.EE:  # e0e1
        ee_distance_hessian(
            x0, x1, x2, x3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, blocks
        )
