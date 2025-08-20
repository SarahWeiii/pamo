import warp as wp


@wp.func
def ee_pair_mollifier_grad(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, dc_dx: wp.array(dtype=wp.vec3)
):
    x01 = x0[0]
    x02 = x0[1]
    x03 = x0[2]

    x11 = x1[0]
    x12 = x1[1]
    x13 = x1[2]

    x21 = x2[0]
    x22 = x2[1]
    x23 = x2[2]

    x31 = x3[0]
    x32 = x3[1]
    x33 = x3[2]

    t2 = -x11
    t3 = -x12
    t4 = -x13
    t5 = -x31
    t6 = -x32
    t7 = -x33
    t8 = t2 + x01
    t9 = t3 + x02
    t10 = t4 + x03
    t11 = t5 + x21
    t12 = t6 + x22
    t13 = t7 + x23
    t14 = t8 * t12
    t15 = t9 * t11
    t16 = t8 * t13
    t17 = t10 * t11
    t18 = t9 * t13
    t19 = t10 * t12
    t20 = -t15
    t21 = -t17
    t22 = -t19
    t23 = t14 + t20
    t24 = t16 + t21
    t25 = t18 + t22
    t26 = t8 * t23 * 2.0
    t27 = t9 * t23 * 2.0
    t28 = t8 * t24 * 2.0
    t29 = t10 * t24 * 2.0
    t30 = t9 * t25 * 2.0
    t31 = t10 * t25 * 2.0
    t32 = t11 * t23 * 2.0
    t33 = t12 * t23 * 2.0
    t34 = t11 * t24 * 2.0
    t35 = t13 * t24 * 2.0
    t36 = t12 * t25 * 2.0
    t37 = t13 * t25 * 2.0

    dc_dx[0] = wp.vec3(t33 + t35, -t32 + t37, -t34 - t36)
    dc_dx[1] = -dc_dx[0]
    dc_dx[2] = wp.vec3(-t27 - t29, t26 - t31, t28 + t30)
    dc_dx[3] = -dc_dx[2]


@wp.func
def ee_pair_mollifier_hessian(
        x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3,
        kappa: float, c_hessian_coeff: float, c_grad_coeff: float,
        dc_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    i0 = 0
    i1 = 1
    i2 = 2
    i3 = 3
    
    energy_hessian[i1][i1] = energy_hessian[i1][i1] + wp.outer(dc_dx[i1], dc_dx[i1] * c_grad_coeff) * kappa
    energy_hessian[i1][i2] = energy_hessian[i1][i2] + wp.outer(dc_dx[i1], dc_dx[i2] * c_grad_coeff) * kappa
    energy_hessian[i1][i3] = energy_hessian[i1][i3] + wp.outer(dc_dx[i1], dc_dx[i3] * c_grad_coeff) * kappa
    energy_hessian[i2][i2] = energy_hessian[i2][i2] + wp.outer(dc_dx[i2], dc_dx[i2] * c_grad_coeff) * kappa
    energy_hessian[i2][i3] = energy_hessian[i2][i3] + wp.outer(dc_dx[i2], dc_dx[i3] * c_grad_coeff) * kappa
    energy_hessian[i3][i3] = energy_hessian[i3][i3] + wp.outer(dc_dx[i3], dc_dx[i3] * c_grad_coeff) * kappa

    energy_hessian[i2][i1] = wp.transpose(energy_hessian[i1][i2])
    energy_hessian[i3][i1] = wp.transpose(energy_hessian[i1][i3])
    energy_hessian[i3][i2] = wp.transpose(energy_hessian[i2][i3])

    energy_hessian[i0][i1] = - energy_hessian[i1][i1] - energy_hessian[i2][i1] - energy_hessian[i3][i1]
    energy_hessian[i0][i2] = - energy_hessian[i1][i2] - energy_hessian[i2][i2] - energy_hessian[i3][i2]
    energy_hessian[i0][i3] = - energy_hessian[i1][i3] - energy_hessian[i2][i3] - energy_hessian[i3][i3]

    energy_hessian[i1][i0] = wp.transpose(energy_hessian[i0][i1])
    energy_hessian[i2][i0] = wp.transpose(energy_hessian[i0][i2])
    energy_hessian[i3][i0] = wp.transpose(energy_hessian[i0][i3])

    energy_hessian[i0][i0] = - energy_hessian[i0][i1] - energy_hessian[i0][i2] - energy_hessian[i0][i3]

    return

    # x0 = x[x0_id]
    x01 = x0[0]
    x02 = x0[1]
    x03 = x0[2]

    # x1 = x[x1_id]
    x11 = x1[0]
    x12 = x1[1]
    x13 = x1[2]

    # x2 = x[x2_id]
    x21 = x2[0]
    x22 = x2[1]
    x23 = x2[2]

    # x3 = x[x3_id]
    x31 = x3[0]
    x32 = x3[1]
    x33 = x3[2]

    t2 = -x11
    t3 = -x12
    t4 = -x13
    t5 = -x31
    t6 = -x32
    t7 = -x33
    t8 = t2+x01
    t9 = t3+x02
    t10 = t4+x03
    t11 = t5+x21
    t12 = t6+x22
    t13 = t7+x23
    t14 = t8*t8
    t15 = t9*t9
    t16 = t10*t10
    t17 = t11*t11
    t18 = t12*t12
    t19 = t13*t13
    t32 = t8*t9*2.0
    t33 = t8*t10*2.0
    t34 = t9*t10*2.0
    t35 = t8*t11*2.0
    t36 = t8*t12*2.0
    t37 = t9*t11*2.0
    t38 = t8*t12*4.0
    t39 = t8*t13*2.0
    t40 = t9*t11*4.0
    t41 = t9*t12*2.0
    t42 = t10*t11*2.0
    t43 = t8*t13*4.0
    t44 = t9*t13*2.0
    t45 = t10*t11*4.0
    t46 = t10*t12*2.0
    t47 = t9*t13*4.0
    t48 = t10*t12*4.0
    t49 = t10*t13*2.0
    t50 = t11*t12*2.0
    t51 = t11*t13*2.0
    t52 = t12*t13*2.0
    t20 = t14*2.0
    t21 = t15*2.0
    t22 = t16*2.0
    t23 = t17*2.0
    t24 = t18*2.0
    t25 = t19*2.0
    t53 = -t32
    t54 = -t33
    t55 = -t34
    t56 = -t35
    t57 = -t36
    t58 = -t37
    t59 = -t38
    t60 = -t39
    t61 = -t40
    t62 = -t41
    t63 = -t42
    t64 = -t43
    t65 = -t44
    t66 = -t45
    t67 = -t46
    t68 = -t47
    t69 = -t48
    t70 = -t49
    t71 = -t50
    t72 = -t51
    t73 = -t52
    t86 = t35+t41
    t87 = t35+t49
    t88 = t41+t49
    t26 = -t20
    t27 = -t21
    t28 = -t22
    t29 = -t23
    t30 = -t24
    t31 = -t25
    t74 = t20+t21
    t75 = t20+t22
    t76 = t21+t22
    t77 = t23+t24
    t78 = t23+t25
    t79 = t24+t25
    t89 = t40+t57
    t90 = t36+t61
    t91 = t37+t59
    t92 = t38+t58
    t93 = t45+t60
    t94 = t39+t66
    t95 = t42+t64
    t96 = t43+t63
    t97 = t48+t65
    t98 = t44+t69
    t99 = t46+t68
    t100 = t47+t67
    t101 = t56+t62
    t102 = t56+t70
    t103 = t62+t70
    t80 = t26+t27
    t81 = t26+t28
    t82 = t27+t28
    t83 = t29+t30
    t84 = t29+t31
    t85 = t30+t31

    energy_hessian[i1][i1] = energy_hessian[i1][i1] + (wp.mat33(t79, t71, t72, t71, t78, t73, t72, t73, t77) * c_hessian_coeff
                              + wp.outer(dc_dx[i1], dc_dx[i1]) * c_grad_coeff) * kappa
    energy_hessian[i1][i2] = energy_hessian[i1][i2] + (wp.mat33(t88, t91, t95, t90, t87, t99, t94, t98, t86) * c_hessian_coeff
                              + wp.outer(dc_dx[i1], dc_dx[i2]) * c_grad_coeff) * kappa
    energy_hessian[i1][i3] = energy_hessian[i1][i3] + (wp.mat33(t103, t92, t96, t89, t102, t100, t93, t97, t101) * c_hessian_coeff
                              + wp.outer(dc_dx[i1], dc_dx[i3]) * c_grad_coeff) * kappa
    energy_hessian[i2][i2] = energy_hessian[i2][i2] + (wp.mat33(t76, t53, t54, t53, t75, t55, t54, t55, t74) * c_hessian_coeff
                              + wp.outer(dc_dx[i2], dc_dx[i2]) * c_grad_coeff) * kappa
    energy_hessian[i2][i3] = energy_hessian[i2][i3] + (wp.mat33(t82, t32, t33, t32, t81, t34, t33, t34, t80) * c_hessian_coeff
                              + wp.outer(dc_dx[i2], dc_dx[i3]) * c_grad_coeff) * kappa
    energy_hessian[i3][i3] = energy_hessian[i3][i3] + (wp.mat33(t76, t53, t54, t53, t75, t55, t54, t55, t74) * c_hessian_coeff
                              + wp.outer(dc_dx[i3], dc_dx[i3]) * c_grad_coeff) * kappa

    energy_hessian[i2][i1] = wp.transpose(energy_hessian[i1][i2])
    energy_hessian[i3][i1] = wp.transpose(energy_hessian[i1][i3])
    energy_hessian[i3][i2] = wp.transpose(energy_hessian[i2][i3])

    energy_hessian[i0][i1] = - energy_hessian[i1][i1] - energy_hessian[i2][i1] - energy_hessian[i3][i1]
    energy_hessian[i0][i2] = - energy_hessian[i1][i2] - energy_hessian[i2][i2] - energy_hessian[i3][i2]
    energy_hessian[i0][i3] = - energy_hessian[i1][i3] - energy_hessian[i2][i3] - energy_hessian[i3][i3]

    energy_hessian[i1][i0] = wp.transpose(energy_hessian[i0][i1])
    energy_hessian[i2][i0] = wp.transpose(energy_hessian[i0][i2])
    energy_hessian[i3][i0] = wp.transpose(energy_hessian[i0][i3])

    energy_hessian[i0][i0] = - energy_hessian[i0][i1] - energy_hessian[i0][i2] - energy_hessian[i0][i3]
