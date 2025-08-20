import warp as wp
from .grad_funcs_struct import *

from ...defs import *


@wp.func
def pt_pair_distance_grad_struct(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    pt_type: int,
):
    if pt_type == PTContactTypes.PT:
        return pt_distance_grad_struct(x0, x1, x2, x3)
    if pt_type == PTContactTypes.PP01:  # Equals x1
        return pp_distance_grad_struct(x0, x1, 0, 1)
    if pt_type == PTContactTypes.PP02:  # Equals x2
        return pp_distance_grad_struct(x0, x2, 0, 2)
    if pt_type == PTContactTypes.PE012:  # On edge x1x2
        return pe_distance_grad_struct(x0, x1, x2, 0, 1, 2)
    if pt_type == PTContactTypes.PP03:  # Equals x3
        return pp_distance_grad_struct(x0, x3, 0, 3)
    if pt_type == PTContactTypes.PE031:  # On edge x3x1
        return pe_distance_grad_struct(x0, x3, x1, 0, 3, 1)
    if pt_type == PTContactTypes.PE023:  # On edge x2x3
        return pe_distance_grad_struct(x0, x2, x3, 0, 2, 3)


@wp.func
def ee_pair_distance_grad_struct(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    ee_type: int,
):
    if ee_type == EEContactTypes.PP02:  # x0x2
        return pp_distance_grad_struct(x0, x2, 0, 2)
    if ee_type == EEContactTypes.PP03:  # x0x3
        return pp_distance_grad_struct(x0, x3, 0, 3)
    if ee_type == EEContactTypes.PP12:  # x1x2
        return pp_distance_grad_struct(x1, x2, 1, 2)
    if ee_type == EEContactTypes.PP13:  # x1x3
        return pp_distance_grad_struct(x1, x3, 1, 3)
    if ee_type == EEContactTypes.PE023:  # x0e1
        return pe_distance_grad_struct(x0, x2, x3, 0, 2, 3)
    if ee_type == EEContactTypes.PE123:  # x1e1
        return pe_distance_grad_struct(x1, x2, x3, 1, 2, 3)
    if ee_type == EEContactTypes.PE201:  # x2e0
        return pe_distance_grad_struct(x2, x0, x1, 2, 0, 1)
    if ee_type == EEContactTypes.PE301:  # x3e0
        return pe_distance_grad_struct(x3, x0, x1, 3, 0, 1)
    if ee_type == EEContactTypes.EE:  # e0e1
        return ee_distance_grad_struct(x0, x1, x2, x3)
