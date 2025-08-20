import warp as wp
from .utils import wp_slice
from .kernels.geometry_kernels import *


def compute_tri_bounds(
    n_F: int, 
    V: wp.array,
    F: wp.array, 
    lowers: wp.array, 
    uppers: wp.array,
    radius: float,
):
    wp.launch(
        kernel=compute_tri_bounds_kernel,
        dim=n_F,
        inputs=[V, F, radius],
        outputs=[lowers, uppers],
    )
    

def compute_edge_bounds(
    n_E: int, 
    V: wp.array,
    E: wp.array, 
    lowers: wp.array, 
    uppers: wp.array,
    radius: float,
):
    wp.launch(
        kernel=compute_edge_bounds_kernel,
        dim=n_E,
        inputs=[V, E, radius],
        outputs=[lowers, uppers],
    )



