import warp as wp


@wp.kernel
def a_plus_k_b_kernel(
    a: wp.array(dtype=wp.vec3),
    k: float,
    b: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3),
):
    """
    input: a, k, b
    output: ret = a + k * b
    """
    tid = wp.tid()

    ret[tid] = a[tid] + k * b[tid]


@wp.kernel
def block_spd_project_kernel(
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    it_max: int,
):
    bid = wp.tid()

    block = blocks[bid]

    wp.spd_project_blocks(3, block, it_max)

    block[3][0] = -block[0][0] - block[1][0] - block[2][0]
    block[3][1] = -block[0][1] - block[1][1] - block[2][1]
    block[3][2] = -block[0][2] - block[1][2] - block[2][2]

    block[0][3] = wp.transpose(block[3][0])
    block[1][3] = wp.transpose(block[3][1])
    block[2][3] = wp.transpose(block[3][2])

    block[3][3] = -block[0][3] - block[1][3] - block[2][3]
