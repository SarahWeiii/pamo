import numpy as np
import warp as wp
from pamo_safe_project.kernels.utils_kernels import block_spd_project_kernel
from pamo_safe_project.utils import wp_slice


def test_wp_slice_is_a_zero_copy_view():
    source = wp.array(np.arange(6, dtype=np.float32), device="cpu")

    view = wp_slice(source, 2, 5)
    view.fill_(9.0)

    np.testing.assert_array_equal(
        source.numpy(), np.array([0, 1, 9, 9, 9, 5], dtype=np.float32)
    )


def test_spd_projection_clamps_negative_eigenvalues():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(9, 9)).astype(np.float32)
    matrix = (matrix + matrix.T) / 2
    blocks = np.zeros((1, 4, 4, 3, 3), dtype=np.float32)
    for block_i in range(3):
        for block_j in range(3):
            blocks[0, block_i, block_j] = matrix[
                block_i * 3 : (block_i + 1) * 3,
                block_j * 3 : (block_j + 1) * 3,
            ]

    blocks_wp = wp.array(blocks, dtype=wp.mat33, device="cpu")
    wp.launch(
        block_spd_project_kernel,
        dim=1,
        inputs=[blocks_wp, 30],
        device="cpu",
    )
    output_blocks = blocks_wp.numpy()[0]
    projected = np.block([[output_blocks[i, j] for j in range(3)] for i in range(3)])

    assert np.linalg.eigvalsh(projected).min() > -1e-4
    np.testing.assert_allclose(projected, projected.T, atol=1e-6)
