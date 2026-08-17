"""Positive-semidefinite projection compatible with warp-lang 1.16+.

The Jacobi iteration is adapted from John Burkardt's LGPL implementation and
the PaMO Warp fork's ``spd_project_blocks`` helper. See the bundled NOTICE and
LGPL license text.
"""

import warp as wp

_SPD_PROJECT_SNIPPET = r"""
    constexpr int n = 9;
    float a[n * n];
    float v[n * n];
    float d[n];
    float bw[n];
    float zw[n];

    // Convert the upper-left 3x3 block matrix (of mat33 values) to a 9x9
    // scalar matrix. The fourth block row/column is reconstructed by the
    // calling Warp kernel after this projection.
    for (int block_i = 0; block_i < 3; ++block_i) {
        for (int block_j = 0; block_j < 3; ++block_j) {
            const auto& block = wp::index(blocks, bid, block_i, block_j);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    a[(block_i * 3 + i) * n + block_j * 3 + j] =
                        block.data[i][j];
                }
            }
        }
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            v[i + j * n] = i == j ? 1.0f : 0.0f;
        }
        d[j] = a[j + j * n];
        bw[j] = d[j];
        zw[j] = 0.0f;
    }

    int iteration = 0;
    while (iteration < it_max) {
        ++iteration;
        float threshold = 0.0f;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) {
                threshold += a[i + j * n] * a[i + j * n];
            }
        }
        threshold = sqrtf(threshold) / (4.0f * n);
        if (threshold == 0.0f) {
            break;
        }

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const float gap = 10.0f * fabsf(a[p + q * n]);
                const float term_p = gap + fabsf(d[p]);
                const float term_q = gap + fabsf(d[q]);
                if (iteration > 4 && term_p == fabsf(d[p]) && term_q == fabsf(d[q])) {
                    a[p + q * n] = 0.0f;
                } else if (threshold <= fabsf(a[p + q * n])) {
                    float h = d[q] - d[p];
                    float t;
                    if (fabsf(h) + gap == fabsf(h)) {
                        t = a[p + q * n] / h;
                    } else {
                        const float theta = 0.5f * h / a[p + q * n];
                        t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                        if (theta < 0.0f) {
                            t = -t;
                        }
                    }
                    const float c = 1.0f / sqrtf(1.0f + t * t);
                    const float s = t * c;
                    const float tau = s / (1.0f + c);
                    h = t * a[p + q * n];
                    zw[p] -= h;
                    zw[q] += h;
                    d[p] -= h;
                    d[q] += h;
                    a[p + q * n] = 0.0f;

                    for (int j = 0; j < p; ++j) {
                        const float g = a[j + p * n];
                        h = a[j + q * n];
                        a[j + p * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }
                    for (int j = p + 1; j < q; ++j) {
                        const float g = a[p + j * n];
                        h = a[j + q * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }
                    for (int j = q + 1; j < n; ++j) {
                        const float g = a[p + j * n];
                        h = a[q + j * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[q + j * n] = h + s * (g - h * tau);
                    }
                    for (int j = 0; j < n; ++j) {
                        const float g = v[j + p * n];
                        h = v[j + q * n];
                        v[j + p * n] = g - s * (h + g * tau);
                        v[j + q * n] = h + s * (g - h * tau);
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            bw[i] += zw[i];
            d[i] = bw[i];
            zw[i] = 0.0f;
        }
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            a[i + j * n] = a[j + i * n];
        }
    }

    for (int i = 0; i < n * n; ++i) {
        a[i] = 0.0f;
    }
    for (int k = 0; k < n; ++k) {
        if (d[k] > 0.0f) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    a[i * n + j] += d[k] * v[k * n + i] * v[k * n + j];
                }
            }
        }
    }

    for (int block_i = 0; block_i < 3; ++block_i) {
        for (int block_j = 0; block_j < 3; ++block_j) {
            auto& block = wp::index(blocks, bid, block_i, block_j);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const int row = block_i * 3 + i;
                    const int column = block_j * 3 + j;
                    block.data[i][j] =
                        0.5f * (a[row * n + column] + a[column * n + row]);
                }
            }
        }
    }
"""


@wp.func_native(
    snippet=_SPD_PROJECT_SNIPPET,
    adj_snippet="// The solver does not differentiate through SPD projection.\n",
)
def project_spd_blocks(
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    bid: int,
    it_max: int,
): ...
