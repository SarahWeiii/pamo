#pragma once
#include "structs.cuh"
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"

__forceinline__ __device__ uint shuffler(uint v, uint bmask)
{
    // return (v * (v + 1) / 2) & bmask;
    return v;  // trade off algorithmic complexity for memory coalesce
}

__forceinline__ __device__ uint cts_find(const uint * parents, const uint i)
{
    // REVIEW: we shall not use path compression?
    // Because parallelism is good without atomic writes.
    uint c = i;
    while (parents[c] != c)
        c = parents[c];
    return c;
}

__forceinline__ __device__ void cts_atomic_union(uint * __restrict__ parents, uint x, uint y)
{
    // REVIEW: I didn't use rank or size based union
    // Because I cannot implement it without locks
    // And we can expect xor-shuffled performance to be the same without
    // Experiment:
    // Using path halving during find here cause overall performance to drop.
    if (x == y) return;
    while (true)
    {
        uint px = parents[x];
        uint py = parents[y];
        if (px == y || x == py || px == py) return;
        if (px == x && py == y)
            atomicCAS(&parents[max(x, y)], max(x, y), min(x, y));
        x = px;
        y = py;
    }
}

__global__ void volume_sign_prescan_kernel(
    const RasterizeResult rast,
    uint * __restrict__ parents,
    const uint N, const int shfBitmask
) {
    const uint3 xy = blockIdx * blockDim + threadIdx;
    if (xy.x >= N || xy.y >= N) return;
    bool skip = false;
    bool flag = false;
    uint shfcn = shuffler(0, shfBitmask);
    if (xy.x == 0 && xy.y == 0)
        parents[shfcn] = shfcn;
    for (uint i = 0; i < N; i++)
    {
        const uint3 xyz = make_uint3(i, xy.y, xy.x);
        const uint access = to_gidx(xyz, N);
        const uint shfm = shuffler(access + 1, shfBitmask);
        if (skip)
        {
            skip = false;
            if (flag)
            {
                flag = false;
                shfcn = shfm;
            }
        }
        else
        {
            const float dist = rast.gridDist[access] * N;
            if (dist < 0.87f)
            {
                flag = true;
                shfcn = shfm;
            }
            else if (flag)
            {
                flag = false;
                shfcn = shfm;
            }
            skip = dist > 2.0f;
        }
        parents[shfm] = shfcn;
    }
}

__global__ void volume_cts_kernel(
    const RasterizeResult rast,
    uint * __restrict__ parents,
    const uint N, const int shfBitmask
) {
    const uint3 tid = blockIdx * blockDim + threadIdx;
    const uint3 xyz = make_uint3(tid.y, tid.z, tid.x);
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);
    const uint shfm = shuffler(access + 1, shfBitmask);

    if (xyz.x == 0 || xyz.y == 0 || xyz.z == 0)
    {
        const int dist = rast.gridDist[access];
        if (dist >= 0.86603f / N)
        {
            const uint shfex = shuffler(0, shfBitmask);
            cts_atomic_union(parents, shfm, shfex);
        }
    }

    const uint a3 = access * 3;
    if (!rast.gridCollide[a3])
    {
        const uint3 nxyz = clamp(xyz + make_uint3(1, 0, 0), 0, N - 1);
        const uint shfn = shuffler(to_gidx(nxyz, N) + 1, shfBitmask);
        cts_atomic_union(parents, shfm, shfn);
    }
    if (!rast.gridCollide[a3 + 1])
    {
        const uint3 nxyz = clamp(xyz + make_uint3(0, 1, 0), 0, N - 1);
        const uint shfn = shuffler(to_gidx(nxyz, N) + 1, shfBitmask);
        cts_atomic_union(parents, shfm, shfn);
    }
    if (!rast.gridCollide[a3 + 2])
    {
        const uint3 nxyz = clamp(xyz + make_uint3(0, 0, 1), 0, N - 1);
        const uint shfn = shuffler(to_gidx(nxyz, N) + 1, shfBitmask);
        cts_atomic_union(parents, shfm, shfn);
    }
}

__global__ void volume_apply_sign_kernel(RasterizeResult rast, const uint * parents, const int N, const int shfBitmask)
{
    const uint root = cts_find(parents, shuffler(0, shfBitmask));
    
    const uint3 tid = blockIdx * blockDim + threadIdx;
    const uint3 xyz = make_uint3(tid.z, tid.y, tid.x);
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);
    const uint shfm = shuffler(access + 1, shfBitmask);

    if (cts_find(parents, shfm) != root)
        rast.gridDist[access] *= -1;
}

static MemoryAllocator cachedAllocatorSign(2 * 1024 * 1024, 1);

inline void clear_sign_alloc_cache()
{
    cachedAllocatorSign.clear();
}

static void fill_signs(const float3 * tris, const int N, RasterizeResult rast, const bool useCachedAllocator)
{
    MemoryAllocator theAllocator(2 * 1024 * 1024, 1);
    MemoryAllocator& ma = useCachedAllocator ? cachedAllocatorSign : theAllocator;

    const uint nodeCount = (N * N * N + 1);
    const uint shfBitmask = npo2(N * N * N + 1) - 1;

    uint * block = ma.alloc<uint>(nodeCount + 1024);
    uint * parents = block + 1023;

    CHECK_CUDA(cudaFuncSetCacheConfig(volume_cts_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(volume_apply_sign_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(volume_sign_prescan_kernel, cudaFuncCachePreferL1));

    dim3 dimBlock2d(ceil_div(N, 32), ceil_div(N, 8), 1);
    dim3 dimGrid2d(32, 8, 1);
    volume_sign_prescan_kernel<<<dimBlock2d, dimGrid2d>>>(rast, parents, N, shfBitmask);
    CHECK_CUDA(cudaGetLastError());
    dim3 dimBlock(ceil_div(N, 32), ceil_div(N, 16), ceil_div(N, 1));
    dim3 dimGrid(32, 16, 1);
    volume_cts_kernel<<<dimBlock, dimGrid>>>(rast, parents, N, shfBitmask);
    CHECK_CUDA(cudaGetLastError());

    volume_apply_sign_kernel<<<dimBlock, dimGrid>>>(rast, parents, N, shfBitmask);
    CHECK_CUDA(cudaGetLastError());

    ma.free(block);
    CHECK_CUDA(cudaDeviceSynchronize());
}
