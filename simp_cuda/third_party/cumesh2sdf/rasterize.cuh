#pragma once
#include "structs.cuh"
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"
#include "allocator.cuh"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

template<bool probe, uint S>
__global__ void rasterize_layer_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const int M, const int N, const float band,
    uint * __restrict__ tempBlockOffset, uint * __restrict__ totalSize,
    uint * __restrict__ outIdx, uint * __restrict__ outGrid, const uint preloc
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // S subdiv
    // N scale of current grid, pre-multiplied by S
    const uint b = blockIdx.x;
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= S * S * S * M) return;

    __shared__ uint blockSize;

    if (threadIdx.x == 0) blockSize = 0;
    __syncthreads();

    const int t = g / (S * S * S);
    const int mo = g - t * (S * S * S);
    const int k = mo / (S * S);
    const int yo = mo - k * (S * S);
    const int j = yo / S;
    const int i = yo - j * S;

    const uint tofs = idx[t];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const uint gid = grid[t];
    const uint3 nxyz = unpack_id(gid) * S + make_uint3(i, j, k);
    const float3 fxyz = (i2f(nxyz) + 0.5f) / (float)N;
    
    const float thresh = 0.87f / N + band;
    const bool intersect = point_to_tri_dist_sqr(v1, v2, v3, fxyz) < thresh * thresh;
    
    uint inblock;
    if (intersect)
        inblock = atomicAdd(&blockSize, 1);
    
    __syncthreads();
    __shared__ uint bofs;
    if (threadIdx.x == 0)
    {
        if constexpr (probe)
            bofs = tempBlockOffset[b] = atomicAdd(totalSize, blockSize);
        else
            bofs = tempBlockOffset[b];
    }
    __syncthreads();

    if (intersect && bofs + inblock < preloc)
    {
        outIdx[bofs + inblock] = tofs;
        outGrid[bofs + inblock] = pack_id(nxyz);
    }
}

template<bool probe>
auto rasterize_layer_kernel_dispatch(const uint S)
{
    switch (S)
    {
        case 1: return rasterize_layer_kernel<probe, 1>;
        case 2: return rasterize_layer_kernel<probe, 2>;
        case 3: return rasterize_layer_kernel<probe, 3>;
        case 4: return rasterize_layer_kernel<probe, 4>;
        case 5: return rasterize_layer_kernel<probe, 5>;
        case 6: return rasterize_layer_kernel<probe, 6>;
        case 7: return rasterize_layer_kernel<probe, 7>;
        case 8: return rasterize_layer_kernel<probe, 8>;
        default: throw std::runtime_error("Rasterize subdivison dispatch failed with S = " + std::to_string(S));
    }
}

__global__ void rasterize_reduce_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const int M, const int N,
    float * __restrict__ outGridDist, bool * __restrict__ outGridCollide
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // N size of target grid
    // outGridDist: [N, N, N] distance
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M) return;
    const uint3 nxyz = unpack_id(grid[g]);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    const uint access = to_gidx(nxyz, N);
    
    const uint tofs = idx[g];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const float finalDist = sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz));
    atomicMin(outGridDist + access, finalDist);

    const float rayth = 1.0f / N + FLT_EPSILON;
    if (finalDist > rayth) return;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(1, 0, 0), finalDist) <= rayth)
        outGridCollide[access * 3] = true;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(0, 1, 0), finalDist) <= rayth)
        outGridCollide[access * 3 + 1] = true;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(0, 0, 1), finalDist) <= rayth)
        outGridCollide[access * 3 + 2] = true;
}

inline uint rasterize_layer_internal(const float3 * tris, const int S, const int M, const int N, uint ** pidx, uint ** pgrid, uint * totalSize, const float band, MemoryAllocator& ma)
{
    uint * idx = *pidx;
    uint * grid = *pgrid;
    
    if (S * S * S * M != S * S * S * (long long)M)
    {
        std::cerr << "Warning: Overflow in rasterize layer. Please reduce batch size." << std::endl;
    }
    uint blocks = ceil_div(S * S * S * M, NTHREAD_1D);
    uint * tempBlockOffset = ma.alloc<uint>(blocks);

    uint * outIdx = ma.alloc<uint>(S * 2 * M);
    uint * outGrid = ma.alloc<uint>(S * 2 * M);
    uint preloc = min(ma.probe<uint>(outIdx), ma.probe<uint>(outGrid));

    // probe & prefill
    common_fill_kernel<uint><<<1, 1>>>(0, 1, totalSize);
    rasterize_layer_kernel_dispatch<true>(S)<<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, M, N, band, tempBlockOffset, totalSize, outIdx, outGrid, preloc
    );
    CHECK_CUDA(cudaGetLastError());

    uint las;
    CHECK_CUDA(cudaMemcpy(&las, totalSize, sizeof(uint), cudaMemcpyDeviceToHost));

    if (las > preloc)
    {
        ma.free(outIdx);
        ma.free(outGrid);
        ma.release_smallest();
        outIdx = ma.alloc<uint>(las);
        outGrid = ma.alloc<uint>(las);
        // reloc & fill
        rasterize_layer_kernel_dispatch<false>(S)<<<blocks, NTHREAD_1D>>>(
            tris, idx, grid, M, N, band, tempBlockOffset, nullptr, outIdx, outGrid, las
        );
        CHECK_CUDA(cudaGetLastError());
    }

    ma.free(idx);
    ma.free(grid);
    ma.free(tempBlockOffset);

    *pidx = outIdx;
    *pgrid = outGrid;
    return las;
}

static RasterizeResult cachedAllocation;
static int cachedSize = 0;
static MemoryAllocator cachedAllocator(4 * 1024 * 1024, 8);

inline void clear_raster_alloc_cache()
{
    if (cachedSize <= 0) return;
    cachedAllocation.free();
    cachedAllocator.clear();
    cachedSize = 0;
}

inline RasterizeResult rasterize_tris_internal(const float3 * tris, const int F, const std::vector<int> SS, const int B, const float band, const bool useCachedAllocator = false)
{
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<true, 2>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<false, 2>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<true, 4>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<false, 4>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<true, 8>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<false, 8>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_reduce_kernel, cudaFuncCachePreferL1));

    int N = 1;
    for (uint i = 0; i < SS.size(); i++)
        N *= SS[i];
    const int R = N;

    RasterizeResult rasterizeResult;
    if (useCachedAllocator && cachedSize >= R)
        rasterizeResult = cachedAllocation;
    else
    {
        CHECK_CUDA(cudaMalloc(&rasterizeResult.gridDist, R * R * R * sizeof(float) + sizeof(uint)));
        CHECK_CUDA(cudaMalloc(&rasterizeResult.gridCollide, R * R * R * sizeof(bool) * 3));
        if (useCachedAllocator)
        {
            clear_raster_alloc_cache();
            cachedAllocation = rasterizeResult;
            cachedSize = R;
        }
    }

    MemoryAllocator theAllocator(2 * 1024 * 1024, 0);
    MemoryAllocator& ma = useCachedAllocator ? cachedAllocator : theAllocator;

    uint * totalSize = (uint*)(rasterizeResult.gridDist + (R * R * R));

    uint startId = pack_id(make_uint3(0, 0, 0));
    int M;
    for (int j = 0; j < F; j += B)
    {
        N = 1;
        M = min(B, F - j);

        uint * idx = ma.alloc<uint>(M);
        uint * grid = ma.alloc<uint>(M);
        
        common_arange_kernel<<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(idx, M, j);
        CHECK_CUDA(cudaGetLastError());
        common_fill_kernel<uint><<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(startId, M, grid);
        CHECK_CUDA(cudaGetLastError());

        for (uint i = 0; i < SS.size(); i++)
        {
            N *= SS[i];
            M = rasterize_layer_internal(tris, SS[i], M, N, &idx, &grid, totalSize, band, ma);
        }

        if (j == 0)
        {
            common_fill_kernel<float><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
                1e9f, R * R * R, rasterizeResult.gridDist
            );
            CHECK_CUDA(cudaGetLastError());
            assert(R % 2 == 0);
            common_fill_kernel<int><<<ceil_div(R * R * R / 4 * 3LL, NTHREAD_1D), NTHREAD_1D>>>(
                0, R * R * R / 4 * 3, (int*)rasterizeResult.gridCollide
            );
            CHECK_CUDA(cudaGetLastError());
        }
        rasterize_reduce_kernel<<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(
            tris, idx, grid, M, R, rasterizeResult.gridDist, rasterizeResult.gridCollide
        );
        CHECK_CUDA(cudaGetLastError());
        ma.free(idx);
        ma.free(grid);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    return rasterizeResult;
}

static RasterizeResult rasterize_tris(const float3 * tris, const int F, const int R, const float band, const int B, const bool useCachedAllocator)
{
    assert(R <= 1024);
    std::vector<int> s;
    int N = R;
    if (N > 8)
    {
        s.push_back(8);
        N /= 8;
    }
    while (N > 4)
    {
        s.push_back(4);
        N /= 4;
    }
    s.push_back(N);
    return rasterize_tris_internal(tris, F, s, B, band, useCachedAllocator);
}
