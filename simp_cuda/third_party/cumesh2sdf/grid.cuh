#pragma once
#include "commons.cuh"
constexpr const uint GRID_BITS = 10;
constexpr const uint GRID_MASK = (1 << GRID_BITS) - 1;

__forceinline__ __device__ __host__ uint3 unpack_id(const uint gridId)
{
    return make_uint3(gridId >> (GRID_BITS << 1), (gridId >> GRID_BITS) & GRID_MASK, gridId & GRID_MASK);
}

__forceinline__ __device__ __host__ uint pack_id(const uint3 gridCoo)
{
    return gridCoo.z | (gridCoo.y << GRID_BITS) | (gridCoo.x << (GRID_BITS << 1));
}

__forceinline__ __device__ uint3 f2i(const float3 gridCoo)
{
    return make_uint3(gridCoo.x, gridCoo.y, gridCoo.z);
}

__forceinline__ __device__ float3 i2f(const uint3 gridCoo)
{
    return make_float3(gridCoo.x, gridCoo.y, gridCoo.z);
}

__forceinline__ __device__ uint to_gidx(const uint3 ijk, const int N)
{
    assert(ijk.x < N && ijk.y < N && ijk.z < N);
    return ijk.x * N * N + ijk.y * N + ijk.z;
}
