#pragma once
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include <utility>
#include "helper_math.h"

#define CHECK_CUDA(ans) _cuda_check<true>((ans), __FILE__, __LINE__)
template<bool abort>
inline void _cuda_check(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if constexpr (abort) exit(code);
   }
}

constexpr const int TILE_SIZE = 8;
constexpr const int NTHREAD_1D = 512;

constexpr int ceil_div(const long long a, const int b)
{
    return ((a) + (b) - 1) / (b);
}

__device__ __forceinline__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<typename T>
__global__ void common_fill_kernel(const T val, const uint L, T * outGrid)
{
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L) return;
    outGrid[g] = val;
}

__global__ void common_arange_kernel(uint * t, uint limit, uint ofs)
{
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= limit) return;
    t[g] = g + ofs;
}

inline uint npo2(const uint n)
{
    uint v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
