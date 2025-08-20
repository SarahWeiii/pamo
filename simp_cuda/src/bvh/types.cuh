#ifndef SELFX_TYPES_H
#define SELFX_TYPES_H

#include "aabb.cuh"
#include <vector_types.h>


namespace selfx{

template <typename T>
struct Vertex
{
    T x, y, z;

    inline __device__ __host__ T *data_ptr() { return &x; }
};

template <typename T>
struct Face
{
    T i, j, k;

    inline __device__ __host__ T *data_ptr() { return &i; }
};

struct aabb_getter
{
    __device__ lbvh::aabb<float> operator()(const selfx::Triangle<float3> &tri) const noexcept
    {
        lbvh::aabb<float> box;

        box.lower.x = min(tri.v0.x, min(tri.v1.x, tri.v2.x));
        box.lower.y = min(tri.v0.y, min(tri.v1.y, tri.v2.y));
        box.lower.z = min(tri.v0.z, min(tri.v1.z, tri.v2.z));

        box.upper.x = max(tri.v0.x, max(tri.v1.x, tri.v2.x));
        box.upper.y = max(tri.v0.y, max(tri.v1.y, tri.v2.y));
        box.upper.z = max(tri.v0.z, max(tri.v1.z, tri.v2.z));

        return box;
    }
};

}
//#endif
#endif