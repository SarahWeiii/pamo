#ifndef LBVH_MATH_CUH
#define LBVH_MATH_CUH
#include "types.cuh"

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

__device__ __host__ float3 operator*(float scalar, const float3& vec) {
    return make_float3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

__device__ __host__ float3 operator*(const float3& vec, float scalar) {
    return scalar * vec;
}

__device__ __host__ bool operator==(const float3 &a, const float3 &b) {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__device__ __host__ float3 operator/(const float3 &a, const float &k) {
        return make_float3(a.x / k, a.y / k, a.z / k);
    }

inline __device__ __host__ float3 cross(const float3 &u, const float3 &v) {
        float3 w;
        w.x = u.y * v.z - u.z * v.y;
        w.y = u.z * v.x - u.x * v.z;
        w.z = u.x * v.y - u.y * v.x;
        return w;
    }
inline __device__ __host__ float dot(const float3 &u, const float3 &v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    }
inline __device__ __host__ float norm(const float3 &u) {
        return sqrtf(dot(u, u));
    }

__device__ __host__ float largest_distance(const float3 &a, const float3 &b){
    float dx = std::abs(b.x - a.x);
    float dy = std::abs(b.y - a.y);
    float dz = std::abs(b.z - a.z);
    float largest_dist = -std::numeric_limits<float>::infinity();
    if (dx > largest_dist) largest_dist = dx;
    if (dy > largest_dist) largest_dist = dy;
    if (dz > largest_dist) largest_dist = dz;
    return largest_dist;
}

__device__ __host__ float distance(const float3 &a, const float3 &b){
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}


#endif