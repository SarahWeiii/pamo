#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>
#include "bvh/bvh.cuh"
#include "bvh/aabb.cuh"

namespace selfx{
  template <typename T>
  struct Triangle
  {
      T v0, v1, v2;

      inline __device__ __host__ T *data_ptr() { return &v0; }
  };
}

namespace cusimp_free
{
  typedef unsigned long long int uint64_cu;
  // typedef uint64_t uint64_cu;

  template <typename T>
  struct Vertex
  {
    T x, y, z;

    inline __device__ __host__ T *data_ptr() { return &x; }

    inline __device__ __host__ Vertex<T> operator+(Vertex<T> const &other) const
    {
      return {x + other.x, y + other.y, z + other.z};
    }
    inline __device__ __host__ T dot(Vertex<T> const &other) const
    {
      return x * other.x + y * other.y + z * other.z;
    }
    inline __device__ __host__ Vertex<T> cross(Vertex<T> const &other) const
    {
      return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }
    inline __device__ __host__ T norm() const
    {
      return sqrt(x * x + y * y + z * z);
    }
    inline __device__ __host__ Vertex<T> operator-(Vertex<T> const &other) const
    {
      return {x - other.x, y - other.y, z - other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(Vertex<T> const &other) const
    {
      return {x * other.x, y * other.y, z * other.z};
    }

    inline __device__ __host__ Vertex<T> operator*(T const &scalar) const
    {
      return {x * scalar, y * scalar, z * scalar};
    }

    inline __device__ __host__ Vertex<T> operator/(T const &scalar) const
    {
      return {x / scalar, y / scalar, z / scalar};
    }

    inline __device__ __host__ Vertex<T> &operator+=(Vertex<T> const &other)
    {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator-=(Vertex<T> const &other)
    {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator*=(T const &scalar)
    {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
    }

    inline __device__ __host__ Vertex<T> &operator/=(T const &scalar)
    {
      x /= scalar;
      y /= scalar;
      z /= scalar;
      return *this;
    }
  };

  template <typename T>
  struct Edge
  {
    T u, v;
    inline __device__ __host__ T *data_ptr() { return &u; }
  };

  template <typename T>
  struct Triangle
  {
    T i, j, k;
    inline __device__ __host__ T *data_ptr() { return &i; }
  };

  template <typename T>
  struct Mat4x4;

  template <typename T>
  struct Vec4
  {
    T x, y, z, w;
    inline __device__ __host__ T *data_ptr() { return &x; }

    inline __device__ __host__ Mat4x4<T> dot_T(Vec4<T> const &other) const
    {
      return {x * other.x, x * other.y, x * other.z, x * other.w,
              y * other.x, y * other.y, y * other.z, y * other.w,
              z * other.x, z * other.y, z * other.z, z * other.w,
              w * other.x, w * other.y, w * other.z, w * other.w};
    }

    inline __device__ __host__ T dot(Vec4<T> const &other) const
    {
      return x * other.x + y * other.y + z * other.z + w * other.w;
    }
  };

  template <typename T>
  struct Mat4x4
  {
    T m00, m01, m02, m03;
    T m10, m11, m12, m13;
    T m20, m21, m22, m23;
    T m30, m31, m32, m33;
    inline __device__ __host__ T *data_ptr() { return &m00; }
    inline __device__ __host__ T vTMv(Vec4<T> const &other) const
    {
      Vec4<T> vec1x4 = {m00 * other.x + m10 * other.y + m20 * other.z + m30 * other.w,
                     m01 * other.x + m11 * other.y + m21 * other.z + m31 * other.w,
                     m02 * other.x + m12 * other.y + m22 * other.z + m32 * other.w,
                     m03 * other.x + m13 * other.y + m23 * other.z + m33 * other.w};
      return vec1x4.dot(other);
    }

    inline __device__ __host__ Mat4x4<T> operator+(Mat4x4<T> const &other) const
    {
      return {m00 + other.m00, m01 + other.m01, m02 + other.m02, m03 + other.m03,
              m10 + other.m10, m11 + other.m11, m12 + other.m12, m13 + other.m13,
              m20 + other.m20, m21 + other.m21, m22 + other.m22, m23 + other.m23,
              m30 + other.m30, m31 + other.m31, m32 + other.m32, m33 + other.m33};
    }

    inline __device__ __host__ Mat4x4<T> &operator+=(Mat4x4<T> const &other)
    {
      m00 += other.m00;
      m01 += other.m01;
      m02 += other.m02;
      m03 += other.m03;
      m10 += other.m10;
      m11 += other.m11;
      m12 += other.m12;
      m13 += other.m13;
      m20 += other.m20;
      m21 += other.m21;
      m22 += other.m22;
      m23 += other.m23;
      m30 += other.m30;
      m31 += other.m31;
      m32 += other.m32;
      m33 += other.m33;
      return *this;
    }
  };

  struct CUSimp_Free
  {
    float tres{};
    uint32_t collapse_t{};
    float edge_s{};
    int n_pts{};
    int n_tris{};
    int n_edges{};
    int n_near_tris{};

    int n_invalid_vertices{}; // number of invalid vertices from iteration before
    int n_vertices_undo{};

    int* debug{};

    // temp storage
    size_t allocated_temp_storage_size{};
    int *__restrict__ temp_storage{}; // used for prefix sum

    // near triangle list
    size_t allocated_near_count{};
    int *__restrict__ first_near_tris{}; // link to the first triangle in the neighboring triangle list
    size_t allocated_near_tris{};
    int *__restrict__ near_tris{}; // neighboring triangle index list
    size_t allocated_near_offset{};
    int *__restrict__ near_offset{}; // help to fill the neighboring triangle list

    // edge list
    size_t allocated_edge_count{};
    int *__restrict__ first_edge{};   // link to the first edge in the neighboring edge list
    size_t allocated_edge{};
    Edge<int> *__restrict__ edges{};  // edge list

    // cost list
    size_t allocated_vert_Q{};
    Mat4x4<float> *__restrict__ vert_Q{};   // Q matrix for each vertex
    size_t allocated_edge_cost{};
    uint32_t *__restrict__ edge_cost{};     // cost for each edge
    size_t allocated_tri_min_cost{};
    uint64_cu *__restrict__ tri_min_cost{};  // the data type is fixed (int32 + int32)

    // output
    size_t allocated_pts{};
    Vertex<float> *__restrict__ points{}; // output points (start from input points)
    int *__restrict__ pts_occ{}; // whether points are valid
    int *__restrict__ pts_map{}; // vert map after removing redundant points
    size_t allocated_tris{};
    Triangle<int> *__restrict__ triangles{}; // output triangles (start from input triangles)

    // undo
    int *n_collapsed{}; //# of collapsed edge
    Vertex<float> *__restrict__ original_points{}; // restore buffer
    Triangle<int> *__restrict__ original_tris{}; // restore buffer
    uint32_t *__restrict__ original_edge_cost{};     // cost for each edge

    int *__restrict__ collapsed_edge_idx{}; // index of collpsed edge
    int * n_edges_undo{}; // number of undo candidate 
    int * edges_undo{}; // edge index of undo candidate

    int *__restrict__ vertices_undo_list{};
    int *__restrict__ tmp_vertices_undo_list{};
    int *__restrict__ vertices_invalid_list{}; // To track invalid edge from iteration before
    bool *__restrict__ vertices_invalid_table{}; // hashmap

    // self-intersected triangles
    int * __restrict__ query_triangle_list{};
    unsigned int *__restrict__ intersected_triangle_idx{};
    int * n_intersect{};

    // for lbvh
    thrust::device_vector<selfx::Triangle<float3>> bvh_triangles; // to construct bvh
    thrust::device_vector<unsigned int> num_found_query; // key : tris, value : # of query found
    thrust::device_vector<unsigned int> first_query_result;
    thrust::device_vector<unsigned int> intersect_candidates;


    inline __host__ void resize(int nPts, int nTris)
    {
      n_pts = nPts;
      n_tris = nTris;
    }

    __host__ void ensure_temp_storage_size(size_t size);
    __host__ void ensure_pts_storage_size(size_t n_pts);
    __host__ void ensure_tris_storage_size(size_t n_tris);
    __host__ void ensure_near_count_storage_size(size_t n_pts);
    __host__ void ensure_near_tris_storage_size(size_t n_near_tris);
    __host__ void ensure_near_offset_storage_size(size_t n_pts);
    __host__ void ensure_edge_count_storage_size(size_t n_tris);
    __host__ void ensure_edge_storage_size(size_t n_edges);
    __host__ void ensure_vert_Q_storage_size(size_t n_pts);
    __host__ void ensure_edge_cost_storage_size(size_t n_edges);
    __host__ void ensure_tri_min_cost_storage_size(size_t n_tris);

    // triangles must start from 0
    __host__ void forward(Vertex<float> *pts, Triangle<int> *tris, int* verts_undo, int n_verts_undo, int nPts, int nTris, float scale, float threshold, bool is_stuck, bool init);
  };
}

