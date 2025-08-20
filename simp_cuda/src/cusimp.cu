#include "cusimp.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/fill.h"
#include <math.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

namespace cusimp
{
    const int BLOCK_SIZE = 512;
    const float COST_RANGE = 10.0;

    bool check_cuda_result(cudaError_t code, const char *file, int line)
    {
        if (code == cudaSuccess)
            return true;

        fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file,
                line);
        return false;
    }

#define CHECK_CUDA(code) check_cuda_result(code, __FILE__, __LINE__);

    template <typename T>
    inline __device__ __host__ T min(T a, T b) { return a < b ? a : b; }
    template <typename T>
    inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }
    template <typename T>
    inline __device__ __host__ T clamp(T x, T a, T b) { return min(max(a, x), b); }

    void CUSimp::ensure_temp_storage_size(size_t size)
    {
        if (size > allocated_temp_storage_size)
        {
            allocated_temp_storage_size = size_t(size + size / 5);
            // fprintf(stderr, "allocated_temp_storage_size %ld\n", allocated_temp_storage_size);
            CHECK_CUDA(cudaFree(temp_storage));
            CHECK_CUDA(cudaMalloc((void **)&temp_storage, allocated_temp_storage_size));
        }
    }

    void CUSimp::ensure_pts_storage_size(size_t point_count)
    {
        if (point_count > allocated_pts)
        {
            allocated_pts = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_pts %ld\n", allocated_pts);
            CHECK_CUDA(cudaFree(points));
            CHECK_CUDA(
                cudaMalloc((void **)&points, (allocated_pts + 1) * sizeof(Vertex<float>)));

            CHECK_CUDA(cudaFree(pts_occ));
            CHECK_CUDA(
                cudaMalloc((void **)&pts_occ, (allocated_pts + 1) * sizeof(int)));

            CHECK_CUDA(cudaFree(pts_map));
            CHECK_CUDA(
                cudaMalloc((void **)&pts_map, (allocated_pts + 1) * sizeof(int)));
        }
    }

    void CUSimp::ensure_tris_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_tris)
        {
            allocated_tris = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_tris %ld\n", allocated_tris);
            CHECK_CUDA(cudaFree(triangles));
            CHECK_CUDA(
                cudaMalloc((void **)&triangles, (allocated_tris + 1) * sizeof(Triangle<int>)));
        }
    }

    void CUSimp::ensure_near_count_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_count)
        {
            allocated_near_count = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_count %ld\n", allocated_near_count);
            CHECK_CUDA(cudaFree(first_near_tris));
            CHECK_CUDA(
                cudaMalloc((void **)&first_near_tris, (allocated_near_count + 1) * sizeof(int)));
        }
    }

    void CUSimp::ensure_near_tris_storage_size(size_t near_tri_count)
    {
        if (near_tri_count > allocated_near_tris)
        {
            allocated_near_tris = size_t(near_tri_count + near_tri_count / 5);
            // fprintf(stderr, "allocated_near_tris %ld\n", allocated_near_tris);
            CHECK_CUDA(cudaFree(near_tris));
            CHECK_CUDA(
                cudaMalloc((void **)&near_tris, (allocated_near_tris + 1) * sizeof(int)));
        }
    }

    void CUSimp::ensure_near_offset_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_offset)
        {
            allocated_near_offset = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_offset %ld\n", allocated_near_offset);
            CHECK_CUDA(cudaFree(near_offset));
            CHECK_CUDA(
                cudaMalloc((void **)&near_offset, (allocated_near_offset + 1) * sizeof(int)));
        }
    }

    void CUSimp::ensure_edge_count_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_edge_count)
        {
            allocated_edge_count = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_edge_count %ld\n", allocated_edge_count);
            CHECK_CUDA(cudaFree(first_edge));
            CHECK_CUDA(
                cudaMalloc((void **)&first_edge, (allocated_edge_count + 1) * sizeof(int)));
        }
    }

    void CUSimp::ensure_edge_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge)
        {
            allocated_edge = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge %ld\n", allocated_edge);
            CHECK_CUDA(cudaFree(edges));
            CHECK_CUDA(
                cudaMalloc((void **)&edges, (allocated_edge + 1) * sizeof(Edge<int>)));
        }
    }

    void CUSimp::ensure_vert_Q_storage_size(size_t point_count)
    {
        if (point_count > allocated_vert_Q)
        {
            allocated_vert_Q = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_vert_Q %ld\n", allocated_vert_Q);
            CHECK_CUDA(cudaFree(vert_Q));
            CHECK_CUDA(
                cudaMalloc((void **)&vert_Q, (allocated_vert_Q + 1) * sizeof(Mat4x4<float>)));
        }
    }

    void CUSimp::ensure_edge_cost_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge_cost)
        {
            allocated_edge_cost = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge_cost %ld\n", allocated_edge_cost);
            CHECK_CUDA(cudaFree(edge_cost));
            CHECK_CUDA(
                cudaMalloc((void **)&edge_cost, (allocated_edge_cost + 1) * sizeof(uint32_t)));
        }
    }

    void CUSimp::ensure_tri_min_cost_storage_size(size_t tri_count)
    {
        if (tri_count > allocated_tri_min_cost)
        {
            allocated_tri_min_cost = size_t(tri_count + tri_count / 5);
            // fprintf(stderr, "allocated_tri_min_cost %ld\n", allocated_tri_min_cost);
            CHECK_CUDA(cudaFree(tri_min_cost));
            CHECK_CUDA(
                cudaMalloc((void **)&tri_min_cost, (allocated_tri_min_cost + 1) * sizeof(uint64_cu)));
        }
    }

    __global__ void count_near_tris_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        atomicAdd(&sp.first_near_tris[tri.i], 1);
        atomicAdd(&sp.first_near_tris[tri.j], 1);
        atomicAdd(&sp.first_near_tris[tri.k], 1);
    }

    __global__ void create_near_tris_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        sp.near_tris[sp.first_near_tris[tri.i] + atomicAdd(&sp.near_offset[tri.i], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.j] + atomicAdd(&sp.near_offset[tri.j], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.k] + atomicAdd(&sp.near_offset[tri.k], 1)] = tri_index;
    }

    __global__ void count_edge_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;
        sp.first_edge[tri_index] += (tri.i > tri.j) + (tri.j > tri.k) + (tri.k > tri.i);
    }

    __global__ void create_edge_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        int first = sp.first_edge[tri_index];
        if (tri.i > tri.j)
            sp.edges[first++] = {int(tri.j), int(tri.i)};
        if (tri.j > tri.k)
            sp.edges[first++] = {int(tri.k), int(tri.j)};
        if (tri.k > tri.i)
            sp.edges[first++] = {int(tri.i), int(tri.k)};
    }

    __device__ Vec4<float> tri2plane(Vertex<float> const *points, Triangle<int> tri)
    {
        Vertex<float> v0 = points[tri.i];
        Vertex<float> v1 = points[tri.j];
        Vertex<float> v2 = points[tri.k];
        // Vertex<float> v0 = {0, 0, 2};
        // Vertex<float> v1 = {1, 0, 0};
        // Vertex<float> v2 = {0, 3, 0};
        Vertex<float> normal = (v1 - v0).cross(v2 - v0);
        normal /= normal.norm();
        float offset = -normal.dot(v0);
        return {normal.x, normal.y, normal.z, offset};
    }

    __global__ void compute_vert_Q_kernel(CUSimp sp)
    {
        int pt_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (pt_index >= sp.n_pts)
            return;
        // printf("pt_index %d sp.n_pts %d\n", pt_index, sp.n_pts);

        int first = sp.first_near_tris[pt_index];
        int last = sp.first_near_tris[pt_index + 1];
        Mat4x4<float> Kp{0};
        for (int i = first; i < last; ++i)
        {
            Vec4<float> p = tri2plane(sp.points, sp.triangles[sp.near_tris[i]]);
            // if (pt_index == 2644)
            //     printf("p %f %f %f %f\n", p.x, p.y, p.z, p.w);
            // Mat4x4<float> temp = p.dot_T(p);
            Kp += p.dot_T(p);
            // printf("p %f %f %f %f\ntemp %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n\n", p.x, p.y, p.z, p.w, temp.m00, temp.m01, temp.m02, temp.m03, temp.m10, temp.m11, temp.m12, temp.m13, temp.m20, temp.m21, temp.m22, temp.m23, temp.m30, temp.m31, temp.m32, temp.m33);
        }
        sp.vert_Q[pt_index] = Kp;
    }

    __device__ float triangle_area(Vertex<float> p0, Vertex<float> p1, Vertex<float> p2)
    {
        float a = (p0 - p1).norm();
        float b = (p1 - p2).norm();
        float c = (p2 - p0).norm();
        float s = (a + b + c) / 2;
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }

    __device__ float edge_length(Vertex<float> p0, Vertex<float> p1)
    {
        return (p0 - p1).norm();
    }

    __global__ void compute_edge_cost_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d sp.n_edges %d\n", edge_index, sp.n_edges);

        Edge<int> edge = sp.edges[edge_index];
        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];

        // check if the collapse is valid
        int dup_num = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            int idx_u;
            // edge i-j; j-k; k-i
            if (sp.triangles[sp.near_tris[i]].i == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].j;
            else if (sp.triangles[sp.near_tris[i]].j == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].k;
            else if (sp.triangles[sp.near_tris[i]].k == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].i;
            else
                printf("error1\n");
            for (int j = sp.first_near_tris[edge.v]; j < sp.first_near_tris[edge.v + 1]; ++j)
            {
                int idx_v;
                // edge i-j; j-k; k-i
                if (sp.triangles[sp.near_tris[j]].i == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].j;
                else if (sp.triangles[sp.near_tris[j]].j == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].k;
                else if (sp.triangles[sp.near_tris[j]].k == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].i;
                else
                    printf("error2\n");
                if (idx_u == idx_v)
                    dup_num++;
                if (dup_num > 2)
                {
                    sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                    return;
                }
            }
        }
        if (dup_num != 2)
        {
            return;
            printf("dup_num %d\n", dup_num);
        }

        // compute near edge length
        float edge_len = 0;
        int num_edge = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        edge_len = edge_len / num_edge / sp.edge_s * sp.tres;
        // printf("edge_len %f\n", edge_len);

        // compute edge length
        edge_len += edge_len + (v0 - v1).norm() / sp.edge_s * sp.tres;
        // edge_len = 0;

        Vertex<float> v = (v0 + v1) / 2;

        // compute skinny triangle cost Q_a
        // test if the triangle normal is flipped
        float Q_a = 0;
        int num_tri = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.v || old_tri.j == edge.v || old_tri.k == edge.v)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.u with v
            if (old_tri.i == edge.u)
                new_v0 = v;
            if (old_tri.j == edge.u)
                new_v1 = v;
            if (old_tri.k == edge.u)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
            {
                sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                return;
            }

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }

        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.u || old_tri.j == edge.u || old_tri.k == edge.u)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.v with v
            if (old_tri.i == edge.v)
                new_v0 = v;
            if (old_tri.j == edge.v)
                new_v1 = v;
            if (old_tri.k == edge.v)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
            {
                sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                return;
            }

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }

        Mat4x4<float> Q = sp.vert_Q[edge.u] + sp.vert_Q[edge.v];
        Vec4<float> v4 = {v.x, v.y, v.z, 1};
        float cost = Q.vTMv(v4) / (sp.edge_s * sp.edge_s);
        sp.edge_cost[edge_index] = uint32_t(clamp(cost + edge_len + Q_a / num_tri * sp.tres, 0.0f, COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());
        // if (edge.v == 2644)
        // {
        //     // Q = sp.vert_Q[edge.v];
        //     // printf("cost_ori %f, cost %f, sp.edge_s %f, edge_len %f, edge.u: f %d 1 2, edge.v: f %d 1 2, v %f %f %f\n Q %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n\n", Q.vTMv(v4), cost, sp.edge_s, edge_len, edge.u+1, edge.v+1, v.x, v.y, v.z, Q.m00, Q.m01, Q.m02, Q.m03, Q.m10, Q.m11, Q.m12, Q.m13, Q.m20, Q.m21, Q.m22, Q.m23, Q.m30, Q.m31, Q.m32, Q.m33);
        //     Vec4<float> vTM = {Q.m00 * v4.x + Q.m10 * v4.y + Q.m20 * v4.z + Q.m30 * v4.w,
        //              Q.m01 * v4.x + Q.m11 * v4.y + Q.m21 * v4.z + Q.m31 * v4.w,
        //              Q.m02 * v4.x + Q.m12 * v4.y + Q.m22 * v4.z + Q.m32 * v4.w,
        //              Q.m03 * v4.x + Q.m13 * v4.y + Q.m23 * v4.z + Q.m33 * v4.w};
        //     float c = vTM.dot(v4);
        //     printf("Q.m00 * v4.x %f \n Q.m30 * v4.w %f \n vTM %f %f %f %f \n c %f\n cost %f \n v4 %f %f %f %f\n Q %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n\n", Q.m00 * v4.x, Q.m30 * v4.w, vTM.x, vTM.y, vTM.z, vTM.w, c, cost, v4.x, v4.y, v4.z, v4.w, Q.m00, Q.m01, Q.m02, Q.m03, Q.m10, Q.m11, Q.m12, Q.m13, Q.m20, Q.m21, Q.m22, Q.m23, Q.m30, Q.m31, Q.m32, Q.m33);
        // }
    }

    __global__ void propagate_edge_cost_kernel(CUSimp sp)
    {
        uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d sp.n_edges %d\n", edge_index, sp.n_edges);
        // printf("sp.n_edges %d\n", sp.n_edges);

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("cost %llu edge_index %d, ((uint64_cu)edge_index) << 32 %llu, sp.edge_cost[edge_index] %u \n", cost, edge_index, ((uint64_cu)edge_index) << 32, sp.edge_cost[edge_index]);

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }
    }

    __global__ void collapse_edge_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d\n", edge_index);
        // atomicAdd(&sp.debug[0], 1);

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("sp.edge_cost[edge_index] %u, sp.collapse_t %u\n", sp.edge_cost[edge_index], sp.collapse_t);
        if (sp.edge_cost[edge_index] > sp.collapse_t)
        {
            // printf("enter\n");
            return;
        }

        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.near_tri[i] %d, sp.tri_min_cost[sp.near_tri[i]] %llu, cost %llu\n", sp.near_tris[i], sp.tri_min_cost[sp.near_tris[i]], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.tri_min_cost[i] %llu, cost %llu\n", sp.tri_min_cost[i], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        // printf("collapse %d - %d\n", edge.u, edge.v);
        Vertex<float> v = (v0 + v1) / 2;
        sp.points[edge.u] = v;
        sp.points[edge.v] = {0, 0, 0};
        sp.pts_occ[edge.v] = 0;
        first = sp.first_near_tris[edge.u];
        last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.v || sp.triangles[sp.near_tris[i]].j == edge.v || sp.triangles[sp.near_tris[i]].k == edge.v)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.u || sp.triangles[sp.near_tris[i]].j == edge.u || sp.triangles[sp.near_tris[i]].k == edge.u)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
            else if (sp.triangles[sp.near_tris[i]].i == edge.v)
                sp.triangles[sp.near_tris[i]].i = edge.u;
            else if (sp.triangles[sp.near_tris[i]].j == edge.v)
                sp.triangles[sp.near_tris[i]].j = edge.u;
            else if (sp.triangles[sp.near_tris[i]].k == edge.v)
                sp.triangles[sp.near_tris[i]].k = edge.u;
        }
    }

    __host__ void CUSimp::forward(Vertex<float> *pts, Triangle<int> *tris, int nPts, int nTris, float scale, float threshold, bool init)
    {
        tres = threshold;
        edge_s = scale;
        collapse_t = uint32_t(clamp(threshold, float(0.0), COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());
        // printf("collapse_t %u\n", collapse_t);

        resize(nPts, nTris);

        ensure_pts_storage_size(n_pts);
        CHECK_CUDA(cudaMemcpy(points, pts, n_pts * sizeof(Vertex<float>),
                              cudaMemcpyHostToDevice));
        std::vector<int> tmp(n_pts, 1);
        CHECK_CUDA(cudaMemcpy(pts_occ, tmp.data(), n_pts * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(pts_occ+n_pts, 0, sizeof(int)));
        ensure_tris_storage_size(n_tris);
        CHECK_CUDA(cudaMemcpy(triangles, tris, n_tris * sizeof(Triangle<int>),
                              cudaMemcpyHostToDevice));

        if (init){
            thrust::device_ptr<Triangle<int>> thrust_triangles(triangles);
            thrust::default_random_engine rng;
            thrust::shuffle(thrust_triangles, thrust_triangles + n_tris, rng);
        }

        size_t temp_storage_bytes = 0;

        ensure_near_count_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(first_near_tris, 0, (n_pts + 1) * sizeof(int)));
        count_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test first_near_tris
        // int *first_near_tris_host;
        // cudaMallocHost((void **)&first_near_tris_host, n_pts * sizeof(int));
        // cudaMemcpy(first_near_tris_host, first_near_tris, n_pts * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     // printf("%d - first_near_tris_host: %d\n", i, first_near_tris_host[i]);
        //     printf("%d - %d\n", i, first_near_tris_host[i]);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_near_tris, first_near_tris, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_near_tris, first_near_tris, n_pts + 1);

        CHECK_CUDA(cudaMemcpy(&n_near_tris, first_near_tris + n_pts, sizeof(int),
                                cudaMemcpyDeviceToHost));
        // fprintf(stderr, "near tris %d\n", n_near_tris);

        // // Test first_near_tris after prefix-sum
        // cudaMallocHost((void **)&first_near_tris_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(first_near_tris_host, first_near_tris, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     // printf("%d - first_near_tris_host: %d\n", i, first_near_tris_host[i]);
        //     printf("%d - %d\n", i, first_near_tris_host[i]);

        ensure_near_tris_storage_size(n_near_tris);
        ensure_near_offset_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(near_offset, 0, (n_pts + 1) * sizeof(int)));
        create_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test near_tris
        // int *near_tris_host;
        // cudaMallocHost((void **)&near_tris_host, n_near_tris * sizeof(int));
        // cudaMemcpy(near_tris_host, near_tris, n_near_tris * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_near_tris;i++)
        //     printf("%d - near_tris_host: %d\n", i, near_tris_host[i]);

        ensure_edge_count_storage_size(n_tris);
        CHECK_CUDA(cudaMemset(first_edge, 0, (n_tris + 1) * sizeof(int)));
        count_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_edge, first_edge, n_tris + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_edge, first_edge, n_tris + 1);

        CHECK_CUDA(cudaMemcpy(&n_edges, first_edge + n_tris, sizeof(int),
                                cudaMemcpyDeviceToHost));

        // printf("n_edges: %d\n", n_edges);

        ensure_edge_storage_size(n_edges);
        create_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test edges
        // Edge<int> *edges_host;
        // cudaMallocHost((void **)&edges_host, n_edges * sizeof(Edge<int>));
        // cudaMemcpy(edges_host, edges, n_edges * sizeof(Edge<int>), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_edges;i++)
        //     printf("%d - edges_host: %d - %d\n", i, edges_host[i].u, edges_host[i].v);

        ensure_vert_Q_storage_size(n_pts);
        compute_vert_Q_kernel<<<(n_pts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test vert_cost
        // float *vert_cost_host;
        // cudaMallocHost((void **)&vert_cost_host, n_pts * sizeof(float));
        // cudaMemcpy(vert_cost_host, vert_cost, n_pts * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     printf("%d - vert_cost_host: %f\n", i, vert_cost_host[i]);

        ensure_edge_cost_storage_size(n_edges);
        compute_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test edge_cost
        // uint32_t *edge_cost_host;
        // cudaMallocHost((void **)&edge_cost_host, n_edges * sizeof(uint32_t));
        // cudaMemcpy(edge_cost_host, edge_cost, n_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < n_edges; i++)
        //   printf("%d - edge_cost_host: %u\n", i, edge_cost_host[i]);

        ensure_tri_min_cost_storage_size(n_tris);
        std::vector<uint64_cu> temp(n_tris, std::numeric_limits<uint64_cu>::max());
        CHECK_CUDA(cudaMemcpy(tri_min_cost, temp.data(), n_tris * sizeof(uint64_cu), cudaMemcpyHostToDevice));
        propagate_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        collapse_edge_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        CHECK_CUDA(cudaMemcpy(pts_map, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToDevice));
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, pts_map, pts_map, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, pts_map, pts_map, n_pts + 1);

        // // Test pts_occ_cost
        // int *pts_occ_host;
        // cudaMallocHost((void **)&pts_occ_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_occ_host, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_occ_host: %u\n", i, pts_occ_host[i]);

        // // Test pts_map_cost
        // int *pts_map_host;
        // cudaMallocHost((void **)&pts_map_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_map_host, pts_map, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_map_host: %u\n", i, pts_map_host[i]);
    }
}