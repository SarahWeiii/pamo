#include <cub/device/device_select.cuh>
#include "../simp_cuda/src/cusimp_free.cu"
#include <algorithm>
#if defined(_MSC_VER)
//  Microsoft 
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

static bool check_cuda_result_2(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#define CHECK_CUDA(code) check_cuda_result_2(code, __FILE__, __LINE__);

__global__ static void face_idx_map_kernel(cusimp_free::Triangle<int>* tris, const int* verts_map, const int n_tris)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n_tris)
    {
        tris[i].i = verts_map[tris[i].i];
        tris[i].j = verts_map[tris[i].j];
        tris[i].k = verts_map[tris[i].k];
    }
}

__global__ static void face_idx_mask_kernel(cusimp_free::Triangle<int>* tris, bool* mask, const int n_tris)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n_tris)
    {
        mask[i] = tris[i].i >= 0;
    }
}

static cusimp_free::CUSimp_Free pamo;

extern "C"
{
	void EXPORT pamo_simplify(float* in_verts, int* in_tris, float* out_verts, int* out_tris, int* in_out_nv_ni, float threshold)
	{
        int& n_verts = in_out_nv_ni[0];
        int& n_tris = in_out_nv_ni[1];

        float bbox[6] = {-1e30f, -1e30f, -1e30f, 1e30f, 1e30f, 1e30f};
        for (int i = 0; i < n_verts; i++)
        {
            bbox[0] = std::max(bbox[0], in_verts[i * 3]);
            bbox[1] = std::max(bbox[1], in_verts[i * 3 + 1]);
            bbox[2] = std::max(bbox[2], in_verts[i * 3 + 2]);

            bbox[3] = std::min(bbox[3], in_verts[i * 3]);
            bbox[4] = std::min(bbox[4], in_verts[i * 3 + 1]);
            bbox[5] = std::min(bbox[5], in_verts[i * 3 + 2]);
        }

        float scale = std::max({ bbox[0] - bbox[3], bbox[1] - bbox[4], bbox[2] - bbox[5] });
        const int tolerance = 4;
        int same_count = 0;
        bool is_stuck = false;
        bool is_init = true;

        cusimp_free::Vertex<float>* init_verts = nullptr;
        cusimp_free::Triangle<int>* init_tris = nullptr;
        bool* temp_tri_mask = nullptr;
        int* init_verts_undo = nullptr;
        int n_verts_undo = 0;

        int* nv_ni_cu = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&init_verts, n_verts * 2 * sizeof(cusimp_free::Vertex<float>)));
        CHECK_CUDA(cudaMalloc((void**)&init_tris, n_tris * 2 * sizeof(cusimp_free::Triangle<int>)));
        CHECK_CUDA(cudaMalloc((void**)&temp_tri_mask, n_tris * 2 * sizeof(bool)));
        CHECK_CUDA(cudaMalloc((void**)&init_verts_undo, sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&nv_ni_cu, sizeof(int) * 2));

        CHECK_CUDA(cudaMemcpy(init_verts, in_verts, n_verts * sizeof(cusimp_free::Vertex<float>), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(init_tris, in_tris, n_tris * sizeof(cusimp_free::Triangle<int>), cudaMemcpyHostToDevice));

        cusimp_free::Vertex<float>* verts = init_verts;
        cusimp_free::Triangle<int>* tris = init_tris;
        int* verts_undo = init_verts_undo;

        void* d_temp_storage = nullptr;
        size_t cur_temp_storage = 0;
        for (int it = 0; it < 1000000; it++)
        {
            pamo.forward(verts, tris, verts_undo, n_verts_undo, n_verts, n_tris, scale, threshold, is_stuck, is_init);
            is_init = false;

            verts = pamo.points;
            tris = pamo.triangles;
            int* verts_occ = pamo.pts_occ;
            int* verts_map = pamo.pts_map;
            verts_undo = pamo.vertices_undo_list;
            n_verts_undo = pamo.n_vertices_undo;
            
            size_t temp_storage_bytes = 0;
            CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, verts, verts_occ, init_verts, nv_ni_cu, pamo.n_pts));
            if (temp_storage_bytes >= cur_temp_storage)
            {
                if (d_temp_storage) CHECK_CUDA(cudaFree(d_temp_storage));
                CHECK_CUDA(cudaMalloc(&d_temp_storage, cur_temp_storage = temp_storage_bytes + temp_storage_bytes / 2));
            }
            CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, verts, verts_occ, init_verts, nv_ni_cu, pamo.n_pts));

            face_idx_mask_kernel<<<(pamo.n_tris + 256) / 256, 256 >>> (tris, temp_tri_mask, pamo.n_tris);
            CHECK_CUDA(cudaGetLastError());

            temp_storage_bytes = 0;
            CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, tris, temp_tri_mask, init_tris, nv_ni_cu + 1, pamo.n_tris));
            if (temp_storage_bytes >= cur_temp_storage)
            {
                if (d_temp_storage) CHECK_CUDA(cudaFree(d_temp_storage));
                CHECK_CUDA(cudaMalloc(&d_temp_storage, cur_temp_storage = temp_storage_bytes + temp_storage_bytes / 2));
            }
            CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, tris, temp_tri_mask, init_tris, nv_ni_cu + 1, pamo.n_tris));

            int old_n_tris = n_tris;
            CHECK_CUDA(cudaMemcpy(in_out_nv_ni, nv_ni_cu, sizeof(int) * 2, cudaMemcpyDeviceToHost));

            face_idx_map_kernel<<<(n_tris + 256) / 256, 256>>>(init_tris, verts_map, n_tris);
            CHECK_CUDA(cudaGetLastError());

            verts = init_verts;
            tris = init_tris;

            if (old_n_tris == n_tris) same_count++;
            else { same_count = 0; is_stuck = false; }

            if (same_count >= 2) is_stuck = true;

            if (same_count >= tolerance) break;
        }

        CHECK_CUDA(cudaMemcpy(out_verts, verts, sizeof(cusimp_free::Vertex<float>) * n_verts, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(out_tris, tris, sizeof(cusimp_free::Triangle<int>) * n_tris, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(init_verts));
        CHECK_CUDA(cudaFree(init_tris));
        CHECK_CUDA(cudaFree(temp_tri_mask));
        CHECK_CUDA(cudaFree(init_verts_undo));
        CHECK_CUDA(cudaFree(nv_ni_cu));
        if (d_temp_storage) CHECK_CUDA(cudaFree(d_temp_storage));
    }
}
