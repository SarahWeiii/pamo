#include "cusimp.h"
#include "cusimp_free.h"
#include <torch/extension.h>

namespace cusimp_free
{

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

  class CUDSP_Free
  {
    CUSimp_Free pamo;

    void release()
    {
      cudaDeviceSynchronize();
      // cudaFree(nullptr) is a no-op; free every owned buffer once and null
      // so a double-free path cannot use a stale pointer.
      cudaFree(pamo.temp_storage); pamo.temp_storage = nullptr;
      cudaFree(pamo.first_near_tris); pamo.first_near_tris = nullptr;
      cudaFree(pamo.near_tris); pamo.near_tris = nullptr;
      cudaFree(pamo.near_offset); pamo.near_offset = nullptr;
      cudaFree(pamo.first_edge); pamo.first_edge = nullptr;
      cudaFree(pamo.edges); pamo.edges = nullptr;
      cudaFree(pamo.vert_Q); pamo.vert_Q = nullptr;
      cudaFree(pamo.edge_cost); pamo.edge_cost = nullptr;
      cudaFree(pamo.tri_min_cost); pamo.tri_min_cost = nullptr;
      cudaFree(pamo.points); pamo.points = nullptr;
      cudaFree(pamo.pts_occ); pamo.pts_occ = nullptr;
      cudaFree(pamo.pts_map); pamo.pts_map = nullptr;
      cudaFree(pamo.triangles); pamo.triangles = nullptr;
      cudaFree(pamo.n_collapsed); pamo.n_collapsed = nullptr;
      cudaFree(pamo.original_points); pamo.original_points = nullptr;
      cudaFree(pamo.original_tris); pamo.original_tris = nullptr;
      cudaFree(pamo.original_edge_cost); pamo.original_edge_cost = nullptr;
      cudaFree(pamo.collapsed_edge_idx); pamo.collapsed_edge_idx = nullptr;
      cudaFree(pamo.n_edges_undo); pamo.n_edges_undo = nullptr;
      cudaFree(pamo.edges_undo); pamo.edges_undo = nullptr;
      cudaFree(pamo.vertices_undo_list); pamo.vertices_undo_list = nullptr;
      cudaFree(pamo.tmp_vertices_undo_list); pamo.tmp_vertices_undo_list = nullptr;
      cudaFree(pamo.vertices_invalid_list); pamo.vertices_invalid_list = nullptr;
      cudaFree(pamo.vertices_invalid_table); pamo.vertices_invalid_table = nullptr;
      cudaFree(pamo.query_triangle_list); pamo.query_triangle_list = nullptr;
      cudaFree(pamo.intersected_triangle_idx); pamo.intersected_triangle_idx = nullptr;
      cudaFree(pamo.n_intersect); pamo.n_intersect = nullptr;
    }

public:
    ~CUDSP_Free()
    {
      release();
    }

    //std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor points, torch::Tensor triangles, int iter, float scale, float epsilon, float threshold)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor points, torch::Tensor triangles, torch::Tensor verts_undo, int n_verts_undo, float scale, float threshold, bool is_stuck, bool init)
    {      
      CHECK_INPUT(points);
      CHECK_INPUT(triangles);
      CHECK_INPUT(verts_undo);

      torch::ScalarType scalarType = torch::kFloat;
      TORCH_INTERNAL_ASSERT(points.dtype() == scalarType,
                            "points type must match the pamo class");
      torch::ScalarType indexType = torch::kInt;
      TORCH_INTERNAL_ASSERT(triangles.dtype() == indexType,
                            "triangles type must match the pamo class");

      int nPts = points.size(0);
      int nTris = triangles.size(0);

      pamo.forward(reinterpret_cast<Vertex<float> *>(points.data_ptr<float>()),
            reinterpret_cast<Triangle<int> *>(triangles.data_ptr<int>()),
            reinterpret_cast<int *>(verts_undo.data_ptr<int>()),
            n_verts_undo,
            nPts, nTris, scale, threshold, is_stuck, init);
      
      auto verts =
          torch::from_blob(
              pamo.points, torch::IntArrayRef{pamo.n_pts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto tris =
          torch::from_blob(
              pamo.triangles, torch::IntArrayRef{pamo.n_tris, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      auto verts_occ =
          torch::from_blob(
              pamo.pts_occ, torch::IntArrayRef{pamo.n_pts, 1},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();
      auto verts_map =
          torch::from_blob(
              pamo.pts_map, torch::IntArrayRef{pamo.n_pts, 1},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      auto vertices_undo =
          torch::from_blob(
              pamo.vertices_undo_list, torch::IntArrayRef{pamo.n_vertices_undo},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, tris, verts_occ, verts_map, vertices_undo};
    }
  }; 

} // namespace cusimp_free

namespace cusimp
{

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

  class CUDSP
  {
    CUSimp sp;

    void release()
    {
      cudaDeviceSynchronize();
      cudaFree(sp.temp_storage); sp.temp_storage = nullptr;
      cudaFree(sp.first_near_tris); sp.first_near_tris = nullptr;
      cudaFree(sp.near_tris); sp.near_tris = nullptr;
      cudaFree(sp.near_offset); sp.near_offset = nullptr;
      cudaFree(sp.first_edge); sp.first_edge = nullptr;
      cudaFree(sp.edges); sp.edges = nullptr;
      cudaFree(sp.vert_Q); sp.vert_Q = nullptr;
      cudaFree(sp.edge_cost); sp.edge_cost = nullptr;
      cudaFree(sp.tri_min_cost); sp.tri_min_cost = nullptr;
      cudaFree(sp.points); sp.points = nullptr;
      cudaFree(sp.pts_occ); sp.pts_occ = nullptr;
      cudaFree(sp.pts_map); sp.pts_map = nullptr;
      cudaFree(sp.triangles); sp.triangles = nullptr;
    }

public:
    ~CUDSP()
    {
      release();
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor points, torch::Tensor triangles, float scale, float threshold, bool init)
    {      
      CHECK_INPUT(points);
      CHECK_INPUT(triangles);

      torch::ScalarType scalarType = torch::kFloat;
      TORCH_INTERNAL_ASSERT(points.dtype() == scalarType,
                            "points type must match the sp class");
      torch::ScalarType indexType = torch::kInt;
      TORCH_INTERNAL_ASSERT(triangles.dtype() == indexType,
                            "triangles type must match the sp class");

      int nPts = points.size(0);
      int nTris = triangles.size(0);

      sp.forward(reinterpret_cast<Vertex<float> *>(points.data_ptr<float>()),
                  reinterpret_cast<Triangle<int> *>(triangles.data_ptr<int>()),
                  nPts, nTris, scale, threshold, init);
      
      auto verts =
          torch::from_blob(
              sp.points, torch::IntArrayRef{sp.n_pts, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(scalarType))
              .clone();
      auto tris =
          torch::from_blob(
              sp.triangles, torch::IntArrayRef{sp.n_tris, 3},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      auto verts_occ =
          torch::from_blob(
              sp.pts_occ, torch::IntArrayRef{sp.n_pts, 1},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();
      auto verts_map =
          torch::from_blob(
              sp.pts_map, torch::IntArrayRef{sp.n_pts, 1},
              torch::TensorOptions().device(torch::kCUDA).dtype(indexType))
              .clone();

      return {verts, tris, verts_occ, verts_map};
    }
  };  

} // namespace cusimp



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<cusimp_free::CUDSP_Free>(m, "CUDSP_Free")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, int, float, float, bool, bool>(&cusimp_free::CUDSP_Free::forward));
      
  pybind11::class_<cusimp::CUDSP>(m, "CUDSP")
      .def(py::init<>())
      .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, float, float, bool>(&cusimp::CUDSP::forward));
}
