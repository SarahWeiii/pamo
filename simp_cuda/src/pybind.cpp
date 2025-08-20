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

public:
    ~CUDSP_Free()
    {
      cudaDeviceSynchronize();
      cudaFree(pamo.temp_storage);
      cudaFree(pamo.first_near_tris);
      cudaFree(pamo.near_tris);
      cudaFree(pamo.near_offset);
      cudaFree(pamo.first_edge);
      cudaFree(pamo.edges);
      cudaFree(pamo.vert_Q);
      cudaFree(pamo.edge_cost);
      cudaFree(pamo.tri_min_cost);
      cudaFree(pamo.points);
      cudaFree(pamo.triangles);
      cudaFree(pamo.n_collapsed);
      cudaFree(pamo.original_points);
      cudaFree(pamo.original_tris);
      cudaFree(pamo.original_edge_cost);
      cudaFree(pamo.collapsed_edge_idx);
      cudaFree(pamo.n_edges_undo);
      cudaFree(pamo.edges_undo);
      cudaFree(pamo.vertices_undo_list);
      cudaFree(pamo.tmp_vertices_undo_list);
      cudaFree(pamo.vertices_invalid_list);
      cudaFree(pamo.vertices_invalid_table);
      cudaFree(pamo.query_triangle_list);
      cudaFree(pamo.intersected_triangle_idx);
      cudaFree(pamo.n_intersect);
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

public:
    ~CUDSP()
    {
      cudaDeviceSynchronize();
      cudaFree(sp.temp_storage);
      cudaFree(sp.first_near_tris);
      cudaFree(sp.near_tris);
      cudaFree(sp.near_offset);
      cudaFree(sp.first_edge);
      cudaFree(sp.edges);
      cudaFree(sp.vert_Q);
      cudaFree(sp.edge_cost);
      cudaFree(sp.tri_min_cost);
      cudaFree(sp.points);
      cudaFree(sp.triangles);
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