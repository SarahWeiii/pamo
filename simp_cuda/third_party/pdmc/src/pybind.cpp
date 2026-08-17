#include "cudualmc.h"
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace cudualmc
{
  template <typename Scalar, typename IndexType>
  class CUDMC
  {
    CUDualMC<Scalar, IndexType> dmc;

    static_assert(std::is_same<Scalar, double>() ||
                  std::is_same<Scalar, float>());
    static_assert(std::is_same<IndexType, long>() ||
                  std::is_same<IndexType, int>());

  public:
    ~CUDMC()
    {
      cudaDeviceSynchronize();
      cudaFree(dmc.temp_storage);
      cudaFree(dmc.first_cell_used);
      cudaFree(dmc.used_to_first_mc_vert);
      cudaFree(dmc.used_to_first_mc_patch);
      cudaFree(dmc.used_cell_code);
      cudaFree(dmc.used_cell_index);
      cudaFree(dmc.mc_vert_to_cell);
      cudaFree(dmc.mc_vert_type);
      cudaFree(dmc.quads);
      cudaFree(dmc.verts);
      cudaFree(dmc.mc_vert_to_edge);
      cudaFree(dmc.tris);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor grid, Scalar iso)
    {
      CHECK_INPUT(grid);

      torch::ScalarType scalarType;
      if constexpr (std::is_same<Scalar, double>())
      {
        scalarType = torch::kDouble;
      }
      else
      {
        scalarType = torch::kFloat;
      }
      TORCH_INTERNAL_ASSERT(grid.dtype() == scalarType,
                            "grid type must match the dmc class");

      torch::ScalarType indexType = torch::kInt;
      if constexpr (std::is_same<IndexType, int>())
      {
        indexType = torch::kInt;
      }
      else
      {
        indexType = torch::kLong;
      }

      IndexType dimX = grid.size(0);
      IndexType dimY = grid.size(1);
      IndexType dimZ = grid.size(2);

      dmc.forward(grid.data_ptr<Scalar>(), dimX, dimY, dimZ, iso, grid.device().index());

      auto verts =
          torch::from_blob(
              dmc.verts, torch::IntArrayRef{dmc.n_verts, 3},
              grid.options().dtype(scalarType))
              .clone();
      auto quads =
          torch::from_blob(
              dmc.quads, torch::IntArrayRef{dmc.n_quads, 4},
              grid.options().dtype(indexType))
              .clone();

      auto tris =
          torch::from_blob(
              dmc.tris, torch::IntArrayRef{dmc.n_tris, 3},
              grid.options().dtype(indexType))
              .clone();

      return {verts, quads, tris};
    }
  };

} // namespace cudualmc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<cudualmc::CUDMC<double, int>>(m, "CUDMCDouble")
      .def(py::init<>())
      .def("forward", &cudualmc::CUDMC<double, int>::forward);

  pybind11::class_<cudualmc::CUDMC<float, int>>(m, "CUDMCFloat")
      .def(py::init<>())
      .def("forward", &cudualmc::CUDMC<float, int>::forward);
}
