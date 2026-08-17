#include <torch/extension.h>
#include "torchcumesh2sdf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_sdf", &get_sdf, "get_sdf (CUDA)", py::arg("tris"), py::arg("R"), py::arg("band"), py::arg("B") = 16384);
    m.def("get_udf", &get_udf, "get_udf (CUDA)", py::arg("tris"), py::arg("R"), py::arg("band"), py::arg("B") = 16384);
    m.def("get_collide", &get_collide, "get_collide (CUDA)", py::arg("tris"), py::arg("R"), py::arg("band"), py::arg("B") = 16384);
    m.def("free_cached_memory", &free_cached_memory, "free_cached_memory (CUDA)");
}
