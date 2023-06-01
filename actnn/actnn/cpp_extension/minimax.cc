#include <paddle/extension.h>
#include "ext_common.h"

std::pair<paddle::Tensor, paddle::Tensor> minimax_cuda(paddle::Tensor data);
std::pair<paddle::Tensor, paddle::Tensor> minimax(paddle::Tensor data) {
  CHECK_CUDA_TENSOR_FLOAT(data);
  return minimax_cuda(data);
}

PYBIND11_MODULE(PYTHON_MODULE_NAME, m) {
  m.def("minimax", &minimax);
}