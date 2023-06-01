#include <paddle/extension.h>
#include <paddle/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ext_common.h"
namespace py = pybind11;

using paddle::Tensor;
using paddle::TensorList;
using paddle::framework::AttributeMap;
using paddle::framework::ExecutionContext;
using paddle::framework::OpKernelType;
using paddle::framework::TensorFromVector;
using paddle::framework::TensorToVector;
using paddle::framework::vectorize;
using paddle::memory::Alloc;
using paddle::memory::Free;
using paddle::memory::MemoryType;

using paddle::autograd::Function;
using paddle::autograd::AutogradContext;
using paddle::IntArrayRef;

using paddle::autograd::tensor_list;

// Declarations for functions in ext_quantization_cuda_kernel.cu
// Pack and unpack
std::pair<Tensor, Tensor> pack_mixed_precision_cuda(
    Tensor data, Tensor min, Tensor max, Tensor bits, bool stochastic);
Tensor unpack_mixed_precision_cuda(
    Tensor data, Tensor bits, Tensor scale, Tensor min, int64_t N, int64_t num_groups, int64_t group_size);
std::pair<Tensor, Tensor> pack_single_precision_cuda(
    Tensor data, Tensor min, Tensor max, int bits, bool stochastic);
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int64_t num_groups, int64_t group_size);
// Pack/Unpack mixed precision
std::pair<Tensor, Tensor> pack_mixed_precision(Tensor data,
                                               Tensor min,
                                               Tensor max,
                                               Tensor bits,
                                               bool stochastic) {
  CHECK_TENSOR_DIM_FLOAT(data, 3);
  CHECK_TENSOR_DIM_FLOAT(min, 3);
  CHECK_TENSOR_DIM_FLOAT(max, 3);
  CHECK_TENSOR_DIM_TYPE(bits, 1, paddle::DataType::INT32);
  return pack_mixed_precision_cuda(data, min, max, bits, stochastic);
}
Tensor unpack_mixed_precision(Tensor data,
                              Tensor bits,
                              Tensor scale,
                              Tensor min,
                              int64_t N,
                              int64_t num_groups,
                              int64_t group_size) {
  CHECK_TENSOR_DIM_TYPE(data, 1, paddle::DataType::INT32);
  CHECK_TENSOR_DIM_TYPE(bits, 1, paddle::DataType::INT32);
  CHECK_TENSOR_DIM_FLOAT(scale, 3);
  CHECK_TENSOR_DIM_FLOAT(min, 3);
  return unpack_mixed_precision_cuda(data, bits, scale, min,
                                     N, num_groups, group_size);
}
// Pack/Unpack single precision
std::pair<Tensor, Tensor> pack_single_precision(Tensor data,
                                                Tensor min,
                                                Tensor max,
                                                int bits,
                                                bool stochastic) {
  CHECK_TENSOR_DIM_FLOAT(data, 3);
  CHECK_TENSOR_DIM_FLOAT(min, 3);
  CHECK_TENSOR_DIM_FLOAT(max, 3);
  return pack_single_precision_cuda(data, min, max, bits, stochastic);
}
Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor min,
                               int64_t N,
                               int64_t num_groups,
                               int64_t group_size) {
  CHECK_TENSOR_DIM_TYPE(data, 1, paddle::DataType::INT8);
  CHECK_TENSOR_DIM_FLOAT(scale, 3);
  CHECK_TENSOR_DIM_FLOAT(min, 3);

  return unpack_single_precision_cuda(data, bits, scale, min,
                                      N, num_groups, group_size);
}


std::pair<Tensor, Tensor> act_quantized_relu_forward_cuda(Tensor data);
Tensor act_quantized_relu_backward_cuda(Tensor grad_output, Tensor mask);

std::pair<Tensor, Tensor> act_quantized_dropout_forward_cuda(Tensor data, float dropout_p);
Tensor act_quantized_dropout_backward_cuda(Tensor grad_output, Tensor mask, float dropout_p);

std::pair<Tensor, Tensor> act_quantized_max_pool2d_forward_cuda(Tensor input,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);
Tensor act_quantized_max_pool2d_backward_cuda(Tensor grad_output, Tensor max_indices,
        IntArrayRef input_shape,
        IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        bool ceil_mode, bool return_indices);

class ActQuantizedReLU : public Function<ActQuantizedReLU> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_relu_forward_cuda(input);
    ctx->SaveForBackward({mask});
    return output;
  }
  static std::vector<Tensor> backward(AutogradContext *ctx, const std::vector<Tensor>& grad_outputs) {
    auto saved = ctx->GetSavedVariables();
    return {act_quantized_relu_backward_cuda(grad_outputs[0], saved[0])};
  }
};
Tensor act_quantized_relu(Tensor input) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedReLU::Apply(input);
}

class ActQuantizedDropout : public Function<ActQuantizedDropout> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, float dropout_p) {
    Tensor output, mask;
    std::tie(output, mask) = act_quantized_dropout_forward_cuda(input, dropout_p);
    ctx->SaveForBackward({mask});
    ctx->SetSavedData("dropout_p", dropout_p);
    return output;
  }
  static std::vector<Tensor> backward(AutogradContext *ctx, const std::vector<Tensor>& grad_outputs) {
    auto saved = ctx->GetSavedVariables();
    float dropout_p = ctx->GetSavedData<float>("dropout_p");
    return {act_quantized_dropout_backward_cuda(grad_outputs[0], saved[0], dropout_p), Tensor()};
  }
};
Tensor act_quantized_dropout(Tensor input, float dropout_p) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedDropout::Apply(input, dropout_p);
}



// Activation quantized max_pool2d: use compressed bit stream to store activation
class ActQuantizedMaxPool2d : public Function<ActQuantizedMaxPool2d> {
 public:
  static Tensor forward(AutogradContext *ctx, Tensor input, IntArrayRef kernel_size,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
    PADDLE_ENFORCE_EQ(kernel_size.size(), 2, "kernel_size size should be 2");
    PADDLE_ENFORCE_EQ(stride.size(), 2, "stride size should be 2");
    PADDLE_ENFORCE_EQ(padding.size(), 2, "padding size should be 2");
    PADDLE_ENFORCE_EQ(dilation.size(), 2, "dilation size should be 2");
    PADDLE_ENFORCE_EQ(ceil_mode, false, "ceil_mode should be false");
    PADDLE_ENFORCE_EQ(return_indices, false, "return_indices should be false");
    PADDLE_ENFORCE_LT(kernel_size[0] * kernel_size[1], 16, "kernel_size is too large");
    Tensor output, max_indices;
    std::tie(output, max_indices) = act_quantized_max_pool2d_forward_cuda(input, kernel_size, stride, padding,
            dilation, ceil_mode, return_indices);
    ctx->SaveForBackward({max_indices});
    ctx->SetSavedTensor("input_shape", input.sizes());
    ctx->SetSavedTensor("kernel_size", kernel_size);
    ctx->SetSavedTensor("stride", stride);
    ctx->SetSavedTensor("padding", padding);
    ctx->SetSavedTensor("dilation", dilation);
    ctx->SetSavedTensor("ceil_mode", paddle::to_tensor(ceil_mode));
    ctx->SetSavedTensor("return_indices", paddle::to_tensor(return_indices));
    return output;
  }
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    return {act_quantized_max_pool2d_backward_cuda(
                grad_outputs[0], saved[0],
                IntArrayRef(saved[1].data<int>()),
                IntArrayRef(saved[2].data<int>()),
                IntArrayRef(saved[3].data<int>()),
                IntArrayRef(saved[4].data<int>()),
                saved[5].data<bool>()[0], saved[6].data<bool>()[0]),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};
Tensor act_quantized_max_pool2d(Tensor input, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, bool return_indices) {
  CHECK_CUDA_TENSOR_FLOAT(input);
  return ActQuantizedMaxPool2d::apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_mixed_precision", &pack_mixed_precision);
  m.def("unpack_mixed_precision", &unpack_mixed_precision);
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
  m.def("act_quantized_relu", &act_quantized_relu);
  m.def("act_quantized_dropout", &act_quantized_dropout);
  m.def("act_quantized_max_pool2d", &act_quantized_max_pool2d);
}
