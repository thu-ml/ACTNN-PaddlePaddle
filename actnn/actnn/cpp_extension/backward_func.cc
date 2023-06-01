#include <paddle/extension.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
namespace at {
namespace native {
std::tuple<Tensor, Tensor, Tensor, int64_t, int64_t> prepare_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int64_t normalized_ndim = normalized_shape.size();
  PADDLE_ENFORCE_GE(
      normalized_ndim, 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., "
      "containing at least one element, but got normalized_shape = %s",
      normalized_shape);
  PADDLE_ENFORCE_EQ(
      !weight.defined() || weight.sizes().equals(normalized_shape), true,
      "Expected weight to be of same shape as normalized_shape, but got "
      "weight of shape %s and normalized_shape = %s",
      weight.sizes(),
      normalized_shape);
  PADDLE_ENFORCE_EQ(
      !bias.defined() || bias.sizes().equals(normalized_shape), true,
      "Expected bias to be of same shape as normalized_shape, but got "
      "bias of shape %s and normalized_shape = %s",
      bias.sizes(),
      normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }
  const int axis = input_ndim - normalized_ndim;
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());
  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();
  return std::make_tuple(X, gamma, beta, M, N);
}
}  // namespace native
}  // namespace at
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cudnn_convolution_backward",           &at::cudnn_convolution_backward);
  m.def("cudnn_convolution_transpose_backward", &at::cudnn_convolution_transpose_backward);
  m.def("prepare_layer_norm_inputs",  &at::native::prepare_layer_norm_inputs);
  m.def("layer_norm_cuda",            &at::native::layer_norm_cuda);
  m.def("layer_norm_backward_cuda",   &at::native::layer_norm_backward_cuda);
  m.def("cudnn_batch_norm",           &at::native::cudnn_batch_norm);
  m.def("cudnn_batch_norm_backward",  &at::native::cudnn_batch_norm_backward);
  m.def("native_batch_norm",          &at::native_batch_norm);
  m.def("native_batch_norm_backward", &at::native_batch_norm_backward);
}