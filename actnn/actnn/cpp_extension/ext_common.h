// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_TYPE(name, n_dim, type)                             \
  PADDLE_ENFORCE_EQ(name.place().is_cuda(), true, paddle::platform::errors::InvalidArgument(#name " must be a CUDA tensor!")); \
  PADDLE_ENFORCE_EQ(name.layout(), paddle::framework::DataLayout::kNCHW, paddle::platform::errors::InvalidArgument(#name " must be contiguous!")); \
  PADDLE_ENFORCE_EQ(name.dims().size(), n_dim, paddle::platform::errors::InvalidArgument("The dimension of " #name " is not correct!")); \
  PADDLE_ENFORCE_EQ(name.type(), type, paddle::platform::errors::InvalidArgument("The type of " #name " is not correct!")); \

// Helper for type check
#define CHECK_CUDA_TENSOR_TYPE(name, type)                                        \
  PADDLE_ENFORCE_EQ(name.place().is_cuda(), true, paddle::platform::errors::InvalidArgument(#name " must be a CUDA tensor!")); \
  PADDLE_ENFORCE_EQ(name.layout(), paddle::framework::DataLayout::kNCHW, paddle::platform::errors::InvalidArgument(#name " must be contiguous!")); \
  PADDLE_ENFORCE_EQ(name.type(), type, paddle::platform::errors::InvalidArgument("The type of " #name " is not correct!")); \

// Helper for type check
#define CHECK_CUDA_TENSOR_FLOAT(name)                                             \
  PADDLE_ENFORCE_EQ(name.place().is_cuda(), true, paddle::platform::errors::InvalidArgument(#name " must be a CUDA tensor!")); \
  PADDLE_ENFORCE_EQ(name.layout(), paddle::framework::DataLayout::kNCHW, paddle::platform::errors::InvalidArgument(#name " must be contiguous!")); \
  PADDLE_ENFORCE_EQ(name.type(), paddle::framework::proto::VarType::FP32 || name.type() == paddle::framework::proto::VarType::FP16, \
              paddle::platform::errors::InvalidArgument("The type of " #name " is not correct!")); \

// Helper for type check
#define CHECK_CUDA_TENSOR_DIM_FLOAT(name, n_dim)                                  \
  PADDLE_ENFORCE_EQ(name.place().is_cuda(), true, paddle::platform::errors::InvalidArgument(#name " must be a CUDA tensor!")); \
  PADDLE_ENFORCE_EQ(name.layout(), paddle::framework::DataLayout::kNCHW, paddle::platform::errors::InvalidArgument(#name " must be contiguous!")); \
  PADDLE_ENFORCE_EQ(name.dims().size(), n_dim, paddle::platform::errors::InvalidArgument("The dimension of " #name " is not correct!")); \
  PADDLE_ENFORCE_EQ(name.type(), paddle::framework::proto::VarType::FP32 || name.type() == paddle::framework::proto::VarType::FP16, \
              paddle::platform::errors::InvalidArgument("The type of " #name " is not correct!")); \



