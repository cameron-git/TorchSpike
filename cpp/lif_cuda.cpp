#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> lif_cuda_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau);

std::vector<at::Tensor> lif_cuda_backward(
    torch::Tensor grad_x,
    torch::Tensor grad_v,
    torch::Tensor x,
    torch::Tensor vh,
    float th,
    float tau);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lif_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau)
{
    CHECK_INPUT(x);
    CHECK_INPUT(v);
    return lif_cuda_forward(x, v, th, tau);
}

std::vector<at::Tensor> lif_backward(
    torch::Tensor grad_x,
    torch::Tensor grad_v,
    torch::Tensor x,
    torch::Tensor vh,
    float th,
    float tau)
{
    CHECK_INPUT(grad_x);
    CHECK_INPUT(grad_v);
    CHECK_INPUT(x);
    CHECK_INPUT(vh);
    return lif_cuda_backward(grad_x, grad_v, x, vh, th, tau);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lif_forward", &lif_forward, "LIF forward (CUDA)");
  m.def("lif_backward", &lif_backward, "LIF backward (CUDA)");
}