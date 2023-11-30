#include <torch/extension.h>

#include <vector>

torch::Tensor d_sigmoid(torch::Tensor z)
{
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

std::vector<at::Tensor> lif_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau)
{
    auto vh = v + (x - v) / tau;
    x = (vh >= th).to(x);
    v = vh * (1 - x);
    return {x, v, vh};
}

std::vector<at::Tensor> lif_backward(
    torch::Tensor grad_x,
    torch::Tensor grad_v,
    torch::Tensor x,
    torch::Tensor vh,
    float th,
    float tau)
{
    grad_x = grad_x + grad_v * -vh;
    grad_v = grad_v * (1 - x) + grad_x * d_sigmoid(vh - th);
    grad_x = grad_v * (1 / tau);
    grad_v = grad_v * (1 - 1 / tau);
    return {grad_x, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lif_forward", &lif_forward, "LIF forward");
    m.def("lif_backward", &lif_backward, "LIF backward");
}