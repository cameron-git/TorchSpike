#include <torch/extension.h>

at::Tensor lif_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau)
{
    v = v + (x - v) / tau;
    x = (v >= th).to(x);
    v = (1 - x) * v;
    return x, v;
}

at::Tensor lif_backward(torch::Tensor x)
{
    return torch::ones(5);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lif_forward", &lif_forward, "LIF Forward");
    m.def("lif_backward", &lif_backward, "LIF Backward");
}