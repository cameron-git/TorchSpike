#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z)
{
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

at::Tensor lif_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau)
{
    auto v_h = v + (x - v) / tau;
    x = (v_h >= th).to(x);
    v = v * (1 - x);
    return x, v, v_h;
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