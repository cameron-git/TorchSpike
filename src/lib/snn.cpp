#include <torch/extension.h>

#include <vector>

// at::Tensor conv2d_if_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor old_voltage)
// {
//     auto currents = torch::conv2d(input, weights);
//     return 5 * input;
// }

// at::Tensor conv2d_if_backward(torch::Tensor input)
// {
//     return torch::ones(5);
// }

at::Tensor if_forward(
    torch::Tensor input,
    torch::Tensor old_voltage,
    float threshold)
{
    auto new_voltage = old_voltage + input;
    auto spikes = (new_voltage >= threshold).to(input);
    new_voltage = (1 - spikes) * new_voltage;
    return spikes, new_voltage;
}

at::Tensor if_backward(torch::Tensor input)
{
    return torch::ones(5);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("if_forward", &if_forward, "IF forward");
    m.def("if_backward", &if_backward, "IF backward");
}