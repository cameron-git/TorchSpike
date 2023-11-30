#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z)
{
    return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z)
{
    const auto s = sigmoid(z);
    return (1.0 - s) * s;
}

template <typename scalar_t>
__global__ void lif_cuda_forward_kernel(
    scalar_t *__restrict__ x,
    scalar_t *__restrict__ v,
    scalar_t *__restrict__ vh,
    size_t neuron_size,
    float th,
    float tau)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * neuron_size + column;
    if (column < neuron_size)
    {
        vh[index] = v[index] + (x[index] - v[index]) / tau;
        x[index] = vh[index] >= th;
        v[index] = vh[index] * (1.0 - x[index]);
    }
}
std::vector<at::Tensor> lif_cuda_forward(
    torch::Tensor x,
    torch::Tensor v,
    float th,
    float tau)
{
    const auto batch_size = x.size(0);
    const auto neuron_size = x.size(1);

    auto vh = torch::zeros_like(v);

    const int threads = 1024;
    // threads - 1 is to make sure we always have a fraction of a block more than needed
    const dim3 blocks((neuron_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(
        x.type(),
        "lif_forward_cuda",
        ([&]
         { lif_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
               x.data<scalar_t>(),
               v.data<scalar_t>(),
               vh.data<scalar_t>(),
               neuron_size,
               th,
               tau); }));
    return {x, v, vh};
}

template <typename scalar_t>
__global__ void lif_cuda_backward_kernel(
    scalar_t *__restrict__ grad_x,
    scalar_t *__restrict__ grad_v,
    scalar_t *__restrict__ x,
    scalar_t *__restrict__ vh,
    size_t neuron_size,
    float th,
    float tau)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * neuron_size + column;
    if (column < neuron_size)
    {
        grad_x[index] = grad_x[index] + grad_v[index] * -vh[index];
        grad_v[index] = grad_v[index] * (1 - x[index]) + grad_x[index] * d_sigmoid(vh[index] - th);
        grad_x[index] = grad_v[index] * (1 / tau);
        grad_v[index] = grad_v[index] * (1 - 1 / tau);
    }
}

std::vector<at::Tensor> lif_cuda_backward(
    torch::Tensor grad_x,
    torch::Tensor grad_v,
    torch::Tensor x,
    torch::Tensor vh,
    float th,
    float tau)
{
    const auto batch_size = x.size(0);
    const auto neuron_size = x.size(1);

    const int threads = 1024;
    // threads - 1 is to make sure we always have a fraction of a block more than needed
    const dim3 blocks((neuron_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(
        x.type(),
        "lif_backward_cuda",
        ([&]
         { lif_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
               grad_x.data<scalar_t>(),
               grad_v.data<scalar_t>(),
               x.data<scalar_t>(),
               vh.data<scalar_t>(),
               neuron_size,
               th,
               tau); }));

    return {grad_x, grad_v};
}
