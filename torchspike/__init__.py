import torch
import torchspike_cpu as snn_cpu
import torchspike_cuda as snn_cuda

class LIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th, tau):
        v = v + (x - v) / tau
        x = (v >= th).to(x)
        ctx.save_for_backward(x, v, th, tau)
        v = v * (1 - x)
        return x, v

    @staticmethod
    def backward(ctx, grad_x, grad_v):
        x, v, th, tau = ctx.saved_tensors
        grad_x = grad_x + grad_v * -v
        sg = torch.sigmoid(v - th)
        grad_v = grad_v * (1 - x) + grad_x * sg * (1 - sg)
        grad_x = grad_v * (1 / tau)
        grad_v = grad_v * (1 - 1 / tau)
        return grad_x, grad_v, None, None

class LIF_CPP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th, tau):
        x, v, vh = snn_cpu.lif_forward(x, v, th, tau)
        ctx.save_for_backward(x, vh, th, tau)
        return x, v

    @staticmethod
    def backward(ctx, grad_x, grad_v):
        x, vh, th, tau = ctx.saved_tensors
        grad_x, grad_v = snn_cpu.lif_backward(grad_x, grad_v, x, vh, th, tau)
        return grad_x, grad_v, None, None

class LIF_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th, tau):
        x, v, vh = snn_cuda.lif_forward(x, v, th, tau)
        ctx.save_for_backward(x, vh, th, tau)
        return x, v

    @staticmethod
    def backward(ctx, grad_x, grad_v):
        x, vh, th, tau = ctx.saved_tensors
        grad_x, grad_v = snn_cuda.lif_backward(grad_x, grad_v, x, vh, th, tau)
        return grad_x, grad_v, None, None