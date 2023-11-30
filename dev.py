import torch


# LIF, hard reset, sigmoid
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


# IF, soft reset, sigmoid
class IF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th):
        v = v + x
        x = (v >= th).to(x)
        ctx.save_for_backward(v, th)
        v = v - x * th
        return x, v

    @staticmethod
    def backward(ctx, grad_x, grad_v):
        v, th = ctx.saved_tensors
        grad_x = grad_x + grad_v * -th
        sg = torch.sigmoid(v - th)
        grad_v = grad_v + grad_x * sg * (1 - sg)
        grad_x = grad_v
        return grad_x, grad_v, None, None


# IF, hard reset, sigmoid
class IF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th, tau):
        v = v + x
        x = (v >= th).to(x)
        ctx.save_for_backward(x, v, th)
        v = v * (1 - x)
        return x, v

    @staticmethod
    def backward(ctx, grad_x, grad_v):
        x, v, th = ctx.saved_tensors
        grad_x = grad_x + grad_v * -v
        sg = torch.sigmoid(v - th)
        grad_v = grad_v * (1 - x) + grad_x * sg * (1 - sg)
        grad_x = grad_v
        return grad_x, grad_v, None, None
