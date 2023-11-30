import torch

# %%
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


# %%
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


# %%
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


# %%
# Experimental IF using like conv1d (not working)
# Instead of having surrogate gradient for heaviside, surrogate for floor is used
class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.floor()

    @staticmethod
    def backward(ctx, grad_x):
        x = ctx.saved_tensors[0]
        x = x % 1
        sg0 = torch.sigmoid(10 * x)
        sg1 = torch.sigmoid(10 * (x - 1))
        grad_x = grad_x * (sg0 * (1 - sg0) + sg1 * (1 - sg1))
        return grad_x


class IF(torch.nn.Module):
    def __init__(self, th=1, tau=2):
        super().__init__()
        self.v = None
        self.th = th
        self.tau = tau

    def forward(self, x):
        self.v = x.cumsum(0).relu()
        x = torch.nn.functional.pad(
            Floor.apply(self.v / self.th).to(x), (0, 0, 0, 0, 1, 0)
        )
        x = x[1:] - x[:-1]
        self.v = self.v - x * self.th
        return x

    # %%
