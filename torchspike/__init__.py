import torch
import torchspike_cpp


class LIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, th, tau):
        x, v = torchspike_cpp.lif_forward(x, v, th, tau)
        ctx.save_for_backward(x, v, th, tau)
        return x

class LIF(torch.nn.Module):
    def __init__(
        self,
        th,
        tau,
    ):
        super().__init__()
        self.th = th
        self.tau = tau
        self.v = torch.zeros(1)

    def forward(self, x):
        x, v = torchspike_cpp.lif_forward(x, self.v, self.th, self.tau)
        self.v = v
        return x
    
    # def backwards
