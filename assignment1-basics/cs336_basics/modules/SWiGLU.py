import math
from cs336_basics.modules.linear import Linear
import torch
from torch import nn


def silu(x: torch.Tensor)->torch.Tensor:
    return x*torch.sigmoid(x)


class SWiGLU(nn.Module):
    def __init__(self, d_model, d_ff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        w1x = self.w1(x)
        w3x = self.w3(x)
        gated = silu(w1x)*w3x
        return self.w2(gated)
