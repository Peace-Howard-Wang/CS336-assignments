import math

import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        result = torch.einsum("...d,d->...d", x,self.weights)/rms_a
        return result.to(in_dtype)