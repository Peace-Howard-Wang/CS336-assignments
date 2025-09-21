import math
from torch import einsum
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None,
                 dtype: torch.dtype | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._reset_parameter()

    def _reset_parameter(self):
        mean = 0
        std = math.sqrt(2/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weights, mean, std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum("oi,bsi->bso", self.weights, x)
