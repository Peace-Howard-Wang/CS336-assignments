import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None,
                 dtype: torch.dtype | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._reset_parameter()

    def _reset_parameter(self):
        mean = 0
        std = 1
        torch.nn.init.trunc_normal_(self.weights, mean, std, a=-3,b=3)

    def forward(self, tokens_id: torch.Tensor)->torch.Tensor:
        return self.weights[tokens_id]