import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert d_k % 2 == 0
        self.d_k = d_k
        # d/2
        inv_freq = 1 / (theta**(torch.arange(0,d_k,2).float()/d_k))
        # max_seq_len
        positions = torch.arange(0, max_seq_len).float()
        # postions[:,None] （max_seq_len, 1)
        # inv_freq[None,:] (1,d/2)
        # *发生广播，得到了(max_seq_len, d/2) * (max_seq_len, d/2)
        angles = positions[:,None] * inv_freq[None,:]
        if device is not None:
            angles = angles.to(device)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor|None = None)->torch.Tensor:
        # x.shape = (B,S,D) or (B,S,H,head_dim)
        if token_positions is None:
            B, S = x.shape[:2]
            token_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)

        if x.size(-1) != self.d_k:
            raise  RuntimeError(f"last dim {x.size(-1)} != rotary dim {self.d_k}")

        if x.ndim == 3:
            x_even = x[...,0::2]
            x_odd = x[...,1::2]
            cos = self.cos[token_positions]
            sin = self.sin[token_positions]
            x_rot_even = x_even*cos - x_odd*sin
            x_rot_odd = x_even*sin + x_odd*cos
            x_rot = torch.stack((x_rot_even,x_rot_odd),dim=-1).view_as(x)
            return x_rot
        elif x.ndim == 4:
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            cos = self.cos[token_positions[..., None]]
            sin = self.sin[token_positions[...,None]]
            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd = x_even * sin + x_odd * cos
            x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).view_as(x)
            return x_rot
        else:
            raise ValueError(f"Unsupported input dims for RoPE with dim == {x.ndim}")