import math
from torch import einsum
import torch
from torch import nn

from cs336_basics.utils import silu, softmax


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
        device = x.device  # x 是输入
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
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

def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.size(-1)
    scores = torch.einsum("b...nd,b...md->b...nm", Q, K)/(d_k**0.5)
    scores = scores.masked_fill(~mask, float('-inf'))
    atte = softmax(scores, -1)
    return atte@V #[b...nm] @ [b...mv] b...nv

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,max_seq_len:int|None=None,
    theta:float|None = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        assert d_model%num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.max_seq_len = None
        self.theta = None
        if max_seq_len:
            self.max_seq_len=max_seq_len
        if theta:
            self.theta = theta
        self.wq = Linear(d_model, self.d_model)
        self.wk = Linear(d_model, self.d_model)
        self.wv = Linear(d_model, self.d_model)
        self.wo = Linear(self.d_model, d_model)

    # x: [...,S,D]
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor|None=None)->torch.Tensor:
        B,S,D = x.shape
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        Q =Q.view(B,S,self.num_heads,self.d_k)
        K =K.view(B,S,self.num_heads,self.d_k)
        V =V.view(B,S,self.num_heads,self.d_k)
        if self.theta is not None and self.max_seq_len is not None:
            rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        mask = torch.tril(torch.ones(S,S,device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        out_heads = scaled_dot_product_attention(Q,K,V,mask)
        out_cont= out_heads.transpose(1,2).contiguous()
        out = out_cont.view(B,S,D)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len,theta)
        self.ffn = SWiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    def forward(self, x, token_positions=None):
        x_norm = self.norm1(x)
        y = x + self.attention(x_norm, token_positions)
        y_norm = self.norm2(y)
        z = self.ffn(y_norm)
        return y + z

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, vocab_size, context_length, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transfomer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_embedding(x)
        for layer in self.transfomer_blocks:
            h = layer(h)
        h = self.norm(h)
        h = self.output_embedding(h)
        return h