import torch
from torch import nn
from cs336_basics.modules.RoPE import RotaryPositionalEmbedding
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.functions import scaled_dot_product_attention

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


