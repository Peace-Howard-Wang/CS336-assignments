from torch import nn

from cs336_basics.modules.RMSNorm import RMSNorm
from cs336_basics.modules.SWiGLU import SWiGLU
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.multihead_self_attention import MultiheadSelfAttention


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