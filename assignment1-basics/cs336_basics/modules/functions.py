import torch

def softmax(x: torch.Tensor, dim=-1)->torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e/torch.sum(e, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.size(-1)
    scores = torch.einsum("b...nd,b...md->b...nm", Q, K)/(d_k**0.5)
    scores = scores.masked_fill(~mask, float('-inf'))
    atte = softmax(scores, -1)
    return atte@V #[b...nm] @ [b...mv] b...nv
