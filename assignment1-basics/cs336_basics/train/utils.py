import math
import os
import typing

import numpy as np
import torch

"""
    logits: [batch, vocab_size]
    targets:[batch]
"""
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits, _ = logits.max(dim=-1,keepdim=True)
    logits = logits - max_logits
    logsumexp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    gather_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (-gather_logits + logsumexp).mean()

def gradient_clipping(params, max_l2_norm,eps=1e-6):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).pow(2)
    l2_norm = math.sqrt(total_norm)

    if l2_norm > max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(max_l2_norm/(l2_norm +eps))

def get_batch(x: np.ndarray, batch_size:int, context_length:int, device:str="cpu"):
    n = len(x)
    assert n > context_length, f"data too short, data size: {n}, context_length: {context_length}"

    starts = np.random.randint(0, n - context_length, size=batch_size)

    inputs = np.stack([x[start:start+context_length] for start in starts])
    targets = np.stack([x[start + 1:start + context_length + 1] for start in starts])

    inputs_t = torch.LongTensor(inputs.astype(np.int64)).to(device)
    targets_t = torch.LongTensor(targets.astype(np.int64)).to(device)

    return inputs_t, targets_t

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src:str|os.PathLike|typing.BinaryIO|typing.IO[bytes], model: torch.nn.Module, optimizer:torch.optim.Optimizer):
    checkpoint = torch.load(src, map_location="mps")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]