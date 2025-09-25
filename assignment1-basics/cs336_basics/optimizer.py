import math
from typing import Union, List

import torch

ParamsT = Union[torch.Tensor, List[torch.Tensor]]
class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                t = state["step"] + 1

                m = beta1*m + (1-beta1)*grad
                v = beta2*v +(1-beta2)*grad**2

                m_hat = m/(1-beta1**t)
                v_hat = v/(1-beta2**t)

                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["step"] = t
        return loss

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t/T_w*alpha_max
    elif T_w <= t <= T_c:
        return alpha_min + (alpha_max - alpha_min)*(1 + math.cos((t-T_w)*math.pi/(T_c -T_w)))/2
    else:
        return alpha_min

