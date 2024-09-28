import torch
import torch.nn as nn

from NeuroPress.functions.rmsnorm import rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return rmsnorm(self.weight.to(x.dtype), x, self.eps).to(x.dtype)
