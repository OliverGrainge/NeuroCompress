import torch
import torch.nn as nn
from NeuroPress.QLayers.Ternary.triton_kernels.rmsnorm_kernel import fast_rms_layernorm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSNorm(torch.nn.Module):
    def __init__(self, shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return fast_rms_layernorm(self.weight.to(x.dtype), x, self.eps).to(x.dtype)

