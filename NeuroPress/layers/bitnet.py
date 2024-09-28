import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuroPress.functions.bitlinear import bitlinear
from NeuroPress.functions.rmsnorm import rmsnorm
from NeuroPress.layers.base import BaseQuantizedLayer


class BaseBitLinear(BaseQuantizedLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(BaseQuantizedLayer, self).__init__()
        nn.Linear.__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    @staticmethod
    def activation_quant(x, dtype=None):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        if dtype is not None:
            y = y.type(dtype)
        return y

    @staticmethod
    def weight_quant(w):
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u


class BitLinear(BaseBitLinear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(BaseQuantizedLayer, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.freeze_state = False

    def train_forward(self, x):
        if self.weight is None:
            raise RuntimeError("Weights are not initialized for training.")
        w = self.weight
        x_norm = self.rmsnorm(x)
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def infer_forward(self, x):
        x = rmsnorm(self.rmsnorm.weight, x, self.rmsnorm.eps)
        scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_quant = (x * scale_x).round().clamp_(-128, 127).type(torch.int8)
        y = bitlinear(x_quant, self.packed_weights)
        y = y / scale_x / self.weight_scale
        return y

    def forward(self, x):
        if self.freeze_state:
            return self.infer_forward(x)
        else:
            return self.train_forward(x)
