import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import ternary_quantize
import NeuroPress.QLayers.Ternary.quant as Q


class BaseTernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearWTA16(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        proj_type="twn"
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.proj_type = proj_type
        self.freeze_state = False

    def freeze(self):
        self.freeze_state = True
        self.q_weights, self.alpha, self.delta = ternary_quantize(self.weight, self.proj_type)

    def unfreeze(self):
        self.freeze_state = False 
        self.q_weights, self.alpha, self.delta = None, None, None

    def forward(self, x):
        if self.freeze_state: 
            q_weights, alpha, delta = self.q_weights, self.alpha, self.delta
        else:
            q_weights, alpha, delta = ternary_quantize(self.weight, self.proj_type)
        out = alpha.view(1, -1) * F.linear(x, q_weights, self.bias / alpha if self.bias is not None else None)
        return out

