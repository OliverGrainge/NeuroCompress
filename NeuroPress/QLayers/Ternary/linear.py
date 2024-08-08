import math

import torch
import torch.nn as nn

from .quant import forward_quantize, setup_quantize


class BaseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, projection_func="TWN"):
        super(BaseLinear, self).__init__(in_features, out_features, bias)
        self.projection_func = projection_func

    def forward(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

    def init(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


# difference between direct and indirect is direct has a non-differentiable projection function
# indirect has a differentiable method that approximates ternary weights


class LinearWTA16(BaseLinear):
    def __init__(self, in_features, out_features, bias=True, method: str = "twn"):
        super(LinearWTA16, self).__init__(in_features, out_features, bias=bias)
        self.method = method
        self.quant_params = setup_quantize(method)

    def forward(self, x):
        q_weight, scale, alpha = forward_quantize(
            self.weight, self.method, self.quant_params
        )
        out = alpha.view(1, -1).repeat(x.shape[0], 1) * nn.functional.linear(
            x, q_weight
        )
        if self.bias is not None:
            out += self.bias
        return out
