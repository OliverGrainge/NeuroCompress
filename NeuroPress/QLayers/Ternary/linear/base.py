import math

import torch
import torch.nn as nn


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
