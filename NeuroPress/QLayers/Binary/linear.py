import torch
import torch.nn as nn
import math
from .quant import SignBinarizeFunction, StochasticBinarySignFunction


class BaseBinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BaseBinaryLinear, self).__init__(in_features, out_features, bias)
        self.init()

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
        if self.bias and linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearW1A16(BaseBinaryLinear):
    def forward(self, x):
        binary_weights = SignBinarizeFunction.apply(self.weight)
        return nn.functional.linear(x, binary_weights, self.bias)


class StochasticLinearW1A16(BaseBinaryLinear):
    def forward(self, x):
        stochastic_binary_weights = StochasticBinarySignFunction.apply(self.weight)
        return nn.functional.linear(x, stochastic_binary_weights, self.bias)


class LinearW1A1(BaseBinaryLinear):
    def forward(self, x):
        binary_weights = SignBinarizeFunction.apply(self.weight)
        binary_inputs = SignBinarizeFunction.apply(x)
        out = nn.functional.linear(binary_inputs, binary_weights)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out


class StochasticLinearW1A1(BaseBinaryLinear):
    def forward(self, x):
        binary_weights = StochasticBinarySignFunction.apply(self.weight)
        binary_inputs = StochasticBinarySignFunction.apply(x)
        out = nn.functional.linear(binary_inputs, binary_weights)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
