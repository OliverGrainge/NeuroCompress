import math

import torch.nn as nn

from .quant import (
    compute_scale,
    dequantize_per_tensor,
    forward_quantize_per_tensor,
    quantize_per_tensor,
)


class BaseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BaseLinear, self).__init__(in_features, out_features, bias)

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


class WeightOnlyQuant(BaseLinear):
    def __init__(self, in_features, out_features, bias=True, bits=8, type="signed"):
        super(WeightOnlyQuant, self).__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.type = type

    def forward(self, x):
        q_weights, scale, zero_point = forward_quantize_per_tensor(self.weight, bits=self.bits, type=self.type)
        q_bias = quantize_per_tensor(self.bias, scale, zero_point, bits=self.bits, type=self.type)
        dq_weights = dequantize_per_tensor(q_weights, scale, zero_point)
        dq_bias = dequantize_per_tensor(q_bias, scale, zero_point)
        out = nn.functional.linear(x, dq_weights, dq_bias)

        return out


class FullQuant(BaseLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_bits=8,
        weight_bits=8,
        weight_type="signed",
        act_type="signed",
    ):
        super(FullQuant, self).__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_type = weight_type
        self.act_type = act_type

    def forward(self, x):
        q_weights, scale_w, zero_point_w = forward_quantize_per_tensor(self.weight, bits=self.weight_bits, type=self.weight_type)
        q_x, scale_x, zero_point_x = forward_quantize_per_tensor(x, bits=self.act_bits, type=self.act_type)
        q_bias = quantize_per_tensor(
            self.bias,
            scale_x * scale_w,
            0.0,
            bits=self.weight_bits,
            type=self.weight_type,
        )
        out = scale_x * scale_w * nn.functional.linear(q_x, q_weights, q_bias)
        if self.weight_type == "unsigned":
            out -= (q_x.sum(1) * scale_x * scale_w * zero_point_w).view(-1, 1)
        if self.act_type == "unsigned":
            out -= scale_x * scale_w * zero_point_x * q_weights
            out += scale_x * scale_w * zero_point_w * zero_point_x
        return out


class LinearW8A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A16, self).__init__(in_features, out_features, bias=bias, bits=8, type="signed")


class LinearW4A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW4A16, self).__init__(in_features, out_features, bias=bias, bits=4, type="unsigned")


class LinearW2A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW2A16, self).__init__(in_features, out_features, bias=bias, bits=2, type="unsigned")


class LinearW8A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=8,
            act_type="signed",
            weight_type="signed",
        )


class LinearW4A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW4A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=4,
            act_type="signed",
            weight_type="unsigned",
        )


class LinearW2A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW2A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=2,
            act_type="signed",
            weight_type="unsigned",
        )
