import math

import torch 
import torch.nn as nn

from .quant import linear_quantize
from .quantize import quantize, dequantize


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
    def __init__(self, in_features, out_features, bias=True, proj_type="minmax", bits=8, per_channel=True, symmetric=True):
        super(WeightOnlyQuant, self).__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.proj_type = proj_type 
        self.per_channel = per_channel 
        self.symmetric = symmetric

        self.freeze_state = False
    
    def freeze(self):
        self.freeze_state = True 
        self.q_weights, self.scale, self.zero_point = linear_quantize(self.weight, proj_type=self.proj_type, bits=self.bits, per_channel=self.per_channel, symmetric=self.symmetric)

    def unfreeze(self):
        self.freeze_state = None 
        self.q_weights, self.scale, self.zero_point = None, None, None

    def forward(self, x):
        if self.freeze_state: 
            q_weights, scale, zero_point = self.q_weights, self.scale, self.zero_point
        else: 
            q_weights, scale, zero_point = linear_quantize(self.weight, proj_type=self.proj_type, bits=self.bits, per_channel=self.per_channel, symmetric=self.symmetric)
        q_bias = quantize(self.bias, scale, zero_point)
        dq_weights = dequantize(q_weights, scale, zero_point)
        dq_bias = dequantize(q_bias, scale, zero_point)
        out = nn.functional.linear(x, dq_weights.to(x.device), dq_bias.to(x.device))
        return out


class FullQuant(BaseLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_bits=8,
        weight_bits=8,
        proj_type="minmax",
        weight_per_channel=True,
        act_per_channel=False,
        symmetric=True,
    ):
        super(FullQuant, self).__init__(in_features, out_features, bias=bias)
        self.proj_type = proj_type 
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_per_channel = weight_per_channel 
        self.act_per_channel = act_per_channel 
        self.symmetric = symmetric 

        self.freeze_state = False
    
    def freeze(self):
        self.freeze_state = True 
        self.q_weights, self.scale, self.zero_point = linear_quantize(self.weight, proj_type=self.proj_type, bits=self.weight_bits, per_channel=self.weight_per_channel, symmetric=self.symmetric)

    def unfreeze(self):
        self.freeze_state = None 
        self.q_weights, self.scale, self.zero_point = None, None, None


    def forward(self, x):
        if self.freeze_state == True: 
            q_weights, scale_w, zero_point_w = self.q_weights, self.scale, self.zero_point
        else: 
            q_weights, scale_w, zero_point_w = linear_quantize(self.weight, bits=self.weight_bits, proj_type=self.proj_type, per_channel=self.weight_per_channel, symmetric=self.symmetric)
        q_x, scale_x, zero_point_x = linear_quantize(x, bits=self.act_bits, proj_type="minmax", symmetric=self.symmetric, per_channel=self.act_per_channel)
        q_bias = quantize(
            self.bias,
            scale_x * scale_w,
            torch.zeros_like(scale_x * scale_w)
        )
        out = scale_x * scale_w * nn.functional.linear(q_x, q_weights, q_bias)
        if self.symmetric == False:
            out -= (q_x.sum(1) * scale_x * scale_w * zero_point_w).view(-1, 1)
        if self.symmetric == False:
            out -= scale_x * scale_w * zero_point_x * q_weights
            out += scale_x * scale_w * zero_point_w * zero_point_x
        return out


class LinearW8A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A16, self).__init__(in_features, out_features, bias=bias, proj_type="minmax", bits=8, per_channel=True, symmetric=True)


class LinearW4A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW4A16, self).__init__(in_features, out_features, bias=bias, proj_type="minmax", bits=4, per_channel=True, symmetric=True)


class LinearW2A16(WeightOnlyQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW2A16, self).__init__(in_features, out_features, bias=bias, proj_type="minmax", bits=2, per_channel=True, symmetric=True)


class LinearW8A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=8,
            proj_type="kldiv",
            weight_per_channel=True, 
            act_per_channel=False,
            symmetric=True,
        )


class LinearW4A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW4A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=4,
            proj_type="kldiv",
            weight_per_channel=True, 
            act_per_channel=False,
            symmetric=True,
        )


class LinearW2A8(FullQuant):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW2A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=2,
            proj_type="kldiv",
            weight_per_channel=True, 
            act_per_channel=False,
            symmetric=True,
        )
