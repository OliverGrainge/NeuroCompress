import torch 
import torch.nn as nn
import math
from .old_quant import quantizelinear, quantize_linear_tensor, compute_linear_symmetric_scale, dequantize_linear_tensor
from collections import OrderedDict

class BaseIntegerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BaseIntegerLinear, self).__init__(in_features, out_features, bias)
        self.act_scale = None
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
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()



class SignedWeightOnlyIntegerLinear(BaseIntegerLinear):
    def __init__(self, in_features, out_features, bias=True, bits=8, weight_dtype=torch.qint8):
        super(SignedWeightOnlyIntegerLinear, self).__init__(in_features, out_features, bias=bias)
        self.bits = bits
        self.weight_dtype = weight_dtype

    def forward(self, x):
        q_weights, scale = quantizelinear(self.weight, bits=self.bits)
        q_bias = quantize_linear_tensor(self.bias, scale=scale, bits=32)
        dq_weights = dequantize_linear_tensor(q_weights, scale)
        dq_bias = dequantize_linear_tensor(q_bias, scale)
        return nn.functional.linear(x, dq_weights, dq_bias)
    
    @torch.no_grad()
    def quantized_forward(self, x):
        return nn.functional.linear(x, self.weight.dequantize(), self.bias.dequantize() if self.bias is not None else None)

    def quantize(self):
        scale = compute_linear_symmetric_scale(self.weight, bits=self.bits)
        q_weight = torch.quantize_per_tensor(self.weight.data, scale.item(), 0, self.weight_dtype)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        if self.bias is not None: 
            q_bias = torch.quantize_per_tensor(self.bias.data, scale.item(), 0, torch.qint32)
            self.bias = torch.nn.Parameter(q_bias, requires_grad=False)


class SignedIntegerLinear(BaseIntegerLinear):
    def __init__(self, in_features, out_features, bias=True, act_bits=8, weight_bits=8, weight_dtype=torch.qint8, act_dtype=torch.qint8):
        super(SignedIntegerLinear, self).__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.act_dtype = act_dtype
        self.act_scale = nn.Parameter(torch.zeros([0]))

    def forward(self, x):
        q_weights, scale_weights = quantizelinear(self.weight, bits=self.weight_bits)
        q_x, scale_x = quantizelinear(x, bits=self.act_bits)
        q_bias = quantize_linear_tensor(self.bias, scale=scale_weights * scale_x, bits=32)
        out = scale_x * scale_weights * nn.functional.linear(q_x, q_weights, q_bias)
        return out
    
    @torch.no_grad()
    def quantized_forward(self, x):
        act_scale = compute_linear_symmetric_scale(x, bits=self.act_bits)
        x = torch.quantize_per_tensor(x, act_scale, 0, dtype=self.act_dtype)
        # ideally 
        # return nnq.functional.linear(x, self.weight, self.bias) with a scale of (x.scale * self.weight.scale)
        # as all are quantized, however this kernel is not supported on cuda, M1 or intel Chips. Only arm
        # We therefore compute this operation in fp32 
        # however since this kernel does not exists we will compute in floating point
        return nn.functional.linear(x.dequantize(), self.weight.dequantize(), self.bias.dequantize() if self.bias is not None else None)

    def quantize(self):
        scale = compute_linear_symmetric_scale(self.weight, bits=self.weight_bits)
        q_weight = torch.quantize_per_tensor(self.weight.data, scale.item(), 0, self.weight_dtype)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        if self.bias is not None: 
            q_bias = torch.quantize_per_tensor(self.bias.data, scale.item(), 0, torch.qint32)
            self.bias = torch.nn.Parameter(q_bias, requires_grad=False)



class UnsignedIntegerLinear(BaseIntegerLinear):
    def __init__(self, in_features, out_features, bias=True, act_bits=8, weight_bits=8, weight_dtype=torch.qint8, act_dtype=torch.qint8):
        super(SignedIntegerLinear, self).__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.act_dtype = act_dtype
        self.act_scale = nn.Parameter(torch.zeros([0]))

    def forward(self, x):
        q_weights, scale_weights = quantizelinear(self.weight, bits=self.weight_bits)
        q_x, scale_x = quantizelinear(x, bits=self.act_bits)
        q_bias = quantize_linear_tensor(self.bias, scale=scale_weights * scale_x, bits=32)
        out = scale_x * scale_weights * nn.functional.linear(q_x, q_weights, q_bias)
        return out
    
    @torch.no_grad()
    def quantized_forward(self, x):
        act_scale = compute_linear_symmetric_scale(x, bits=self.act_bits)
        x = torch.quantize_per_tensor(x, act_scale, 0, dtype=self.act_dtype)
        # ideally 
        # return nnq.functional.linear(x, self.weight, self.bias) with a scale of (x.scale * self.weight.scale)
        # as all are quantized, however this kernel is not supported on cuda, M1 or intel Chips. Only arm
        # We therefore compute this operation in fp32 
        # however since this kernel does not exists we will compute in floating point
        return nn.functional.linear(x.dequantize(), self.weight.dequantize(), self.bias.dequantize() if self.bias is not None else None)

    def quantize(self):
        scale = compute_linear_symmetric_scale(self.weight, bits=self.weight_bits)
        q_weight = torch.quantize_per_tensor(self.weight.data, scale.item(), 0, self.weight_dtype)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        if self.bias is not None: 
            q_bias = torch.quantize_per_tensor(self.bias.data, scale.item(), 0, torch.qint32)
            self.bias = torch.nn.Parameter(q_bias, requires_grad=False)


class LinearW8A16(SignedWeightOnlyIntegerLinear): 
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A16, self).__init__(
            in_features,
            out_features,
            bias=bias,
            bits=8,
            weight_dtype=torch.qint8)


    
class LinearW4A16(BaseIntegerLinear): 
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A16, self).__init__(
            in_features,
            out_features,
            bias=bias,
            bits=8,
            weight_dtype=torch.quint4x2)



class LinearW8A8(SignedIntegerLinear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=8,
            weight_dtype=torch.qint8, 
            act_dtype=torch.qint8)


    
class LinearW4A8(SignedIntegerLinear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearW8A8, self).__init__(
            in_features,
            out_features,
            bias=bias,
            act_bits=8,
            weight_bits=8,
            weight_dtype=torch.qint8, 
            act_dtype=torch.qint8)





