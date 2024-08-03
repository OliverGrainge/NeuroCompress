import torch 
import torch.nn as nn
import math
from .quant import quantizelinear, quantize_linear_tensor

class BaseIntegerLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BaseIntegerLinear, self).__init__(in_features, out_features, bias)
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


class LinearW8A16(BaseIntegerLinear): 
    def forward(self, x):
        quantized_weights, scale = quantizelinear(self.weight, bits=8)
        dequantized_weights = scale * quantized_weights
        return nn.functional.linear(x, dequantized_weights, self.bias)
    
class LinearW4A16(BaseIntegerLinear): 
    def forward(self, x):
        quantized_weights, scale = quantizelinear(self.weight, bits=4)
        dequantized_weights = scale * quantized_weights
        return nn.functional.linear(x, dequantized_weights, self.bias)

class LinearW2A16(BaseIntegerLinear): 
    def forward(self, x):
        quantized_weights, scale = quantizelinear(self.weight, bits=2)
        dequantized_weights = scale * quantized_weights
        return nn.functional.linear(x, dequantized_weights, self.bias)
    

class LinearW8A8(BaseIntegerLinear):
    def forward(self, x):
        quantized_weights, scale_weights = quantizelinear(self.weight, bits=8)
        quantized_x, scale_x = quantizelinear(x, bits=8)
        quantized_bias = quantize_linear_tensor(self.bias, scale=scale_weights * scale_x, bits=8)
        out = scale_x * scale_weights * nn.functional.linear(quantized_x, quantized_weights, quantized_bias)
        return out
    
class LinearW4A8(BaseIntegerLinear):
    def forward(self, x):
        quantized_weights, scale_weights = quantizelinear(self.weight, bits=8)
        quantized_x, scale_x = quantizelinear(x, bits=4)
        quantized_bias = quantize_linear_tensor(self.bias, scale=scale_weights * scale_x, bits=4)
        out = scale_x * scale_weights * nn.functional.linear(quantized_x, quantized_weights, quantized_bias)
        return out

class LinearW2A8(BaseIntegerLinear):
    def forward(self, x):
        quantized_weights, scale_weights = quantizelinear(self.weight, bits=8)
        quantized_x, scale_x = quantizelinear(x, bits=2)
        quantized_bias = quantize_linear_tensor(self.bias, scale=scale_weights * scale_x, bits=2)
        out = scale_x * scale_weights * nn.functional.linear(quantized_x, quantized_weights, quantized_bias)
        return out






