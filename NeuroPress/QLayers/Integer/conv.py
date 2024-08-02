import torch
import torch.nn as nn
import math
from .quant import quantizeconv2d, quantizelinear

class BaseIntegerConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BaseIntegerConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.init()

    def forward(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

    def init(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def setup(self, conv_layer: nn.Conv2d):
        self.weight.data = conv_layer.weight.data.detach()
        if conv_layer.bias is not None:
            self.bias.data = conv_layer.bias.data.detach()

class Conv2dW8A16(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_weights, scale = quantizeconv2d(self.weight, bits=8)
        dequantized_weights = quantized_weights * scale
        return nn.functional.conv2d(x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dW4A16(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_weights, scale = quantizeconv2d(self.weight, bits=4)
        dequantized_weights = quantized_weights * scale
        return nn.functional.conv2d(x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dW2A16(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_weights, scale = quantizeconv2d(self.weight, bits=2)
        dequantized_weights = quantized_weights * scale
        return nn.functional.conv2d(x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dW8A8(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_x, scale_x = quantizelinear(x, bits=8)
        quantized_weights, scale_w = quantizeconv2d(self.weight, bits=8)
        dequantized_weights = quantized_weights * scale_w
        dequantized_x = quantized_x * scale_x
        return nn.functional.conv2d(dequantized_x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dW4A8(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_x, scale_x = quantizelinear(x, bits=8)
        quantized_weights, scale_w = quantizeconv2d(self.weight, bits=4)
        dequantized_weights = quantized_weights * scale_w
        dequantized_x = quantized_x * scale_x
        return nn.functional.conv2d(dequantized_x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dW2A8(BaseIntegerConv2d): 
    def forward(self, x):
        quantized_x, scale_x = quantizelinear(x, bits=8)
        quantized_weights, scale_w = quantizeconv2d(self.weight, bits=2)
        dequantized_weights = quantized_weights * scale_w
        dequantized_x = quantized_x * scale_x
        return nn.functional.conv2d(dequantized_x, dequantized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)