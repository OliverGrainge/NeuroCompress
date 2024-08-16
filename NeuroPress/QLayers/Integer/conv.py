import math

import torch.nn as nn

from .quant import (
    compute_scale,
    dequantize_per_channel,
    dequantize_per_tensor,
    forward_quantize_per_channel,
    forward_quantize_per_tensor,
    quantize_per_tensor,
)


class BaseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(BaseConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

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


class WeightOnlyQuant(BaseConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits=8,
        type="signed",
    ):
        super(WeightOnlyQuant, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bits = bits
        self.type = type

    def forward(self, x):
        q_weights, scale, zero_point = forward_quantize_per_channel(self.weight, bits=self.bits, type=self.type)
        q_bias = quantize_per_tensor(self.bias, scale.max(), zero_point.min(), bits=self.bits, type=self.type)
        dq_weights = dequantize_per_channel(q_weights, scale, zero_point)
        dq_bias = dequantize_per_tensor(q_bias, scale.max(), zero_point.min())
        out = nn.functional.conv2d(
            x,
            dq_weights,
            dq_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out


class FullQuant(BaseConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        act_bits=8,
        weight_bits=8,
        weight_type="signed",
        act_type="signed",
    ):
        super(FullQuant, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_type = weight_type
        self.act_type = act_type

    def forward(self, x):
        q_weights, scale_w, zero_point_w = forward_quantize_per_tensor(self.weight, bits=self.weight_bits, type=self.weight_type)
        q_x, scale_x, zero_point_x = forward_quantize_per_tensor(x, bits=self.act_bits, type=self.act_type)
        q_bias = (
            quantize_per_tensor(
                self.bias,
                scale_x.max() * scale_w.max(),
                zero_point_w.min(),
                bits=self.weight_bits,
                type=self.weight_type,
            )
            if self.bias is not None
            else None
        )
        out = (
            scale_x
            * scale_w
            * nn.functional.conv2d(
                q_x,
                q_weights,
                q_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        )
        return out


class Conv2dW8A16(WeightOnlyQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW8A16, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            bits=8,
            type="signed",
        )


class Conv2dW4A16(WeightOnlyQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW4A16, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            bits=4,
            type="unsigned",
        )


class Conv2dW2A16(WeightOnlyQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW2A16, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            bits=2,
            type="unsigned",
        )


class Conv2dW8A8(FullQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW8A8, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_bits=8,
            act_bits=8,
            weight_type="signed",
            act_type="signed",
        )


class Conv2dW4A8(FullQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW4A8, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_bits=4,
            act_bits=8,
            weight_type="signed",
            act_type="signed",
        )


class Conv2dW2A8(FullQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dW8A8, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight_bits=2,
            act_bits=8,
            weight_type="signed",
            act_type="signed",
        )
