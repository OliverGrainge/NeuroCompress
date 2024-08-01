import torch
import torch.nn as nn
import math
from NeuroPress.QLayers.utils import SignBinarizeFunction, StochasticBinarySignFunction


class BaseBinaryConv2d(nn.Conv2d):
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
        binarize_function=None,
    ):
        super(BaseBinaryConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.binarize_function = binarize_function
        self.init()  # Initialize weights and biases when the layer is created

    def forward(self, x):
        binary_weights = self.binarize_function.apply(self.weight)
        return nn.functional.conv2d(
            x,
            binary_weights,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def init(self):
        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )  # He initialization for weights
        if self.bias is not None:
            # Initialize bias uniformly within an interval that depends on fan-in
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def setup(self, conv_layer: nn.Conv2d):
        # Copy weights and biases from another Conv2d layer
        self.weight.data = conv_layer.weight.data.clone()
        if self.bias is not None and conv_layer.bias is not None:
            self.bias.data = conv_layer.bias.data.clone()


class Conv2dW1A16(BaseBinaryConv2d):
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
        super(Conv2dW1A16, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            SignBinarizeFunction,
        )


class StochastiConv2dW1A16(BaseBinaryConv2d):
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
        super(StochastiConv2dW1A16, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            StochasticBinarySignFunction,
        )


class Conv2dW1A1(BaseBinaryConv2d):
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
        # Use super() without specifying the parent class explicitly
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.binarize_function = SignBinarizeFunction

    def forward(self, x):
        binary_weights = self.binarize_function.apply(self.weight)
        binary_inputs = self.binarize_function.apply(x)
        out = nn.functional.conv2d(
            binary_inputs,
            binary_weights,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class StochastiConv2dW1A1(BaseBinaryConv2d):
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
        # Use super() without specifying the parent class explicitly
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.binarize_function = StochasticBinarySignFunction

    def forward(self, x):
        binary_weights = self.binarize_function.apply(self.weight)
        binary_inputs = self.binarize_function.apply(x)
        out = nn.functional.conv2d(
            binary_inputs,
            binary_weights,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out
