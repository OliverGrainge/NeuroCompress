import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Binary.quant as Q


def get_binarize(projection="deterministic", backward="ste"):
    if projection == "deterministic" and backward == "clipped_ste":
        return Q.binarize_deterministic_clipped_ste
    elif projection == "stochastic" and backward == "clipeed_ste":
        return Q.binarize_stochastic_clipped_ste
    elif projection == "xnor":
        return Q.binarize_spatialconv_clipped_ste
    else:
        raise NotImplementedError


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
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def setup(self, conv_layer: nn.Conv2d):
        self.weight.data = conv_layer.weight.data.detach()
        if conv_layer.bias is not None:
            self.bias.data = conv_layer.bias.data.detach()


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
        padding_mode="zeros",
        projection="deterministic",
        backward="clipped_ste",
        per_channel=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.binarize = get_binarize(projection=projection, backward=backward)
        self.per_channel = per_channel

    def forward(self, x):
        qw, alpha = self.binarize.apply(self.weight)
        return alpha.view(1, len(alpha), 1, 1) * F.conv2d(
            x,
            qw,
            self.bias / alpha,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
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
        padding_mode="zeros",
        projection="deterministic",
        backward="clipped_ste",
        per_channel=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.binarize_weight = get_binarize(projection=projection, backward=backward)
        self.binarize_activation = get_binarize(projection="xnor")
        self.per_channel = per_channel

    def forward(self, x):
        qw, alpha_w = self.binarize_weight.apply(self.weight)
        qx, alpha_x = self.binarize_activation.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        out = (
            alpha_w.view(1, len(alpha_w), 1, 1)
            * alpha_x
            * F.conv2d(
                qx,
                qw,
                None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        )

        #
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out
