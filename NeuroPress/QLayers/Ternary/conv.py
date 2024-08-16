import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Ternary.quant as Q


class BaseTernaryConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class Conv2dWTA16_TWN(BaseTernaryConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.ternarize = Q.ternarize_twn

    def forward(self, x):
        qw, alpha, delta = self.ternarize.apply(self.weight)
        out = alpha.view(1, -1, 1, 1) * F.conv2d(
            x,
            qw,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out


class Conv2dWTA16_TTN(BaseTernaryConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.ternarize = Q.ternarize_ttn
        self.wp = nn.Parameter(self._init_scales(self.weight, pos=True, t=0.05))
        self.wn = nn.Parameter(self._init_scales(self.weight, pos=False, t=0.05))

    @staticmethod
    def _init_scales(tensor: torch.tensor, pos: bool, t: float):
        delta = torch.tensor([t]).to(tensor.device) * tensor.view(tensor.shape[0], -1).abs().mean(dim=1)
        if pos:
            mask = tensor > delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
        else:
            mask = tensor < -delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
        masked_tensor = tensor * mask.float()
        totals = masked_tensor.abs().view(masked_tensor.shape[0], -1).sum(dim=1)

        nums = mask.view(mask.shape[0], -1).sum(dim=1)
        if pos:
            param = totals.flatten() / nums.flatten()
            return param.to(tensor.device)
        else:
            param = -totals.flatten() / nums.flatten()
            return param.to(tensor.device)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()

        self.wp = nn.Parameter(self._init_scales(self.weight, pos=True, t=0.05))
        self.wn = nn.Parameter(self._init_scales(self.weight, pos=False, t=0.05))

    def forward(self, x):
        qweight, wp, wn = self.ternarize.apply(self.weight, self.wp, self.wn)
        out = F.conv2d(x, qweight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out


def Conv2dWTA16(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, method="ttn"):
    if method == "twn":
        return Conv2dWTA16_TWN(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
    elif method == "ttn":
        return Conv2dWTA16_TTN(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
    else:
        raise NotImplementedError
