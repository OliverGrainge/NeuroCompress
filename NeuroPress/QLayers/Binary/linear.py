import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Binary.quant as Q


def get_binarize(projection="deterministic", backward="ste"):
    if projection == "deterministic" and backward == "clipped_ste":
        return Q.binarize_deterministic_clipped_ste
    elif projection == "stochastic" and backward == "clipeed_ste":
        return Q.binarize_stochastic_clipped_ste
    else:
        raise NotImplementedError


class BaseBinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearW1A16(BaseBinaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        projection="deterministic",
        backward="clipped_ste",
        per_channel=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.binarize = get_binarize(projection=projection, backward=backward)
        self.per_channel = per_channel

    def forward(self, x):
        qw, alpha = self.binarize.apply(self.weight)
        out = alpha.view(1, -1) * F.linear(x, qw, self.bias / alpha if self.bias is not None else None)
        return out


class LinearW1A1(BaseBinaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        projection="deterministic",
        backward="clipped_ste",
        per_channel=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.binarize = get_binarize(projection=projection, backward=backward)
        self.per_channel = per_channel

    def forward(self, x):
        qw, alpha_w = self.binarize.apply(self.weight)
        qx, alpha_x = self.binarize.apply(x)
        out = F.linear(qx, qw, None)
        out *= alpha_w.view(1, -1) * alpha_x.view(-1, 1)
        if self.bias is not None:
            out += self.bias.view(1, -1)
        return out
