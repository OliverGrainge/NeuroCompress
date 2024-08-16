import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Ternary.quant as Q


def get_ternarize(projection="deterministic", backward="ste"):
    if projection == "deterministic" and backward == "ste":
        return Q.ternarize_deterministic_ste
    else:
        raise NotImplementedError


class BaseTernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearWTA16_TWN(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        projection="deterministic",
        backward="ste",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.ternarize = get_ternarize(projection=projection, backward=backward)

    def forward(self, x):
        qw, alpha, delta = self.ternarize.apply(self.weight)
        out = alpha.view(1, -1) * F.linear(x, qw, self.bias / alpha if self.bias is not None else None)
        return out


def LinearWTA18_TWM(in_features, out_features, bias=True, method="twn"):
    if method == "twn":
        return LinearWTA16_TWN(in_features, out_features, bias=bias)
    else:
        raise NotImplementedError
