import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Ternary.quant as Q


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
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.ternarize = Q.ternarize_twn

    def forward(self, x):
        qw, alpha, delta = self.ternarize.apply(self.weight)
        out = alpha.view(1, -1) * F.linear(x, qw, self.bias / alpha if self.bias is not None else None)
        return out


class LinearWTA16_TTN(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(in_features, out_features, bias=bias)
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
        nums = mask.sum(dim=1)
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
        out = F.linear(x, qweight)
        if self.bias is not None:
            out += self.bias.view(1, -1)
        return out


def LinearWTA16(in_features, out_features, bias=True, method="twn"):
    if method == "twn":
        return LinearWTA16_TWN(in_features, out_features, bias=bias)
    elif method == "ttn":
        return LinearWTA16_TTN(in_features, out_features, bias=bias)
    else:
        raise NotImplementedError
