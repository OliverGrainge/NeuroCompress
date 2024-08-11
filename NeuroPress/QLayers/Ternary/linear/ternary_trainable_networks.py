import torch
import torch.nn as nn

from .base import BaseLinear


def initial_scales(weight):
    return nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))


def quantize(weight, w_p, w_n, t=0.05):
    """
    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_p, -w_n}.
    """
    delta = t * weight.abs().max()
    a = (weight > delta).float()
    b = (weight < -delta).float()
    return w_p * a + (-w_n * b)


def get_grads(weight_grad, weight, w_p, w_n, t):
    """
    Arguments:
        weight_grad: gradient with respect to quantized weight.
        weight: corresponding full precision weight.
        w_p, w_n: scaling factors.
        t: hyperparameter for quantization.

    Returns:
        1. gradient for the full precision weight.
        2. gradient for w_p.
        3. gradient for w_n.
    """
    delta = t * weight.abs().max()
    # masks
    a = (weight > delta).float()
    b = (weight < -delta).float()
    c = torch.ones(weight.size()).to(weight.device) - a - b
    # scaled weight grad and grads for scaling factors (w_p, w_n)
    return (
        w_p * a * weight_grad + w_n * b * weight_grad + 1.0 * c * weight_grad,
        (a * weight_grad).sum(),
        (b * weight_grad).sum(),
    )


class QuantizeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, w_p, w_n):
        q_w = quantize(weight, w_p, w_n, t=0.05)
        ctx.save_for_backward(weight, w_p, w_n)
        ctx.t = 0.05
        return q_w

    @staticmethod
    def backward(ctx, grad_output):
        weight, w_p, w_n = ctx.saved_tensors
        t = ctx.t
        grad_input = grad_output.clone()
        return get_grads(grad_input, weight, w_p, w_n, t)


def forward_quantize_ttn(weight, w_p, w_n, t=0.05):
    return QuantizeLinear.apply(weight, w_p, w_n)


class LinearWTA16_TTN(BaseLinear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWTA16_TTN, self).__init__(in_features, out_features, bias=bias)
        self.w_p, self.w_n = initial_scales(self.weight)

    def forward(self, x):
        q_weight = forward_quantize_ttn(self.weight, self.w_p, self.w_n)
        return nn.functional.linear(x, q_weight, self.bias)
