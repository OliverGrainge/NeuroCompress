import torch

from .projections import compute_scale_and_zeropoint
from .quantize import dequantize, quantize


class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, proj_type, bits, per_channel, symmetric):
        ctx.save_for_backward(x)
        scale, zero_point = compute_scale_and_zeropoint(x, proj_type=proj_type, bits=bits, per_channel=per_channel, symmetric=symmetric)
        qtensor = quantize(x, scale, zero_point)
        ctx.scale = scale
        return qtensor, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_zero_point):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        if scale.numel() == 1:
            grad_input = grad_output / scale
        else:
            grad_input = grad_output / scale.view(
                grad_output.shape[0], *[1 for _ in range(grad_output.ndim - 1)]
            )  # Adjusting gradients based on the scale
        return grad_input, None, None, None, None


def linear_quantize(tensor: torch.tensor, proj_type="minmax", bits=8, per_channel=True, symmetric=True):
    return QuantFunc.apply(tensor, proj_type, bits, per_channel, symmetric)
