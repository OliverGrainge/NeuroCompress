import torch 
from typing import Tuple

def compute_linear_scale(tensor: torch.Tensor, bits=8) -> float:
    real_range = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
    quantized_range = 2**(bits-1) - 1
    return real_range / quantized_range 


def quantize_linear_weights(tensor: torch.Tensor, bits: int=8) -> Tuple[torch.Tensor, float]:
    scale = compute_linear_scale(tensor)
    qtensor = torch.round(tensor/scale)
    qtensor = torch.clip(qtensor, -2**(bits-1), 2**(bits-1) -1)
    return qtensor, scale


def quantize_conv2d_weights(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    # Calculate the scale
    # Max absolute value per output channel, keeping input channels and kernel dimensions intact
    max_abs_per_channel = tensor.abs().view(tensor.shape[0], -1).max(dim=1)[0]
    scale = max_abs_per_channel / (2**(bits-1) - 1)
    scale = scale.view(-1, 1, 1, 1)
    
    # Quantize
    qtensor = torch.round(tensor / scale)
    qtensor = torch.clip(qtensor, -2**(bits-1), 2**(bits-1) -1)
    return qtensor, scale


class QuantizeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits=8):
        ctx.save_for_backward(x)
        qtensor, scale = quantize_linear_weights(x, bits=bits)
        ctx.scale = scale
        return qtensor, scale  # Returning both quantized tensor and scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        x, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output / scale  # Adjusting gradients based on the scale
        return grad_input, None

class QuantizeConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits=8):
        ctx.save_for_backward(x)
        qtensor, scale = quantize_conv2d_weights(x, bits=bits)
        ctx.scale = scale
        return qtensor, scale  # Similarly returning both outputs

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        x, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output / scale  # Assuming pass-through again
        return grad_input, None  # Same reasoning as above

def quantizelinear(tensor: torch.tensor, bits: int): 
    return QuantizeLinear.apply(tensor, bits)

def quantizeconv2d(tensor: torch.tensor, bits: int):
    return QuantizeConv2d.apply(tensor, bits)

