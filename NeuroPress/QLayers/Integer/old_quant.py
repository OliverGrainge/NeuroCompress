import torch 
from typing import Tuple

def compute_linear_symmetric_scale(tensor: torch.Tensor, bits=8) -> float:
    real_range = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
    quantized_range = 2**(bits-1) - 1
    return real_range / quantized_range 

def compute_linear_scale_and_zeropoint(tensor: torch.Tensor, bits=8):
    min_val = tensor.min()
    max_val = tensor.max()
    quantized_range = 2**bits - 1
    scale = (max_val - min_val) / quantized_range
    zero_point = quantized_range * min_val / (min_val - max_val)
    zero_point = max(0, min(quantized_range, round(zero_point)))
    return scale, zero_point

def quantize_linear_tensor(tensor: torch.Tensor, scale: torch.Tensor, zero_point=None, bits=8):
    if tensor is None: 
        return tensor
    qtensor = torch.round(tensor/scale)
    if zero_point is not None: 
        qtensor += zero_point
    qtensor = torch.clip(qtensor, -2**(bits-1), 2**(bits-1) -1)
    return qtensor 

def dequantize_linear_tensor(tensor: torch.Tensor, scale: float, zero_point: float=0.0):
    return scale * (tensor - zero_point)

def quantize_linear_weights(tensor: torch.Tensor, bits: int=8) -> Tuple[torch.Tensor, float]:
    scale = compute_linear_symmetric_scale(tensor)
    qtensor = quantize_linear_tensor(tensor, scale, bits=bits)
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

