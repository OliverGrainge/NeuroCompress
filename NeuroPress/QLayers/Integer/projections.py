import torch 
import torch.nn.functional as F 
from .quantize import quantize, dequantize


def minmax(tensor: torch.Tensor, bits: int = 8, per_channel: bool = False, symmetric: bool = True):
    # Symmetric Quantization
    if not per_channel and symmetric:  # Global symmetric quantization
        scale = tensor.abs().max() / (2 ** (bits - 1) - 1)
        return scale, torch.tensor([0], device=tensor.device, dtype=tensor.dtype)
    
    elif per_channel and symmetric:  # Per-channel symmetric quantization
        scale = tensor.abs().max(dim=1)[0] / (2 ** (bits - 1) - 1)  # dim=1: quantize per channel
        return scale, torch.zeros_like(scale, device=tensor.device, dtype=tensor.dtype)
    
    # Asymmetric Quantization
    elif not per_channel and not symmetric:  # Global asymmetric quantization
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = torch.round(-min_val / scale).clamp(0, 2 ** bits - 1)
        return scale, zero_point
    
    elif per_channel and not symmetric:  # Per-channel asymmetric quantization
        min_val = tensor.min(dim=1)[0]  # min per channel
        max_val = tensor.max(dim=1)[0]  # max per channel
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = torch.round(-min_val / scale).clamp(0, 2 ** bits - 1)
        return scale, zero_point
    

def kl_div(tensor: torch.Tensor, bits: int = 8, per_channel=False, symmetric=True, num_bins: int = 1024, eps=1e-10):
    min_val, max_val = tensor.min(), tensor.max()
    num_bins = min(num_bins, tensor.numel())
    
    # Create histogram of the original tensor
    hist = torch.histc(tensor, bins=num_bins, min=min_val.item(), max=max_val.item())
    hist = hist.float() / hist.sum()  # Normalize histogram

    #quantize using minmax: 
    scale, zero_point = minmax(tensor, bits=bits, per_channel=per_channel, symmetric=symmetric)

    best_kl_div = float('inf')
    best_scale = None
    best_zero_point = None

    # Try different scales in the range
    for scale_mult in torch.linspace(0.5, 1.0, num_bins):
        scale_val = scale * scale_mult
        if not symmetric: 
            zero_point = torch.round(-min_val / scale_val).clamp(0, 2 ** bits - 1)

        # quantization
        qtensor = quantize(tensor, scale_val, zero_point)
        # dequantize the tensor 
        dqtensor = dequantize(qtensor, scale_val, zero_point)

        # Create histogram of quantized values
        dqhist = torch.histc(dqtensor, bins=num_bins, min=min_val.item(), max=max_val.item())
        dqhist = dqhist.float() / dqhist.sum()  # Normalize histogram
        dqhist += eps
        # Calculate KL Divergence between original and quantized distributions
        kl_divergence = F.kl_div(dqhist.log(), hist, reduction='batchmean')
        
        if kl_divergence < best_kl_div:
            best_kl_div = kl_divergence
            best_scale = scale_val
            best_zero_point = zero_point
    return best_scale, best_zero_point


def mse(tensor: torch.Tensor, bits: int = 8, num_steps: int = 1000, per_channel=False, symmetric=True):
    min_val, max_val = tensor.min(), tensor.max()

    scale, zero_point = minmax(tensor, bits=bits, per_channel=per_channel, symmetric=symmetric)
    
    best_mse = float('inf')
    best_scale = None
    best_zero_point = None
    
    # Try different scales around a base value
    for scale_mult in torch.linspace(0.5, 1.5, num_steps):
        if symmetric:
            scale_var = scale * scale_mult
        else:
            scale_var = scale * scale_mult
            zero_point = torch.round(-min_val / scale_var).clamp(0, 2 ** bits - 1)
        
        # Quantize and dequantize
        qtensor = quantize(tensor, scale_var, zero_point)
        dqtensor = dequantize(qtensor, scale_var, zero_point)

        # Compute Mean Squared Error
        mse = torch.mean((tensor - dqtensor) ** 2)

        # Track the best scale and zero-point with the lowest MSE
        if mse < best_mse:
            best_mse = mse
            best_scale = scale_var
            best_zero_point = zero_point

    return best_scale, best_zero_point.to(tensor.device)


def compute_scale_and_zeropoint(tensor: torch.Tensor, proj_type: str="minmax", bits: int = 8, per_channel=False, symmetric=True):
    if "minmax" in proj_type.lower():
        return minmax(tensor, bits=bits, per_channel=per_channel, symmetric=symmetric)
    elif "kl" in proj_type.lower():
        return kl_div(tensor, bits=bits, per_channel=per_channel, symmetric=symmetric)
    elif "mse" in proj_type.lower():
        return mse(tensor, bits=bits, per_channel=per_channel, symmetric=symmetric)
    else: 
        raise Exception(f"projection type {proj_type} is not available")
