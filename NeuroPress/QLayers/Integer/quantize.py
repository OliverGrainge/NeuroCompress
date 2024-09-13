import torch
import torch.nn.functional as F


def quantize(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    if scale.numel() == 1:  # Scalar scale
        qtensor = torch.round(tensor / scale.to(tensor.device)) + zero_point.to(tensor.device)
    else:  # Per-channel scale, reshape for broadcasting
        scale = scale.view(tensor.shape[0], *[1 for _ in range(tensor.ndim - 1)])
        zero_point = zero_point.view(tensor.shape[0], *[1 for _ in range(tensor.ndim - 1)])
        qtensor = torch.round(tensor / scale.to(tensor.device)) + zero_point.to(tensor.device)
    return qtensor


def dequantize(qtensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    if scale.numel() == 1:  # Scalar scale
        dqtensor = scale * (qtensor - zero_point)
    else:  # Per-channel scale, reshape for broadcasting
        scale = scale.view(qtensor.shape[0], *[1 for _ in range(qtensor.ndim - 1)])
        zero_point = zero_point.view(qtensor.shape[0], *[1 for _ in range(qtensor.ndim - 1)])
        dqtensor = scale.to(qtensor.device) * (qtensor - zero_point.to(qtensor.device))
    return dqtensor
