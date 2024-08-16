import torch


def compute_scale(tensor: torch.Tensor, bits=8, type: str = "signed", per_channel=False):
    if per_channel:
        max_per_channel = tensor.abs().view(tensor.shape[0], -1).max(dim=1)[0]
        scale = max_per_channel / (2 ** (bits - 1) - 1)
        scale = scale.view(-1, 1, 1, 1)
        if type == "signed":
            return scale.to(tensor.device), torch.zeros(scale.shape[0], dtype=torch.int).view(-1, 1, 1, 1).to(tensor.device)
        else:
            zero_point = -torch.round(tensor.view(tensor.shape[0], -1).min(dim=1)[0] / scale[0]).view(-1, 1, 1, 1)
            return scale.to(tensor.device), zero_point.to(tensor.device)

    real_range = tensor.abs().max()
    quantized_range = 2 ** (bits - 1) - 1
    scale = torch.Tensor(real_range / quantized_range)
    if type == "unsigned":
        zero_point = -torch.round(tensor.min() / scale)
    else:
        zero_point = torch.zeros(1)
    return scale.to(tensor.device), zero_point.to(tensor.device)


def dequantize_per_tensor(tensor: torch.Tensor, scale: float, zero_point: int):
    if tensor == None:
        return None
    return scale * (tensor - zero_point)


def dequantize_per_channel(tensor: torch.Tensor, scale: float, zero_point: int):
    if tensor == None:
        return None
    return scale * (tensor - zero_point)


def quantize_per_tensor(tensor: torch.Tensor, scale: float, zero_point: int, bits: int, type: str):
    if tensor is None:
        return None
    if type == "signed":
        qtensor = torch.round(tensor / scale)
        return torch.clamp(qtensor, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    elif type == "unsigned":
        qtensor = torch.round(tensor / scale) + zero_point
        out = torch.clamp(qtensor, 0, 2**bits - 1)
        return out
    else:
        raise Exception(f"{type} is not implemented")


class QuantizeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, type):
        ctx.save_for_backward(x)
        scale, zero_point = compute_scale(x, bits, type, per_channel=False)
        qtensor = quantize_per_tensor(x, scale, zero_point, bits, type)
        ctx.scale = scale
        return qtensor, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_zero_point):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output / scale  # Adjusting gradients based on the scale
        return grad_input, None, None


class QuantizeConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, type):
        ctx.save_for_backward(x)
        scale, zero_point = compute_scale(x, bits, type, per_channel=True)
        qtensor = quantize_per_tensor(x, scale, zero_point, bits, type)
        ctx.scale = scale
        return qtensor, scale, zero_point

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_zero_point):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output / scale  # Adjusting gradients based on the scale
        return grad_input, None, None


def forward_quantize_per_tensor(tensor: torch.tensor, bits=8, type="signed"):
    return QuantizeLinear.apply(tensor, bits, type)


def forward_quantize_per_channel(tensor: torch.tensor, bits=8, type="signed"):
    return QuantizeConv2d.apply(tensor, bits, type)
