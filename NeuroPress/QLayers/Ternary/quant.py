import numpy as np
import torch

# =================== Ternary Weight Networks =======================
# URL: https://arxiv.org/pdf/1605.04711


def threshold_projection_rowise_vector(
    tensor: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    expanded_scale = scale.expand_as(tensor)
    gt_scale = tensor > expanded_scale
    lt_negative_scale = tensor < -expanded_scale

    new_tensor = torch.where(
        gt_scale,
        torch.tensor(1, dtype=tensor.dtype, device=tensor.device),
        torch.where(
            lt_negative_scale,
            torch.tensor(-1, dtype=tensor.dtype, device=tensor.device),
            torch.tensor(0, dtype=tensor.dtype, device=tensor.device),
        ),
    )
    return new_tensor


def compute_scale_twn(weights, per_channel=False):
    n = weights[0].nelement()
    scale = 0.75 * torch.sum(weights.abs(), dim=(1,)) / n
    Alpha = []
    for i in range(weights.size()[0]):
        count = 0
        abssum = 0
        absvalue = weights[i].view(1, -1).abs()
        truth_value = absvalue > scale[i]
        count = truth_value.sum()
        abssum = torch.matmul(absvalue, truth_value.to(torch.float32).view(-1, 1))
        Alpha.append(abssum / count)
    alpha = torch.cat(Alpha, dim=0)
    return scale.view(-1, 1), alpha


class QuantizeTWN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        scale, alpha = compute_scale_twn(x)
        q_x = threshold_projection_rowise_vector(x, scale)
        return q_x, scale, alpha

    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_alpha):
        grad_input = grad_output.clone()
        return grad_input, None, None


def setupTWN():
    return None


def forward_quantize_twn(tensor: torch.tensor, scale: torch.tensor):
    return QuantizeTWN.apply(tensor, scale)


# ==================================================================

# ==================================================================


def forward_quantize(tensor: torch.tensor, method: str, quant_params):
    if method.lower() == "twn":
        return QuantizeTWN.apply(tensor)
    else:
        raise NotImplementedError


def setup_quantize(method: str):
    if method == "twn":
        return setupTWN()
    else:
        raise NotImplementedError
