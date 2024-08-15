import torch
import torch.nn.functional as F

# ========================== Projection Functions ===========================


def deterministic(tensor: torch.tensor, per_channel=True, batched=False):
    qtensor = torch.where(tensor > 0, torch.tensor(1.0), torch.tensor(-1.0))
    if not per_channel:
        alpha = tensor.abs().mean().view(1)
    else:
        alpha = tensor.view(tensor.shape[0], -1).abs().mean(dim=1)
    return qtensor, alpha


def stochastic(tensor: torch.tensor, per_channel=True):
    p = torch.clip((tensor + 1) / 2, 0, 1)
    rand_tensor = torch.rand_like(tensor)
    mask = rand_tensor < p
    qtensor = -torch.ones_like(tensor)
    qtensor[mask] = 1.0

    if not per_channel:
        alpha = tensor.abs().mean().view(1)
    else:
        alpha = tensor.view(tensor.shape[0], -1).abs().mean(dim=1)
    return qtensor, alpha


def xnor_net_activation_quant(
    tensor: torch.tensor, kernel_size, stride, padding, dilation, groups
):
    qtensor = torch.where(tensor > 0, torch.tensor(1.0), torch.tensor(-1.0))
    spatial_norms = (
        tensor.T.reshape(tensor.shape[2], tensor.shape[3], -1)
        .abs()
        .mean(2)
        .unsqueeze(0)
    )
    kernel = torch.ones(kernel_size) / (kernel_size[0] * kernel_size[1])
    alpha = F.conv2d(
        spatial_norms,
        kernel,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return qtensor, alpha.unsqueeze(0)
