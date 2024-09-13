import torch

from .quantize import dequantize, quantize

# ================= projection =============================


def twn(tensor: torch.tensor, per_channel: bool = True):
    if per_channel:

        delta = tensor.view(tensor.shape[0], -1).abs().mean(dim=1) * 0.75
        qtensor = quantize(tensor, delta)
        mask = qtensor.abs().type(torch.bool)
        alpha = torch.mean(tensor.abs() * mask, dim=1)
    else:
        delta = tensor.abs().mean() * 0.75
        qtensor = quantize(tensor, delta)
        mask = qtensor.abs().type(torch.bool)
        alpha = qtensor[mask].mean()
    return delta, alpha


"""
def deterministic(tensor: torch.tensor):
    shape = tensor.shape
    delta = tensor.view(tensor.shape[0], -1).abs().mean(dim=1) * 0.75
    qtensor = torch.zeros_like(tensor).to(tensor.device)
    pos_mask = tensor > delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    neg_mask = tensor < -delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    qtensor[pos_mask] = 1.0
    qtensor[neg_mask] = -1.0
    masked_tensor = tensor * torch.logical_or(pos_mask, neg_mask).float()
    alpha = (1 / qtensor.abs().view(shape[0], -1).sum(dim=1)) * masked_tensor.abs().view(shape[0], -1).sum(dim=1)
    return qtensor.float(), alpha, delta


def learned(tensor: torch.tensor, wp: torch.tensor, wn: torch.tensor, t=0.05):
    delta = torch.tensor([t]).to(tensor.device) * tensor.view(tensor.shape[0], -1).abs().mean(dim=1)
    qtensor_pos_mask = tensor > delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    qtensor_neg_mask = tensor < -delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    qtensor = torch.zeros_like(tensor)
    wp = torch.ones_like(qtensor) * wp.view(-1, *torch.ones(qtensor.ndim - 1).type(torch.int))
    wn = torch.ones_like(qtensor) * wn.view(-1, *torch.ones(qtensor.ndim - 1).type(torch.int))
    qtensor[qtensor_pos_mask] = wp[qtensor_pos_mask]
    qtensor[qtensor_neg_mask] = wn[qtensor_neg_mask]
    return qtensor, qtensor_pos_mask, qtensor_neg_mask
"""


def compute_scales(tensor, proj_type="twn"):
    if proj_type == "twn":
        return twn(tensor, per_channel=True)


if __name__ == "__main__":
    x = torch.randn(10, 5)
    delta, alpha = twn(x)
    qtensor = quantize(x, delta)
    dqtensor = dequantize(qtensor, alpha)
    print(dqtensor)
    print(x)
