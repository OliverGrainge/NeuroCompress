import torch

# ================= projection =============================


def deterministic(tensor: torch.tensor):
    shape = tensor.shape
    delta = tensor.view(tensor.shape[0], -1).abs().mean(dim=1) * 0.75
    qtensor = tensor > delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    masked_tensor = tensor * qtensor.float()
    alpha = (1 / qtensor.view(shape[0], -1).sum(dim=1)) * masked_tensor.view(shape[0], -1).sum(dim=1)
    return qtensor.float(), alpha, delta


def learned(tensor: torch.tensor, wp: torch.tensor, wn: torch.tensor, t=0.05):
    delta = torch.tensor([t]) * tensor.view(tensor.shape[0], -1).abs().mean(dim=1)
    qtensor_pos_mask = tensor > delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    qtensor_neg_mask = tensor < -delta.view(-1, *torch.ones(tensor.ndim - 1).type(torch.int))
    qtensor = torch.zeros_like(tensor)
    wp = wp.unsqueeze(1).expand(-1, tensor.shape[1])
    wn = wn.unsqueeze(1).expand(-1, tensor.shape[1])
    qtensor[qtensor_pos_mask] = wp[qtensor_pos_mask]
    qtensor[qtensor_neg_mask] = wn[qtensor_neg_mask]
    return qtensor
