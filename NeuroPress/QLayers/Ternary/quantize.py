import torch 
import torch.nn as nn


def quantize(tensor, delta):
    if delta.numel() == 1: 
        qtensor = torch.where(tensor > delta, torch.tensor(1.0), 
                                torch.where(tensor < -delta, torch.tensor(-1.0), torch.tensor(0.0)))
    else: 
        assert len(delta) == tensor.shape[0]
        delta = delta.view(-1,*[1 for _ in range(tensor.ndim-1)])
        qtensor = torch.where(tensor > delta, torch.tensor(1.0), 
                        torch.where(tensor < -delta, torch.tensor(-1.0), torch.tensor(0.0)))
    return qtensor



def dequantize(qtensor, alpha):
    if alpha.numel() == 1: 
        dqtensor = alpha * qtensor
    else: 
        dqtensor = alpha.view(qtensor.shape[0], *[1 for _ in range(qtensor.ndim - 1)]) * qtensor
    return dqtensor
