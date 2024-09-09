import torch
import torch.nn.functional as F
from torch.autograd import Function
from .projection import compute_scales
from .quantize import quantize


class Quantize(Function):
    @staticmethod
    def forward(ctx, x, proj_type):
        alpha, delta = compute_scales(x)
        qtensor = quantize(x, delta)
        return qtensor, alpha, delta
    
    @staticmethod 
    def backward(ctx, grad_qtensor, grad_alpha, grad_delta):
        grad_input = grad_qtensor.clone()
        return grad_input, None, None
    

def ternary_quantize(x, proj_type="twn"):
    if proj_type == "twn":
        return Quantize.apply(x, "twn")



"""
class ternarize_twn(Function):
    @staticmethod
    def forward(ctx, x):
        qtensor, alpha, delta = proj.deterministic(x)
        return qtensor, alpha, delta

    @staticmethod
    def backward(ctx, grad_qtensor, grad_alpha, grad_delta):
        grad_input = grad_qtensor.clone()
        return grad_input, None, None


class ternarize_ttn(Function):
    @staticmethod
    def forward(ctx, x, wp, wn):
        qtensor, pos_mask, neg_mask = proj.learned(x, wp, wn)
        ctx.save_for_backward(pos_mask, neg_mask, wp, wn)
        return qtensor, wp, wn

    @staticmethod
    def backward(ctx, grad_qtensor, grad_wp, grad_wn):
        pos_mask, neg_mask, wp, wn = ctx.saved_tensors
        grad_input = grad_qtensor.clone()
        wp = torch.ones_like(grad_qtensor) * wp.view(-1, *torch.ones(grad_qtensor.ndim - 1).type(torch.int))
        wn = torch.ones_like(grad_qtensor) * wn.view(-1, *torch.ones(grad_qtensor.ndim - 1).type(torch.int))
        grad_input[pos_mask] *= wp[pos_mask]
        grad_input[neg_mask] *= wn[neg_mask]
        grad_wp_input, grad_wn_input = grad_wp.clone(), grad_wn.clone()
        grad_wp_input = (grad_qtensor * pos_mask.float()).view(grad_qtensor.shape[0], -1).sum(dim=1)
        grad_wn_input = (grad_qtensor * neg_mask.float()).view(grad_qtensor.shape[0], -1).sum(dim=1)
        return grad_input, grad_wp_input, grad_wn_input
"""