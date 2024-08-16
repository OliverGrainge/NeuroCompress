import torch
import torch.nn.functional as F
from torch.autograd import Function

import NeuroPress.QLayers.Ternary.projection as proj


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
        wp = wp.unsqueeze(1).expand(-1, grad_input.shape[1])
        wn = wn.unsqueeze(1).expand(-1, grad_input.shape[1])
        grad_input[pos_mask] *= wp[pos_mask]
        grad_input[neg_mask] *= wn[neg_mask]
        grad_wp_input, grad_wn_input = grad_wp.clone(), grad_wn.clone()
        grad_wp_input = (grad_qtensor * pos_mask.float()).view(grad_qtensor.shape[0], -1).sum(dim=1)
        grad_wn_input = (grad_qtensor * neg_mask.float()).view(grad_qtensor.shape[0], -1).sum(dim=1)
        return grad_input, grad_wp_input, grad_wn_input
