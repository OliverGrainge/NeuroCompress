import torch
import torch.nn.functional as F
from torch.autograd import Function

import NeuroPress.QLayers.Binary.projection as proj


class binarize_deterministic_clipped_ste(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        qtensor, alpha = proj.deterministic(x)
        return qtensor, alpha

    @staticmethod
    def backward(ctx, grad_qtensor, grad_alpha):
        (x,) = ctx.saved_tensors
        mask = torch.abs(x) > 1
        grad_inputs = grad_qtensor.clone()
        grad_inputs[mask] = 0.0
        return grad_inputs, None


class binarize_stochastic_clipped_ste(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        qtensor, alpha = proj.stochastic(x)
        return qtensor, alpha

    @staticmethod
    def backward(ctx, grad_qtensor, grad_alpha):
        (x,) = ctx.saved_tensors
        mask = torch.abs(x) > 1
        grad_inputs = grad_qtensor.clone()
        grad_inputs[mask] = 0.0
        return grad_inputs, None


class binarize_spatialconv_clipped_ste(Function):
    @staticmethod
    def forward(ctx, x, kerenl_size, stride, padding, dilation, groups):
        ctx.save_for_backward(x)
        qtensor, alpha = proj.xnor_net_activation_quant(x, kerenl_size, stride, padding, dilation, groups)
        return qtensor, alpha

    @staticmethod
    def backward(ctx, grad_qtensor, grad_alpha):
        (x,) = ctx.saved_tensors
        mask = torch.abs(x) > 1
        grad_inputs = grad_qtensor.clone()
        grad_inputs[mask] = 0.0
        return grad_inputs, None, None, None, None, None
