import projection as proj
import torch
import torch.autograd.Function as Function
import torch.nn.functional as F


class binarize_deterministic_clipped_ste(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return proj.deterministic(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_tensors
        mask = torch.abs(x) < 1
        grad_inputs = grad_outputs.clone()
        grad_inputs[mask] = 0.0
        return grad_inputs


class binarize_stochastic_clipped_ste(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return proj.stochastic(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_tensors
        mask = torch.abs(x) < 1
        grad_inputs = grad_outputs.clone()
        grad_inputs[mask] = 0.0
        return grad_inputs
