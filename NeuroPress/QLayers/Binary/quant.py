import torch


class SignBinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(
            x > 0,
            torch.tensor(1.0, device=x.device),
            torch.tensor(-1.0, device=x.device),
        )

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.clamp(grad_input, -1, 1)
        return grad_input



class StochasticBinarySignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, temp=10000.0):
        ctx.save_for_backward(input)
        input = input * temp
        prob = torch.sigmoid((input + 1) / 2)  # Use sigmoid to smooth the transition
        return torch.where(
            torch.rand_like(input) < prob,
            torch.tensor(1.0, device=input.device),
            torch.tensor(-1.0, device=input.device),
        )

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.clamp(grad_input, -1, 1)
        return grad_input
