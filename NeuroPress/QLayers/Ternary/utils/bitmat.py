from typing import Tuple

import torch

from NeuroPress.QLayers.Ternary.triton_kernels.bitmat_kernel import bitmat_
from NeuroPress.QLayers.Ternary.utils.packing import pack_ternary


def terniarize(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Terniarizes the weights and returns the scale.
    """
    dtype = weights.dtype
    scale = 1 / torch.max(weights.abs().mean(), torch.tensor(1e-5))
    
    
    return torch.clamp((weights * scale).round().to(torch.int8), -1, 1), scale.to(dtype)

def quantize_activations(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the activations and returns the scale for each row.
    """
    dtype = x.dtype
    scale = (127 / torch.max(x.abs().max(dim=-1).values, torch.tensor(1e-5))).unsqueeze(-1)
    return torch.clamp((x * scale).round(), -128, 127).to(torch.int8), scale.to(dtype)


class BitMat(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, W, X, scale_w=None):
        """
        During the forward pass, we ternarize the weights, pack them and then quantize the activations.
        We then perform the bit matrix multiplication and return the scaled results.
        ternarization:
        scale_w = 1 / mean(abs(W))                              | STE
        W = clip(round(W * scale_w), -1, 1)                     | STE
        packing:
        packed_w = 4 int8 -> 1 int8                             | STE
        quantization:
        scale_x = 127 / max(abs(X))                             | STE
        X = clip(round(X * scale_x), -127, 128)                 | STE
        bit matrix multiplication:
        Y = X @ w_packed.t()                                    | dot product
        Y = Y / scale_w / scale_x)                              | STE
        """
        if scale_w is None:
            dtype = W.dtype
            W, scale_w = terniarize(W)
            #packed_w = pack_ternary(W, 4) -> this is actually not efficent atm
            ctx.save_for_backward(X)
            X, scale_x = quantize_activations(X)
            y = X.to(dtype) @ W.to(dtype).t()
            #y = batched_bitmat(X, packed_w) -> this is actually not efficent atm
            return y / scale_w / scale_x
        else:
            X, scale_x = quantize_activations(X)
            print(X.dtype, W.dtype)
            print(X.shape, W.shape)
            y = bitmat_(X, W.t().contiguous())
            return y / scale_w / scale_x


    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        # Compute the gradient of the weights
        if grad_output.ndim == 2:
            grad_W = grad_output.t() @ X  # This should give a tensor of shape [256, 512]
        else: 
            grad_W =  (grad_output.transpose(1,2) @ X).mean(dim=0)
        
        
        # Return the correct gradient for the weights and None for the other saved tensors
        return grad_W, None, None
    
#def bitmat(W: torch.Tensor, X: torch.Tensor, scale_w) -> torch.Tensor:
#    return BitMat.apply(W, X, scale_w)


def bitmat(W, X, scale_w=None):
    if scale_w is not None:
        return BitMat.apply(W, X, scale_w)
    else:
        return BitMat.apply(W, X)