"""
Module: rmsnorm

This module defines the `RMSNorm` class, a normalization layer based on Root Mean Square (RMS) normalization.
RMSNorm normalizes the input tensor by its root mean square, scaled by a learnable weight parameter.
This normalization technique is an alternative to Layer Normalization and is computationally efficient.
"""

import torch
import torch.nn as nn

from NeuroPress.functions.rmsnorm import rmsnorm


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization Layer.

    `RMSNorm` normalizes the input tensor based on the root mean square of its elements, scaled by a learnable weight.
    This normalization technique stabilizes the hidden states in neural networks, promoting faster convergence
    and improved performance.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (torch.nn.Parameter): Learnable scaling parameter of shape `shape`.

    Args:
        shape (int or tuple): The shape of the weight parameter. Typically matches the last dimension of the input.
        eps (float, optional): A small value to prevent division by zero. Default is `1e-6`.
    """

    def __init__(self, shape, eps=1e-6):
        """
        Initialize the RMSNorm layer.

        Args:
            shape (int or tuple): The shape of the weight parameter. Typically matches the last dimension of the input.
            eps (float, optional): A small value to prevent division by zero. Default is `1e-6`.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(shape))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Applies Root Mean Square (RMS) normalization to the input tensor `x`, scaling it by the learnable `weight` parameter.

        Args:
            x (torch.Tensor): The input tensor to normalize. Expected shape is `(*, shape)`, where `shape` matches
                              the last dimension of `self.weight`.

        Returns:
            torch.Tensor: The RMS-normalized tensor, scaled by `self.weight`. The output shape matches the input shape.

        """
        return rmsnorm(self.weight.to(x.dtype), x, self.eps).to(x.dtype)
