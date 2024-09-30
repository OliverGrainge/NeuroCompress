"""
Module: bitnet

This module defines quantized linear layers for neural networks,
utilizing bit-level quantization techniques to optimize performance
and memory usage. It includes the `BaseBitLinear` class, which serves
as a foundation for quantized linear layers, and the `BitLinear` class,
which implements training and inference behaviors with quantization.

Dependencies:
    - torch
    - torch.nn
    - torch.nn.functional
    - NeuroPress.functions.bitlinear.bitlinear
    - NeuroPress.layers.base.BaseQuantizedLayer
    - NeuroPress.functions.rmsnorm.rmsnorm
    - NeuroPress.utils.pack_ternary
    - NeuroPress.layers.rmsnorm.RMSNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuroPress.functions.bitlinear import bitlinear
from NeuroPress.functions.rmsnorm import rmsnorm
from NeuroPress.layers.base import BaseQuantizedLayer
from NeuroPress.layers.rmsnorm import RMSNorm
from NeuroPress.utils import pack_ternary


class BaseBitLinear(BaseQuantizedLayer, nn.Linear):
    """
    Base class for quantized linear layers using bit-level quantization.

    This class inherits from `BaseQuantizedLayer` and `nn.Linear`, providing
    foundational quantization functionalities for linear layers in neural networks.
    It includes methods for activation and weight quantization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
        device (torch.device, optional): The device on which the layer's parameters will be allocated.
        dtype (torch.dtype, optional): The desired data type of the layer's parameters.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """
        Initialize the BaseBitLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to `False`, the layer will not learn an additive bias. Default: `True`.
            device (torch.device, optional): The device on which the layer's parameters will be allocated.
            dtype (torch.dtype, optional): The desired data type of the layer's parameters.
        """
        super(BaseQuantizedLayer, self).__init__()
        nn.Linear.__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    @staticmethod
    def activation_quant(x, dtype=None):
        """
        Quantize the activation tensor.

        This method scales the input tensor `x`, rounds it to the nearest integer,
        clamps the values to the range [-128, 127], and optionally casts it to the specified dtype.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            dtype (torch.dtype, optional): The desired data type of the output tensor.

        Returns:
            torch.Tensor: The quantized activation tensor.

        """
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        if dtype is not None:
            y = y.type(dtype)
        return y

    @staticmethod
    def weight_quant(w):
        """
        Quantize the weight tensor.

        This method scales the input weight tensor `w` based on its mean absolute value,
        rounds it to the nearest integer, clamps the values to the range [-1, 1],
        and returns the quantized weights.

        Args:
            w (torch.Tensor): The weight tensor to quantize.

        Returns:
            torch.Tensor: The quantized weight tensor.


        """
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u


class BitLinear(BaseBitLinear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(BaseQuantizedLayer, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.freeze_state = False

        self.rmsnorm = RMSNorm(in_features)

    def train_forward(self, x):
        if self.weight is None:
            raise RuntimeError("Weights are not initialized for training.")
        w = self.weight
        x_norm = self.rmsnorm(x)
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def infer_forward(self, x):
        x = rmsnorm(self.rmsnorm.weight, x, self.rmsnorm.eps)
        scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_quant = (x * scale_x).round().clamp_(-128, 127).type(torch.int8)
        y = bitlinear(x_quant, self.packed_weights)
        y = y / scale_x / self.weight_scale
        return y

    def forward(self, x):
        if self.freeze_state:
            return self.infer_forward(x)
        else:
            return self.train_forward(x)

    def freeze_layer(self):
        """
        Freeze the layer for inference.

        This method quantizes the weights, packs them into ternary format, and removes
        the floating-point weights to optimize for inference.

        Returns:
            None
        """
        self.freeze_state = True
        w = self.weight
        device = self.weight.device
        self.weight_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        q_weights = self.weight_quant(w)
        q_weights = torch.clamp((q_weights * self.weight_scale).round(), -1, 1).type(
            torch.int8
        )
        self.packed_weights = nn.Parameter(
            pack_ternary(q_weights).t().contiguous().to(device), requires_grad=False
        )
        self.float_weight = self.weight.data
        del self.weight

    def unfreeze_layer(self):
        """
        Unfreeze the layer for training.

        This method restores the floating-point weights from the stored data and removes
        the packed ternary weights and scaling factor.

        Returns:
            None
        """
        self.freeze_state = False
        self.packed_weights = None
        self.weight_scale = None
        self.weight = nn.Parameter(self.float_weight)

    @staticmethod
    def __repr__():
        """
        Return the string representation of the BitLinear layer.

        Returns:
            str: The string "BitLinear".
        """
        return "BitLinear"

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Save the layer's state to the state dictionary.

        Overrides the base method to exclude the floating-point weights when the layer is frozen.

        Args:
            destination (dict): The destination dictionary.
            prefix (str): The prefix for the state keys.
            keep_vars (bool): Whether to keep variables.

        Returns:
            None
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.freeze_state:
            key = prefix + "weight"
            if key in destination:
                del destination[key]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load the layer's state from the state dictionary.

        Overrides the base method to handle both frozen and unfrozen states.

        Args:
            state_dict (dict): The state dictionary containing parameters and buffers.
            prefix (str): The prefix for the state keys.
            local_metadata (dict): Metadata of the state.
            strict (bool): Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's `state_dict` function.
            missing_keys (list): A list to append missing keys to.
            unexpected_keys (list): A list to append unexpected keys to.
            error_msgs (list): A list to append error messages to.

        Returns:
            None
        """
        key_weight = prefix + "weight"
        key_packed_weights = prefix + "packed_weights"
        key_weight_scale = prefix + "weight_scale"

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if (
            key_weight in missing_keys
            and key_packed_weights in state_dict.keys()
            and key_weight_scale in state_dict.keys()
        ):

            self.freeze_state = True
            self.packed_weights = state_dict[key_packed_weights]
            self.weight_scale = state_dict[key_weight_scale]

            missing_keys.remove(key_weight)
            unexpected_keys.remove(key_packed_weights)
            unexpected_keys.remove(key_weight_scale)

        elif (
            key_weight in state_dict.keys()
            and key_packed_weights not in state_dict.keys()
            and key_weight_scale not in state_dict.keys()
        ):
            self.freeze_state = False
