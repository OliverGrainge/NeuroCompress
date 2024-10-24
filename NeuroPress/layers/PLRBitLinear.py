import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuroPress.functions.bitlinear import bitlinear
from NeuroPress.functions.rmsnorm import rmsnorm
from NeuroPress.layers.base import BaseQuantizedLayer
from NeuroPress.layers.rmsnorm import RMSNorm
from NeuroPress.utils import pack_ternary


class PLRBitLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.rmsnorm = RMSNorm(in_features)
        self.freeze_state = False
        self.register_buffer("packed_weights", None)
        self.scale = nn.Parameter(1 / self.weight.data.abs().mean())
        self.q_lambda = 1.0

    @staticmethod
    def activation_quant(x, dtype=None):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        if dtype is not None:
            y = y.type(dtype)
        return y

    def weight_quant(self, w):
        u = (w * self.scale).round().clamp_(-1, 1) / self.scale
        return u

    def activation_norm_quant(self, x, dtype=None):
        x = self.rmsnorm(x)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127)
        if dtype is not None:
            y = y.type(dtype)
        return y, scale

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
        q_weights = self.weight_quant(w)
        q_weights = torch.clamp((q_weights * self.scale).round(), -1, 1).type(
            torch.int8
        )
        self.packed_weights = nn.Parameter(
            pack_ternary(q_weights).t().contiguous().to(device), requires_grad=False
        )
        self.float_weight = self.weight.data
        del self.weight

    def unfreeze_layer(self):
        self.freeze_state = False
        self.packed_weights = None
        self.scale = None
        self.weight = nn.Parameter(self.float_weight)

    def train_forward(self, x):
        if self.weight is None:
            raise RuntimeError("Weights are not initialized for training.")
        w = self.weight
        x_norm = self.rmsnorm(x)
        x_quant = (
            x_norm + self.q_lambda * (self.activation_quant(x_norm) - x_norm).detach()
        )
        w_quant = w + self.q_lambda * (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def infer_forward(self, x):
        x = rmsnorm(self.rmsnorm.weight, x, self.rmsnorm.eps)
        scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_quant = (x * scale_x).round().clamp_(-128, 127).type(torch.int8)
        y = bitlinear(x_quant, self.packed_weights)
        y = y / scale_x / self.scale
        return y

    def forward(self, x):
        if self.freeze_state:
            return self.infer_forward(x)
        else:
            return self.train_forward(x)

    def __repr__(
        self,
    ):
        return "PLRBitLinear1"

    def weight_decay_layer(self, lr, weight_decay_scale):
        q_weight = (self.weight * self.scale).round().clamp_(-1, 1) / self.scale
        squared_res = (self.weight - q_weight) ** 2
        res = (self.weight - q_weight)
        decay_factor = res * 0.1 + 0.9 * (res.sign() * squared_res * 0.9) 
        self.weight = self.weight - lr * weight_decay_scale * decay_factor  # Apply decay directly to the weight
        return self.weight

    def _save_to_state_dict(self, destination, prefix, keep_vars):
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
            self.scale = state_dict[key_weight_scale]

            missing_keys.remove(key_weight)
            unexpected_keys.remove(key_packed_weights)
            unexpected_keys.remove(key_weight_scale)

        elif (
            key_weight in state_dict.keys()
            and key_packed_weights not in state_dict.keys()
            and key_weight_scale not in state_dict.keys()
        ):
            self.freeze_state = False


