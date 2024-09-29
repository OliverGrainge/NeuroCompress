import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuroPress.functions.bitlinear import bitlinear
from NeuroPress.layers.base import BaseQuantizedLayer
from NeuroPress.functions.rmsnorm import rmsnorm
from NeuroPress.utils import pack_ternary
from NeuroPress.layers.rmsnorm import RMSNorm


class BaseBitLinear(BaseQuantizedLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(BaseQuantizedLayer, self).__init__()
        nn.Linear.__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    @staticmethod
    def activation_quant(x, dtype=None):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        if dtype is not None:
            y = y.type(dtype)
        return y

    @staticmethod
    def weight_quant(w):
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
        self.freeze_state = True
        w = self.weight
        device = self.weight.device
        self.weight_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        q_weights = self.weight_quant(w)
        q_weights = torch.clamp((q_weights * self.weight_scale).round(), -1, 1).type(torch.int8)
        self.packed_weights = nn.Parameter(pack_ternary(q_weights).t().contiguous().to(device), requires_grad=False)
        self.float_weight = self.weight.data 
        del self.weight 


    def unfreeze_layer(self):
        self.freeze_state = False
        self.packed_weights = None
        self.weight_scale = None
        self.weight = nn.Parameter(self.float_weight)

    @staticmethod
    def __repr__(): 
        return "BitLinear"
    
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
