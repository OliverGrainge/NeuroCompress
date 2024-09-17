import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Ternary.quant as Q
from NeuroPress.Utils import RMSNorm
from NeuroPress.QLayers.Ternary.utils.bitmat import bitmat
from NeuroPress.QLayers.Ternary.utils.rmsnorm import RMSLayerNorm
from NeuroPress.QLayers.Ternary.utils.packing import pack_ternary, unpack_ternary
from NeuroPress.QLayers.Ternary.utils.bitmat import terniarize
from NeuroPress.QLayers.Ternary.triton_kernels.bitmat_kernel import bitmat_

from .quant import ternary_quantize


class BaseTernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearWTA16(BaseTernaryLinear):
    def __init__(self, in_features, out_features, bias=True, proj_type="twn"):
        super().__init__(in_features, out_features, bias=bias)
        self.proj_type = proj_type
        self.freeze_state = False

    def freeze(self):
        self.freeze_state = True
        self.q_weights, self.alpha, self.delta = ternary_quantize(self.weight, self.proj_type)

    def unfreeze(self):
        self.freeze_state = False
        self.q_weights, self.alpha, self.delta = None, None, None

    def forward(self, x):
        if self.freeze_state:
            q_weights, alpha, delta = self.q_weights, self.alpha, self.delta
        else:
            q_weights, alpha, delta = ternary_quantize(self.weight, self.proj_type)

        out = alpha.view(1, -1) * F.linear(x, q_weights, self.bias / alpha if self.bias is not None else None)
        return out




from NeuroPress.QLayers.Ternary.triton_kernels.rmsnorm_kernel import fast_rms_layernorm


class LinearWTA8(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        proj_type="bitnet",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.proj_type = proj_type
        self.freeze_state = False
        self.rmsnorm = RMSNorm(in_features)
        # Register 'packed_weights' and 'weight_scale' as buffers
        self.register_buffer('packed_weights', None)
        self.register_buffer('weight_scale', None)

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

    def activation_norm_quant(self, x, dtype=None):
        x = self.rmsnorm(x)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127)
        if dtype is not None: 
            y = y.type(dtype)
        return y, scale

    def freeze(self):
        self.freeze_state = True
        w = self.weight
        self.weight_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        q_weights = self.weight_quant(w).cuda()
        q_weights = torch.clamp((q_weights * self.weight_scale).round(), -1, 1).type(torch.int8)
        self.packed_weights = pack_ternary(q_weights).t().contiguous().cuda()
        # Optionally delete the full-precision weights to save memory
        # del self.weight

    def unfreeze(self):
        self.freeze_state = False
        self.packed_weights = None
        self.weight_scale = None

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
        x = fast_rms_layernorm(self.rmsnorm.weight, x, self.rmsnorm.eps)
        scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_quant = (x * scale_x).round().clamp_(-128, 127).type(torch.int8)
        y = bitmat_(x_quant, self.packed_weights)
        y = y / scale_x / self.weight_scale
        return y 

    def forward(self, x):
        if self.freeze_state:
            return self.infer_forward(x)
        else:
            return self.train_forward(x)

    # Override _save_to_state_dict to handle prefixes and exclude 'weight' when frozen
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.freeze_state:
            # Remove 'weight' from destination to avoid saving it
            key = prefix + 'weight'
            if key in destination:
                del destination[key]

    # Override _load_from_state_dict without popping keys
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Adjust missing_keys based on freeze_state
        key_weight = prefix + 'weight'
        key_packed_weights = prefix + 'packed_weights'
        key_weight_scale = prefix + 'weight_scale'



        
        # Proceed with the standard loading process
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        
        if key_weight in missing_keys and key_packed_weights in state_dict.keys() and key_weight_scale in state_dict.keys():
            self.freeze_state = True 
            print("loading in frozen mode")
            self.packed_weights = state_dict[key_packed_weights]
            self.weight_scale = state_dict[key_weight_scale]
            missing_keys.remove(key_weight)
            unexpected_keys.remove(key_packed_weights)
            unexpected_keys.remove(key_weight_scale)

        elif key_weight in state_dict.keys() and key_packed_weights not in state_dict.keys() and key_weight_scale not in state_dict.keys():
            self.freeze_state = False 
            print("loading in unfrozen mode")



        

