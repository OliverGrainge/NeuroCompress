import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuroPress.QLayers.Ternary.quant as Q
from NeuroPress.Utils import RMSNorm

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





from NeuroPress.QLayers.Ternary.utils.bitmat import bitmat
from NeuroPress.QLayers.Ternary.utils.rmsnorm import RMSLayerNorm
from NeuroPress.QLayers.Ternary.utils.packing import pack_ternary, unpack_ternary
from NeuroPress.QLayers.Ternary.utils.bitmat import terniarize

"""
class LinearWTA8(torch.nn.Module):


    def __init__(self, in_features, out_features, bias=None, eps=1e-5, keep_rms_in_32b=False, dtype=torch.float16):
        super(LinearWTA8, self).__init__()
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.scale_w = torch.nn.Parameter(torch.Tensor(1))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSLayerNorm(in_features, eps)
        self.keep_rms_in_32b = keep_rms_in_32b
        self._post_init()



    def _post_init(self):
        #crea un var dei parametri del modello così da poter inizializzare i pesi e i bias
        # Inizializza i pesi utilizzando l'inizializzazione di Kaiming
        params = torch.nn.Parameter(torch.zeros((self.out_features, self.in_features), dtype=self.dtype))
        torch.nn.init.kaiming_normal_(params, mode='fan_out', nonlinearity='relu')
        terniarized_val, self.scale_w.data = terniarize(params)
        del params
        self.register_buffer('weight',pack_ternary(terniarized_val))

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def convert_weights_to_parameters(self):
        # Converti i pesi in torch.nn.Parameter di tipo float16 per il training.
        if self.weight.dtype == torch.int8:
            unpacked_weight = unpack_ternary(self.weight)
            half_weight = (unpacked_weight / self.scale_w).to(self.dtype)
            self.weight = torch.nn.Parameter(half_weight)
            self.scale_w = None# <- this is done so that the bitmat kernel knows we're training

    def convert_weights_to_packed(self):
        # Converti i pesi indietro in PackedParameter di tipo int8 dopo il training.
        if isinstance(self.weight, torch.nn.Parameter):
            terniarized_weight, scale_weight = terniarize(self.weight.data)
            packed_weights = pack_ternary(terniarized_weight)
            self.scale_w = torch.nn.Parameter(scale_weight)
            del self.weight # <- this is done so that torch doesn't trow an error when trying to convert the nn.Parameter to PackedParameter
            self.register_buffer('weight', packed_weights)

    def train(self, mode=True):
        super().train(mode)
        device = next(self.parameters()).device
        if mode:
            self.convert_weights_to_parameters()
        else:
            self.convert_weights_to_packed()
        return self.to(device)

    def forward(self, x):
        if self.training and (self.weight.dtype == torch.int8):
            # Just to make sure the weights are in the right format even if the user forgot to call train()
            self.convert_weights_to_parameters()
        x_dtype = x.dtype
        x = self.norm(x.to(self.norm.weight.dtype)).to(x_dtype)
        output = bitmat(self.weight, x, scale_w=self.scale_w)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
    

"""

"""
class MyLinearWTA8(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        proj_type="bitnet",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.proj_type = proj_type
        self.freeze_state = False
        self.rmsnorm = RMSNorm(in_features)

    @staticmethod
    def activation_quant(x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y

    @staticmethod
    def weight_quant(w):
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

    def activation_norm_quant(self, x):
        x = self.rmsnorm(x)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127)
        return y, scale

    def freeze(self):
        self.freeze_state = True
        w = self.weight
        self.weight_scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        self.q_weights = self.weight_quant(w)

    def unfreeze(self):
        self.q_weights, self.weight_scale = None, None
        self.freeze_state = False

    def train_forward(self, x):
        w = self.weight
        x_norm = self.rmsnorm(x)
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

    def infer_forward(self, x):
        w_quant = self.q_weights
        w_scale = self.weight_scale
        x_quant, x_scale = self.activation_norm_quant(x)
        y = F.linear(x_quant, w_quant) / w_scale / x_scale
        return y

    def forward(self, x):
        if self.freeze_state:
            return self.infer_forward(x)
        else:
            return self.train_forward(x)
"""
        
from NeuroPress.QLayers.Ternary.triton_kernels.bitmat_kernel import bitmat_

class MyLinearWTA8(BaseTernaryLinear):
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
    def activation_quant(x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127) / scale
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
        
        x_quant, scale_x = self.activation_norm_quant(x, dtype=torch.int8)
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
            #missing_keys.remove(key_packed_weights)
            #missing_keys.remove(key_weight_scale)



        

class LinearWTA8(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5, keep_rms_in_32b=True, dtype=torch.float32):
        super(LinearWTA8, self).__init__()
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Initialize full-precision weights
        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), dtype=self.dtype))
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize bias if applicable
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=self.dtype))
            torch.nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        
        # Initialize other components
        self.norm = RMSLayerNorm(in_features, eps)
        self.keep_rms_in_32b = keep_rms_in_32b
        
        # Initialize placeholders for packed weights and scale
        self.register_buffer('packed_weight', None)
        self.register_buffer('scale_w', None)



    def convert_weights_to_parameters(self):
        if self.packed_weight is not None:
            # Remove the packed weights and scale
            del self.packed_weight
            self.packed_weight = None
            del self.scale_w
            self.scale_w = None

    def convert_weights_to_packed(self):
        if isinstance(self.weight, torch.nn.Parameter):
            # Ternarize and pack the weights
            ternarized_weight, scale_weight = terniarize(self.weight.data)
            # Assign to packed_weight and scale_w (already registered as buffers)
            self.packed_weight = pack_ternary(ternarized_weight)
            self.scale_w = scale_weight
            # Delete the full-precision weights to save memory
            del self.weight
            self.weight = None

    def freeze(self):
        # Convert weights to packed format and delete full-precision weights
        self.convert_weights_to_packed()
        # After freezing, the model should not be trained
        self.training = False  # Ensure the model is set to eval mode

    def forward(self, x):
        x_dtype = x.dtype
        x = self.norm(x.to(self.norm.weight.dtype)).to(x_dtype)
        
        if self.packed_weight is not None and self.scale_w is not None:
            # Use packed weights during inference after freezing
            output = bitmat(self.packed_weight, x, scale_w=self.scale_w)
        elif self.weight is not None:
            # Use full-precision weights during training and evaluation
            output = bitmat(self.weight, x)
        else:
            raise RuntimeError("Weights are not initialized.")
        
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Include packed_weight and scale_w in the state dict
        state = super(LinearWTA8, self).state_dict(destination, prefix, keep_vars)
        
        # Remove 'weight' from state dict if it's None
        if 'weight' in state and self.weight is None:
            del state[prefix + 'weight']
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Load state dict and handle missing 'weight' key
        if 'packed_weight' in state_dict:
            self.packed_weight = state_dict['packed_weight']
        if 'scale_w' in state_dict:
            self.scale_w = state_dict['scale_w']
        if 'weight' not in state_dict:
            self.weight = None
        super(LinearWTA8, self).load_state_dict(state_dict, strict)

    #def train(self, mode=True):
    #    if self.weight is None:
    #        raise RuntimeError("Cannot train a frozen model. The weights have been deleted after freezing.")
    #    super().train(mode)
    #    return self

    #def eval(self):
    #    super().eval()
    #    return self