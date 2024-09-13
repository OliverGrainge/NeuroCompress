import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import ternary_quantize
import NeuroPress.QLayers.Ternary.quant as Q
from NeuroPress.Utils import RMSNorm

class BaseTernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def setup(self, linear_layer: nn.Linear):
        self.weight.data = linear_layer.weight.data.detach()
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data.detach()


class LinearWTA16(BaseTernaryLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        proj_type="twn"
    ):
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





class LinearWTA8(BaseTernaryLinear):
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
        u = (w*scale).round().clamp_(-1, 1)/scale
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
        w = self.q_weights
        w_scale = self.weight_scale 
        x_quant, x_scale = self.activation_norm_quant(x)
        y = F.linear(x_quant, w) / w_scale / x_scale
        return y
    
    def forward(self, x):
        if self.freeze_state: 
            return self.infer_forward(x)
        else: 
            return self.train_forward(x)
    
