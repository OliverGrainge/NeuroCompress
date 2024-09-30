from .bitlinear import BitLinear
from .learned_bitlinear import LBitLinear
from .learned_reg_bitlinear import LRBitLinear1, LRBitLinear2
from .progressive_bitlinear import PLRBitLinear1, PLRBitLinear2

from .rmsnorm import RMSNorm

LINEAR_LAYERS = (BitLinear, LBitLinear, LRBitLinear1, LRBitLinear2, PLRBitLinear1, PLRBitLinear2)
