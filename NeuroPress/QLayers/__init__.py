from .Binary import (
    Conv2dW1A1,
    Conv2dW1A16,
    LinearW1A1,
    LinearW1A16,
    StochasticLinearW1A1,
    StochasticLinearW1A16,
    StochastiConv2dW1A1,
    StochastiConv2dW1A16,
)
from .Integer import (
    Conv2dW2A8,
    Conv2dW2A16,
    Conv2dW4A8,
    Conv2dW4A16,
    Conv2dW8A8,
    Conv2dW8A16,
    LinearW2A8,
    LinearW2A16,
    LinearW4A8,
    LinearW4A16,
    LinearW8A8,
    LinearW8A16,
)
from .Ternary import LinearWTA16_TTN, LinearWTA16_TWN

ALL_INTQLAYERS = (LinearW8A16, LinearW4A16, LinearW8A8, LinearW4A8)
