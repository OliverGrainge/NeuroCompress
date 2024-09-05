from .Binary import Conv2dW1A1, Conv2dW1A16, LinearW1A1, LinearW1A16
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
from .Ternary import Conv2dWTA16, LinearWTA16

ALL_QLINEARS = (LinearW8A16, LinearW4A16, LinearW8A8, LinearW4A8)
ALL_QCONVS = (Conv2dW8A16, Conv2dW8A8)
