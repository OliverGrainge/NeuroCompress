from .Binary import (
    Conv2dW1A16,
    StochastiConv2dW1A16,
    LinearW1A16,
    StochasticLinearW1A16,
    LinearW1A1,
    StochasticLinearW1A1,
    Conv2dW1A1,
    StochastiConv2dW1A1,
)


from .Integer import LinearW8A16, LinearW4A16, LinearW8A8, LinearW4A8

ALL_INTQLAYERS = (LinearW8A16, LinearW4A16,LinearW8A8, LinearW4A8)