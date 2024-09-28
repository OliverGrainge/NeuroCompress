from abc import ABC, abstractmethod

import torch.nn as nn


class BaseQuantizedLayer(nn.Module, ABC):
    def __init__(self):
        super(BaseQuantizedLayer, self).__init__()

    @abstractmethod
    def train_forward(self, x):
        """forward method during training."""
        pass

    @abstractmethod
    def infer_forward(self, x):
        """forward method during inference."""
        pass

    @abstractmethod
    def freeze(self, x):
        """permenantly quantize the layer for deployment."""
        pass

    @abstractmethod
    def unfreeze(self, x):
        """unfreeze the layer for training."""
        pass

    @abstractmethod
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """adapt the state dictionary for inference or training ."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict, strict=True):
        """loading the state dict, either in frozen model or training mode."""
        pass
