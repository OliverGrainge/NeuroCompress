from abc import ABC, abstractmethod

import torch.nn as nn


class BaseQuantizedLayer(ABC):
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
    def forward(self, x):
        """forward method which calls either train_forward or infer_forward depending on the
        situation.
        """
        pass

    @abstractmethod
    def freeze(self, x):
        """permenantly quantize the layer for deployment."""
        pass

    @abstractmethod
    def unfreeze(self, x):
        """unfreeze the layer for training."""
        pass
