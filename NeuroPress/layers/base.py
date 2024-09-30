from abc import ABC, abstractmethod

import torch.nn as nn


class BaseQuantizedLayer(ABC):
    def __init__(self):
        super(BaseQuantizedLayer, self).__init__()

    @abstractmethod
    def train_forward(self, x):
        pass

    @abstractmethod
    def infer_forward(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def freeze_layer(self, x):
        pass

    @abstractmethod
    def unfreeze_layer(self, x):
        pass
