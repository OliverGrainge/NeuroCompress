from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils import data
from typing import Callable
import torch


class Compressor(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataset: data.Dataset,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._model = model
        self._compressed_model = None
        self._dataset = dataset

    @abstractmethod
    def compress(self) -> None:
        """
        Apply compression to the model and return the compressed model.
        """
        pass

    @abstractmethod
    def get_compressed_model(self) -> nn.Module:
        """
        Optionally, implement a common method that all compressors can use.
        """
        return self._compressed_model
