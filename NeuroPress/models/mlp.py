"""
Module: mlp

This module defines the MLP class, a Multi-Layer Perceptron (MLP) model for classification tasks.
The MLP class inherits from Qmodel and allows for flexible configuration of the network's architecture
by specifying the type of layers, input size, hidden layer size, number of classes, and number of layers.
"""

import torch.nn as nn
import torch.nn.functional as F

from NeuroPress.models.base import Qmodel


class MLP(Qmodel):
    """
    Multi-Layer Perceptron (MLP) Model for Classification.

    The MLP class implements a fully connected neural network with a configurable number of layers.
    It is designed for classification tasks and allows the use of different layer types, such as
    standard linear layers or quantized linear layers, by specifying the ``layer_type`` during initialization.

    Attributes:
        layers (nn.ModuleList): A list of layers comprising the MLP model.

    Args:
        layer_type (class): The class of the layer to be used (e.g., ``nn.Linear``, ``BitLinear``).
        input_size (int): The size of each input sample.
        hidden_size (int): The size of each hidden layer.
        num_classes (int): The number of output classes for classification.
        num_layers (int): The total number of layers in the MLP (including input and output layers).

    """

    def __init__(self, layer_type, input_size, hidden_size, num_classes, num_layers):
        """
        Initialize the MLP model.

        Constructs an MLP with the specified number of layers, input size, hidden layer size,
        and number of output classes. The model is built using the provided ``layer_type`` for each layer.

        Args:
            layer_type (class): The class of the layer to be used (e.g., ``nn.Linear``, ``BitLinear``).
            input_size (int): The size of each input sample.
            hidden_size (int): The size of each hidden layer.
            num_classes (int): The number of output classes for classification.
            num_layers (int): The total number of layers in the MLP (including input and output layers).

        Raises:
            ValueError: If ``num_layers`` is less than 2, as at least an input and an output layer are required.

        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([layer_type(input_size, hidden_size)])

        for _ in range(1, num_layers):
            self.layers.append(layer_type(hidden_size, hidden_size))

        self.layers.append(layer_type(hidden_size, num_classes))

    def forward(self, x):
        """
        Perform a forward pass through the MLP.

        Passes the input tensor ``x`` through each layer of the MLP, applying the ReLU activation
        function after each hidden layer. The output layer does not use an activation function,
        as it is assumed to be used with a classification loss function like CrossEntropyLoss,
        which internally applies softmax.

        Args:
            x (torch.Tensor): The input tensor of shape ``(batch_size, input_size)``.

        Returns:
            torch.Tensor: The output logits tensor of shape ``(batch_size, num_classes)``.

        """
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        x = self.layers[-1](x)
        return x
