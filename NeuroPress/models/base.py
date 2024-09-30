"""
Module: base

This module defines the `Qmodel` class, a base class for quantized neural network models.
`Qmodel` extends PyTorch's `nn.Module` and provides functionality to freeze and unfreeze
quantized layers within the model. Freezing layers is essential for inference, where
quantized weights are utilized for efficient computation, while unfreezing allows for
training or fine-tuning the model with floating-point weights.
"""

import torch 
import torch.nn as nn 


class Qmodel(nn.Module):
    """
    Base Class for Quantized Neural Network Models.

    The `Qmodel` class serves as a foundational class for creating quantized neural network
    models. It extends PyTorch's `nn.Module` and provides methods to freeze and unfreeze
    quantized layers within the model. Freezing a layer typically involves converting
    its parameters to a quantized format suitable for efficient inference, while unfreezing
    reverts them back to a trainable floating-point format.

    Attributes:
        None

    Args:
        None

    Example:
        ```python
        import torch.nn as nn
        from qmodel_base import Qmodel
        from NeuroPress.layers.bitlinear import BitLinear

        class SampleModel(Qmodel):
            def __init__(self):
                super(SampleModel, self).__init__()
                self.fc1 = BitLinear(784, 256)
                self.fc2 = BitLinear(256, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        # Initialize the model
        model = SampleModel()

        # Freeze the quantized layers for inference
        model.freeze()

        # Unfreeze the layers for training
        model.unfreeze()
        ```
    """ 
    def __init__(self,):
        """
        Initialize the Qmodel.

        Constructs the base `Qmodel` by initializing the parent `nn.Module`.
        This base class does not define any layers itself but provides methods
        to manage quantized layers within derived models.

        Args:
            None

        Example:
            ```python
            from qmodel_base import Qmodel

            # Initialize the base model
            model = Qmodel()
            ```
        """
        super().__init__()

    def freeze(self): 
        """
        Freeze Quantized Layers in the Model.

        Iterates through all submodules of the model and invokes the `freeze_layer`
        method on modules that possess this attribute. Freezing a layer typically
        converts its parameters to a quantized format suitable for efficient inference.

        This method is essential for preparing the model for deployment, ensuring that
        all quantized layers are in their optimized state for inference.

        Args:
            None

        Raises:
            AttributeError: If a module intended to be frozen does not have a `freeze_layer` method.

        Example:
            ```python
            # Freeze all quantized layers in the model
            model.freeze()
            ```
        """
        for module in self.modules(): 
            if hasattr(module, 'freeze_layer'):
                module.freeze_layer()
    
    def unfreeze(self): 
        """
        Unfreeze Quantized Layers in the Model.

        Iterates through all submodules of the model and invokes the `unfreeze_layer`
        method on modules that possess this attribute. Unfreezing a layer typically
        reverts its parameters back to a floating-point format, allowing for training
        or fine-tuning.

        This method is useful when you need to update the model's weights or perform
        further training after deployment.

        Args:
            None

        Raises:
            AttributeError: If a module intended to be unfrozen does not have an `unfreeze_layer` method.

        Example:
            ```python
            # Unfreeze all quantized layers in the model for training
            model.unfreeze()
            ```
        """
        for module in self.modules(): 
            if hasattr(module, 'unfreeze_layer'):
                module.unfreeze_layer()