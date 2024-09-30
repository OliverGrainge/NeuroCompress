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
    """

    def __init__(
        self,
    ):
        """
        Initialize the Qmodel.

        Constructs the base `Qmodel` by initializing the parent `nn.Module`.
        This base class does not define any layers itself but provides methods
        to manage quantized layers within derived models.

        Args:
            None
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
        """
        for module in self.modules():
            if hasattr(module, "freeze_layer"):
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

        """
        for module in self.modules():
            if hasattr(module, "unfreeze_layer"):
                module.unfreeze_layer()


    def compute_reg(self): 
        """
        Computes the total regularization loss for all applicable layers in the model.

        This method iterates through all the submodules of the model and accumulates the 
        regularization loss from layers that implement the `compute_reg_layer` method. 
        The `compute_reg_layer` method typically computes a regularization term based 
        on the quantization or weight properties of the layer.

        Regularization helps in preventing overfitting and encourages model weights to 
        adhere to certain constraints, such as being close to their quantized versions 
        in quantized models.

        Returns:
            torch.Tensor: The total regularization loss accumulated from all relevant layers.
        
        Note:
            - Only submodules with a `compute_reg_layer` method will contribute to the 
            regularization loss.
            - The method assumes that each layer's `compute_reg_layer` returns a scalar 
            tensor representing the loss for that layer.

        """
        reg_loss = torch.tensor([0.0]).to(next(self.parameters()).device).float()
        for module in self.modules():
            if hasattr(module, "compute_reg_layer"):
                reg_loss += module.compute_reg_layer()
        return reg_loss
    


