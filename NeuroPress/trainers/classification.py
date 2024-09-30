"""
Module: classification

This module defines the `ClassificationTrainer` class, a PyTorch Lightning module designed
for training, validating, and testing classification models. The `ClassificationTrainer`
encapsulates the model, loss function, optimizer configuration, and the necessary steps
for training and evaluation, streamlining the workflow for classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

class ClassificationTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for Classification Tasks.

    The `ClassificationTrainer` class encapsulates a classification model along with its
    training, validation, and testing routines. It leverages PyTorch Lightning's
    abstractions to simplify the training loop and manage optimization.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        lr (float): Learning rate for the optimizer.
        criterion (nn.CrossEntropyLoss): Loss function for classification.

    Args:
        model (nn.Module): The neural network model to be trained.
        lr (float, optional): Learning rate for the optimizer. Defaults to `0.001`.

    Example:
        ```python
        import torch.nn as nn
        from classification_trainer import ClassificationTrainer

        # Define a simple neural network
        class SimpleNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super(SimpleNet, self).__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                return self.fc(x)

        # Initialize the model
        model = SimpleNet(input_size=784, num_classes=10)

        # Initialize the trainer
        trainer = ClassificationTrainer(model=model, lr=0.001)

        # Train the model using PyTorch Lightning's Trainer
        from pytorch_lightning import Trainer
        pl_trainer = Trainer(max_epochs=10)
        pl_trainer.fit(trainer, train_dataloader, val_dataloader)
        ```
    """
    def __init__(self, model: nn.Module, lr: float = 0.001):
        """
        Initialize the ClassificationTrainer module.

        Constructs the `ClassificationTrainer` by initializing the parent `LightningModule`,
        setting up the model, learning rate, and loss function.

        Args:
            model (nn.Module): The neural network model to be trained.
            lr (float, optional): Learning rate for the optimizer. Defaults to `0.001`.

        Example:
            ```python
            # Assuming `model` is an instance of nn.Module
            trainer = ClassificationTrainer(model=model, lr=0.001)
            ```
        """
        super(ClassificationTrainer, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Executes the forward pass of the encapsulated model on the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_features)`.

        Returns:
            torch.Tensor: Output logits tensor of shape `(batch_size, num_classes)`.

        Example:
            ```python
            # Assuming `trainer` is an instance of ClassificationTrainer
            input_tensor = torch.randn(64, 784)  # Batch of 64 samples
            logits = trainer(input_tensor)
            print(logits.shape)  # Output: torch.Size([64, 10])
            ```
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Sets up the optimizer used for updating the model's parameters during training.

        Returns:
            torch.optim.Optimizer: The configured Adam optimizer.

        Example:
            ```python
            # This method is called automatically by PyTorch Lightning
            optimizer = trainer.configure_optimizers()
            ```
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Processes a batch of training data, computes the loss, and logs the training loss.

        Args:
            batch (tuple): A tuple containing input data and target labels `(x, y)`.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.

        Example:
            ```python
            # This method is called automatically by PyTorch Lightning during training
            loss = trainer.training_step(batch, batch_idx)
            ```
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Processes a batch of validation data, computes the validation loss and accuracy,
        and logs them.

        Args:
            batch (tuple): A tuple containing input data and target labels `(x, y)`.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed validation loss for the batch.

        Example:
            ```python
            # This method is called automatically by PyTorch Lightning during validation
            val_loss = trainer.validation_step(batch, batch_idx)
            ```
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.

        Processes a batch of test data, computes the test loss and accuracy,
        and logs them.

        Args:
            batch (tuple): A tuple containing input data and target labels `(x, y)`.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed test loss for the batch.

        Example:
            ```python
            # This method is called automatically by PyTorch Lightning during testing
            test_loss = trainer.test_step(batch, batch_idx)
            ```
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', val_loss)
        self.log('test_acc', acc)
        return val_loss

