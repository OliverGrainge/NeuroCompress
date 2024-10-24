import os
import sys

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from NeuroPress.layers import LINEAR_LAYERS
from NeuroPress.models import MLP
from NeuroPress.trainers import ClassificationTrainer
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    batch_size = 256
    hidden_size = 1024
    num_layers = 5
    lr = 0.001
    max_epochs = 10
    accelerator = "gpu"

    for qlayer in LINEAR_LAYERS:
        logger = TensorBoardLogger(
            "tb_logs", name=f"MLP_Layer-{qlayer(12, 12).__repr__()}"
        )

        # Adjust the input size for CIFAR-10 (32x32 images)
        model = MLP(
            qlayer,
            input_size=32 * 32 * 3,  # CIFAR-10 has 3 color channels
            hidden_size=hidden_size,
            num_classes=10,
            num_layers=num_layers,
        )

        module = ClassificationTrainer(model=model, lr=lr, reg=1e-4)

        trainer = pl.Trainer(
            accelerator=accelerator,
            precision="32",
            max_epochs=max_epochs,
            logger=logger,
            log_every_n_steps=2,
        )

        # Adjust the transforms for CIFAR-10 normalization
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
                    std=[0.2470, 0.2435, 0.2616],  # CIFAR-10 std
                ),
                transforms.Lambda(
                    lambda x: x.view(-1)
                ),  # Flatten the image to match input_size
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="data", train=False, transform=transform, download=True
        )

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False
        )

        trainer.fit(module, train_loader, val_loader)

        model.freeze()
        trainer.test(module, val_loader)


if __name__ == "__main__":
    main()
