import os
import sys

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pytorch_lightning.loggers import TensorBoardLogger

from NeuroPress.layers import BitLinear
from NeuroPress.models import MLP
from NeuroPress.trainers import ClassificationTrainer


def main():
    batch_size = 64
    hidden_size = 128
    num_layers = 5
    lr = 0.001
    qlayer = BitLinear

    logger = TensorBoardLogger("tb_logs", name=f"MLP_Layer-{qlayer.__repr__()}")

    model = MLP(
        qlayer,
        input_size=28 * 28,
        hidden_size=hidden_size,
        num_classes=10,
        num_layers=num_layers,
    )

    module = ClassificationTrainer(model=model, lr=lr)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="32",
        max_epochs=1,
        logger=logger,
        log_every_n_steps=2,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    val_dataset = datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    trainer.fit(module, train_loader, val_loader)

    model.freeze()
    trainer.test(module, val_loader)


if __name__ == "__main__":
    main()
