import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import NeuroPress.QLayers as Q
from NeuroPress import postquantize
from NeuroPress.Utils import get_device

torch.manual_seed(42)

# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_sizes = [128, 64]  # hidden layer sizes of the network
output_size = 10  # 10 classes for the digits 0-9
batch_size = 512  # You can modify this as needed
epochs = 1  # Number of training epochs
learning_rate = 0.01  # learning rate
device = get_device()  # Setting the device

qlayer = Q.LinearWTA16  # qunatized layer example
layer = nn.Linear


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = layer(input_size, hidden_sizes[0])
        self.ln_1 = nn.LayerNorm(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = layer(hidden_sizes[0], hidden_sizes[1])
        self.ln_2 = nn.LayerNorm(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = layer(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.relu1(self.ln_1(self.fc1(x)))

        x = self.relu2(self.ln_2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

Qmodel = MLP().to(device)
Qcriterion = nn.CrossEntropyLoss()
Qoptimizer = optim.SGD(Qmodel.parameters(), lr=learning_rate)


# Training the model
def train_model(model, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
    return model


# Evaluating the model
def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as ttest:
            for data, target in ttest:
                ttest.set_description("Test Set")
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")


train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel, Qcriterion)
postquantize(Qmodel, qlinear=qlayer)
evaluate_model(Qmodel, Qcriterion)
Qoptimizer = optim.SGD(Qmodel.parameters(), lr=learning_rate)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel, Qcriterion)
