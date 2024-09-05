import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from colorama import Fore, Style, init  # For colorful output

# Initialize colorama for colored output
init(autoreset=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import NeuroPress.QLayers as Q
from NeuroPress import postquantize
from NeuroPress.Utils import get_device

# Hyperparameters
batch_size = 225
epochs = 1
learning_rate = 0.01

# Setting the device
device = get_device()

qconv = Q.Conv2dWTA16
qlinear = Q.LinearW4A8


# Colorful print functions
def print_info(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)


def print_warning(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)


def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Adding padding
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.hardtanh = nn.Hardtanh()

        self.fc1 = nn.Linear(2304, 1024)
        self.fcbn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fcbn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.pool(self.conv1(x))))
        x = F.relu(self.bn2(self.pool(self.conv2(x))))
        x = F.relu(self.bn3(self.pool(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Data loading with transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss criterion, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Function to train the model
def train_model(model, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
    return model


# Function to evaluate the model and return accuracy
def evaluate_model(model):
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / total
    return test_loss / total, accuracy


if __name__ == "__main__":
    # Dictionary to store results
    results = {}

    # Train and evaluate the model
    print_info("Training Fully Precision Model...")
    train_model(model, optimizer, criterion)
    print_info("Evaluating Fully Precision Model...")
    loss, accuracy = evaluate_model(model)
    results["Fully Precision"] = accuracy

    # Post-quantization
    print_warning("Post Quantizing Model...")
    postquantize(model, qlinear=qlinear, qconv=qconv)
    print_info("Evaluating Post Quantized Model...")
    loss, accuracy = evaluate_model(model)
    results["Post Quantized"] = accuracy

    # Retrain and evaluate the quantized model
    print_info("Retraining Quantized Model...")
    train_model(model, optimizer, criterion)
    print_info("Evaluating QAT Trained Model...")
    loss, accuracy = evaluate_model(model)
    results["QAT Trained"] = accuracy

    # Print summary of accuracies
    print_info("\n\n ====================== Summary of Model Accuracies: =======================")
    for model_type, acc in results.items():
        print(f"                    {model_type} Model Accuracy: {acc:.2f}%")
    print_info("==============================================================================")