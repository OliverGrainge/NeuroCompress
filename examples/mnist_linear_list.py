import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from colorama import Fore, Style, init  # For colorful output

# Initialize colorama for colored output
init(autoreset=True)

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
num_workers=8
device = get_device()  # Setting the device

qlayer = Q.LinearW8A8  # Quantized layer example
layer = nn.Linear


# MLP model definition
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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

Qmodel = MLP().to(device)
Qcriterion = nn.CrossEntropyLoss()
Qoptimizer = optim.SGD(Qmodel.parameters(), lr=learning_rate)


# Colorful print functions
def print_info(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)


def print_warning(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)


def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)


# Training the model
def train_model(model, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

                accuracy = 100.0 * correct / total
                tepoch.set_postfix(loss=total_loss/(tepoch.n+1), accuracy=accuracy)

    return model


# Evaluating the model and return accuracy
def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as ttest:
            for data, target in ttest:
                ttest.set_description("Evaluating")
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / total
    return test_loss / total, accuracy


if __name__ == "__main__":
    layer_map = {}
    for name, layer in Qmodel.named_modules():
        if "fc3" in name: 
            layer_map[layer] = Q.LinearW8A16

    print(layer_map)
    # Dictionary to store accuracies for different models
    results = {}

    # Training and evaluating the fully precision model
    print_info("Training Fully Precision Model...")
    train_model(Qmodel, Qoptimizer, Qcriterion)
    print_info("Evaluating Fully Precision Model...")
    loss, accuracy = evaluate_model(Qmodel, Qcriterion)
    results["Fully Precision"] = accuracy

    # Post quantizing and evaluating the model
    print_warning("Post Quantizing Model...")
    postquantize(Qmodel, layer_map=layer_map)
    print(Qmodel)
    print_info("Evaluating Post Quantized Model...")
    loss, accuracy = evaluate_model(Qmodel, Qcriterion)
    results["Post Quantized"] = accuracy

    # Retraining with QAT and evaluating the model
    print_info("Retraining with QAT (Quantization Aware Training)...")
    Qoptimizer = optim.SGD(Qmodel.parameters(), lr=learning_rate)
    train_model(Qmodel, Qoptimizer, Qcriterion)
    print_info("Evaluating QAT Trained Model...")
    loss, accuracy = evaluate_model(Qmodel, Qcriterion)
    results["QAT Trained"] = accuracy

    # Print summary of accuracies
    print_info("\n\n ====================== Summary of Model Accuracies: =======================")
    for model_type, acc in results.items():
        print(f"                    {model_type} Model Accuracy: {acc:.2f}%")
    print_info("==============================================================================")