import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os 
import torch.nn.functional as F


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NeuroPress.QLayers import LinearW1A16, StochasticLinearW1A16, LinearW1A1, StochasticLinearW1A1, LinearW8A16, LinearW4A16, LinearW8A8, LinearW2A16
from NeuroPress.Utils import get_device


# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_sizes = [128, 64] # hidden layer sizes of the network 
output_size = 10   # 10 classes for the digits 0-9
batch_size = 512   # You can modify this as needed
epochs = 3       # Number of training epochs
learning_rate = 0.01 # learning rate
device = get_device() # Setting the device

qlayer = LinearW4A16 # qunatized layer example

class QuantMLP(nn.Module):
    def __init__(self):
        super(QuantMLP, self).__init__()
        self.fc1 = qlayer(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = qlayer(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = qlayer(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

Qmodel = QuantMLP().to(device)
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
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel, Qcriterion)





