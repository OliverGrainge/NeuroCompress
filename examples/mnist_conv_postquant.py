import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys 
import torch.nn.functional as F
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import NeuroPress.QLayers as Q
from NeuroPress.Utils import get_device

# Hyperparameters
batch_size = 512
epochs = 1
learning_rate = 0.01

# Setting the device
device = get_device()

qconv = Q.Conv2dW4A8
qlinear = Q.LinearW8A16

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


def postquantize(model: nn.Module):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            has_bias = False if layer.bias is None else True
            new_layer = qlinear(layer.in_features, layer.out_features, has_bias)
            new_layer.setup(layer)
            setattr(model, name, new_layer)
        elif isinstance(layer, nn.Conv2d):
            has_bias = False if layer.bias is None else True
            new_layer = qconv(layer.in_channels,
                              layer.out_channels, 
                              layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              dilation=layer.dilation,
                              groups=layer.groups,
                              bias=has_bias)
            new_layer.setup(layer)
            setattr(model, name, new_layer)



# Data loading with transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

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

# Function to evaluate the model
def evaluate_model(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Train and evaluate the model
train_model(model, optimizer, criterion)
evaluate_model(model)
postquantize(model)
print(model)
evaluate_model(model)
#train_model(model, optimizer, criterion)
#evaluate_model(model)