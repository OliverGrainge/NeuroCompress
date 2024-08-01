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
from NeuroPress.QLayers import LinearW1A16, StochasticLinearW1A16, LinearW1A1, StochasticLinearW1A1


# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_sizes = [6144, 6144, 6144, 6144]
hidden_sizes = [512, 512, 512, 512]
output_size = 10   # 10 classes for the digits 0-9
batch_size = 512   # You can modify this as needed
epochs = 1       # Number of training epochs
learning_rate = 0.01

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps"
    

class BinaryMLP(nn.Module):
    def __init__(self):
        super(BinaryMLP, self).__init__()
        self.act = nn.Hardtanh()
        self.fc1 = LinearW1A1(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = LinearW1A1(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = LinearW1A1(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2], hidden_sizes[3])

        self.fc4 = LinearW1A1(hidden_sizes[3], output_size)
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image tensor
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.fc4(x)
        x = self.drop(x)
        return x



# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss criterion and optimizer
model = BinaryMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

Qmodel = BinaryMLP().to(device)
Qcriterion = nn.CrossEntropyLoss()
Qoptimizer = optim.SGD(Qmodel.parameters(), lr=learning_rate)

def quantize_model(qmodel, model):
    for name, module in model.named_modules():
        for qname, qmodule in qmodel.named_modules():
            if name == qname:
                qmodule.setup(module)

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
def evaluate_model(model):
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


def analyse_weights(model):
    weight = model.fc1.weight.data.cpu().detach().clone().numpy().flatten()
    import matplotlib.pyplot as plt
    plt.hist(weight)
    plt.show()

train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)
train_model(Qmodel, Qoptimizer, Qcriterion)
evaluate_model(Qmodel)


