import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from NeuroPress.QLayers.Ternary import LinearWTA8, MyLinearWTA8
from NeuroPress.Utils import RMSNorm

qlayer = LinearWTA8


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), qlayer(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), qlayer(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = qlayer(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(qlayer(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), FeedForward(dim, mlp_dim, dropout=dropout)])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            qlayer(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = qlayer(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class StackImages:
    def __init__(self, n_images):
        self.n_images = n_images

    def __call__(self, img):
        # Generate a list of `n_images` copies of the input image
        img_stack = [img] * self.n_images
        # Stack the images along the channel dimension
        return torch.cat(img_stack, dim=0)
    
# Example transform pipeline with image stacking
n_stacked_images = 3  # For example, stack 3 images

train_transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        StackImages(n_stacked_images),  # Stack images along channel dimension
        transforms.Normalize((0.5,) * n_stacked_images, (0.5,) * n_stacked_images),  # Normalize the stacked images
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        StackImages(n_stacked_images),  # Stack images along channel dimension
        transforms.Normalize((0.5,) * n_stacked_images, (0.5,) * n_stacked_images),  # Normalize the stacked images
    ]
)



train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=6)

# Update model parameters for CIFAR-10


# Initialize the model
model = ViT(
        image_size=224,        # Smaller image size for reduced complexity
        patch_size=8,         # More patches for better granularity
        dim=128,               # Reduced embedding dimension
        depth=4,               # Fewer transformer layers
        heads=4,               # Fewer attention heads
        mlp_dim=128 * 4,          # MLP layer dimension (4x dim)
        dropout=0.1,           # Regularization via dropout
        emb_dropout=0.1,       # Dropout for the embedding layer
        channels=3,            # RGB images
        dim_head=128//4,           # Dimension of each attention head)
        num_classes=10,
)

# Loss
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train(model, train_loader, criterion, optimizer, epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"epoch: {epoch}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute running loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Test loop to evaluate the model
def test(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


# Train the model
for _ in range(100):
    train(model, train_loader, criterion, optimizer, epochs=5)
    test(model, test_loader)
