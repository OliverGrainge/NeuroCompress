import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from einops.layers.torch import Rearrange
from torchmetrics.functional import accuracy
import os 
import sys
torch.set_float32_matmul_precision('medium')
                                   
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from NeuroPress.Utils import RMSNorm
from NeuroPress.QLayers.Ternary import LinearWTA8


qlayer = LinearWTA8


# Define the ViT LightningModule
class ViTLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super(ViTLightningModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=10)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=10)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

# Model definition (same as before)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            qlayer(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            qlayer(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = qlayer(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            qlayer(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
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
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

# Transforms and data loaders
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=312, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=312, shuffle=False, num_workers=16)

# Initialize ViT model and LightningModule
model = ViT(
    image_size=224,   # Keep this to ensure compatibility with common datasets
    patch_size=16,    # You could increase to 32 for even faster training
    num_classes=10,   # Depends on your dataset (10 for CIFAR-10)
    dim=256,          # Reduced embedding dimension for faster training
    depth=4,          # Fewer transformer layers
    heads=4,          # Fewer attention heads
    mlp_dim=512,      # Smaller feedforward network dimension
    dropout=0.1,      # Keep some dropout for regularization
    emb_dropout=0.1   # Embedding dropout
)

vit_lightning_model = ViTLightningModule(model)
#vit_lightning_model = torch.compile(vit_lightning_model)

# Set up TensorBoard logger
logger = TensorBoardLogger("tb_logs", name=f"ViT_CIFAR10_ternary")

# Set up PyTorch Lightning trainer
trainer = pl.Trainer(
    max_epochs=50,
    logger=logger,
    precision='32'
)

# Train and validate the model
trainer.fit(vit_lightning_model, train_loader, test_loader)
