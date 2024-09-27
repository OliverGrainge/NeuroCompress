import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
torch.autograd.set_detect_anomaly(True)


torch.set_float32_matmul_precision("medium")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from NeuroPress.QLayers.Ternary import LinearWTA8, PLinearWTA8, P2LinearWTA8, P3LinearWTA8
from NeuroPress import postquantize
from NeuroPress.Utils import RMSNorm

#qlayer = nn.Linear
#qlayer = LinearWTA8
qlayer = P3LinearWTA8


# Define the ViT LightningModule
class ViTLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lambda_=0.02):
        super(ViTLightningModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.lambda_scale = lambda_
        self.lambda_ = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("NaN or Inf detected in outputs")
        loss = self.criterion(outputs, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf detected in loss")
        reg_loss = self.model.compute_reg_loss()
        print(self.lambda_ * reg_loss.item(), loss.item())
        acc = accuracy(outputs, labels, task="multiclass", num_classes=10)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("reg_loss", self.lambda_ * reg_loss)
        loss = loss + self.lambda_ * reg_loss
        self.log("total loss", loss)
        self.training_step_end()
        return loss
    
    def training_step_end(self):
        max_steps = self.trainer.max_epochs * 196 
        t = self.trainer.global_step/max_steps
        t = torch.tensor((t * 8) - 4)
        def sigmoid(x): 
            return 1/(1 + torch.exp(-x))
        
        for module in self.model.modules(): 
            if isinstance(module, nn.Linear) and hasattr(module, 'q_lambda'):
                module.q_lambda = sigmoid(t)
        self.lambda_ = sigmoid(t) * self.lambda_scale



    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        return optimizer



# Model definition (same as before)
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
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = qlayer(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(qlayer(inner_dim, dim), nn.Dropout(dropout))

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
        i = 0
        for attn, ff in self.layers:
            i += 1
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
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
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
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.mlp_head(x)
    

    def compute_reg_loss(self): 
        # Initialize total_weight_reg on the correct device
        total_weight_reg = torch.zeros(1, device=next(self.parameters()).device)
        
        # Iterate through modules to compute total weight regularization
        for module in self.modules(): 
            #print(module)
            if isinstance(module, nn.Linear) and hasattr(module, 'compute_reg'):
                #print(total_weight_reg)
                total_weight_reg = total_weight_reg + module.compute_reg()
        
        return total_weight_reg
            




# Transforms and data loaders
val_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

# Initialize ViT model and LightningModule
model = ViT(
    image_size=224,        # Smaller image size for reduced complexity
    patch_size=16,         # More patches for better granularity
    dim=384,               # Reduced embedding dimension
    depth=4,               # Fewer transformer layers
    heads=4,               # Fewer attention heads
    mlp_dim=1536,          # MLP layer dimension (4x dim)
    dropout=0.1,           # Regularization via dropout
    emb_dropout=0.1,       # Dropout for the embedding layer
    channels=3,            # RGB images
    dim_head=96,           # Dimension of each attention head
    num_classes=10
)



vit_lightning_model = ViTLightningModule(model, lr=1e-4, lambda_=0.00001)
# vit_lightning_model = torch.compile(vit_lightning_model)

# Set up TensorBoard logger
q = qlayer(10, 10)
logger = TensorBoardLogger("tb_logs", name=q.__repr__() + f"_lambda-{vit_lightning_model.lambda_}")
#logger = TensorBoardLogger("tb_logs", name="bitnet")



# Set up PyTorch Lightning trainer
trainer = pl.Trainer(
    max_epochs=30,                      # Reduce the number of epochs (usually less than 10 is enough for this test)
    logger=logger,
    precision="32",              # Mixed precision for faster training
    num_sanity_val_steps=0,              # Skip validation sanity checks
    #limit_train_batches=1,               # Limit training to a single batch
    #limit_val_batches=1,                 # Limit validation to a single batch
    #overfit_batches=1,                    # Overfit to a single batch of data
    #log_every_n_steps=1

)

# Train and validate the model
trainer.fit(vit_lightning_model, train_loader, test_loader)
