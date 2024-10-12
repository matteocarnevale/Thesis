import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
from tqdm import tqdm

from mamba_ssm import Mamba, Mamba2

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models.layers import lecun_normal_, Mlp
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.registry import register_model

from models.layers import Sequencer2DBlock, PatchEmbed, LSTM2D, GRU2D, RNN2D, Downsample2D, MAMBA2_2D, MAMBA2D
from models.two_dim_sequencer import Sequencer2D
from models.layers import MAMBA2_2D


BATCH = 512
IMG_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 4

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.backends.cudnn.benchmark = True

# Data augmentation and normalization for CIFAR100
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
    }

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transforms['train'])
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)


# Define the model hyperparameters
num_classes = 100
img_size = IMG_SIZE
in_chans = 3
layers = [1, 1, 1, 1]
patch_sizes = [1, 2, 1, 1]
embed_dims=[48, 48, 96, 96]
hidden_sizes=[12, 12, 24, 24]
mlp_ratios = [3.0, 3.0, 3.0, 3.0]
drop_rate = 0.0
drop_path_rate = 0.1
num_rnn_layers = 1
union="cat",   
with_fc=True
nlhb=False
stem_norm=False
state_expansion = [8, 16, 16, 32]  # Example state expansion
block_expansion = [2, 2, 2, 2]      # Example block expansion
conv_dim = [4, 4, 4, 4]             # Example conv dimensions
headdim=[24, 24, 48, 48]           


# Instantiate different models
def create_model(layer=MAMBA2D):
    if (layer==LSTM2D):
        layers = [1, 2, 3, 2]
    else:
        layers = [1, 1, 1, 1]

    return Sequencer2D(
        num_classes=num_classes,
        img_size=img_size,
        in_chans=in_chans,
        layers=layers,
        patch_sizes=patch_sizes,
        embed_dims=embed_dims,
        hidden_sizes=hidden_sizes,
        mlp_ratios=mlp_ratios,
        block_layer=Sequencer2DBlock,
        rnn_layer=layer,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        num_rnn_layers=num_rnn_layers,
        bidirectional=True,
        union="cat",
        with_fc=with_fc,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        nlhb=nlhb,
        stem_norm=stem_norm,
        state_expansion=state_expansion,
        block_expansion=block_expansion,
        conv_dim=conv_dim,
        headdim=headdim
    ).to(device)

def compute_metrics(preds, labels):
    pred_classes = torch.argmax(preds, dim=1)
    accuracy = (pred_classes == labels).float().mean().item()
    return accuracy * 100

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
   
    num_batches = len(data_loader)
    batch_idx = 0

    # Mixed precision gradient scaler
    scaler = torch.amp.GradScaler("cuda")

    progress_bar = tqdm(data_loader, desc='Training')
    for images, targets in progress_bar:

        images = images.to(device)
        targets = targets.to(device) 

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss = nn.functional.cross_entropy(preds, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient update with scaled optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        accuracy = compute_metrics(preds, targets)
        
        total_accuracy += accuracy
        progress_bar.set_postfix(loss=f'{total_loss / (progress_bar.n + 1):.4f}', accuracy=f'{total_accuracy / (progress_bar.n + 1):.4f}%')
        
        batch_idx += 1

    
    return total_loss / num_batches, total_accuracy / num_batches


def validate(model, val_loader, device, epoch):
    model.eval()
    val_loss, val_accuracy = 0.0, 0.0
   
    num_batches = len(val_loader)
    progress_bar = tqdm(val_loader, desc='Validating')
    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass with autocast for mixed precision
            with torch.amp.autocast("cuda"):
                preds = model(images)
                loss = nn.functional.cross_entropy(preds, targets)
            
            val_loss += loss.item()
            accuracy = compute_metrics(preds, targets)
        
            val_accuracy += accuracy
            progress_bar.set_postfix(loss=f'{val_loss / (progress_bar.n + 1):.4f}', accuracy=f'{val_accuracy / (progress_bar.n + 1):.4f}%')

    return val_loss / num_batches, val_accuracy / num_batches


def train_model(model, train_loader, test_loader, device, num_epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    main_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_accuracy = validate(model, val_loader, device, epoch)
        
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        
        main_scheduler.step()
        scheduler.step(val_loss)

    return train_losses, val_losses, train_accuracies, val_accuracies

# Train the models and collect results
MAMBA_model = create_model(MAMBA2D)

# Set the number of epochs
num_epochs = 50

# Train bidirectional MAMBA
print("Training Bidirectional MAMBA block...")

train_loss_bi, val_loss_bi, train_acc_bi, val_acc_bi = train_model(MAMBA_model, train_loader, val_loader, device, num_epochs)

del MAMBA_model

torch.cuda.empty_cache()

LSTM_model = create_model(LSTM2D)

# Train unidirectional MAMBA
print("Training Bidirectional LSTM2D block...")
train_loss_uni, val_loss_uni, train_acc_uni, val_acc_uni = train_model(LSTM_model, train_loader, val_loader, device, num_epochs)


def plot_results(train_acc_bi, val_acc_bi, train_acc_uni, val_acc_uni, 
                 train_loss_bi, val_loss_bi, train_loss_uni, val_loss_uni, 
                 num_epochs, save_path=results_dir):
    epochs = range(1, num_epochs + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_bi, 'bo-', label='Train Accuracy (MAMBA)')
    plt.plot(epochs, val_acc_bi, 'b--', label='Val Accuracy (MAMBA)')
    plt.plot(epochs, train_acc_uni, 'ro-', label='Train Accuracy (LSTM)')
    plt.plot(epochs, val_acc_uni, 'r--', label='Val Accuracy (LSTM)')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_bi, 'bo-', label='Train Loss (MAMBA)')
    plt.plot(epochs, val_loss_bi, 'b--', label='Val Loss (MAMBA)')
    plt.plot(epochs, train_loss_uni, 'ro-', label='Train Loss (LSTM)')
    plt.plot(epochs, val_loss_uni, 'r--', label='Val Loss (LSTM)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PNG file in the specified directory
    accuracy_loss_graph_path = os.path.join(save_path, "accuracy_loss_graph.png")
    plt.savefig(accuracy_loss_graph_path)
    print(f"Graph saved at: {accuracy_loss_graph_path}")
    plt.show()

# Plot the results of the training
results_dir = "/media/HDD/carnevale/results"
os.makedirs(results_dir, exist_ok=True)
plot_results(train_acc_bi, val_acc_bi, train_acc_uni, val_acc_uni, train_loss_bi, val_loss_bi, train_loss_uni, val_loss_uni)
