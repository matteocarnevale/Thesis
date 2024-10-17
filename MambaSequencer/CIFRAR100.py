import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from models.layers import Sequencer2DBlock, PatchEmbed, Downsample2D, MAMBA2_2D, MAMBA2D
from models.MambaSequencer import Sequencer2D
from mamba_ssm import Mamba, Mamba2
from timm.models.layers import lecun_normal_, Mlp
from functools import partial

# Define constants
BATCH = 256
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
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
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
layers = [1, 1, 2, 2]
patch_sizes = [4, 2, 1, 1]
embed_dims = [96, 96, 192, 192]
mlp_ratios = [2.0, 2.0, 2.0, 2.0]
drop_rate = 0.1
drop_path_rate = 0.1
num_rnn_layers = 1
union = "cat"
with_fc = True
nlhb = False
stem_norm = False
state_expansion = [16, 32, 64, 128]
block_expansion = [2, 2, 2, 2]
conv_dim = [4, 4, 4, 4]

# Instantiate the model
def create_model(model=MAMBA2D):
    if model == MAMBA2D:
        return Sequencer2D(
            num_classes=num_classes,
            img_size=img_size,
            in_chans=in_chans,
            layers=layers,
            patch_sizes=patch_sizes,
            embed_dims=embed_dims,
            mlp_ratios=mlp_ratios,
            block_layer=Sequencer2DBlock,
            rnn_layer=MAMBA2D,
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
            conv_dim=conv_dim
        ).to(device)

def compute_metrics(preds, labels):
    pred_classes = torch.argmax(preds, dim=1)
    accuracy = (pred_classes == labels).float().mean().item()
    return accuracy * 100, pred_classes

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    main_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    gpu_usages = []
    epoch_times = []
    all_labels = []
    all_preds = []
    
    total_training_time = time.time()
    
    for epoch in range(num_epochs):
        # Start timing
        epoch_start_time = time.time()

        # Training Phase
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        scaler = GradScaler()
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast():
                preds = model(images)
                loss = nn.functional.cross_entropy(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            accuracy, _ = compute_metrics(preds, targets)
            total_accuracy += accuracy

            progress_bar.set_postfix(loss=f'{total_loss / (progress_bar.n + 1):.4f}', accuracy=f'{total_accuracy / (progress_bar.n + 1):.4f}%')

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        # Validation Phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Validating Epoch {epoch+1}')
            for images, targets in progress_bar:
                images, targets = images.to(device), targets.to(device)
                with autocast():
                    preds = model(images)
                    loss = nn.functional.cross_entropy(preds, targets)

                val_loss += loss.item()
                accuracy, pred_classes = compute_metrics(preds, targets)
                val_accuracy += accuracy
                
                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(pred_classes.cpu().numpy())

                progress_bar.set_postfix(loss=f'{val_loss / (progress_bar.n + 1):.4f}', accuracy=f'{val_accuracy / (progress_bar.n + 1):.4f}%')

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # GPU usage
        gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
        gpu_usages.append(gpu_usage)

        # End timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.2f}%, GPU Usage: {gpu_usage:.2f} GB')

        main_scheduler.step()
        scheduler.step(avg_val_loss)

    total_training_time = time.time() - total_training_time
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return train_losses, val_losses, train_accuracies, val_accuracies, gpu_usages, epoch_times, total_training_time, confusion_mat

# Set the number of epochs
num_epochs = 50

# Train the model and collect results
MAMBA_model = create_model(MAMBA2D)
pytorch_total_params = sum(p.numel() for p in MAMBA_model.parameters())
print(f'Total parameters in MAMBA model: {pytorch_total_params}')

print("Training MAMBA model...")
train_loss, val_loss, train_acc, val_acc, gpu_usages, epoch_times, total_training_time, confusion_mat = train_model(MAMBA_model, train_loader, val_loader, device, num_epochs)

# Save the model to calculate size
model_path = os.path.join("/media/HDD/carnevale/results", "MAMBA_model.pth")
torch.save(MAMBA_model.state_dict(), model_path)
model_size = os.path.getsize(model_path) / (1024 ** 2)  # Size in MB

# Save results to CSV
results_dir = "/media/HDD/carnevale/results"
os.makedirs(results_dir, exist_ok=True)

def save_results_to_csv(train_acc, val_acc, train_loss, val_loss, num_epochs, pytorch_total_params, gpu_usages, epoch_times, total_training_time, model_size, confusion_mat, save_path):
    data = {
        'Epoch': range(1, num_epochs + 1),
        'Train Accuracy': train_acc,
        'Validation Accuracy': val_acc,
        'Train Loss': train_loss,
        'Validation Loss': val_loss,
        'Parameters': [pytorch_total_params] * num_epochs,
        'GPU Usage (GB)': gpu_usages,
        'Epoch Time (s)': epoch_times,
    }

    df = pd.DataFrame(data)
    csv_file_path = os.path.join(save_path, "training_results_MAMBA.csv")
    df.to_csv(csv_file_path, index=False)

    # Append additional metrics
    with open(csv_file_path, 'a') as f:
        f.write(f"\nTotal Training Time (s),{total_training_time:.2f}\n")
        f.write(f"Model Size (MB),{model_size:.2f}\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, confusion_mat, fmt='%d', delimiter=",")

    print(f"Results saved to: {csv_file_path}")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Model Size: {model_size:.2f} MB")
    print("Confusion Matrix:")
    print(confusion_mat)

# Save the results
save_results_to_csv(train_acc, val_acc, train_loss, val_loss, num_epochs, pytorch_total_params, gpu_usages, epoch_times, total_training_time, model_size, confusion_mat, results_dir)