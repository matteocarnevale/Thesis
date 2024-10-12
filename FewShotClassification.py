import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader, Subset
import csv

# Importa il tuo modello Sequencer2D
from sequencer.models.two_dim_sequencer import Sequencer2D

# Funzione per congelare i layer del modello
def freeze_model_backbone(model):
    for name, param in model.named_parameters():
        # Congela tutti i parametri del backbone
        if "fc" not in name and "classifier" not in name:
            param.requires_grad = False

# Carica il dataset e prepara la versione few-shot
def load_few_shot_dataset(dataset, num_examples_per_class=5):
    class_counts = [0] * len(dataset.classes)
    few_shot_indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < num_examples_per_class:
            few_shot_indices.append(idx)
            class_counts[label] += 1

        if sum(class_counts) >= num_examples_per_class * len(dataset.classes):
            break

    few_shot_dataset = Subset(dataset, few_shot_indices)
    return few_shot_dataset

# Funzione per salvare i log del training in un CSV
def save_training_logs(log_file_path, epoch, loss, dataset_name):
    # Se il file non esiste, crea il file CSV con le intestazioni
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Epoch', 'Loss', 'Dataset'])

    # Aggiungi i log al CSV
    with open(log_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, loss, dataset_name])

# Training del modello con few-shot learning e salvataggio dei log
def few_shot_training(model, train_loader, num_classes, dataset_name, epochs=10, learning_rate=0.001, log_file_path='few_shot_training_log.csv'):
    # Definisci un nuovo layer fully connected per adattarsi alle nuove classi
    model.fc = nn.Linear(model.embed_dims[-1], num_classes)  # Adatta output alle nuove classi

    # Usa un ottimizzatore come Adam e una funzione di perdita
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Ottimizzazione
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calcola la loss media
        avg_loss = running_loss / len(train_loader)
        
        # Stampa del progresso per ogni epoca
        print(f"Epoca {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Salva i log nel CSV
        save_training_logs(log_file_path, epoch + 1, avg_loss, dataset_name)

    print(f"Fine del training few-shot su {dataset_name}")

# Main script per few-shot learning
if __name__ == "__main__":
    # Percorso del modello pre-addestrato
    model_path = 'path/to/sequencer2d_pretrained.pth'

    # Carica il modello Sequencer2D
    model = Sequencer2D(num_classes=1000)  # Specifica i parametri per il modello
    model.load_state_dict(torch.load(model_path))

    # Congela il backbone del modello
    freeze_model_backbone(model)

    # Dataset 1: CIFAR-100
    print("\n--- Training Few-shot CIFAR-100 ---\n")
    transform_cifar100 = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensiona le immagini CIFAR-100 a 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar100_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar100)
    few_shot_cifar100 = load_few_shot_dataset(cifar100_dataset, num_examples_per_class=5)
    few_shot_loader_cifar100 = DataLoader(few_shot_cifar100, batch_size=8, shuffle=True, num_workers=4)

    # Adatta il numero di classi del dataset few-shot
    num_classes_cifar100 = len(few_shot_cifar100.dataset.classes)

    # Trasferisci il modello su GPU, se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fine-tuning del modello con few-shot learning su CIFAR-100
    few_shot_training(model, few_shot_loader_cifar100, num_classes_cifar100, dataset_name="CIFAR-100", epochs=10, learning_rate=0.001, log_file_path='few_shot_training_log.csv')

    # Salva il modello fine-tunato per CIFAR-100
    torch.save(model.state_dict(), 'sequencer2d_few_shot_cifar100.pth')
    print("Modello fine-tunato su CIFAR-100 salvato come 'sequencer2d_few_shot_cifar100.pth'")

    # Dataset 2: Stanford Cars
    print("\n--- Training Few-shot Stanford Cars ---\n")
    transform_cars = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensiona le immagini Cars a 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cars_dataset = datasets.StanfordCars(root='./data', split='train', download=True, transform=transform_cars)
    few_shot_cars = load_few_shot_dataset(cars_dataset, num_examples_per_class=5)
    few_shot_loader_cars = DataLoader(few_shot_cars, batch_size=8, shuffle=True, num_workers=4)

    # Adatta il numero di classi del dataset few-shot
    num_classes_cars = len(few_shot_cars.dataset.classes)

    # Fine-tuning del modello con few-shot learning su Stanford Cars
    few_shot_training(model, few_shot_loader_cars, num_classes_cars, dataset_name="Stanford Cars", epochs=10, learning_rate=0.001, log_file_path='few_shot_training_log.csv')

    # Salva il modello fine-tunato per Stanford Cars
    torch.save(model.state_dict(), 'sequencer2d_few_shot_cars.pth')
    print("Modello fine-tunato su Stanford Cars salvato come 'sequencer2d_few_shot_cars.pth'")
