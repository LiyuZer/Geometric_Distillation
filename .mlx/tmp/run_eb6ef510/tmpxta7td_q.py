import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models.vision_transformer import EncoderBlock
from torchvision import transforms
from torch.optim import Adam, AdamW
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from model import ModifiedViTB32, FactorizedViTB32, FactorizedMultiheadAttention, FactorizedMLPBlock, replace_encoder_with_factorized
import json
import sys

SETUP = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_hook(container):
    def hook(module, input, output):
        container.append(output)
    return hook

def setup_base_model():
    print('Starting base model training...')
    folder_path = './data/setup_base_model'
    os.makedirs(folder_path, exist_ok=True)
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = ModifiedViTB32(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(dataloader)
        train_acc = 100 * correct / total
        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_acc)
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_dataloader:
                v_inputs, v_labels = (v_inputs.to(device), v_labels.to(device))
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_labels)
                v_running_loss += v_loss.item()
                _, v_predicted = torch.max(v_outputs.data, 1)
                v_total += v_labels.size(0)
                v_correct += (v_predicted == v_labels).sum().item()
        v_epoch_loss = v_running_loss / len(val_dataloader)
        v_epoch_acc = 100 * v_correct / v_total
        val_losses.append(v_epoch_loss)
        val_accuracies.append(v_epoch_acc)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, Val Loss: {v_epoch_loss:.4f}, Val Accuracy: {v_epoch_acc:.2f}%')
    print('Training complete.')
    epochs = list(range(1, num_epochs + 1))
    with open(f'{folder_path}/training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
        for e, l, a, vl, va in zip(epochs, epoch_losses, epoch_accuracies, val_losses, val_accuracies):
            writer.writerow([e, l, a, vl, va])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, epoch_losses, marker='o', label='Train')
    ax[0].plot(epochs, val_losses, marker='o', color='orange', label='Val')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(epochs, epoch_accuracies, marker='o', label='Train')
    ax[1].plot(epochs, val_accuracies, marker='o', color='orange', label='Val')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f'{folder_path}/training_metrics.png')
    torch.save(model.state_dict(), f'{folder_path}/vit_cifar100_state_dict.pth')
    return {'teacher': model, 'final_val_accuracy': val_accuracies[-1], 'torch_seed': 42, 'random_seed': 42, 'seed': 42}

def wrapper_setup_base_model():
    try:
        result = setup_base_model()
        print(json.dumps({"result": result}))
    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)

wrapper_setup_base_model()