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
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from model import ModifiedViTB32, FactorizedViTB32, FactorizedEncoderBlock, FactorizedMultiheadAttention, FactorizedMLPBlock, replace_encoder_with_factorized
from decorators import trial, verify, VerificationResult, setup
import csv

# Data is stored under ./data by default
# Each trial, setup etc function saves under ./data/function_name/

#+++++++++++++++++++++++++++++++++++++++++ SETUP METHOD FOR EXPERIMENTS +++++++++++++++++++++++++++++++++++++++++

# This setups the base model, fine-tuning a ViT on CIFAR-100, also sets seeds for reproducibility
# The result of this function is used by subsequent trials
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@setup()
def setup_base_model():
    print("Starting base model training...")
    # Setup the seeds for reproducibility
    torch_seed = 42
    random_seed = 42
    numpy_seed = 42

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)

    folder_path = './data/setup_base_model'
    os.makedirs(folder_path, exist_ok=True)

    # Get the transforms of the pretrained model
    # Using the weights transforms ensures we use the same normalization etc as the pretrained model
    weights = ViT_B_32_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Add augmentations for training, then  use the preprocess for normalization
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomCrop(32, padding=4),
        preprocess        
    ])

    val_transforms = transforms.Compose([
        preprocess
    ])

    # Get CIFAR-100 train and validation (test) datasets
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

    # Dataloaders
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate model
    model = ModifiedViTB32(num_classes=100).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=0.0001, weight_decay=0.01
    )

    # Track metrics
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []
    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Using tqdm
        for inputs, labels in tqdm(dataloader):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Record train metrics for this epoch
        train_loss = running_loss / len(dataloader)
        train_acc = 100 * correct / total
        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_acc)

        # Validation
        model.eval()
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_dataloader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
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

        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, '
            f'Val Loss: {v_epoch_loss:.4f}, Val Accuracy: {v_epoch_acc:.2f}%'
        )

    print('Training complete.')

    # Save metrics to CSV
    epochs = list(range(1, num_epochs + 1))
    with open(f'{folder_path}/training_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
        for e, l, a, vl, va in zip(epochs, epoch_losses, epoch_accuracies, val_losses, val_accuracies):
            writer.writerow([e, l, a, vl, va])

    # Plot and save loss/accuracy curves
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

    # Save model checkpoint to repository
    torch.save(model.state_dict(), f'{folder_path}/vit_cifar100_state_dict.pth')
    return {"final_val_accuracy": val_accuracies[-1], "Seed": {"torch_seed": torch_seed, "random_seed": random_seed, "numpy_seed": numpy_seed}}


def make_hook(container):
    def hook(module, input, output):
        container.append(output)
    return hook

#++++++++++++++++++++++++++++++++++++++++++ TRIAL METHOD FOR EXPERIMENTS +++++++++++++++++++++++++++++++++++++++++
# This runs the training of the factorized ViT with a given rank, and saves the results
@trial(order=1, values={"rank":[2, 4, 8 ,16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256], "__parallel__":13}) 
def train_factorized(rank):
    # Setup the seeds for reproducibility
    torch_seed = SETUP.get("Seed", {}).get("torch_seed", 42)
    random_seed = SETUP.get("Seed", {}).get("random_seed", 42)
    numpy_seed = SETUP.get("Seed", {}).get("numpy_seed", 42)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)

    folder_path = f'./data/train_factorized_rank_{rank}'
    os.makedirs(folder_path, exist_ok=True)

    print("Starting training with factorized MLPs in ViT...")
    # Hyperparameters (kept simple/realistic)
    batch_size = 24
    num_workers = 2
    lr = 1e-4
    num_epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Teacher (fine-tuned) model, load the state dict
    teacher_dict = torch.load(f'./data/setup_base_model/vit_cifar100_state_dict.pth', map_location=device)
    teacher = ModifiedViTB32(num_classes=100)
    teacher.load_state_dict(teacher_dict)

    # Student: pretrained ViT with factorized MLPs and 100-class head
    student = FactorizedViTB32(num_classes=100, rank=rank, default_dropout=0.0)
    print(f"Original model architecture:\n{student}\n")
    print(f"Student model with factorized MLPs architecture:\n{student}\n")

    # Parameter count for teacher and student
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher model parameters: {teacher_params}")
    print(f"Student model parameters: {student_params}")

    # Move to device
    teacher = teacher.to(device)
    student = student.to(device)

    # Freeze teacher and set to eval
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Hooks to capture intermediate outputs
    mlp_outputs = []
    factorized_mlp_outputs = []

    for name, module in teacher.named_modules():
        if isinstance(module, EncoderBlock):
            # Register hooks for the multihead attention and MLP layers
            module.self_attention.register_forward_hook(make_hook(mlp_outputs))
            module.mlp.register_forward_hook(make_hook(mlp_outputs))

    for name, module in student.named_modules():
        if isinstance(module, FactorizedEncoderBlock):
            module.self_attention.register_forward_hook(make_hook(factorized_mlp_outputs))
            module.mlp.register_forward_hook(make_hook(factorized_mlp_outputs))

    # Data
    preprocess = ViT_B_32_Weights.IMAGENET1K_V1.transforms()

    # Add augmentations for training, then  use the preprocess for normalization
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomCrop(32, padding=4),
        preprocess        
    ])

    val_transforms = transforms.Compose([
        preprocess
    ])

    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Optimization
    mse_loss = nn.MSELoss()
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # Training loop (MSE-only distillation)
    final_accuracy = 0.0
    final_loss = 0.0
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)

            # Clear previous captured outputs
            mlp_outputs.clear()
            factorized_mlp_outputs.clear()

            # Forward teacher without grad to capture teacher features
            with torch.no_grad():
                _ = teacher(images)
                teacher_logits = _

            # Forward student to capture factorized features
            student_logits = student(images)

            # Distillation loss: MSE on intermediate features + logits
            loss = 0.0
            for orig_out, fact_out in zip(mlp_outputs, factorized_mlp_outputs):
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0] # In the case of multihead attention, output is a tuple
                loss = loss + mse_loss(fact_out, orig_out.detach())
                # Cosine similarity as well
                loss = loss + 1 - nn.functional.cosine_similarity(fact_out, orig_out.detach(), dim=-1).mean()
            loss = loss + mse_loss(student_logits, teacher_logits.detach()) + (1 - nn.functional.cosine_similarity(student_logits, teacher_logits.detach(), dim=1).mean())
            denom = (len(mlp_outputs) + 1)
            loss = loss / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        final_loss = avg_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


        # Validation accuracy of the student
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                mlp_outputs.clear()
                factorized_mlp_outputs.clear()
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        final_accuracy = accuracy
        print(f'Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%')
        # Save the current epoch loss and accuracy to a log file
        with open(f'{folder_path}/vit_cifar100_factorized_{rank}_training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}\n')

    # Save student checkpoint
    torch.save(student.state_dict(), f'{folder_path}/vit_cifar100_factorized_{rank}_state_dict.pth')
    print(f'Training complete. Saved factorized student checkpoint to {folder_path}/vit_cifar100_factorized_{rank}_state_dict.pth')
    return {"final_accuracy" : final_accuracy, "final_loss": final_loss, "rank": rank}

#++++++++++++++++++++++++++++++++++++++++++ VERIFICATION METHOD FOR EXPERIMENTS +++++++++++++++++++++++++++++++++++++++++
# This verifies the results of the factorized training trials, and saves a CSV for human analysis
@verify(["train_factorized"])
def factorized_data_verification(results):
    # We have multiple trials with different ranks, for each we calculate the delta_accuracy/delta_rank and delta_loss/delta_rank
    runs = results.get("train_factorized", {})

    accuracies = []
    losses = []
    ranks = []
    for run in runs:
        if not run.get("success"):
            ok = False
            break
        res = run.get("result", {})
        final_accuracy = res.get("final_accuracy", [])
        final_loss = res.get("final_loss", [])
        rank = res.get("rank", None)

        accuracies.append(final_accuracy)
        losses.append(final_loss)
        ranks.append(rank)

    delta_accuracy = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
    delta_loss = [losses[i] - losses[i-1] for i in range(1, len(losses))]
    delta_rank = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]

    delta_accuracy_per_delta_rank = [da/dr if dr != 0 else 0.0 for da, dr in zip(delta_accuracy, delta_rank)]
    delta_loss_per_delta_rank = [dl/dr if dr != 0 else 0.0 for dl, dr in zip(delta_loss, delta_rank)]

    #Save to CSV
    folder_path = f'./data/verify_factorized_data'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/factorized_verification_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'accuracy', 'loss', 'delta_accuracy', 'delta_loss', 'delta_rank', 'delta_accuracy_per_delta_rank', 'delta_loss_per_delta_rank'])
        for i in range(len(ranks)):
            da = delta_accuracy[i-1] if i > 0 else 0.0
            dl = delta_loss[i-1] if i > 0 else 0.0
            dr = delta_rank[i-1] if i > 0 else 0.0
            dapdr = delta_accuracy_per_delta_rank[i-1] if i > 0 else 0.0
            dlpdr = delta_loss_per_delta_rank[i-1] if i > 0 else 0.0
            writer.writerow([ranks[i], accuracies[i], losses[i], da, dl, dr, dapdr, dlpdr])
    print(f"Saved verification metrics to {folder_path}/factorized_verification_metrics.csv")
    return VerificationResult.INCONCLUSIVE # This requires human analysis, we do not automatically SUPPORT or REFUTE, but we do through MLX commands later

