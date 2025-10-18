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
import json
import sys
import os

SETUP = {"final_val_accuracy": 76.96, "Seed": {"torch_seed": 42, "random_seed": 42, "numpy_seed": 42}}
# --- MLX auto-seeding (best-effort) ---
seed = (SETUP.get('seed') if isinstance(SETUP, dict) else None)
numpy_seed = SETUP.get('numpy_seed', seed) if isinstance(SETUP, dict) else None
torch_seed = SETUP.get('torch_seed', seed) if isinstance(SETUP, dict) else None
pythonhashseed = SETUP.get('pythonhashseed', seed) if isinstance(SETUP, dict) else None
try:
    import os as _os_seed
    import random as _random_seed
    if pythonhashseed is not None:
        _os_seed.environ['PYTHONHASHSEED'] = str(int(pythonhashseed))
    if seed is not None:
        _random_seed.seed(int(seed))
except Exception:
    pass
try:
    import numpy as _np_seed
    if numpy_seed is not None:
        _np_seed.random.seed(int(numpy_seed))
except Exception:
    pass
try:
    import torch as _torch_seed
    if torch_seed is not None:
        _torch_seed.manual_seed(int(torch_seed))
        if hasattr(_torch_seed, 'cuda') and _torch_seed.cuda.is_available():
            _torch_seed.cuda.manual_seed_all(int(torch_seed))
        try:
            import torch.backends.cudnn as _cudnn
            _cudnn.deterministic = True
            _cudnn.benchmark = False
        except Exception:
            pass
except Exception:
    pass
# --- end auto-seeding ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_hook(container):
    def hook(module, input, output):
        container.append(output)
    return hook

def train_factorized(rank):
    torch_seed = SETUP.get('Seed', {}).get('torch_seed', 42)
    random_seed = SETUP.get('Seed', {}).get('random_seed', 42)
    numpy_seed = SETUP.get('Seed', {}).get('numpy_seed', 42)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    folder_path = f'./data/train_factorized_rank_{rank}'
    os.makedirs(folder_path, exist_ok=True)
    print('Starting training with factorized MLPs in ViT...')
    batch_size = 24
    num_workers = 2
    lr = 0.0001
    num_epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_dict = torch.load(f'./data/setup_base_model/vit_cifar100_state_dict.pth', map_location=device)
    teacher = ModifiedViTB32(num_classes=100)
    teacher.load_state_dict(teacher_dict)
    student = FactorizedViTB32(num_classes=100, rank=rank, default_dropout=0.0)
    print(f'Original model architecture:\n{student}\n')
    print(f'Student model with factorized MLPs architecture:\n{student}\n')
    teacher_params = sum((p.numel() for p in teacher.parameters()))
    student_params = sum((p.numel() for p in student.parameters()))
    print(f'Teacher model parameters: {teacher_params}')
    print(f'Student model parameters: {student_params}')
    teacher = teacher.to(device)
    student = student.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    mlp_outputs = []
    factorized_mlp_outputs = []
    for (name, module) in teacher.named_modules():
        if isinstance(module, EncoderBlock):
            module.self_attention.register_forward_hook(make_hook(mlp_outputs))
            module.mlp.register_forward_hook(make_hook(mlp_outputs))
    for (name, module) in student.named_modules():
        if isinstance(module, FactorizedEncoderBlock):
            module.self_attention.register_forward_hook(make_hook(factorized_mlp_outputs))
            module.mlp.register_forward_hook(make_hook(factorized_mlp_outputs))
    preprocess = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), preprocess])
    val_transforms = transforms.Compose([preprocess])
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mse_loss = nn.MSELoss()
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    final_accuracy = 0.0
    final_loss = 0.0
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0.0
        for (images, labels) in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images = images.to(device)
            mlp_outputs.clear()
            factorized_mlp_outputs.clear()
            with torch.no_grad():
                _ = teacher(images)
                teacher_logits = _
            student_logits = student(images)
            loss = 0.0
            for (orig_out, fact_out) in zip(mlp_outputs, factorized_mlp_outputs):
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0]
                loss = loss + mse_loss(fact_out, orig_out.detach())
                loss = loss + 1 - nn.functional.cosine_similarity(fact_out, orig_out.detach(), dim=-1).mean()
            loss = loss + mse_loss(student_logits, teacher_logits.detach()) + (1 - nn.functional.cosine_similarity(student_logits, teacher_logits.detach(), dim=1).mean())
            denom = len(mlp_outputs) + 1
            loss = loss / denom
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        final_loss = avg_loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in val_loader:
                mlp_outputs.clear()
                factorized_mlp_outputs.clear()
                (images, labels) = (images.to(device), labels.to(device))
                outputs = student(images)
                (_, predicted) = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        final_accuracy = accuracy
        print(f'Validation Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')
        with open(f'{folder_path}/vit_cifar100_factorized_{rank}_training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, Loss: {avg_loss}, Accuracy: {accuracy}\n')
    torch.save(student.state_dict(), f'{folder_path}/vit_cifar100_factorized_{rank}_state_dict.pth')
    print(f'Training complete. Saved factorized student checkpoint to {folder_path}/vit_cifar100_factorized_{rank}_state_dict.pth')
    return {'final_accuracy': final_accuracy, 'final_loss': final_loss, 'rank': rank}

def wrapper_train_factorized():
    try:
        result = train_factorized(rank=24)
        print(json.dumps({"result": result}))
    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)

wrapper_train_factorized()