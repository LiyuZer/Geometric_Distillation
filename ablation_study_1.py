import os
import csv
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.models import ViT_B_32_Weights
from torchvision.models.vision_transformer import EncoderBlock
from torch.optim import AdamW
from tqdm import tqdm

from model import ModifiedViTB32, FactorizedViTB32, FactorizedEncoderBlock
from decorators import trial, verify, VerificationResult, setup

"""
Ablation Study 1: Geometric vs Pure KD vs Logit MSE+Cosine

- Student: FactorizedViTB32(rank=34)
- Teacher: ModifiedViTB32 fine-tuned on CIFAR-100 (checkpoint expected at ./data/setup_base_model/vit_cifar100_state_dict.pth)
- Modes:
  * geometric: MSE + (1 - cosine) on intermediate features and logits (same as vit_factorized)
  * pure: Hinton et al. KD (KL with temperature T and CE with labels), no feature hooks
  * logit_mse: MSE + (1 - cosine) on final logits only, no feature hooks

- Training: 200 epochs, AdamW lr=1e-4, weight_decay=0.01, batch_size=24, num_workers=2
- Data: CIFAR-100 with IMAGENET1K_V1 preprocess + flips/crops for train
- Seeding: Use SETUP seeds if available, else defaults to 42; enforce deterministic worker seeding and DataLoader generator for consistent ordering
- Initialization fairness: All modes load the same initial student weights from a shared stored state_dict at ./data/ablation1_initial_student_state_dict.pth
- Verification: Compare only geometric vs pure by final val accuracy
  * SUPPORTS if geom_acc >= 1.1 * pure_acc
  * REFUTES if geom_acc < pure_acc
  * INCONCLUSIVE otherwise
"""


def seed_everything(torch_seed: int = 42, numpy_seed: int = 42, random_seed: int = 42):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _init_fn


def make_torch_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_hook(container):
    def hook(module, input, output):
        container.append(output)
    return hook


@setup()
def ablation1_setup():
    # Provide seeds for the run; do not train or modify the teacher.
    torch_seed = 42
    random_seed = 42
    numpy_seed = 42
    return {
        "Seed": {
            "torch_seed": torch_seed,
            "random_seed": random_seed,
            "numpy_seed": numpy_seed,
        }
    }


def kd_loss_hinton(student_logits, teacher_logits, T: float = 4.0):
    # KL divergence between softened distributions; batchmean is the standard reduction
    log_p_student = torch.log_softmax(student_logits / T, dim=1)
    p_teacher = torch.softmax(teacher_logits / T, dim=1)
    kl = nn.functional.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    return kl


@trial(order=1, values={"mode": ["geometric", "pure", "logit_mse"], "__parallel__": 3})
def ablation1_train(mode: str):
    # Seeds
    torch_seed = SETUP.get("Seed", {}).get("torch_seed", 42)
    random_seed = SETUP.get("Seed", {}).get("random_seed", 42)
    numpy_seed = SETUP.get("Seed", {}).get("numpy_seed", 42)
    seed_everything(torch_seed, numpy_seed, random_seed)

    # Paths
    folder_path = f"./data/ablation1_train_{mode}"
    os.makedirs(folder_path, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher: expect checkpoint to exist
    teacher_ckpt = "./data/setup_base_model/vit_cifar100_state_dict.pth"
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {teacher_ckpt}. Please run setup_base_model first."
        )

    teacher = ModifiedViTB32(num_classes=100)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)
    teacher.eval()

    # Student: ensure identical initialization across modes by saving a common initial state
    init_student_path = "./data/ablation1_initial_student_state_dict.pth"
    if not os.path.exists(init_student_path):
        proto = FactorizedViTB32(num_classes=100, rank=34, default_dropout=0.0)
        torch.save(proto.state_dict(), init_student_path)
        del proto

    student = FactorizedViTB32(num_classes=100, rank=34, default_dropout=0.0)
    student.load_state_dict(torch.load(init_student_path, map_location="cpu"))
    student = student.to(device)

    # Data
    weights = ViT_B_32_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        preprocess,
    ])
    val_transforms = transforms.Compose([
        preprocess,
    ])

    # Deterministic data ordering and augment seeds via generator + worker_init
    g_train = make_torch_generator(torch_seed)
    g_val = make_torch_generator(torch_seed)
    worker_init = make_worker_init_fn(torch_seed)

    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR100(root="./data", train=False, download=True, transform=val_transforms)

    batch_size = 24
    num_workers = 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=g_train, worker_init_fn=worker_init
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        generator=g_val, worker_init_fn=worker_init
    )

    # Optimization
    optimizer = AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
    criterion_ce = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Hooks for geometric mode
    teacher_feats = []
    student_feats = []
    if mode == "geometric":
        for name, module in teacher.named_modules():
            if isinstance(module, EncoderBlock):
                module.self_attention.register_forward_hook(make_hook(teacher_feats))
                module.mlp.register_forward_hook(make_hook(teacher_feats))
        for name, module in student.named_modules():
            if isinstance(module, FactorizedEncoderBlock):
                module.self_attention.register_forward_hook(make_hook(student_feats))
                module.mlp.register_forward_hook(make_hook(student_feats))

    # Training
    num_epochs = 200
    final_accuracy = 0.0
    final_loss = 0.0

    log_path = os.path.join(folder_path, f"ablation1_{mode}_training_log.txt")

    T = 4.0
    alpha = 0.5

    for epoch in range(num_epochs):
        student.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{mode}]"):
            images = images.to(device)
            labels = labels.to(device)

            # Clear captured outputs for geometric mode
            if mode == "geometric":
                teacher_feats.clear()
                student_feats.clear()

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            if mode == "geometric":
                loss = 0.0
                # features alignment (teacher -> student)
                for orig_out, fact_out in zip(teacher_feats, student_feats):
                    if isinstance(orig_out, tuple):
                        orig_out = orig_out[0]
                    # MSE + (1 - cosine)
                    loss = loss + mse_loss(fact_out, orig_out.detach())
                    loss = loss + (1.0 - nn.functional.cosine_similarity(fact_out, orig_out.detach(), dim=-1).mean())
                # logits alignment
                loss = loss + mse_loss(student_logits, teacher_logits.detach())
                loss = loss + (1.0 - nn.functional.cosine_similarity(student_logits, teacher_logits.detach(), dim=1).mean())
                denom = (len(teacher_feats) + 1)
                loss = loss / max(1, denom)

            elif mode == "pure":
                # Hinton KD: alpha * T^2 * KL + (1-alpha) * CE
                kd = kd_loss_hinton(student_logits, teacher_logits.detach(), T=T)
                ce = criterion_ce(student_logits, labels)
                loss = alpha * kd + (1.0 - alpha) * ce

            elif mode == "logit_mse":
                # MSE + (1 - cosine) on logits only
                loss = mse_loss(student_logits, teacher_logits.detach())
                loss = loss + (1.0 - nn.functional.cosine_similarity(student_logits, teacher_logits.detach(), dim=1).mean())

            else:
                raise ValueError(f"Unknown mode: {mode}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        final_loss = avg_loss

        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        final_accuracy = accuracy

        # Log epoch metrics
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, Val Accuracy: {accuracy:.2f}%\n")

    # Save student checkpoint
    ckpt_path = os.path.join(folder_path, f"ablation1_{mode}_student_state_dict.pth")
    torch.save(student.state_dict(), ckpt_path)

    return {
        "mode": mode,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "rank": 34,
        "epochs": num_epochs,
        "checkpoint": ckpt_path,
    }


@verify(["ablation1_train"])
def ablation1_verify(results):
    runs = results.get("ablation1_train", [])

    # Aggregate final metrics by mode
    by_mode = {}
    ok = True
    for run in runs:
        if not run.get("success"):
            ok = False
            continue
        res = run.get("result", {})
        mode = res.get("mode")
        acc = res.get("final_accuracy")
        loss = res.get("final_loss")
        by_mode[mode] = {"accuracy": acc, "loss": loss}

    # Save CSV for human analysis
    folder_path = "./data/verify_ablation1"
    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, "ablation1_verification_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "final_accuracy", "final_loss"])
        for mode in sorted(by_mode.keys()):
            writer.writerow([mode, by_mode[mode]["accuracy"], by_mode[mode]["loss"]])

    # Decision only on geometric vs pure
    geom = by_mode.get("geometric", {}).get("accuracy")
    pure = by_mode.get("pure", {}).get("accuracy")

    if geom is None or pure is None:
        print("Geometric and/or Pure KD results missing — cannot verify.")
        return VerificationResult.INVALID

    if geom >= 1.1 * pure:
        print(f"SUPPORTS: Geometric ({geom:.2f}%) >= 1.1 * Pure ({pure:.2f}%).")
        return VerificationResult.SUPPORTS
    elif geom < pure:
        print(f"REFUTES: Geometric ({geom:.2f}%) < Pure ({pure:.2f}%).")
        return VerificationResult.REFUTES
    else:
        print(f"INCONCLUSIVE: Geometric ({geom:.2f}%) >= Pure ({pure:.2f}%) but < 10% relative improvement.")
        return VerificationResult.INCONCLUSIVE
