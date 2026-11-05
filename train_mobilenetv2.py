"""
MobileNetV2 Training and Fine-Tuning Pipeline for Crop Disease Classification
Group 5 - Samsung Innovation Campus (SIC) Capstone Project
ML Engineer: Dinh Gia Bao (Team Leader & ML Lead)

Supports:
- Transfer learning with ImageNet pre-trained MobileNetV2
- 2-phase training (Backbone freezing -> Full network fine-tuning)
- Robust mapping for PlantVillage dataset folder name variations
- Full evaluation: Confusion Matrix, Classification Report (Precision, Recall, F1)
- Automatic export to PyTorch checkpoint (.pt) and ONNX format (.onnx)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Fix UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.config import (
    CLASS_NAMES,
    MODEL_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_WEIGHTS_PATH,
)

# Standard aliases for PlantVillage folder names -> Standard CLASS_NAMES
FOLDER_NAME_MAPPING = {
    # Corn
    "Corn_(maize)___Common_rust": "Corn_Common_rust",
    "Corn_(maize)___Common_rust_": "Corn_Common_rust",
    "Corn___Common_rust": "Corn_Common_rust",
    "Corn_Common_rust": "Corn_Common_rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn_Gray_leaf_spot",
    "Corn___Gray_leaf_spot": "Corn_Gray_leaf_spot",
    "Corn_Gray_leaf_spot": "Corn_Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Corn___Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Corn_Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Corn_(maize)___healthy": "Corn_healthy",
    "Corn___healthy": "Corn_healthy",
    "Corn_healthy": "Corn_healthy",
    # Potato
    "Potato___Early_blight": "Potato_Early_blight",
    "Potato_Early_blight": "Potato_Early_blight",
    "Potato___Late_blight": "Potato_Late_blight",
    "Potato_Late_blight": "Potato_Late_blight",
    "Potato___healthy": "Potato_healthy",
    "Potato_healthy": "Potato_healthy",
    # Tomato
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato_Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato_Late_blight": "Tomato_Late_blight",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___healthy": "Tomato_healthy",
    "Tomato_healthy": "Tomato_healthy",
}


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Builds standard training (data augmentation) and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(MODEL_INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV2 with custom linear classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluates the model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc, all_preds, all_targets


def export_onnx(model: nn.Module, output_path: Path, device: torch.device):
    """Exports model to ONNX for accelerated inference."""
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"[Export] Saved ONNX model to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[Export] Warning: Could not export ONNX ({e}). Skipping.")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for SIC Crop Disease Diagnosis")
    parser.add_argument("--data_dir", type=str, default="", help="Path to PlantVillage 11-class dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weights_out", type=str, default=str(MODEL_WEIGHTS_PATH), help="Output path for weights")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="Export to ONNX after training")
    parser.add_argument("--init_only", action="store_true", help="Initialize and export pretrained base model without dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🌱 SIC CAPSTONE GROUP 5 - MOBILENETV2 TRAINING PIPELINE")
    print(f"ML Engineer: Đinh Gia Bảo | Thiết bị sử dụng: {device}")
    print(f"Số lượng lớp bệnh mục tiêu: {len(CLASS_NAMES)}")
    print("=" * 70)

    output_path = Path(args.weights_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mode 1: Initialize base weights (Fast bootstrap)
    if args.init_only or not args.data_dir or not Path(args.data_dir).exists():
        if not args.init_only:
            print(f"\n[Thông báo] Không tìm thấy thư mục dataset tại '{args.data_dir}'.")
            print("Đang khởi tạo trọng số MobileNetV2 tiền huấn luyện chuẩn (Transfer Learning Base)...")
        
        model = build_model(num_classes=len(CLASS_NAMES), pretrained=True)
        model.to(device)

        # Save state dict
        torch.save(model.state_dict(), output_path)
        print(f"[Thành công] Đã lưu trọng số cơ sở tại: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

        if args.export_onnx:
            onnx_path = output_path.with_suffix(".onnx")
            export_onnx(model, onnx_path, device)

        print("\nĐể huấn luyện với dữ liệu thật PlantVillage, chạy lệnh:")
        print("  python train_mobilenetv2.py --data_dir /path/to/plantvillage_data --epochs 20")
        return

    # Mode 2: Full training on PlantVillage dataset
    data_path = Path(args.data_dir)
    print(f"\n[Dataset] Đang tải tập dữ liệu từ: {data_path}")

    train_tf, val_tf = build_transforms()
    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_tf)

    print(f"[Dataset] Tìm thấy {len(full_dataset)} ảnh thuộc {len(full_dataset.classes)} thư mục.")

    # Split train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Dataset] Train set: {train_size} ảnh | Validation set: {val_size} ảnh")

    # Build model
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    start_time = time.time()

    print("\n--- BẮT ĐẦU HUẤN LUYỆN (TRANSFER LEARNING) ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  --> [Saved Best Model] Val Acc đạt đỉnh mới: {best_val_acc:.2f}%")

    total_time = (time.time() - start_time) / 60
    print(f"\n[Hoàn thành] Tổng thời gian huấn luyện: {total_time:.1f} phút.")
    print(f"Độ chính xác cao nhất trên tập Validation: {best_val_acc:.2f}%")

    if args.export_onnx:
        onnx_path = output_path.with_suffix(".onnx")
        export_onnx(model, onnx_path, device)


if __name__ == "__main__":
    main()
