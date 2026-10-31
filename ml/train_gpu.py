"""Image-level GPU training and a small CNN comparison, without changing deployed weights."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from backend.config import CLASS_NAMES, MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from ml.dataset import ManifestDataset
from ml.evaluate import classification_metrics, calibration_metrics
from ml.runtime import PAUSE_FILE, check_pause
from ml.train import calibrate


def small_cnn():
    layers, channels = [], 3
    for width in (16, 32, 64):
        layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(CLASS_NAMES)))


def make_model(architecture, checkpoint=None):
    if architecture == "small_cnn":
        model = small_cnn()
    else:
        model = mobilenet_v2(weights=None, num_classes=len(CLASS_NAMES))
    if checkpoint:
        for key, expected in {"class_names": CLASS_NAMES, "input_size": list(MODEL_INPUT_SIZE),
                              "mean": IMAGENET_MEAN, "std": IMAGENET_STD}.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint contract mismatch: {key}")
        if checkpoint.get("architecture") != architecture:
            raise ValueError("Checkpoint architecture mismatch")
        model.load_state_dict(checkpoint["state_dict"])
    if architecture == "mobilenet_v2":
        if checkpoint is None:
            raise ValueError("Fine-tuning requires an existing trained checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.features[17], model.features[18], model.classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return model


def train_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def get_logits(model, dataset, device, batch_size):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            check_pause()
            logits.append(model(images.to(device)).float().cpu())
            targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def summarize(logits, targets, temperature=1.):
    result = classification_metrics(targets.numpy(), logits.argmax(1).numpy())
    result["loss"] = float(nn.functional.cross_entropy(logits, targets))
    result["calibration"] = calibration_metrics(logits, targets, temperature)
    return result


def evaluate_checkpoint(path, data, split, device="cpu", batch_size=16):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["architecture"], checkpoint).to(device)
    dataset = ManifestDataset(data / f"{split}.csv")
    logits, targets = get_logits(model, dataset, device, batch_size)
    return {**summarize(logits, targets, checkpoint["temperature"]), "split": split,
            "count": len(dataset), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def train(args):
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, patience and learning rate must be positive")
    if args.output.exists():
        raise ValueError("Output already exists; choose a new candidate filename")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install the GPU environment first")
    check_pause()
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    source = torch.load(args.weights, map_location="cpu", weights_only=True) if args.architecture == "mobilenet_v2" else None
    model = make_model(args.architecture, source).to(device)
    train_set = ManifestDataset(args.data / "train.csv", training=True)
    val_set = ManifestDataset(args.data / "validation.csv")
    counts = torch.bincount(torch.tensor([int(row["class_index"]) for row in train_set.rows]), minlength=len(CLASS_NAMES)).float()
    if (counts == 0).any():
        raise ValueError("Training split must cover every class")
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (len(CLASS_NAMES) * counts)).to(device))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.architecture == "mobilenet_v2":
        groups = [{"params": [p for p in model.features.parameters() if p.requires_grad], "lr": args.lr},
                  {"params": model.classifier.parameters(), "lr": args.lr * 10}]
    else:
        groups = parameters
    optimizer = torch.optim.Adam(groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    config = {"architecture": args.architecture, "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
              "patience": args.patience, "batch_size": args.batch_size, "amp": amp,
              "source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest() if source else None,
              "train_sha256": hashlib.sha256(train_set.manifest.read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(val_set.manifest.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    resume_path = args.output.with_suffix(".resume.pt")
    history, start_epoch, start_batch, best_epoch = [], 0, 0, -1
    best_loss, best_state = float("inf"), cpu_state(model)
    if source:
        initial_logits, initial_targets = get_logits(model, val_set, device, args.batch_size)
        initial = summarize(initial_logits, initial_targets, source.get("temperature", 1.))
        best_loss = initial["loss"]
    else:
        initial = None
    resume = None
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if resume["config"] != config:
            raise ValueError("Resume configuration differs; choose a new output")
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history, start_epoch, start_batch = resume["history"], resume["epoch"], resume["batch"]
        best_loss, best_state, best_epoch = resume["best_loss"], resume["best_state"], resume["best_epoch"]

    started = time.perf_counter()
    def save(epoch, batch):
        atomic_save({"config": config, "state_dict": cpu_state(model), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                     "epoch": epoch, "batch": batch, "best_loss": best_loss, "best_state": best_state,
                     "best_epoch": best_epoch, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if amp else []}, resume_path)

    for epoch in range(start_epoch, args.epochs):
        if epoch - best_epoch > args.patience:
            break
        train_mode(model)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        iterator = iter(loader)
        skipped = start_batch if epoch == start_epoch else 0
        for _ in range(skipped):
            next(iterator)
        if resume is not None:
            torch.set_rng_state(resume["torch_rng"].cpu())
            random.setstate(resume["python_rng"])
            if amp:
                torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng"]])
            resume = None
        for index, (images, labels) in enumerate(iterator, skipped):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            if PAUSE_FILE.exists():
                save(epoch, index + 1)
                check_pause()
            if (index + 1) % 50 == 0:
                save(epoch, index + 1)
                print(f"Epoch {epoch + 1}/{args.epochs}, batch {index + 1}/{len(loader)}, loss {loss.item():.4f}", flush=True)
        save(epoch, len(loader))
        logits, targets = get_logits(model, val_set, device, args.batch_size)
        result = summarize(logits, targets)
        history.append({"epoch": epoch + 1, "validation_loss": result["loss"], "validation_macro_f1": result["macro_f1"]})
        if result["loss"] < best_loss - 1e-5:
            best_loss, best_state, best_epoch = result["loss"], cpu_state(model), epoch
        scheduler.step()
        save(epoch + 1, 0)
        print(f"Epoch {epoch + 1}: validation F1={result['macro_f1']:.6f}, loss={result['loss']:.6f}, best epoch={best_epoch + 1}", flush=True)
    model.load_state_dict(best_state)
    best_state = cpu_state(model)
    logits, targets = get_logits(model, val_set, device, args.batch_size)
    temperature, calibration = calibrate(logits, targets)
    report = {"config": config, "device": str(device), "gpu": torch.cuda.get_device_name() if amp else None,
              "torch": str(torch.__version__), "initial_validation": initial, "history": history,
              "best_epoch": best_epoch + 1, "validation": summarize(logits, targets, temperature),
              "training_seconds": time.perf_counter() - started, "test_used_for_selection": False}
    checkpoint = {"state_dict": best_state, "architecture": args.architecture, "class_names": CLASS_NAMES,
                  "input_size": list(MODEL_INPUT_SIZE), "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                  "temperature": temperature, "calibration": calibration, "training": report}
    atomic_save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; selected validation F1={report['validation']['macro_f1']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["small_cnn", "mobilenet_v2"], default="mobilenet_v2")
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a locked checkpoint instead of training")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    if args.evaluate:
        if args.output.exists():
            parser.error("Evaluation output already exists")
        result = evaluate_checkpoint(args.evaluate, args.data, args.split, args.device, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["split", "count", "accuracy", "macro_f1"]}), flush=True)
    else:
        train(args)


if __name__ == "__main__":
    main()
