"""CPU-friendly transfer-learning baseline with cached augmented backbone features.

Only train and validation are read. Test is reserved for ml.evaluate.
The ImageNet backbone stays frozen; this is a reproducible baseline for the ML lead.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from backend.config import CLASS_NAMES, MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from ml.dataset import ManifestDataset
from ml.runtime import check_pause


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_features(model, loader, device, views, cache_dir):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for view in range(views):
            for index, (images, targets) in enumerate(loader):
                check_pause()
                shard = cache_dir / f"view{view}_batch{index}.pt"
                if shard.exists():
                    saved = torch.load(shard, map_location="cpu", weights_only=True)
                    if not torch.equal(saved["targets"], targets):
                        raise ValueError("Feature shard labels changed")
                    values = saved["features"]
                else:
                    values = model.features(images.to(device, memory_format=torch.channels_last)).mean(dim=(2, 3)).cpu()
                    shard.parent.mkdir(parents=True, exist_ok=True)
                    temporary = shard.with_suffix(".partial")
                    torch.save({"features": values, "targets": targets}, temporary)
                    temporary.replace(shard)
                features.append(values)
                labels.append(targets)
                if index % 50 == 0:
                    print(f"Feature view {view + 1}/{views}, batch {index + 1}/{len(loader)}", flush=True)
    return torch.cat(features), torch.cat(labels)


def calibrate(logits, targets):
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_temperature.exp().clamp(.05, 10), targets)
        loss.backward()
        return loss

    before = nn.functional.cross_entropy(logits, targets).item()
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(.05, 10).item())
    after = nn.functional.cross_entropy(logits / temperature, targets).item()
    if after > before:
        temperature, after = 1., before
    return temperature, {"method": "temperature_scaling_validation_only", "temperature": temperature,
                         "validation_nll_before": before, "validation_nll_after": after}


def train(args):
    if min(args.epochs, args.batch_size, args.views, args.threads, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Training counts, patience, threads and learning rate must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Checkpoint already exists; choose a new --output: {output}")
    train_set = ManifestDataset(args.data / "train.csv", args.source, training=True)
    validation_set = ManifestDataset(args.data / "validation.csv", args.source)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device, memory_format=torch.channels_last)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(CLASS_NAMES)).to(device)
    metadata = {
        "seed": args.seed, "views": args.views, "torch": str(torch.__version__),
        "feature_memory_format": "channels_last",
        "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
        "train_manifest_sha256": hashlib.sha256((args.data / "train.csv").read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256((args.data / "validation.csv").read_bytes()).hexdigest(),
        "class_names": CLASS_NAMES, "input_size": list(MODEL_INPUT_SIZE),
        "batch_size": args.batch_size, "workers": args.workers,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "augmentation": {"horizontal_flip_probability": .5, "vertical_flip_probability": .5,
                         "rotation_degrees": 20, "brightness_jitter": .2},
    }
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path("artifacts") / f"features_{cache_key}.pt"
    started = time.perf_counter()
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Feature-cache metadata mismatch")
        features, targets = saved["train_features"], saved["train_targets"]
        val_features, val_targets = saved["val_features"], saved["val_targets"]
        print(f"Loaded verified feature cache: {cache}", flush=True)
    else:
        features, targets = extract_features(model, train_loader, device, args.views, cache.with_suffix("") / "train")
        val_features, val_targets = extract_features(model, validation_loader, device, 1, cache.with_suffix("") / "validation")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(metadata=metadata, train_features=features, train_targets=targets,
                        val_features=val_features, val_targets=val_targets), cache)
    # Cached augmentation is used only for training; validation remains deterministic.
    seed_everything(args.seed + 1)
    feature_loader = DataLoader(TensorDataset(features, targets), batch_size=256, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))
    counts = torch.bincount(targets, minlength=len(CLASS_NAMES)).float()
    if (counts == 0).any():
        raise ValueError("Training data must contain all supported classes")
    class_weights = (counts.sum() / (len(CLASS_NAMES) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    val_features, val_targets = val_features.to(device), val_targets.to(device)
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    resume_path = output.with_suffix(".resume.pt")
    resume_config = {"metadata": metadata, "epochs": args.epochs, "patience": args.patience, "lr": args.lr}
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume["config"] != resume_config:
            raise ValueError("Resume configuration differs; choose a new output checkpoint.")
        model.classifier.load_state_dict(resume["classifier"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["rng_state"])
        feature_loader.generator.set_state(resume["loader_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        best_loss, best_epoch, best_state, history = resume["best_loss"], resume["best_epoch"], resume["best_state"], resume["history"]
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs_to_run = [] if history and len(history) - best_epoch >= args.patience else range(len(history) + 1, args.epochs + 1)
    for epoch in epochs_to_run:
        check_pause()
        model.classifier.train()
        running = 0.
        for batch_features, batch_targets in feature_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.classifier(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch_targets)
        model.classifier.eval()
        with torch.no_grad():
            logits = model.classifier(val_features)
            val_loss = criterion(logits, val_targets).item()
            accuracy = float((logits.argmax(1) == val_targets).float().mean().item())
        history.append(dict(epoch=epoch, train_loss=running / len(targets), validation_loss=val_loss,
                            validation_accuracy=accuracy, learning_rate=optimizer.param_groups[0]["lr"]))
        print(f"Epoch {epoch}: train_loss={history[-1]['train_loss']:.4f}, val_loss={val_loss:.4f}, val_acc={accuracy:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.classifier.state_dict())
        scheduler.step()
        temporary_resume = resume_path.with_suffix(".partial")
        torch.save({"config": resume_config, "classifier": model.classifier.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(), "loader_rng_state": feature_loader.generator.get_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "best_loss": best_loss, "best_epoch": best_epoch, "best_state": best_state,
                    "history": history}, temporary_resume)
        temporary_resume.replace(resume_path)
        check_pause()
        if epoch - best_epoch >= args.patience:
            break
    model.classifier.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits = model.classifier(val_features).cpu()
    temperature, calibration = calibrate(validation_logits, val_targets.cpu())
    report = {
        **metadata, "architecture": "mobilenet_v2", "strategy": "frozen_backbone_cached_augmented_features",
        "optimizer": "Adam", "scheduler": "CosineAnnealingLR", "loss": "class_weighted_cross_entropy",
        "device": str(device), "threads": args.threads, "batch_size": args.batch_size,
        "learning_rate": args.lr, "patience": args.patience, "max_epochs": args.epochs,
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history,
        "training_seconds": time.perf_counter() - started, "calibration": calibration,
        "test_data_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "class_names": CLASS_NAMES, "input_size": list(MODEL_INPUT_SIZE),
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "temperature": temperature, "calibration": calibration,
        "architecture": "mobilenet_v2", "training": report,
    }
    temporary = output.with_suffix(".partial")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--source", type=Path, help="Override raw-data path after moving the project")
    parser.add_argument("--output", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.001)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
