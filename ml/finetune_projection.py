"""Fine-tune MobileNetV2's final 1x1 projection on disk-backed cached activations.

One seeded augmented training view; BatchNorm remains frozen. No test data is read.
Streaming shards bound RAM use on the 8GB CPU host. All outputs are new candidates.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from backend.inference_engine import CropDiseaseInferenceEngine
from ml.dataset import ManifestDataset
from ml.optimize_head import atomic_save, metrics
from ml.runtime import PAUSE_FILE, check_pause
from ml.train import calibrate


def collect_shards(model, dataset, root, batch_size, training):
    paths = []
    for start in range(0, len(dataset), batch_size):
        check_pause()
        path = root / f"{start:05d}.pt"
        paths.append(path)
        if path.exists():
            continue
        torch.manual_seed(42 + start)
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([image for image, _ in samples]).to(memory_format=torch.channels_last)
        with torch.no_grad():
            features = model.features[:-1](images)
        # shortcut: FP16 disk cache saves RAM/storage; final validation uses full FP32 image inference.
        root.mkdir(parents=True, exist_ok=True)
        atomic_save({"features": features.to(torch.float16), "targets": torch.tensor([label for _, label in samples])}, path)
        if len(paths) % 25 == 0:
            print(f"{'Train' if training else 'Validation'} activation shards {len(paths)}", flush=True)
    return paths


def tail_logits(model, features):
    return model.classifier(model.features[-1](features).mean((2, 3)))


def validation_logits(model, paths):
    values, targets = [], []
    model.eval()
    with torch.no_grad():
        for path in paths:
            check_pause()
            shard = torch.load(path, weights_only=True)
            values.append(tail_logits(model, shard["features"].float().to(memory_format=torch.channels_last)))
            targets.append(shard["targets"])
    return torch.cat(values), torch.cat(targets)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Candidate output exists; choose a new output")
    baseline = torch.load(args.weights, weights_only=True)
    engine = CropDiseaseInferenceEngine(args.weights, device="cpu", runtime="eager")
    if not engine.weights_loaded:
        raise ValueError(engine.load_error or "Invalid input checkpoint")
    model = engine.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.features[-1][0].parameters():
        parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    training = ManifestDataset(args.data / "train.csv", training=True)
    validation = ManifestDataset(args.data / "validation.csv")
    config = {"input_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(), "epochs": args.epochs,
              "lr": args.lr, "batch_size": args.batch_size, "seed": 42, "cache_dtype": "float16",
              "train_sha256": hashlib.sha256(training.manifest.read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(validation.manifest.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    root = args.activation_cache or Path("artifacts") / f"projection_{cache_key}"
    if args.activation_cache:
        previous = torch.load(root / "resume.pt", weights_only=True)["config"]
        for key in ("input_sha256", "train_sha256", "validation_sha256", "batch_size", "seed", "cache_dtype"):
            if previous[key] != config[key]:
                raise ValueError(f"Activation cache contract mismatch: {key}")
    root.mkdir(parents=True, exist_ok=True)
    train_paths = collect_shards(model, training, root / "train", args.batch_size, True)
    val_paths = collect_shards(model, validation, root / "validation", args.batch_size, False)
    targets = torch.tensor([int(row["class_index"]) for row in training.rows])
    counts = torch.bincount(targets, minlength=engine.num_classes).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (len(counts) * counts)).sqrt())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    logits, val_targets = validation_logits(model, val_paths)
    reference = metrics(logits, val_targets)
    best_score = (reference["macro_f1"], -reference["nll"])
    best_state = copy.deepcopy(model.state_dict())
    history, start_epoch, start_batch, best_epoch = [], 0, 0, 0
    resume_path = root / (f"resume_{cache_key}.pt" if args.activation_cache else "resume.pt")
    if resume_path.exists():
        resume = torch.load(resume_path, weights_only=True)
        if resume["config"] != config:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best_state, best_score, best_epoch = resume["best_state"], tuple(resume["best_score"]), resume["best_epoch"]
        start_epoch, start_batch, history = resume["epoch"], resume["batch"], resume["history"]
        torch.set_rng_state(resume["rng"])

    def save(epoch, batch):
        atomic_save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "best_state": best_state, "best_score": best_score,
                     "best_epoch": best_epoch, "epoch": epoch, "batch": batch,
                     "history": history, "rng": torch.get_rng_state()}, resume_path)

    for epoch in range(start_epoch, args.epochs):
        model.eval()
        model.classifier.train()
        order = torch.randperm(len(train_paths), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        for position in range(start_batch if epoch == start_epoch else 0, len(order)):
            if PAUSE_FILE.exists():
                save(epoch, position)
                check_pause()
            shard = torch.load(train_paths[order[position]], weights_only=True)
            features = shard["features"].float().to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(tail_logits(model, features), shard["targets"])
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            if (position + 1) % 20 == 0:
                save(epoch, position + 1)
            if (position + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, batch {position + 1}/{len(order)}", flush=True)
        # A pause during validation resumes from the completed training epoch.
        save(epoch, len(order))
        logits, val_targets = validation_logits(model, val_paths)
        result = metrics(logits, val_targets)
        history.append({"epoch": epoch + 1, "macro_f1": result["macro_f1"], "nll": result["nll"]})
        score = (result["macro_f1"], -result["nll"])
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
        save(epoch + 1, 0)
        print(f"Epoch {epoch + 1}: validation F1={score[0]:.6f}, best epoch={best_epoch}", flush=True)
        if epoch + 1 - best_epoch >= 3:
            break
    model.load_state_dict(best_state)
    logits, val_targets = validation_logits(model, val_paths)
    temperature, calibration = calibrate(logits.detach(), val_targets)
    report = {"config": config, "baseline_cached_validation": reference, "history": history,
              "best_epoch": best_epoch, "selected_cached_validation": metrics(logits, val_targets),
              "strategy": "train final 1x1 convolution and linear head; all BatchNorm frozen",
              "test_used_for_selection": False, "requires_full_fp32_validation": True,
              "calibration": calibration, "resume_path": str(resume_path)}
    candidate = copy.deepcopy(baseline)
    candidate.update(state_dict=best_state, temperature=temperature, calibration=calibration,
                     projection_finetune=report)
    atomic_save(candidate, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; best cached validation F1={best_score[0]:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_head_v2_20261024.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mobilenetv2_projection_20261024.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--activation-cache", type=Path, help="Reuse compatible existing activation shards")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0:
        parser.error("Training arguments must be positive")
    run(args)


if __name__ == "__main__":
    main()
