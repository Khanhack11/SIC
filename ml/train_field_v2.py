"""Train MobileNetV3-Large field v2 candidate on expanded multi-source field dataset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.mobile_model import (
    FIELD_ARCHITECTURE,
    FIELD_CLASSES,
    FIELD_PREPROCESSING,
    build_field_model,
)
from ml.evaluate import calibration_metrics, classification_metrics
from ml.runtime import PAUSE_FILE, check_pause
from ml.train import calibrate
from ml.train_field import FieldDataset, logits_for
from ml.train_gpu import atomic_save, cpu_state, train_mode


def multi_domain_metrics(logits, targets, rows, temperature=1.0):
    result = {}
    probabilities = (logits / temperature).softmax(1)
    scores, predicted = probabilities.max(1)
    domains = sorted({r["domain"] for r in rows})
    for domain in domains:
        mask = torch.tensor([r["domain"] == domain for r in rows])
        if not mask.any():
            continue
        values = classification_metrics(
            targets[mask].numpy(),
            logits[mask].argmax(1).numpy(),
            FIELD_CLASSES,
        )
        present = [v["f1"] for v in values["per_class"].values() if v["support"]]
        known = [
            v["f1"]
            for k, v in values["per_class"].items()
            if v["support"] and k != "Unsupported"
        ]
        values["macro_f1_present"] = sum(present) / len(present) if present else 0.0
        values["known_macro_f1_present"] = sum(known) / len(known) if known else 0.0
        values["calibration"] = calibration_metrics(
            logits[mask], targets[mask], temperature
        )

        diseased = (
            torch.tensor(
                [
                    r["label"] != "Unsupported" and not r["label"].endswith("_healthy")
                    for r in rows
                ]
            )
            & mask
        )
        healthy_pred = torch.tensor(
            [FIELD_CLASSES[i].endswith("_healthy") for i in predicted.tolist()]
        )
        accepted = (scores >= 0.75) & (predicted != FIELD_CLASSES.index("Unsupported"))
        values["score_gate_coverage"] = float(accepted[mask].float().mean())
        values["score_gate_accuracy"] = (
            float((predicted[accepted & mask] == targets[accepted & mask]).float().mean())
            if (accepted & mask).any()
            else None
        )
        values["known_diseased_count"] = int(diseased.sum())
        values["known_diseased_high_confidence_healthy_count"] = int(
            (diseased & healthy_pred & accepted).sum()
        )
        values["known_diseased_high_confidence_healthy_rate"] = (
            float((diseased & healthy_pred & accepted).sum() / diseased.sum())
            if diseased.any()
            else None
        )
        result[domain] = values
    return result


def weights_for_v2(rows):
    """Domain-and-class balanced sampling weights emphasizing field known & OOD discrimination."""
    hard_field_classes = {
        "Potato_Early_blight": 1.35,
        "Potato_Late_blight": 1.35,
        "Tomato_Early_blight": 1.35,
        "Tomato_Late_blight": 1.35,
        "Tomato_Septoria_leaf_spot": 1.25,
        "Tomato_healthy": 1.20,
        "Potato_healthy": 1.10,
    }
    bucket_counts = Counter()
    for r in rows:
        is_village = r["domain"] == "PlantVillage"
        is_crop = r.get("source_dataset") == "plantdoc_detection_crop"
        bucket = ("village" if is_village else ("pd_crop" if is_crop else r["domain"]), r["label"])
        bucket_counts[bucket] += 1

    weights = []
    for r in rows:
        label = r["label"]
        if r["domain"] == "PlantVillage":
            bucket = ("village", label)
            # PlantVillage gets ~22% total share across 11 classes (2% per class)
            bucket_share = 0.02
        else:
            is_crop = r.get("source_dataset") == "plantdoc_detection_crop"
            sub = "pd_crop" if is_crop else r["domain"]
            bucket = (sub, label)
            if label == "Unsupported":
                # Field Unsupported gets strong representation to sharpen OOD boundary
                bucket_share = 0.08 if sub == "PlantDoc" else (0.04 if sub == "pd_crop" else 0.06)
            else:
                boost = hard_field_classes.get(label, 1.0)
                base_share = 0.032 if sub == "PlantDoc" else (0.022 if sub == "pd_crop" else 0.022)
                bucket_share = base_share * boost
        weights.append(bucket_share / max(1, bucket_counts[bucket]))
    return torch.tensor(weights, dtype=torch.double)


def validation_loss_v2(logits, labels, rows):
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    domains = [d for d in ("PlantVillage", "PlantDoc", "MultiSourceField") if any(r["domain"] == d for r in rows)]
    weights = {"PlantVillage": 0.25, "PlantDoc": 0.50, "MultiSourceField": 0.25}
    total_w = sum(weights[d] for d in domains)
    return sum(
        weights[d] * float(losses[torch.tensor([r["domain"] == d for r in rows])].mean())
        for d in domains
    ) / total_w


def calibrate_domain_balanced(logits, labels, rows):
    """Select temperature on validation split balancing NLL and PlantDoc score-gate reliability."""
    nll_temp, _ = calibrate(logits, labels)
    pd_mask = torch.tensor([r["domain"] == "PlantDoc" for r in rows])
    pv_mask = torch.tensor([r["domain"] == "PlantVillage" for r in rows])
    if not pd_mask.any():
        return nll_temp, calibration_metrics(logits, labels, nll_temp)

    best_temp = nll_temp
    best_score = -1e9
    for candidate_t in torch.linspace(0.55, 1.60, 211):
        t = float(candidate_t)
        probs = (logits / t).softmax(1)
        scores, pred = probs.max(1)
        accepted = (scores >= 0.75) & (pred != FIELD_CLASSES.index("Unsupported"))
        pd_acc_mask = accepted & pd_mask
        cov = float(pd_acc_mask.sum() / pd_mask.sum())
        acc = (
            float((pred[pd_acc_mask] == labels[pd_acc_mask]).float().mean())
            if pd_acc_mask.any()
            else 0.0
        )
        pd_nll = float(nn.functional.cross_entropy(logits[pd_mask] / t, labels[pd_mask]))
        pv_nll = float(nn.functional.cross_entropy(logits[pv_mask] / t, labels[pv_mask])) if pv_mask.any() else 0.0
        # Reward meeting coverage>=0.25 and accuracy>=0.85 on validation while keeping NLL low
        gate_bonus = 0.0
        if cov >= 0.25 and acc >= 0.85:
            gate_bonus = 2.0 + acc + 0.5 * cov
        elif cov >= 0.25:
            gate_bonus = 0.5 + acc
        score = gate_bonus - (0.65 * pd_nll + 0.35 * pv_nll)
        if score > best_score:
            best_score = score
            best_temp = t
    return best_temp, calibration_metrics(logits, labels, best_temp)


def train_v2(args):
    if args.output.exists():
        raise ValueError(f"Output checkpoint {args.output} already exists; preserve existing candidates")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU environment is required")

    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)

    manifest = args.data / "records.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit = json.loads((args.data / "audit.json").read_text(encoding="utf-8"))
    if audit["records_sha256"] != digest:
        raise ValueError("Audited manifest hash mismatch")

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    for key, expected in {
        "architecture": FIELD_ARCHITECTURE,
        "class_names": FIELD_CLASSES,
        "preprocessing": FIELD_PREPROCESSING,
    }.items():
        if source.get(key) != expected:
            raise ValueError("Parent checkpoint contract mismatch: " + key)

    config = {
        "parent": str(args.source),
        "parent_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "records_sha256": digest,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "seed": 42,
        "unfrozen_feature_blocks": 9,
        "mid_backbone_lr": 2.5e-5,
        "top_backbone_lr": 6e-5,
        "head_lr": 2e-4,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transform_sha256": hashlib.sha256(Path("backend/mobile_model.py").read_bytes()).hexdigest(),
    }

    records = json.loads(manifest.read_text(encoding="utf-8"))
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "validation"]

    model = build_field_model().to("cuda")
    model.load_state_dict(source["state_dict"])

    feature_blocks = list(model.features.children())
    for p in model.features.parameters():
        p.requires_grad_(False)
    for block in feature_blocks[-9:]:
        for p in block.parameters():
            p.requires_grad_(True)

    mid_params = [p for b in feature_blocks[-9:-4] for p in b.parameters() if p.requires_grad]
    top_params = [p for b in feature_blocks[-4:] for p in b.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": mid_params, "lr": config["mid_backbone_lr"]},
            {"params": top_params, "lr": config["top_backbone_lr"]},
            {"params": list(model.classifier.parameters()), "lr": config["head_lr"]},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    logits, targets = logits_for(model, val_rows, "cuda", args.batch_size)
    init_metrics = multi_domain_metrics(logits, targets, val_rows, 1.0)
    init_loss = validation_loss_v2(logits, targets, val_rows)
    init_score = (
        init_metrics.get("PlantDoc", {}).get("known_macro_f1_present", 0.0)
        + 0.4 * init_metrics.get("MultiSourceField", {}).get("known_macro_f1_present", 0.0)
        + 0.2 * init_metrics.get("PlantVillage", {}).get("known_macro_f1_present", 0.0)
    )
    print(
        "Initial validation known F1:",
        {d: round(m["known_macro_f1_present"], 4) for d, m in init_metrics.items()},
        flush=True,
    )

    best_score = init_score
    best_loss = init_loss
    best_state = cpu_state(model)
    selected_stage = "parent"
    history = []

    dataset = FieldDataset(train_rows, training=True)
    weights = weights_for_v2(train_rows)
    started = time.perf_counter()

    # Construct class weights to penalize false positives on known classes from Unsupported
    class_weights = torch.ones(len(FIELD_CLASSES), device="cuda")
    class_weights[FIELD_CLASSES.index("Unsupported")] = 1.15

    for epoch in range(args.epochs):
        check_pause()
        gen = torch.Generator().manual_seed(42 + epoch)
        sample_indices = torch.multinomial(
            weights, args.samples_per_epoch, replacement=True, generator=gen
        ).tolist()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sample_indices,
            generator=torch.Generator().manual_seed(1042 + epoch),
        )
        train_mode(model)
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(images.to("cuda"))
                loss = nn.functional.cross_entropy(
                    out,
                    labels.to("cuda"),
                    weight=class_weights,
                    label_smoothing=0.04,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss encountered")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            if batch_idx % 50 == 0 or batch_idx == len(loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs}, batch {batch_idx}/{len(loader)}, loss {running_loss / batch_idx:.4f}",
                    flush=True,
                )
            check_pause()

        scheduler.step()
        logits, targets = logits_for(model, val_rows, "cuda", args.batch_size)
        val_loss = validation_loss_v2(logits, targets, val_rows)
        metrics = multi_domain_metrics(logits, targets, val_rows, 1.0)
        pd_f1 = metrics.get("PlantDoc", {}).get("known_macro_f1_present", 0.0)
        msf_f1 = metrics.get("MultiSourceField", {}).get("known_macro_f1_present", 0.0)
        pv_f1 = metrics.get("PlantVillage", {}).get("known_macro_f1_present", 0.0)
        composite = pd_f1 + 0.4 * msf_f1 + 0.2 * pv_f1

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": round(running_loss / max(1, len(loader)), 5),
            "validation_loss": round(val_loss, 5),
            "known_f1": {d: round(m["known_macro_f1_present"], 4) for d, m in metrics.items()},
            "macro_f1": {d: round(m["macro_f1_present"], 4) for d, m in metrics.items()},
            "composite_score": round(composite, 5),
        }
        history.append(epoch_entry)
        print("Validation summary:", epoch_entry, flush=True)

        if composite > best_score and pv_f1 >= 0.95:
            best_score = composite
            best_loss = val_loss
            best_state = cpu_state(model)
            selected_stage = f"field_v2/epoch_{epoch + 1}"
            atomic_save(
                {
                    "state_dict": best_state,
                    "config": config,
                    "history": history,
                    "selected_stage": selected_stage,
                },
                args.output.with_suffix(".progress.pt"),
            )

    model.load_state_dict(best_state)
    logits, targets = logits_for(model, val_rows, "cuda", args.batch_size)
    temperature, calibration = calibrate_domain_balanced(logits, targets, val_rows)
    final_val_metrics = multi_domain_metrics(logits, targets, val_rows, temperature)

    report = {
        "config": config,
        "architecture": FIELD_ARCHITECTURE,
        "selected_stage": selected_stage,
        "history": history,
        "validation": final_val_metrics,
        "temperature": temperature,
        "training_seconds": time.perf_counter() - started,
        "test_used_for_selection": False,
        "test_evaluated": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "parent": str(args.source),
    }
    atomic_save(
        {
            "architecture": FIELD_ARCHITECTURE,
            "class_names": FIELD_CLASSES,
            "preprocessing": FIELD_PREPROCESSING,
            "state_dict": best_state,
            "temperature": temperature,
            "calibration": calibration,
            "training": report,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Saved v2 field candidate:", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/candidate/mobilenetv3_field.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_runs/field_v2/model.pt"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/field_mobile_v2")
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=32)
    train_v2(parser.parse_args())
