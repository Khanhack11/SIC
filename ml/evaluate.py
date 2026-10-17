"""One-shot held-out evaluation with macro metrics, confusion matrix and calibration."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.config import CLASS_NAMES, CONFIDENCE_THRESHOLD
from backend.inference_engine import CropDiseaseInferenceEngine, ModelNotReadyError
from ml.dataset import ManifestDataset
from ml.runtime import check_pause


def classification_metrics(targets, predictions, class_names=CLASS_NAMES):
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(len(matrix), dtype=float), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(matrix), dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(matrix), dtype=float), where=(precision + recall) > 0)
    return {
        "accuracy": float(true_positive.sum() / max(1, matrix.sum())),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(class_names)},
        "confusion_matrix": matrix.tolist(), "confusion_matrix_axes": "rows=true, columns=predicted",
        "class_names": class_names,
    }


def calibration_metrics(logits, targets, temperature, bins=15):
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted.eq(targets).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= 1)
        if mask.any():
            ece += float(mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs())
    accepted = confidence >= CONFIDENCE_THRESHOLD
    return {
        "nll": float(torch.nn.functional.cross_entropy(logits / temperature, targets)),
        "ece": ece, "ece_bins": bins, "confidence_threshold": CONFIDENCE_THRESHOLD,
        "accepted_count": int(accepted.sum()), "accepted_coverage": float(accepted.float().mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
    }


def evaluate(args):
    torch.set_num_threads(args.threads)
    dataset = ManifestDataset(args.data / "test.csv", args.source)
    engine = CropDiseaseInferenceEngine(args.weights, device=args.device, runtime=args.runtime)
    if not engine.weights_loaded:
        raise ModelNotReadyError(engine.load_error or "Trained checkpoint required")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    outputs, targets = [], []
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            check_pause()
            outputs.append(engine.model(images.to(engine.device, memory_format=engine.memory_format)).cpu())
            targets.append(labels)
            if index % 20 == 0:
                print(f"Evaluation batch {index + 1}/{len(loader)}", flush=True)
    logits, targets = torch.cat(outputs), torch.cat(targets)
    predictions = logits.argmax(dim=1)
    report = classification_metrics(targets.numpy(), predictions.numpy())
    report.update({
        "test_count": len(dataset), "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "test_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
        "device": str(engine.device), "runtime": args.runtime, "threads": args.threads,
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "temperature": engine.temperature, "calibrated": engine.calibrated,
        "calibration_before": calibration_metrics(logits, targets, 1.),
        "calibration_after": calibration_metrics(logits, targets, engine.temperature),
        "macro_f1_target": .85, "macro_f1_target_met": report["macro_f1"] >= .85,
        "limitations": "PlantVillage controlled-background test subset; field generalization is unmeasured.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, report["confusion_matrix"]):
            writer.writerow([label, *row])
    print(json.dumps({key: report[key] for key in ("test_count", "accuracy", "macro_f1", "macro_f1_target_met")}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_optimization/results/evaluation.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--runtime", choices=["eager", "torchscript"], default="eager")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
