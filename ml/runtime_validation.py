"""Validate deployment candidates on validation only, before choosing a runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.inference_engine import CropDiseaseInferenceEngine
from ml.dataset import ManifestDataset
from ml.evaluate import classification_metrics, calibration_metrics
from ml.runtime import check_pause


def compare(data, weights, output):
    torch.set_num_threads(2)
    dataset = ManifestDataset(data / "validation.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    candidates = [("eager", "eager", None), ("torchscript", "torchscript", None),
                  ("onnx_fp32", "onnx", Path("models/mobilenetv2_crop_disease.onnx")),
                  ("onnx_int8", "onnx", Path("models/mobilenetv2_crop_disease_int8.onnx"))]
    reference, results = None, {}
    for name, runtime, path in candidates:
        check_pause()
        kwargs = {"onnx_path": path} if path else {}
        engine = CropDiseaseInferenceEngine(weights, device="cpu", runtime=runtime, **kwargs)
        if not engine.weights_loaded:
            raise ValueError(engine.load_error)
        logits, targets = [], []
        with torch.inference_mode():
            for images, labels in loader:
                check_pause()
                logits.append(engine.logits(images.to(memory_format=engine.memory_format)).cpu())
                targets.append(labels)
        logits, targets = torch.cat(logits), torch.cat(targets)
        if reference is None:
            reference = logits
        metrics = classification_metrics(targets.numpy(), logits.argmax(1).numpy())
        metrics.update({"prediction_agreement_with_eager": float(logits.argmax(1).eq(reference.argmax(1)).float().mean()),
                        "max_logit_difference": float((logits - reference).abs().max()),
                        "calibration": calibration_metrics(logits, targets, engine.temperature),
                        "eligible_quality": metrics["macro_f1"] >= results.get("eager", metrics)["macro_f1"] - .005})
        results[name] = metrics
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {"validation_images": len(dataset), "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                  "validation_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                  "max_allowed_macro_f1_drop": .005, "test_used_for_selection": False, "results": results}
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(name, {key: metrics[key] for key in ("macro_f1", "prediction_agreement_with_eager", "max_logit_difference", "eligible_quality")}, flush=True)
        del engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_optimization/results/optimization_20261017/runtime_validation.json"))
    args = parser.parse_args()
    compare(args.data, args.weights, args.output)


if __name__ == "__main__":
    main()
