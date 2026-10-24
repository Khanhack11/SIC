"""Bounded validation-only search of regularized linear heads using audited features.

Standardization is folded into the saved linear layer, so serving stays MobileNetV2.
The deployed checkpoint is never overwritten. Completed candidates are resumable.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from backend.config import CLASS_NAMES
from backend.inference_engine import CropDiseaseInferenceEngine
from ml.dataset import ManifestDataset
from ml.evaluate import classification_metrics, calibration_metrics
from ml.runtime import check_pause
from ml.train import calibrate


def fold_standardization(weight, bias, mean, scale):
    folded = weight / scale
    return folded, bias - folded @ mean


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.2)


def metrics(logits, targets):
    result = classification_metrics(targets.numpy(), logits.argmax(1).numpy())
    result["nll"] = float(nn.functional.cross_entropy(logits, targets))
    return result


def optimize(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Candidate checkpoint already exists; choose a new output")
    saved = torch.load(args.cache, map_location="cpu", weights_only=True)
    baseline = torch.load(args.weights, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    for key in ("class_names", "input_size", "mean", "std"):
        if metadata[key] != baseline[key]:
            raise ValueError(f"Feature-cache contract mismatch: {key}")
    for split in ("train", "validation"):
        digest = hashlib.sha256((args.data / f"{split}.csv").read_bytes()).hexdigest()
        if digest != metadata[f"{split}_manifest_sha256"]:
            raise ValueError(f"Feature cache uses different {split} manifest")
    validation = ManifestDataset(args.data / "validation.csv")
    if not torch.equal(saved["val_targets"], torch.tensor([int(row["class_index"]) for row in validation.rows])):
        raise ValueError("Feature-cache labels differ from validation manifest")
    engine = CropDiseaseInferenceEngine(args.weights, device="cpu", runtime="eager")
    if not engine.weights_loaded:
        raise ValueError(engine.load_error or "Baseline unavailable")
    # Recompute distributed validation samples to check cache/backbone provenance.
    indices = torch.linspace(0, len(validation) - 1, 16).long().tolist()
    with torch.inference_mode():
        images = torch.stack([validation[index][0] for index in indices])
        actual = engine.model.features(images.to(memory_format=torch.channels_last)).mean((2, 3))
        torch.testing.assert_close(actual, saved["val_features"][indices], atol=1e-4, rtol=1e-4)
    x, y = saved["train_features"], saved["train_targets"]
    vx, vy = saved["val_features"], saved["val_targets"]
    if len(x) != len(y) or len(vx) != len(vy) or x.shape[1] != vx.shape[1]:
        raise ValueError("Invalid cached tensor dimensions")
    if not torch.isfinite(x).all() or not torch.isfinite(vx).all():
        raise ValueError("Non-finite cached features")
    mean, scale = x.mean(0), x.std(0).clamp_min(.01)
    x, vx_standard = (x - mean) / scale, (vx - mean) / scale
    counts = torch.bincount(y, minlength=len(CLASS_NAMES)).float()
    if (counts == 0).any():
        raise ValueError("Each class must be present in training data")
    state = baseline["state_dict"]
    with torch.no_grad():
        baseline_logits = nn.functional.linear(vx, state["classifier.1.weight"], state["classifier.1.bias"])
    reference = metrics(baseline_logits, vy)
    baseline_macro = reference["macro_f1"]
    run_dir = args.output.parent / (args.output.stem + "_search")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = {"baseline_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = {**fingerprint, "baseline_validation": reference, "candidates": [], "selected": "baseline",
              "selection": "macro F1; NLL tie-break; require macro F1 improvement >=0.002 for promotion",
              "test_used_for_selection": False, "validation_reused_from_prior_rounds": True}
    best = (reference["macro_f1"], -reference["nll"])
    best_head = (state["classifier.1.weight"].clone(), state["classifier.1.bias"].clone())
    for exponent in (.5, 1.):
        for regularization in (.0001, .001, .01):
            check_pause()
            candidate = f"weight{exponent:g}_l2{regularization:g}"
            progress = run_dir / f"{candidate}.pt"
            torch.manual_seed(42)
            head = nn.Linear(x.shape[1], len(CLASS_NAMES))
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            weights = (counts.sum() / (len(CLASS_NAMES) * counts)).pow(exponent)
            optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=10, history_size=20,
                                         line_search_fn="strong_wolfe")
            start_chunk = 0
            if progress.exists():
                resume = torch.load(progress, weights_only=True)
                if resume["fingerprint"] != fingerprint:
                    raise ValueError("Search inputs changed; choose a new output name")
                head.load_state_dict(resume["head"])
                optimizer.load_state_dict(resume["optimizer"])
                start_chunk = resume["chunks"]

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(head(x), y, weight=weights)
                loss = loss + regularization * head.weight.square().sum() / 2
                loss.backward()
                return loss

            for chunk in range(start_chunk, 10):
                check_pause()
                optimizer.step(closure)
                atomic_save({"fingerprint": fingerprint, "head": head.state_dict(),
                             "optimizer": optimizer.state_dict(), "chunks": chunk + 1}, progress)
                check_pause()
            with torch.no_grad():
                logits = head(vx_standard)
                result = metrics(logits, vy)
                folded_weight, folded_bias = fold_standardization(head.weight, head.bias, mean, scale)
                torch.testing.assert_close(nn.functional.linear(vx, folded_weight, folded_bias), logits,
                                           atol=1e-4, rtol=1e-4)
            result.update(name=candidate, class_weight_exponent=exponent, l2=regularization)
            report["candidates"].append(result)
            score = (result["macro_f1"], -result["nll"])
            if score > best and result["macro_f1"] >= baseline_macro + .002:
                best = score
                best_head = (folded_weight.detach().clone(), folded_bias.detach().clone())
                report["selected"] = candidate
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{candidate}: validation macro F1={result['macro_f1']:.6f}, NLL={result['nll']:.4f}", flush=True)
    check_pause()
    with torch.no_grad():
        logits = nn.functional.linear(vx, *best_head)
    temperature, calibration = calibrate(logits.detach(), vy)
    candidate_checkpoint = copy.deepcopy(baseline)
    candidate_checkpoint["state_dict"]["classifier.1.weight"] = best_head[0]
    candidate_checkpoint["state_dict"]["classifier.1.bias"] = best_head[1]
    candidate_checkpoint.update(temperature=temperature, calibration=calibration)
    report["selected_validation"] = metrics(logits, vy)
    report["selected_calibration"] = calibration_metrics(logits.detach(), vy, temperature)
    report["calibration"] = calibration
    candidate_checkpoint["optimization"] = report
    atomic_save(candidate_checkpoint, args.output)
    report["candidate_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected {report['selected']}; validation macro F1={best[0]:.6f}; saved {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/features_f96fe288aaf4be0d.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mobilenetv2_head_20261024.pt"))
    parser.add_argument("--report", type=Path, default=Path("docs/data_optimization/results/optimization_20261024/head_search.json"))
    optimize(parser.parse_args())


if __name__ == "__main__":
    main()
