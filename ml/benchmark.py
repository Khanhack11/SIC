"""Compare CPU thread/runtime settings using validation images and simulated load.

Measures validation, preprocessing, trained inference and probability conversion.
HTTP/network and agricultural advice retrieval are outside the measured region.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from backend.config import CLASS_NAMES

from backend.inference_engine import CropDiseaseInferenceEngine, ModelNotReadyError, validate_image_file
from ml.dataset import ManifestDataset
from ml.runtime import check_pause


def summarize(values, elapsed):
    values = np.asarray(values)
    return {"count": len(values), "mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(values.max()),
            "throughput_requests_per_second": len(values) / elapsed,
            "mean_under_200ms": bool(values.mean() < 200), "p95_under_200ms": bool(np.percentile(values, 95) < 200)}


def benchmark(args):
    dataset = ManifestDataset(args.data / "validation.csv", args.source)
    # Cycle across all classes, not just the first alphabetically sorted images.
    selected = []
    for label in range(len(CLASS_NAMES)):
        selected.extend(row for row in dataset.rows if int(row["class_index"]) == label)
    selected = [next(row for row in selected if int(row["class_index"]) == label) for label in range(len(CLASS_NAMES))]
    inputs = [(dataset.source_root / row["path"]).read_bytes() for row in selected]
    filenames = [Path(row["path"]).name for row in selected]
    rows, references = [], None
    for runtime in args.runtimes:
        for threads in args.threads:
            check_pause()
            torch.set_num_threads(threads)
            engine = CropDiseaseInferenceEngine(args.weights, device="cpu", runtime=runtime)
            if not engine.weights_loaded:
                raise ModelNotReadyError(engine.load_error or "Trained checkpoint required")
            predictions = [engine.predict(data) for data in inputs]
            if references is None:
                references = predictions
            parity = all(a["predicted_class"] == b["predicted_class"] and
                         max(abs(a["all_probabilities"][name] - b["all_probabilities"][name])
                             for name in a["all_probabilities"]) <= .0002
                         for a, b in zip(references, predictions))
            if not parity:
                raise ValueError("Optimized runtime differs from the reference predictions")
            for index in range(args.warmup):
                engine.predict(inputs[index % len(inputs)])

            def request(index):
                check_pause()
                index %= len(inputs)
                started = time.perf_counter()
                validate_image_file(inputs[index], filenames[index])
                engine.predict(inputs[index])
                return (time.perf_counter() - started) * 1000

            for concurrency in args.concurrency:
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    latencies = list(pool.map(request, range(args.runs)))
                elapsed = time.perf_counter() - started
                result = {"runtime": runtime, "threads": threads, "concurrency": concurrency,
                          "prediction_parity": parity, **summarize(latencies, elapsed)}
                rows.append(result)
                print(json.dumps(result), flush=True)
    serial = [row for row in rows if row["concurrency"] == 1]
    recommended = min(serial or rows, key=lambda row: row["p95_ms"])
    report = {
        "device": "cpu", "processor": platform.processor(), "platform": platform.platform(),
        "torch_version": str(torch.__version__), "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
        "warmup_requests": args.warmup, "images": len(inputs), "results": rows,
        "recommended_serial_setting": {key: recommended[key] for key in ("runtime", "threads", "p95_ms")},
        "scope": "In-process image validation + preprocess + trained inference + probabilities; no HTTP/network/advisory.",
        "load_definition": "Closed-loop ThreadPoolExecutor clients; wall time includes task execution, not client queue wait.",
        "parity_scope": "One validation image per class; probabilities compared after 4-decimal rounding, tolerance 0.0002.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_optimization/results/benchmark.json"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--runtimes", nargs="+", choices=["eager", "torchscript", "onnx"], default=["eager", "torchscript"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(args.runs, args.warmup, *args.threads, *args.concurrency) < 1:
        parser.error("runs, warmup, threads and concurrency must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
