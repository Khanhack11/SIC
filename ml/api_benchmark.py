"""Benchmark real localhost HTTP with owned subprocesses and request-local thresholds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from backend.config import CLASS_NAMES
from ml.benchmark import summarize
from ml.dataset import ManifestDataset
from ml.runtime import check_pause


def measure(args):
    dataset = ManifestDataset(args.data / "validation.csv")
    selected = [next(row for row in dataset.rows if row["label"] == label) for label in CLASS_NAMES]
    images = [(dataset.source_root / row["path"]).read_bytes() for row in selected]
    results = []
    for runtime in args.runtimes:
        for threads in args.threads:
            for slots in args.slots:
                check_pause()
                with socket.socket() as candidate:
                    candidate.bind(("127.0.0.1", 0))
                    port = candidate.getsockname()[1]
                environment = {**os.environ, "SIC_INFERENCE_RUNTIME": "onnx" if runtime == "onnx_int8" else runtime,
                               "SIC_TORCH_THREADS": str(threads),
                               "SIC_INFERENCE_CONCURRENCY": str(slots)}
                if runtime in {"onnx", "onnx_int8"}:
                    graph_name = "mobilenetv2_crop_disease_int8.onnx" if runtime == "onnx_int8" else "mobilenetv2_crop_disease.onnx"
                    environment["SIC_ONNX_PATH"] = str((Path("models") / graph_name).resolve())
                log_path = Path("artifacts") / f"api_{runtime}_{threads}_{slots}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("wb") as log:
                    command = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
                               "--port", str(port), "--no-access-log"]
                    server = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    ownership = {"harness_pid": os.getpid(), "server_pid": server.pid, "command": command,
                                 "cwd": str(Path.cwd()), "started_at_unix": time.time(), "active": True}
                    registry = Path("artifacts/api_active_process.json")
                    registry.write_text(json.dumps(ownership, indent=2), encoding="utf-8")
                    try:
                        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30,
                                          limits=httpx.Limits(max_connections=max(args.concurrency))) as client:
                            deadline = time.monotonic() + 60
                            while True:
                                check_pause()
                                if server.poll() is not None:
                                    raise RuntimeError(f"Server exited; inspect {log_path}")
                                try:
                                    health = client.get("/api/health").json()
                                    if not health["weights_loaded"]:
                                        raise RuntimeError(f"Model not ready; inspect {log_path}")
                                    break
                                except httpx.ConnectError:
                                    if time.monotonic() > deadline:
                                        raise TimeoutError("Server startup deadline exceeded")
                                    time.sleep(.2)

                            def request(index):
                                check_pause()
                                image_index = index % len(images)
                                threshold = .1 if index % 2 == 0 else .99
                                started = time.perf_counter()
                                response = client.post("/api/diagnose", params={"custom_threshold": threshold},
                                                       files={"file": (Path(selected[image_index]["path"]).name,
                                                                       images[image_index],
                                                                       "image/png" if selected[image_index]["path"].lower().endswith(".png") else "image/jpeg")})
                                elapsed = (time.perf_counter() - started) * 1000
                                response.raise_for_status()
                                payload = response.json()
                                if payload["prediction"]["threshold_applied"] != threshold:
                                    raise AssertionError("Cross-request threshold contamination")
                                timing = response.headers.get("server-timing", "")
                                processing = next((float(part.split(";dur=")[1]) for part in timing.split(",")
                                                   if part.strip().startswith("processing;dur=")), None)
                                return {"http_ms": elapsed, "inference_ms": payload["prediction"]["latency_ms"],
                                        "processing_ms": processing}

                            for index in range(args.warmup):
                                request(index)
                            for concurrency in args.concurrency:
                                started = time.perf_counter()
                                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                    values = list(pool.map(request, range(args.runs)))
                                elapsed = time.perf_counter() - started
                                row = {"runtime": runtime, "threads": threads, "inference_slots": slots,
                                       "clients": concurrency, **summarize([value["http_ms"] for value in values], elapsed),
                                       "mean_server_inference_ms": sum(value["inference_ms"] for value in values) / len(values)}
                                if all(value["processing_ms"] is not None for value in values):
                                    row["mean_server_processing_ms"] = sum(value["processing_ms"] for value in values) / len(values)
                                results.append(row)
                                args.output.parent.mkdir(parents=True, exist_ok=True)
                                report = {"scope": "Real localhost HTTP multipart upload, validation, inference, advisory and JSON response",
                                          "load": "Closed-loop clients; client wall latency includes server queueing, excludes client executor queue",
                                          "checkpoint_sha256": hashlib.sha256(Path("models/mobilenetv2_crop_disease.pt").read_bytes()).hexdigest(),
                                          "validation_manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
                                          "warmup": args.warmup, "images": len(images), "results": results,
                                          "custom_threshold_isolation": True}
                                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                                print(json.dumps(row), flush=True)
                    finally:
                        if server.poll() is None:
                            server.terminate()
                            try:
                                server.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                server.kill()
                                server.wait(timeout=5)
                        registry.write_text(json.dumps({**ownership, "active": False}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--runtimes", nargs="+", choices=["eager", "torchscript", "onnx", "onnx_int8"], default=["eager", "torchscript", "onnx"])
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/data_optimization/results/optimization_20261017/api_benchmark.json"))
    args = parser.parse_args()
    if min(args.runs, args.warmup, *args.threads, *args.slots, *args.concurrency) < 1:
        parser.error("All counts must be positive")
    measure(args)


if __name__ == "__main__":
    main()
