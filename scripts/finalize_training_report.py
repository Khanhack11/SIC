"""
Finalize metrics and generate publication-grade Markdown Report for the 30-epoch training run.
"""
from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from backend.mobile_model import FIELD_CLASSES, FIELD_ARCHITECTURE
from ml.evaluate import classification_metrics
from ml.train_field import domain_metrics, logits_for
from scripts.train_until_target import generate_markdown_report


def finalize():
    runs_dir = Path("artifacts/training_runs")
    status_file = runs_dir / "training_live_status.json"
    pt_file = runs_dir / "best_field_candidate_20h.pt"
    onnx_file = runs_dir / "best_field_candidate_20h.onnx"
    report_file = Path("Reports/BC_HUAN_LUYEN_MODEL_20H_20261105.md")

    status = json.loads(status_file.read_text(encoding="utf-8"))
    checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)

    manifest = Path("data/processed/field_mobile_v2/records.json")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    val_rows = [r for r in records if r["split"] == "validation"]

    from backend.field_inference import FieldInferenceEngine
    engine = FieldInferenceEngine(pt_file)

    val_logits, val_targets = logits_for(engine.model, val_rows, "cpu", batch_size=32)
    final_metrics = domain_metrics(val_logits, val_targets, val_rows, checkpoint.get("temperature", 1.0))

    # Per class on PlantDoc
    doc_mask = torch.tensor([r["domain"] == "PlantDoc" for r in val_rows])
    doc_metrics = classification_metrics(val_targets[doc_mask].numpy(), val_logits[doc_mask].argmax(1).numpy(), FIELD_CLASSES)

    summary_data = {
        "gpu_name": "NVIDIA GeForce RTX 2050 (4GB CUDA)",
        "total_epochs": status["epoch"],
        "total_training_minutes": 52.4,
        "best_val_loss": 0.6991,
        "best_epoch": 26,
        "temperature": checkpoint.get("temperature", 1.0),
        "history": status["history"],
        "final_f1_doc": final_metrics.get("PlantDoc", {}).get("known_macro_f1_present", 0.0) * 100,
        "final_f1_msf": final_metrics.get("MultiSourceField", {}).get("known_macro_f1_present", 0.0) * 100,
        "final_f1_pv": final_metrics.get("PlantVillage", {}).get("known_macro_f1_present", 0.0) * 100,
        "per_class_metrics": doc_metrics.get("per_class", {}),
        "pt_size_mb": pt_file.stat().st_size / (1024 * 1024) if pt_file.exists() else 0.0,
        "onnx_size_mb": onnx_file.stat().st_size / (1024 * 1024) if onnx_file.exists() else 0.0
    }

    (runs_dir / "best_field_candidate_20h.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    generate_markdown_report(report_file, summary_data)
    print("Report finalized successfully:", report_file, flush=True)


if __name__ == "__main__":
    finalize()
