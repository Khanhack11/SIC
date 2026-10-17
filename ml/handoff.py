"""Bundle reproducible manifests and measured results for the team handoff."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

from backend.config import CLASS_NAMES


def bundle(data, weights, destination):
    data, weights, destination = Path(data), Path(weights), Path(destination)
    results = destination / "results"
    audit = json.loads((data / "audit.json").read_text(encoding="utf-8"))
    evaluation = json.loads((results / "evaluation.json").read_text(encoding="utf-8"))
    benchmark = json.loads((results / "benchmark.json").read_text(encoding="utf-8"))
    training = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    checkpoint_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if evaluation["checkpoint_sha256"] != checkpoint_hash or benchmark["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Reports were produced with a different checkpoint")
    if evaluation["test_manifest_sha256"] != audit["manifest_sha256"]["test"]:
        raise ValueError("Evaluation used a different test manifest")
    if benchmark["validation_manifest_sha256"] != audit["manifest_sha256"]["validation"]:
        raise ValueError("Benchmark used a different validation manifest")
    for split in ("train", "validation"):
        if training[f"{split}_manifest_sha256"] != audit["manifest_sha256"][split]:
            raise ValueError("Training used different manifests")
    manifests = destination / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = data / f"{split}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["manifest_sha256"][split]:
            raise ValueError("Manifest changed after audit")
        shutil.copyfile(path, manifests / path.name)
    for name in ("audit.json", "labels.json", "excluded.json"):
        shutil.copyfile(data / name, results / name)
    shutil.copyfile(weights.with_suffix(".json"), results / "training.json")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "dependencies": {name: version(name) for name in
                                    ("torch", "torchvision", "pillow", "numpy", "fastapi", "uvicorn", "httpx")}}
    (results / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (results / "checkpoint.sha256").write_text(f"{checkpoint_hash}  {weights.name}\n", encoding="utf-8")
    best = benchmark["recommended_serial_setting"]
    best_rows = [row for row in benchmark["results"] if row["runtime"] == best["runtime"] and row["threads"] == best["threads"]]
    serial = next((row for row in best_rows if row["concurrency"] == 1), best_rows[0])
    per_class = evaluation["per_class"]
    lines = [
        "# Kết quả thực hiện phần Data và Optimization", "",
        "Người phụ trách Trần Đặng Công Tâm, Group 5. Nhánh local `tran-dang-cong-tam`.", "",
        "## Dữ liệu đã chuẩn bị", "",
        f"Đã tải {audit['input_valid_count']:,} ảnh RGB, loại {audit['excluded_count']} ảnh và giữ {audit['curated_count']:,} ảnh. "
        f"Các tập có {audit['splits']['train']['count']:,} train, {audit['splits']['validation']['count']:,} validation "
        f"và {audit['splits']['test']['count']:,} test. Không giao nhau về nhóm đã xác định, hash file hoặc hash pixel.", "",
        f"Có {audit['leaf_map_matches']:,} ảnh khớp leaf-map và {audit['leaf_map_fallbacks']:,} ảnh dùng identifier dự phòng. "
        "Không thể khẳng định toàn bộ lá vật lý đã tách biệt chỉ bằng metadata; cần giữ giới hạn này khi báo cáo kết quả.", "",
        "| Lớp | Train | Validation | Test | Test F1 |", "|---|---:|---:|---:|---:|",
    ]
    for label in CLASS_NAMES:
        counts = [audit["splits"][split]["per_class"][label] for split in ("train", "validation", "test")]
        lines.append(f"| {label} | {counts[0]} | {counts[1]} | {counts[2]} | {per_class[label]['f1']:.2%} |")
    lines += ["", "## Baseline và chất lượng trên test", "",
              "MobileNetV2 với ImageNet backbone đóng băng, hai augmented feature views mỗi ảnh train; "
              "head được huấn luyện bằng Adam, cosine annealing, class-weighted cross-entropy và early stopping. "
              "Temperature scaling được fit trên validation. Test không dùng để chọn model hoặc cấu hình.", "",
              f"Epoch tốt nhất {training['best_epoch']}; đã chạy {len(training['history'])} epoch. "
              f"Temperature {evaluation['temperature']:.4f}.", "",
              "| Chỉ số | Kết quả |", "|---|---:|",
              f"| Test accuracy | {evaluation['accuracy']:.2%} |",
              f"| Macro precision | {evaluation['macro_precision']:.2%} |",
              f"| Macro recall | {evaluation['macro_recall']:.2%} |",
              f"| Macro F1 | {evaluation['macro_f1']:.2%} |",
              f"| Mục tiêu macro F1 ≥85% | {'Đạt' if evaluation['macro_f1_target_met'] else 'Chưa đạt'} |",
              f"| Test ECE trước temperature scaling | {evaluation['calibration_before']['ece']:.4f} |",
              f"| Test ECE sau temperature scaling | {evaluation['calibration_after']['ece']:.4f} |",
              "", "Confusion matrix đầy đủ ở `results/evaluation.csv`; chỉ số từng lớp ở `results/evaluation.json`.", "",
              "## Hiệu năng đo trên CPU", "",
              f"Cấu hình serial tốt nhất trong các cấu hình đã đo là runtime `{best['runtime']}`, {best['threads']} CPU threads. "
              f"Mean {serial['mean_ms']:.2f} ms, p95 {serial['p95_ms']:.2f} ms. "
              f"Mục tiêu mean dưới 200ms: {'đạt' if serial['mean_under_200ms'] else 'chưa đạt'}.", "",
              "| Concurrency | Mean ms | P95 ms | Requests mỗi giây |", "|---:|---:|---:|---:|"]
    for row in best_rows:
        lines.append(f"| {row['concurrency']} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} | {row['throughput_requests_per_second']:.2f} |")
    lines += ["", "Thời gian benchmark gồm validation ảnh, preprocessing, inference và xác suất trong process; "
              "không bao gồm HTTP, mạng hoặc tư vấn. Benchmark dùng một ảnh validation mỗi lớp và warmup. "
              "Đây là phép đo trên máy này với tải đóng vòng; cần đo lại trên máy triển khai.", "",
              "## Bàn giao và giới hạn", "",
              "- Bảo nhận manifests, thứ tự nhãn và pipeline để tiếp tục fine-tune nếu cần.",
              "- Khánh nhận checkpoint và cấu hình runtime; backend kiểm tra metadata và trả 503 nếu thiếu model hợp lệ.",
              "- Hải dùng API để thử ảnh thực địa và giao diện phản hồi confidence thấp.",
              "- Trọng số và ảnh nằm local, bị Git bỏ qua; cần bàn giao riêng hoặc tái tạo bằng lệnh trong README.",
              "- Kết quả trên ảnh nền kiểm soát PlantVillage chưa chứng minh chất lượng trên ảnh ngoài đồng.",
              "- Potato_healthy có ít mẫu hơn các lớp khác; F1 của lớp này cần được đọc cùng support.", "",
              f"Checkpoint SHA-256 `{checkpoint_hash}`.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Handoff report: {destination / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--destination", type=Path, default=Path("docs/data_optimization"))
    args = parser.parse_args()
    bundle(args.data, args.weights, args.destination)


if __name__ == "__main__":
    main()
