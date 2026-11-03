# Chuẩn bị dữ liệu, train và đánh giá

Chạy từ thư mục SIC bằng `python -m ml.<module>`. Model phục vụ nằm trong `models/`; lượt train mới lưu tại `artifacts/training_runs/`. Một lượt train không tự thay model chạy web.

| Công việc | Mã nguồn |
|---|---|
| Tải và kiểm kê dữ liệu | `download_recommended_datasets`, `download_additional_field_datasets`, `analyze_recommended_datasets` |
| Chuẩn hóa nhãn | `labels`, `prepare_dataset`, `prepare_field_v2` |
| Xem nhãn thực tế và class index | `inspect_labels` |
| Train baseline V2 | `train`, `train_gpu` |
| Train candidate V3 từ đầu | `train_field` |
| Thí nghiệm fine-tune đa nguồn | `train_field_v2` |
| Tiếp tục V3, có checkpoint khôi phục | `continue_field` |
| Đánh giá trên split đã khóa | `evaluate`, `evaluate_field` |
| Kiểm tiêu chí phát hành | `field_release` |
| Export và đối chiếu PyTorch/ONNX | `export_onnx`, `export_field_onnx`, `validate_field_runtime` |
| Đo forward, pipeline CPU/CUDA | `benchmark_deployment` |
| Đo HTTP thực, tải đồng thời | `api_benchmark` |
| Đóng gói source và hai model đang dùng | `package_demo` |

## Dữ liệu và nhãn

`data/processed/field_mobile_v2/records.json` có 20.902 record: 16.225 train, 2.740 validation, 1.937 test. Nhãn nguồn được chuẩn hóa sang 12 chỉ số, gồm 11 lớp đã biết và Unsupported. Nhãn hiện chưa được chuyên gia của nhóm duyệt đầy đủ. Crop train từ PlantDoc dùng annotation của nguồn; crop trong web là thao tác thủ công.

```powershell
.venv/Scripts/python.exe -m ml.inspect_labels --output artifacts/label_review/review_new.csv
```

Lệnh từ chối ghi đè CSV có sẵn. Cột đề xuất sửa nhãn chỉ phục vụ duyệt, chưa tự cập nhật manifest. [Quy trình gán nhãn](../docs/DATA_LABELING.md).

## Train thực trên NVIDIA

Ví dụ tiếp tục từ candidate hiện tại, tạo lượt mới và giữ nguyên weights phục vụ:

```powershell
.venv-gpu/Scripts/python.exe -m ml.continue_field --source models/candidate/mobilenetv3_field.pt --data data/processed/field_mobile_v2 --output artifacts/training_runs/next_field/model.pt --epochs 4 --samples-per-epoch 6400 --unfreeze-blocks 4 --backbone-lr 0.000005 --head-lr 0.00005 --min-village-f1 0.95 --balanced-calibration --cosine-lr
```

Trainer đọc ảnh, augmentation, forward CUDA, cross-entropy với nhãn thật, backward và AdamW. Validation chọn checkpoint; test không tham gia chọn model. `continue_field` lưu optimizer, scaler, RNG và thứ tự/cursor lấy mẫu để khôi phục cùng lượt. Chỉ mở rộng dữ liệu qua `--expanded-data` khi audit đáp ứng hợp đồng parent của trainer.

`train_field_v2` là mã thí nghiệm fine-tune đa nguồn, không có đầy đủ cơ chế khôi phục optimizer giữa chừng như `continue_field`. Các module `optimize_head`, `finetune_projection`, `recover_field_candidate` giữ để tái hiện thí nghiệm cũ; khi cần parent đã lưu trữ, truyền đường dẫn rõ ràng.

Train lâu hơn chưa bảo đảm tăng chất lượng. Hai lượt tiếp tục gần nhất không vượt parent theo tiêu chí chọn checkpoint; candidate giữ weights parent với metadata/calibration đã cập nhật. Không tạo thêm bản model phục vụ nếu chưa vượt tiêu chí đã khóa.

## Đánh giá và benchmark

```powershell
.venv/Scripts/python.exe -m ml.evaluate_field --data data/processed/field_mobile_v2 --weights models/candidate/mobilenetv3_field.pt --split validation --output artifacts/evaluation/next_validation.json
.venv/Scripts/python.exe -m ml.benchmark_deployment --device cpu --runtime onnx --output artifacts/benchmarks/cpu_new.json
.venv-gpu/Scripts/python.exe -m ml.benchmark_deployment --device cuda --runtime eager --output artifacts/benchmarks/gpu_new.json
```

Chọn tên output mới cho mỗi bằng chứng. Không chạy train hoặc kiểm thử nặng song song khi đo latency. [Benchmark và giới hạn](../docs/BENCHMARKS.md), [runtime CPU/GPU](../docs/RUNTIME.md).
