# SIC · Nhận diện bệnh lá

Ứng dụng nhận ảnh lá ngô, khoai tây hoặc cà chua, cho phép crop thủ công rồi phân tích qua FastAPI. Một MobileNetV2 phục vụ mặc định; một MobileNetV3-Large là candidate nghiên cứu, chưa qua tiêu chí phát hành thực địa.

## Chạy trên CPU hoặc GPU

| Chế độ | Lệnh Windows | Model thực thi | Giao diện |
|---|---|---|---|
| CPU x86-64 | `./run_demo.cmd` | V2 ONNX, CPUExecutionProvider | http://127.0.0.1:8000/app |
| GPU NVIDIA | `./run_gpu_demo.cmd` | V2 `.pt`, PyTorch CUDA | http://127.0.0.1:8000/app |
| Candidate CPU | `./run_field_demo.cmd` | V3 `.pt`, PyTorch CPU | http://127.0.0.1:8003/app |
| Candidate GPU | `./run_field_demo.cmd gpu` | V3 `.pt`, PyTorch CUDA | http://127.0.0.1:8003/app |
| Candidate ONNX CPU | `./run_field_demo.cmd onnx` | V3 ONNX, CPUExecutionProvider | http://127.0.0.1:8003/app |

Chọn một chế độ cho mỗi cổng. Giữ server chạy khi dùng web. `/api/health` báo `device`, `inference_runtime`, `weights_loaded`; GPU phải báo `cuda`, không tự đổi sang CPU. Máy hiện tại: Intel Core i5-12450H 64-bit và NVIDIA RTX 2050 4 GB.

Cài môi trường CPU:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-demo.txt
```

Cài môi trường GPU:

```powershell
python -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-gpu-demo.txt
```

Weights bàn giao riêng, không nằm trong Git. [Các file đang dùng](models/README.md).

## Mã nào thực thi model?

| Thành phần | Nhiệm vụ |
|---|---|
| [backend/inference_engine.py](backend/inference_engine.py) | Tạo mạng V2, nạp `.pt`/`.onnx`, chạy `logits()` và `predict()` |
| [backend/mobile_model.py](backend/mobile_model.py) | Kiến trúc V3-Large, head 960 → 256 → 12 |
| [backend/field_inference.py](backend/field_inference.py) | Thực thi V3 trên CPU/CUDA hoặc ONNX CPU |
| [backend/config.py](backend/config.py) | `SIC_DEVICE=cpu/cuda`, `SIC_INFERENCE_RUNTIME=onnx/eager` |
| [backend/main.py](backend/main.py) | Upload, kiểm ảnh, gọi model và cổng tư vấn |

`.pt` lưu tensor trọng số và metadata PyTorch; `.onnx` lưu graph và trọng số đã export. Thư viện native x86-64 hoặc CUDA thực thi các phép toán. Định dạng file model và kiến trúc CPU là hai thông tin khác nhau. [Giải thích runtime và mã native](docs/RUNTIME.md).

## Gán nhãn ở đâu?

Nhãn hiện tại chủ yếu đến từ dữ liệu công khai. Code chuẩn hóa tên lớp rồi ghi vào manifest; nhóm chưa xác nhận lại toàn bộ nhãn bằng chuyên gia.

- V2: `data/processed/plantvillage/{train,validation,test}.csv` có `label`, `class_index`.
- V3: `data/processed/field_mobile_v2/records.json` có ảnh, label, SHA, source, domain và split.
- [ml/labels.py](ml/labels.py) ánh xạ tên lớp; [ml/prepare_field_v2.py](ml/prepare_field_v2.py) chuẩn bị đa nguồn; `FieldDataset` trong [ml/train_field.py](ml/train_field.py) đưa ảnh và nhãn vào tensor.

```powershell
.venv/Scripts/python.exe -m ml.inspect_labels --output artifacts/label_review/review_new.csv
```

CSV xuất nhãn thật, chỉ số lớp, nguồn, split, SHA và cột đề xuất duyệt lại. Sửa CSV không tự thay manifest đã khóa. [Quy trình và ví dụ thật](docs/DATA_LABELING.md).

## Benchmark

[docs/BENCHMARKS.md](docs/BENCHMARKS.md) tập trung kết quả và JSON gốc. Chất lượng nhận diện gồm accuracy, precision, recall, F1 từng lớp, confusion matrix, ECE và accuracy/coverage qua ngưỡng. Hiệu năng gồm CPU/GPU, FP32, batch, warmup, số lần đo, mean/P50/P95, throughput và VRAM.

Validation CPU đã khóa trên cùng manifest:

| Tập | Ảnh | V2 macro F1 lớp hiện diện | V3 macro F1 lớp hiện diện |
|---|---:|---:|---:|
| PlantVillage | 1.731 | 94,48% | 98,17% |
| PlantDoc, lớp đã biết | 414 | 15,95% | 54,98% |
| MultiSourceField | 595 | 21,74% | 66,33% |

V2 từng đạt accuracy 94,19%, F1 93,76% trên 1.720 test ảnh PlantVillage nền kiểm soát. Đây không phải accuracy ảnh thực địa. V3 có score-gate accuracy 66,67%, coverage 18,84% trên PlantDoc, chưa đạt yêu cầu 85% và 25%.

## Cấu trúc

```text
backend/                 API, mạng và inference
frontend/                Upload, crop thủ công và kết quả
models/                  Một baseline và một candidate
ml/                      Chuẩn bị dữ liệu, train, đánh giá, export
artifacts/training_runs/ Đầu ra lượt train mới, không tự đưa vào models
artifacts/label_review/  Bảng xuất nhãn để xem và duyệt
data/raw/               Ảnh nguồn và metadata, local
data/processed/         Manifest/split đã khóa, local
docs/                   Hướng dẫn và bằng chứng
tests/                  Kiểm thử phần mềm
```

Các model lịch sử/progress cũ được lưu nguyên vẹn ngoài project tại `Reports/SIC_Model_Archive_20261103_222709`; `archive_manifest.json` ghi đường dẫn cũ/mới và SHA. Hồ sơ đo cũ giữ nguyên để đối chiếu. [Quản lý hai model hiện tại](models/README.md).

## Kiểm thử và giới hạn

```powershell
.venv/Scripts/python.exe tests/run_all.py --web --mobile
```

Chuẩn bị Node/Playwright theo [tests/README.md](tests/README.md). Log nằm trong `tests/results/`.

Ảnh người dùng mặc định là ảnh thực địa; dự đoán healthy vẫn cần kiểm tra lại. Confidence là điểm model, không phải accuracy chẩn đoán đã được chứng minh. Crop hiện thủ công; camera là chức năng bổ sung. Nhãn thuốc tại Việt Nam chưa được duyệt nên API ẩn hướng dẫn liều dùng. [Phạm vi và các phase](docs/MOBILE_FIELD_PHASES.md).

Nhóm 5, Samsung Innovation Campus. Báo cáo 04–05/11/2026, thời lượng 20 phút.
