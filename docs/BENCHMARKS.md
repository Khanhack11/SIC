# Kết quả nhận diện và hiệu năng thực thi

Bằng chứng dưới đây gắn với hai checkpoint đang giữ trong project. Đo runtime ngày 03/11/2026 trên Windows x86-64, Intel Core i5-12450H và NVIDIA RTX 2050 4 GB. Chất lượng nhận diện và tốc độ là hai phép đánh giá riêng.

## Chất lượng trên cùng validation

Manifest SHA-256 `4678faca1433bbb79d31185cdb6600994e29b777c9fbe8ed23bfb58fe92f5f24`, 2.740 ảnh. JSON gốc được sao chép nguyên byte từ lượt đánh giá đã có; hash weights được kiểm lại với file hiện tại. Không chạy lại test để chọn model.

| Domain | Số ảnh | V2 accuracy | V3 accuracy | V2 F1 lớp đã biết | V3 F1 lớp đã biết |
|---|---:|---:|---:|---:|---:|
| PlantVillage | 1731 | 94.63% | 98.27% | 94.48% | 98.17% |
| PlantDoc | 414 | 10.14% | 75.85% | 15.95% | 54.98% |
| MultiSourceField | 595 | 15.46% | 77.98% | 21.74% | 66.33% |

Accuracy tính trên toàn bộ ảnh domain, gồm Unsupported. Macro F1 lớp đã biết lấy trung bình các lớp SIC có ground truth hiện diện, giữ các nhầm lẫn sang đủ 12 nhãn. PlantDoc có 8 lớp SIC hiện diện; không diễn giải support = 0 thành model đã được đánh giá cho lớp đó. V2 không có đầu ra Unsupported, nên accuracy toàn domain chịu ảnh hưởng của ảnh ngoài 11 lớp.

| Cổng score ≥ 75%, dự đoán khác Unsupported | PlantDoc V2 | PlantDoc V3 | Yêu cầu V3 |
|---|---:|---:|---:|
| Accuracy trên ảnh qua cổng | 12.28% | 66.67% | ≥85% |
| Coverage trên toàn PlantDoc | 55.07% | 18.84% | ≥25% |

V3 chỉ đúng 52/78 dự đoán qua cổng trên 414 ảnh PlantDoc. Candidate chưa đạt tiêu chí phát hành. Đây là cổng score trước bước yêu cầu kiểm tra lại ảnh thực địa healthy trong API, chưa phải tỷ lệ chẩn đoán được chấp nhận cuối cùng. Tiêu chí đầy đủ ở [FIELD_RELEASE_CRITERIA.json](FIELD_RELEASE_CRITERIA.json).

Baseline từng được đánh giá trên test PlantVillage nền kiểm soát, 1.720 ảnh: accuracy 94,19%; macro precision 93,32%; macro recall 94,43%; macro F1 93,76%. ECE sau calibration 0,005654 (15 bins). Kết quả lịch sử này không chứng minh accuracy ngoài đồng; không so trực tiếp test đó với validation đa nguồn.

### Mở kết quả từng lớp

- [Precision, recall, F1 và support từng lớp, 83 dòng CSV](benchmarks/quality/per_class.csv).
- [V2 validation, đầy đủ confusion matrix/calibration](benchmarks/quality/baseline_validation.json).
- [V3 validation, đầy đủ confusion matrix/calibration](benchmarks/quality/candidate_validation.json).
- [V2 test PlantVillage lịch sử](benchmarks/quality/baseline_plantvillage_test_historical.json), [nguồn và checksum báo cáo](benchmarks/quality/provenance.json).

Confusion matrix có hàng là nhãn thật, cột là dự đoán theo class_names. Precision hỏi trong các ảnh model gọi là bệnh X có bao nhiêu ảnh đúng; recall hỏi bao nhiêu ảnh bệnh X được tìm đúng; F1 kết hợp hai tỷ lệ. Accuracy là số dự đoán đúng chia số ảnh. ECE đo chênh lệch giữa điểm tin cậy và độ đúng quan sát theo bin; ECE thấp không bảo đảm ảnh lạ được nhận ra.

Nhãn chủ yếu từ các bộ công khai, chưa được chuyên gia của nhóm duyệt đầy đủ. Holeta đã có trong train đa nguồn, không dùng làm external test độc lập của candidate này. Chưa có kết quả xác nhận trên ảnh nông hộ Việt Nam được thu mới và duyệt độc lập.

## Pipeline ảnh trên CPU/CUDA

Batch 1; FP32, không autocast; 2 thread CPU; warmup 10, đo 50 lần mỗi model/runtime trên 4 JPEG demo. Cột này gồm decode, preprocessing, chuyển tensor lên thiết bị, forward và xác suất; không gồm HTTP/tư vấn. CUDA được đồng bộ trước/sau timer, TF32 tắt. Ảnh demo chỉ dùng đo tốc độ.

| Model | Runtime | Mean (ms) | P95 (ms) |
|---|---|---:|---:|
| baseline | CPU ONNX | 44.27 | 56.14 |
| candidate | CPU ONNX | 41.81 | 64.61 |
| baseline | CPU PyTorch | 103.73 | 163.11 |
| candidate | CPU PyTorch | 95.01 | 136.52 |
| baseline | GPU PyTorch CUDA | 47.32 | 66.43 |
| candidate | GPU PyTorch CUDA | 55.52 | 74.22 |

CUDA chạy tensor thật trên GPU. ONNX CPU nhanh hơn PyTorch CUDA ở luồng từng ảnh trong phép đo này; GPU nhanh hơn PyTorch CPU. Thời gian còn phụ thuộc tải nền, nhiệt độ và điện năng. Không suy ra tốc độ điện thoại từ số đo máy chủ.

V2 có 2.237.963 tham số, V3 có 3.221.052. JSON còn ghi forward riêng ở batch 1/4, throughput ảnh/giây, DLL/PYD thực thi, Torch/CUDA và VRAM tensor peak. VRAM này không gồm toàn bộ bộ nhớ driver/CUDA context.

[CPU ONNX](benchmarks/deployment_20261103/cpu_onnx.json), [CPU PyTorch](benchmarks/deployment_20261103/cpu_pytorch.json), [GPU CUDA](benchmarks/deployment_20261103/gpu_cuda.json).

## API HTTP thực

Multipart upload qua localhost, gồm kiểm ảnh, hàng đợi, inference, tư vấn và JSON. Mỗi mức tải: 80 request, warmup 10, 11 ảnh validation, 2 thread và 2 inference slot. Closed-loop 1/4/8 client; không gồm mạng Internet. Ngưỡng request luân phiên 0,1 và 0,99 để kiểm tra tính độc lập, không phải ngưỡng mặc định của giao diện.

| Model/runtime | Client | Mean (ms) | P95 (ms) | Request/s |
|---|---:|---:|---:|---:|
| V2 CPU ONNX | 1 | 51.51 | 83.75 | 19.22 |
| V2 CPU ONNX | 4 | 156.99 | 234.98 | 24.67 |
| V2 CPU ONNX | 8 | 287.13 | 410.84 | 26.91 |
| V2 CUDA | 1 | 58.43 | 85.16 | 16.94 |
| V2 CUDA | 4 | 158.08 | 240.98 | 24.85 |
| V2 CUDA | 8 | 266.73 | 350.01 | 28.78 |
| V3 CPU ONNX | 1 | 48.10 | 61.43 | 20.61 |
| V3 CPU ONNX | 4 | 142.10 | 245.96 | 27.22 |
| V3 CPU ONNX | 8 | 245.93 | 338.51 | 30.99 |
| V3 CUDA | 1 | 64.66 | 87.25 | 15.31 |
| V3 CUDA | 4 | 189.68 | 316.66 | 20.62 |
| V3 CUDA | 8 | 334.91 | 424.75 | 22.64 |

P95 là mốc 95% request không vượt quá trong mẫu đo. Ở tải đồng thời, một số cấu hình vượt mục tiêu 200 ms; không báo toàn hệ thống luôn dưới 200 ms. Throughput request/s không phải FPS camera.

JSON HTTP: [V2 CPU](benchmarks/deployment_20261103/http_baseline_cpu.json), [V2 GPU](benchmarks/deployment_20261103/http_baseline_gpu.json), [V3 CPU](benchmarks/deployment_20261103/http_candidate_cpu.json), [V3 GPU](benchmarks/deployment_20261103/http_candidate_gpu.json).

## Đối chiếu V3 PyTorch và ONNX

Cùng 2,740 ảnh validation: khác nhãn 0, khác cổng score 0; chênh xác suất lớn nhất 7.7486038e-06, dung sai 0.0001. Parity: True; điều kiện runtime của phép đo: True.

[JSON đối chiếu đầy đủ](benchmarks/deployment_20261103/candidate_runtime_parity.json). Đây là đối chiếu cách thực thi cùng model, không nâng điều kiện phát hành chất lượng ảnh thực địa.

## Đo lại và xem nhãn

[Lệnh CPU/GPU](RUNTIME.md), [mã benchmark_deployment](../ml/benchmark_deployment.py), [mã api_benchmark](../ml/api_benchmark.py), [quy trình nhãn](DATA_LABELING.md). Đo từng cấu hình riêng khi không train/test nặng. Giữ output mỗi lượt, không ghi đè bằng chứng.

## Kiểm thử phần mềm sau khi sắp xếp model

61 unit/integration, 10 backend, 17 stress/edge, 25 frontend và 4 async checks đã đạt. Chromium/WebKit đạt ở viewport 360/390/412/844 px; đây là mô phỏng trên Windows, chưa phải điện thoại thật. API candidate đã kiểm riêng trên CPU và CUDA, gồm crop/ảnh lỗi. [Bằng chứng kiểm thử](benchmarks/deployment_20261103/software_checks.json). Log gốc nằm trong `tests/results/20261103_225746_236046/`.

## Thí nghiệm bổ sung nguồn dữ liệu

[Lượt CCMT/BARI/Ecuador bắt đầu 03/11/2026](final_release/alternative_sources_20261103/README.md) có manifest mới và báo cáo đối chiếu riêng. Các bảng phía trên vẫn là bằng chứng của hai model được đo trước lượt thử nghiệm này.
