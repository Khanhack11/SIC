# Kết quả thực hiện phần Data và Optimization

Người phụ trách Trần Đặng Công Tâm, Group 5. Nhánh local `tran-dang-cong-tam`.

## Dữ liệu đã chuẩn bị

Đã tải 11,504 ảnh RGB, loại 10 ảnh và giữ 11,494 ảnh. Các tập có 8,043 train, 1,731 validation và 1,720 test. Không giao nhau về nhóm đã xác định, hash file hoặc hash pixel.

Có 5,671 ảnh khớp leaf-map và 5,833 ảnh dùng identifier dự phòng. Không thể khẳng định toàn bộ lá vật lý đã tách biệt chỉ bằng metadata; cần giữ giới hạn này khi báo cáo kết quả.

| Lớp | Train | Validation | Test | Test F1 |
|---|---:|---:|---:|---:|
| Corn_Common_rust | 834 | 179 | 179 | 99.44% |
| Corn_Gray_leaf_spot | 359 | 77 | 77 | 84.85% |
| Corn_Northern_Leaf_Blight | 689 | 148 | 148 | 91.61% |
| Corn_healthy | 814 | 174 | 174 | 99.71% |
| Potato_Early_blight | 700 | 152 | 148 | 99.66% |
| Potato_Late_blight | 700 | 152 | 148 | 96.14% |
| Potato_healthy | 104 | 24 | 24 | 94.12% |
| Tomato_Early_blight | 700 | 151 | 149 | 86.18% |
| Tomato_Late_blight | 1047 | 225 | 224 | 91.34% |
| Tomato_Septoria_leaf_spot | 1050 | 225 | 225 | 90.99% |
| Tomato_healthy | 1046 | 224 | 224 | 97.35% |

## Baseline và chất lượng trên test

MobileNetV2 với ImageNet backbone đóng băng, hai augmented feature views mỗi ảnh train; head được huấn luyện bằng Adam, cosine annealing, class-weighted cross-entropy và early stopping. Temperature scaling được fit trên validation. Test không dùng để chọn model hoặc cấu hình.

Epoch tốt nhất 47; đã chạy 59 epoch. Temperature 1.0313.

| Chỉ số | Kết quả |
|---|---:|
| Test accuracy | 94.19% |
| Macro precision | 93.32% |
| Macro recall | 94.43% |
| Macro F1 | 93.76% |
| Mục tiêu macro F1 ≥85% | Đạt |
| Test ECE trước temperature scaling | 0.0084 |
| Test ECE sau temperature scaling | 0.0057 |

Confusion matrix đầy đủ ở `results/evaluation.csv`; chỉ số từng lớp ở `results/evaluation.json`.

## Hiệu năng đo trên CPU

Cấu hình serial tốt nhất trong các cấu hình đã đo là runtime `torchscript`, 2 CPU threads. Mean 14.93 ms, p95 19.45 ms. Mục tiêu mean dưới 200ms: đạt.

| Concurrency | Mean ms | P95 ms | Requests mỗi giây |
|---:|---:|---:|---:|
| 1 | 14.93 | 19.45 | 66.18 |
| 4 | 42.38 | 64.74 | 92.42 |

Thời gian benchmark gồm validation ảnh, preprocessing, inference và xác suất trong process; không bao gồm HTTP, mạng hoặc tư vấn. Benchmark dùng một ảnh validation mỗi lớp và warmup. Đây là phép đo trên máy này với tải đóng vòng; cần đo lại trên máy triển khai.

## Bàn giao và giới hạn

- Bảo nhận manifests, thứ tự nhãn và pipeline để tiếp tục fine-tune nếu cần.
- Khánh nhận checkpoint và cấu hình runtime; backend kiểm tra metadata và trả 503 nếu thiếu model hợp lệ.
- Hải dùng API để thử ảnh thực địa và giao diện phản hồi confidence thấp.
- Trọng số và ảnh nằm local, bị Git bỏ qua; cần bàn giao riêng hoặc tái tạo bằng lệnh trong README.
- Kết quả trên ảnh nền kiểm soát PlantVillage chưa chứng minh chất lượng trên ảnh ngoài đồng.
- Potato_healthy có ít mẫu hơn các lớp khác; F1 của lớp này cần được đọc cùng support.

Checkpoint SHA-256 `ea7f52fc69187c7e5ee701b943bf80046ce26ff90d12aecb559967815a62a423`.
