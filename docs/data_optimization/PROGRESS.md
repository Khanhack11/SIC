# Tiến độ phần Data và Optimization của Trần Đặng Công Tâm

Ngày làm việc 16/10/2026. Nhánh local `tran-dang-cong-tam`.

## Trạng thái hiện tại

- Đã tải 11.504 ảnh RGB của 11 lớp từ PlantVillage, kiểm tra Git blob hash từng ảnh.
- Nguồn được ghim ở commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` của `spMohanty/PlantVillage-Dataset`.
- Đã loại 10 ảnh trùng pixel, giữ 11.494 ảnh; chia 8.043 train / 1.731 validation / 1.720 test. Hash file, pixel và nhóm đã xác định không giao nhau giữa các tập.
- Có 5.671 ảnh khớp leaf-map, 5.833 ảnh dùng identifier dự phòng; không khẳng định mọi lá vật lý đã được tách biệt.
- Đã viết pipeline augmentation, tiền xử lý dùng chung với backend, huấn luyện baseline, đánh giá và benchmark.
- Đã hoàn thành trích xuất feature và huấn luyện baseline 59 epoch, best epoch 47. Checkpoint cuối: `models/mobilenetv2_crop_disease.pt`.
- Đã kiểm tra cách bố trí tensor: batch 64 dùng channels-last khoảng 1,5 giây, contiguous khoảng 3,1 giây ở 4 threads. Đây là microbenchmark kỹ thuật, chưa phải latency API.
- Test 1.720 ảnh: accuracy 94,19%, macro F1 93,76%, đạt mục tiêu F1 ≥85% trên subset PlantVillage này.
- Benchmark serial được khuyến nghị: TorchScript, 2 CPU threads; mean 14,93 ms, p95 19,45 ms. Chưa gồm HTTP/network/advisory. Cấu hình mặc định API vẫn là eager 4 threads; dùng biến môi trường trong README để chọn cấu hình benchmark.
- 8 kiểm thử mới của pipeline/API đã pass. Kết quả backend và stress test được lưu trong `results/`.
- Đã tạo `REPORT.md`, sao chép manifests và lưu audit, training, evaluation, benchmark, environment, checksum checkpoint trong bộ bàn giao này.
- Không còn tiến trình huấn luyện/đánh giá/benchmark của chat đang chạy tại thời điểm hoàn thành chuỗi lệnh.

## Lệnh tiếp tục

Chạy trong thư mục gốc SIC. Python trên máy hiện có torch và torchvision bản CPU.

```powershell
python -m ml.download_dataset
python -m ml.prepare_dataset
$env:TORCH_HOME = "$PWD/artifacts/torch"
python -u -m ml.train
python -u -m ml.evaluate
python -u -m ml.benchmark
python -m unittest discover -s tests -v
python test_backend.py
python stress_and_edge_test.py
```

Downloader kiểm tra lại và tiếp tục các file chưa đủ. Nếu đã có `data/processed/plantvillage/audit.json`, không chạy lại prepare vào cùng thư mục; dùng manifest đã có hoặc chọn thư mục output mới. Huấn luyện lưu feature theo batch và trạng thái optimizer theo epoch; chạy lại cùng tham số để tiếp tục khi chưa có checkpoint cuối cùng.

## Yêu cầu về hạn mức

Người dùng yêu cầu khi Codex còn không quá 5% hạn mức thì lưu tiến độ, tạm dừng các tiến trình của chat này rồi tắt máy. Lần kiểm tra gần nhất: còn 7% cửa sổ 5 giờ, 69% cửa sổ tuần. Chưa đạt ngưỡng tắt máy. Lịch theo dõi nền chưa tạo được do công cụ automation cần phê duyệt trong khi phiên không cho phép luồng phê duyệt đó. Kiểm tra trực tiếp vẫn thực hiện trong lúc chat đang làm việc.

Để yêu cầu các job ML dừng có lưu trạng thái, tạo `artifacts/PAUSE_REQUESTED`. Xóa đúng file này trước khi chạy tiếp. Không dùng lệnh tắt máy cho đến khi đã đọc hạn mức thực tế và đạt ngưỡng.

## Phạm vi Git

Nhánh `tran-dang-cong-tam` chứa phần Data & Optimization. Ngày 17/10/2026, người dùng yêu cầu commit và đẩy các thay đổi lên `https://github.com/Khanhack11/SIC.git`. Ảnh gốc, feature cache và checkpoint được giữ local theo `.gitignore`; bộ bàn giao trên Git gồm mã nguồn, manifests, kết quả đo và tài liệu tái lập.
