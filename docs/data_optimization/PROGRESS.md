# Tiến độ phần Data và Optimization của Trần Đặng Công Tâm

Ngày làm việc 16/10/2026. Nhánh local `tran-dang-cong-tam`.

## Vòng tối ưu ngày 17 tháng 10 năm 2026

- Người dùng yêu cầu tiếp tục tối ưu và tắt máy khi Limits còn không quá 5%. Theo dõi nền đã được tạo thành công, automation `l-u-sic-v-t-t-m-y-khi-codex-c-n-5`, kiểm tra mỗi 5 phút. Tiếp tục kiểm tra trực tiếp khi làm việc.
- Đã sửa API để CPU inference chạy trong threadpool, giới hạn số lượt inference đồng thời bằng `SIC_INFERENCE_CONCURRENCY` (mặc định 2), và áp dụng ngưỡng tư vấn riêng cho request thay vì đổi biến dùng chung. Upload được đọc tối đa giới hạn file cộng 1 byte.
- Đã bổ sung ONNX Runtime FP32 với metadata kiểm tra nhãn, input contract và SHA-256 checkpoint. Model local: `models/mobilenetv2_crop_disease.onnx`.
- Validation đủ 1.731 ảnh: eager, TorchScript và ONNX FP32 cùng macro F1 94,4801%, prediction agreement 100%. INT8 chỉ đạt 90,7411%, đã loại khỏi lựa chọn triển khai. Không đổi checkpoint baseline.
- Perceptual audit trên 11.494 ảnh không gắn cờ cặp gần trùng khác split theo pHash <=2 và dHash <=4. Không suy ra mọi lá vật lý đã tách biệt; metadata dự phòng vẫn là giới hạn.
- Các kết quả mới được ghi riêng ở `results/optimization_20261017/`. Các thay đổi REPORT.md, benchmark.json và environment.json có sẵn đầu lượt được giữ lại; bản sao nằm trong `artifacts/optimization_20261017_baseline/`.
- Đang đo HTTP localhost bằng `python -m ml.api_benchmark`; registry server do công cụ sở hữu: `artifacts/api_active_process.json`. Công cụ tự dừng server của mình trong finally khi được yêu cầu pause.
- Đã chọn ONNX FP32, 2 threads, 2 inference slots. Vòng xác nhận 150 request mỗi mức tải: HTTP mean/p95 19,28/35,63 ms với 1 client; 45,15/54,39 ms với 4 client; 81,31/93,96 ms với 8 client.
- 14 kiểm thử unit/integration, 10 kiểm thử backend và 17 kiểm thử stress cuối đều pass. Server benchmark đã được dừng; registry active=false.
- Mặc định API tự chọn ONNX nếu graph và dependency có sẵn, còn lại dùng TorchScript. Báo cáo đầy đủ: `OPTIMIZATION_20261017.md`. Vòng tối ưu đang được commit và push trên nhánh `tran-dang-cong-tam`; các thay đổi REPORT.md, benchmark.json và environment.json có sẵn đầu lượt vẫn được giữ riêng trong working tree.

Lệnh tiếp tục sau khi kiểm tra/xóa marker pause cũ nếu đã được người dùng cho phép tiếp tục:

```powershell
python -m ml.runtime_validation
python -m ml.near_duplicate_audit
python -m ml.api_benchmark --threads 2 4 --slots 1 2 --concurrency 1 4 8 --runs 80
python -m unittest discover -s tests -v
```

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
