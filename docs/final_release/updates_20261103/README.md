# Cập nhật train và kiểm thử ngày 03/11/2026

Fine-tune từ baseline trên NVIDIA RTX 2050 bằng CUDA, 4 epoch, batch 16, backbone learning rate 0.000003 và classifier 0.00003. Chọn checkpoint bằng validation loss; không đọc test ứng viên. Validation đã được dùng trong các vòng trước nên không phải một tập đánh giá mới độc lập.

Baseline validation macro F1 đo trên CUDA trong lượt train: 94.3907%. Checkpoint chọn epoch 3 đạt 94.6360%. Chênh lệch 0.2452 điểm phần trăm. Recall giảm lớn nhất 1.3158 điểm. Gate yêu cầu tăng ít nhất 0,5 điểm F1 và không giảm quá 2 điểm recall mỗi lớp: không đạt. Model demo vẫn là baseline; checkpoint mới lưu riêng tại `models/mobilenetv2_gpu_20261103.pt` và trong ZIP bàn giao.

## Sửa lỗi demo

Tái hiện được phản hồi tải ảnh mẫu cũ về muộn ghi đè ảnh local vừa chọn. Client hiện kiểm tra phiên lựa chọn trước khi nhận ảnh hoặc hiển thị lỗi tải. Nút phân tích khóa khi tải mẫu để tránh gửi ảnh trước đó. Test hồi quy đã thất bại trước sửa và đạt sau sửa.

Toàn bộ mã test đã gom vào `tests/`. Chạy `tests/run_tests.cmd` hoặc `.venv/Scripts/python.exe tests/run_all.py --web`. Kết quả lưu vào `tests/results/<thời điểm>/`. Bộ stress hiện trả exit code khác 0 nếu có lỗi. Browser đợi đúng server của lượt chạy sẵn sàng, tránh kiểm thử nhầm server đang chiếm port.

25 unit/integration, 10 backend, 17 stress/edge và 19 tình huống browser đều đạt. Đã cài và kiểm tra Playwright từ `tests/package-lock.json` trên máy này. Camera mô phỏng, HTTP 400/500/503, mất kết nối và retry đã được kiểm tra. Không có lỗi JavaScript ở trang. [Kết quả](test_results/summary.json) và screenshot đi kèm.

Báo cáo Word và slide ngày 31/10 giữ nguyên như bản lịch sử; tài liệu này bổ sung kết quả ngày 03/11. Camera thật và ảnh ngoài đồng vẫn cần kiểm tra riêng. Các con số latency trong log là điều kiện của lần chạy đó, không gộp với benchmark HTTP trước đây.

## Đối chiếu CPU sau train

Đo cùng 1.731 ảnh validation bằng eager PyTorch CPU, batch 16 và 2 threads: baseline 94.4801%, ứng viên 94.6360% macro F1. Chênh lệch 0.1559 điểm phần trăm. Gate không đạt. [Quyết định CPU](cpu_decision.json). Các số đo CUDA trong lượt train được giữ riêng; không coi chúng là số đo CPU phục vụ web. Không đọc test ứng viên.

Gói ZIP đã giải nén sang thư mục mới và chạy lại đạt 25 unit/integration, 10 backend và 17 stress/edge. Bộ test có đủ fixtures. Gói dùng môi trường Python đã cài trên máy này; chưa kiểm tra trên máy vật lý khác. Bản ZIP cuối cập nhật thêm hồ sơ kiểm chứng, giữ nguyên mã đã kiểm thử và xác minh toàn bộ checksum manifest.
