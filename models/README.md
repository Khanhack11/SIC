# Model và trọng số

| Nhóm file | Vai trò |
|---|---|
| mobilenetv2_crop_disease.pt, .onnx | Baseline phục vụ mặc định, cấu hình trong backend/config.py |
| mobilenetv2_crop_disease.json | Hồ sơ train baseline |
| mobilenetv3_field_continued_20261103.pt, .onnx, .json | V3 thử nghiệm mới nhất, chạy riêng bằng run_field_demo.cmd |
| mobilenetv3_field_20261103.* | V3 trước lượt train tiếp tục |
| mobilenetv2_gpu_*, mobilenetv2_head_*, mobilenetv2_projection_* | Các thí nghiệm V2 trước |
| small_cnn_20261031.* | CNN đối chứng |
| *.resume.pt, *.progress.pt | Trạng thái phục hồi; khả năng resume phụ thuộc từng trainer |
| *_int8.onnx | Graph thử nghiệm lượng tử hóa; không tự chọn để phục vụ |
| *_search/ | Kết quả tìm cấu hình của các thí nghiệm trước |

Tên file và vị trí trọng số được giữ để khớp checkpoint metadata, lệnh chạy và hồ sơ đánh giá. ONNX là định dạng chạy của cùng model, không phải một mô hình được train riêng.

V3 chưa qua tiêu chí triển khai trên dữ liệu thực tế. Đợt bổ sung dữ liệu ngày 03/11 chưa train lại. Trọng số không lưu trong Git; có trong gói bàn giao local. Không đổi tên hoặc ghép metadata giữa các checkpoint.
