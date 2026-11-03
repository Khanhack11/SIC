# Bản chuẩn bị báo cáo ngày 04 và 05 tháng 11 năm 2026

[Kết quả V3 và sửa lỗi camera/crop ngày 03/11](mobile_field/continuation_20261103/README.md). Gói bàn giao mới là `Reports/SIC_mobile_field_20261103.zip`. Toàn bộ mã test nằm trong `tests/`, mở `run_tests.cmd` ở thư mục project để chạy chung. Báo cáo Word đã cập nhật ngày 03/11 và giữ 11 trang. Slide/video giữ bản 31/10 cho kết quả baseline; dùng kịch bản và hồ sơ mới khi giải thích V3.

Bản demo dùng checkpoint MobileNetV2 baseline, cùng ONNX FP32 đã kiểm tra. Frontend và API chạy chung tại `/app`. Có camera, upload, validation phía client/server, trạng thái model chưa sẵn sàng, panel tham khảo và nguồn cho 11 record. Các hướng dẫn thuốc/công thức sinh học chưa duyệt bị ẩn tại API.

## Tài liệu bàn giao

- [Final Report](<SIC_AI_Capstone Project_Final Report.docx>) giữ cấu trúc của mẫu Samsung.
- [Slide 20 phút](SIC_Final_Presentation_20min.pptx) có 14 trang và speaker notes.
- [Kịch bản demo và câu hỏi](PRESENTATION_20_MIN.md).
- [Ba sơ đồ và giải thích pipeline bằng tiếng Việt](../../.archify/architecture-sic-pipeline-20261103-114010/README.md).
- `SIC_Demo.webm` là video thao tác trên ứng dụng thật, không có thuyết minh; ảnh minh họa từ validation.
- `SIC_mobile_field_20261103.zip` được bàn giao riêng trong Reports, gồm source, model phục vụ, V3 thí nghiệm, sơ đồ và checksum manifest. Không có Python environment hoặc raw dataset.

## Kết quả và bằng chứng

| Kết quả | Hồ sơ |
|---|---|
| V3 tiếp tục: F1 PlantDoc validation CPU 46,59%, chưa qua cổng triển khai | [Kết quả lượt tiếp tục](mobile_field/continuation_20261103/README.md) |
| 38 unit/integration, backend/stress, frontend và mobile đều đạt | [Regression](mobile_field/continuation_20261103/regression_summary.json) |
| Baseline accuracy 94,19%, macro F1 93,76% trên 1.720 test images | [evaluation.json](../data_optimization/results/evaluation.json) |
| CNN nhỏ, 8 epoch, validation F1 75,87%, chọn epoch 7 | [small_cnn_training.json](small_cnn_training.json) |
| CNN nhỏ test F1 75,47%, accuracy 77,03% | [small_cnn_test.json](small_cnn_test.json) |
| GPU MobileNetV2 chọn epoch 5, validation F1 95,01% | [mobilenetv2_gpu_training.json](mobilenetv2_gpu_training.json) |
| Giữ baseline vì recall đốm lá xám giảm 9,09 điểm phần trăm | [gpu_decision.json](gpu_decision.json) |
| HTTP ONNX mean/p95 11,07/13,63 ms ở một client; 55,51/62,70 ms ở tám client | [api_benchmark.json](api_benchmark.json) |
| Kết quả kiểm thử và hash bản phát hành | [verification.json](verification.json) |
| Các kiểm tra browser, camera mô phỏng và responsive | [frontend_verification.json](frontend_verification.json) |

GPU candidate không được dùng test để chọn tiếp sau khi không đạt recall gate. Seed cố định không bảo đảm mọi phép toán CUDA giống bit trên mọi máy. Sau lượt train, mã trainer được bổ sung bước đưa best state về CPU khi ghi checkpoint sau resume; không thay số liệu hoặc weights đã đo.

Kiểm thử camera dùng thiết bị mô phỏng, chưa xác nhận camera vật lý. PlantDoc là ảnh Internet có nền phức tạp, chưa thay thế bộ ảnh điện thoại được chuyên gia địa phương gán nhãn. V3 có lớp Unsupported nhưng chưa phát hiện được mọi ảnh lạ. Nhãn thuốc tại Việt Nam chưa được người chuyên môn duyệt.

Tài liệu trong `docs/final_report` là bản lưu trước lượt hoàn thiện này; dùng thư mục hiện tại khi trình bày.
