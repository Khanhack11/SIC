# Bản chuẩn bị báo cáo ngày 04 và 05 tháng 11 năm 2026

Bản demo dùng checkpoint MobileNetV2 baseline, cùng ONNX FP32 đã kiểm tra. Frontend và API chạy chung tại `/app`. Có camera, upload, validation phía client/server, trạng thái model chưa sẵn sàng, panel tham khảo và nguồn cho 11 record. Các hướng dẫn thuốc/công thức sinh học chưa duyệt bị ẩn tại API.

## Tài liệu bàn giao

- [Final Report](<SIC_AI_Capstone Project_Final Report.docx>) giữ cấu trúc của mẫu Samsung.
- [Slide 20 phút](SIC_Final_Presentation_20min.pptx) có 14 trang và speaker notes.
- [Kịch bản demo và câu hỏi](PRESENTATION_20_MIN.md).
- `SIC_Demo.webm` là video thao tác trên ứng dụng thật, không có thuyết minh; ảnh minh họa từ validation.
- `SIC_demo_20261104.zip` được bàn giao riêng trong Reports, gồm source, model phục vụ, checkpoint thí nghiệm và checksum manifest. Không có Python environment hoặc raw dataset.

## Kết quả và bằng chứng

| Kết quả | Hồ sơ |
|---|---|
| Baseline accuracy 94,19%, macro F1 93,76% trên 1.720 test images | [evaluation.json](../data_optimization/results/evaluation.json) |
| CNN nhỏ, 8 epoch, validation F1 75,87%, chọn epoch 7 | [small_cnn_training.json](small_cnn_training.json) |
| CNN nhỏ test F1 75,47%, accuracy 77,03% | [small_cnn_test.json](small_cnn_test.json) |
| GPU MobileNetV2 chọn epoch 5, validation F1 95,01% | [mobilenetv2_gpu_training.json](mobilenetv2_gpu_training.json) |
| Giữ baseline vì recall đốm lá xám giảm 9,09 điểm phần trăm | [gpu_decision.json](gpu_decision.json) |
| HTTP ONNX mean/p95 11,07/13,63 ms ở một client; 55,51/62,70 ms ở tám client | [api_benchmark.json](api_benchmark.json) |
| Kết quả kiểm thử và hash bản phát hành | [verification.json](verification.json) |
| Các kiểm tra browser, camera mô phỏng và responsive | [frontend_verification.json](frontend_verification.json) |

GPU candidate không được dùng test để chọn tiếp sau khi không đạt recall gate. Seed cố định không bảo đảm mọi phép toán CUDA giống bit trên mọi máy. Sau lượt train, mã trainer được bổ sung bước đưa best state về CPU khi ghi checkpoint sau resume; không thay số liệu hoặc weights đã đo.

Kiểm thử camera dùng thiết bị mô phỏng, chưa xác nhận camera vật lý. Chưa đo chất lượng ngoài đồng, chưa có bộ phát hiện ảnh ngoài miền và chưa có người chuyên môn duyệt nhãn thuốc tại Việt Nam. Các giới hạn này được giữ trong report và slide.

Tài liệu trong `docs/final_report` là bản lưu trước lượt hoàn thiện này; dùng thư mục hiện tại khi trình bày.
