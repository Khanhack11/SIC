# Kịch bản báo cáo 20 phút

Ngày dự kiến: 04–05/11/2026. Deck `SIC_Final_Presentation_20min.pptx` có 14 trang; speaker notes chứa thời lượng, gợi ý nói và nguồn cho từng trang. Mở slide cùng `http://127.0.0.1:8000/app` trước khi trình bày.

| Phút | Slide | Nội dung |
|---|---:|---|
| 0:00–0:30 | 1 | Tên đề tài, Group 5, mục đích hỗ trợ nhận diện ảnh lá |
| 0:30–1:00 | 2 | Thứ tự trình bày và phần demo |
| 1:00–2:00 | 3 | Ba cây, 11 lớp, mục tiêu F1 và latency |
| 2:00–3:30 | 4 | Nguồn dữ liệu, loại trùng và split theo nhóm |
| 3:30–5:00 | 5 | Luồng ảnh, API, model, confidence gate và kho tri thức |
| 5:00–6:30 | 6 | Baseline, augmentation, loss, chọn epoch, calibration và lượt GPU |
| 6:30–8:00 | 7 | Test baseline và CNN đối chứng |
| 8:00–10:00 | 8 | Ứng viên GPU tăng F1 nhưng giảm recall một lớp; vì sao chưa phát hành |
| 10:00–11:00 | 9 | Confidence, ECE, coverage và accepted accuracy |
| 11:00–12:00 | 10 | Mean/p95 HTTP, tải 1/4/8 client và phạm vi phép đo |
| 12:00–13:00 | 11 | Nguồn tư vấn, phạm vi kiểm tra và phần thuốc bị ẩn |
| 13:00–17:00 | 12 | Demo trực tiếp |
| 17:00–19:00 | 13 | Giới hạn dữ liệu, ảnh ngoài miền và việc cần đánh giá thêm |
| 19:00–20:00 | 14 | Câu hỏi hoặc phần dự phòng |

## Trước buổi báo cáo

1. Cắm nguồn, mở `run_demo.cmd`, đợi trạng thái Model sẵn sàng. Không đổi Python environment hoặc weights sát giờ báo cáo.
2. Mở `/app`, thử ảnh ngô gỉ sắt, lá khoai tây và ảnh khó. Kiểm tra có nguồn tư vấn, không có liều thuốc chưa duyệt.
3. Thử camera thật trên máy sẽ trình bày. Nếu không có camera hoặc bị từ chối quyền, dùng ảnh từ máy; không nói rằng camera vật lý đã được kiểm chứng bằng test mô phỏng.
4. Mở deck và video `SIC_Demo.webm` để dự phòng. Video không có lời thuyết minh; dùng lời giải thích trực tiếp khi phát.
5. Giữ một bản ZIP demo và tài liệu trên USB. Môi trường Python không nằm trong ZIP; cài dependency trước khi chuyển sang máy khác.

## Demo trong bốn phút

Mở phần ảnh mẫu và chọn Ngô, gỉ sắt. Nhấn Phân tích ảnh, chỉ ra confidence và thời gian xử lý model. Mở panel nguồn, giải thích tài liệu hỗ trợ triệu chứng và canh tác. Mở phần sử dụng thuốc để cho thấy hệ thống chưa đưa liều thuốc chưa duyệt.

Chọn Khoai tây, lá khỏe. Nói rõ model xếp ảnh vào lớp khỏe trong 11 lớp đã học; kết quả không chứng nhận toàn bộ cây không có bệnh. Sau đó dùng ảnh khó, quan sát confidence khoảng 59,6%, thông báo chưa đủ tin cậy và hướng dẫn chụp lại.

Chọn một file hỏng hoặc file trên 10 MB để minh họa validation. Nếu camera đã thử trước buổi trình bày, chụp một ảnh để cho thấy luồng capture và preview. Không dùng ảnh vật thể ngoài phạm vi để khẳng định model biết từ chối ảnh lạ: hệ thống hiện chưa có chức năng đó.

## Giải thích các câu hỏi kỹ thuật

**Lý do giữ baseline.** Validation macro F1 của ứng viên GPU tăng từ 94,48% lên 95,01%. Recall đốm lá xám giảm từ 93,51% xuống 84,42%, vượt giới hạn giảm hai điểm phần trăm đã đặt trước. Ứng viên bị loại ở validation và chưa đọc test để chọn tiếp. Tham chiếu `gpu_decision.json`.

**CNN đối chứng và MobileNetV2.** CNN nhỏ đạt test macro F1 75,47%; baseline đạt 93,76%. Đây là hai chiến lược với pretrained weights và budget khác nhau. Không kết luận chênh lệch chỉ do kiến trúc. CNN đối chứng chạy tám epoch, còn baseline dùng ImageNet và train classifier 59 epoch.

**Chống rò rỉ dữ liệu.** Audit loại 10 ảnh trùng pixel, giữ các nhóm lá đã biết trong một split và kiểm tra giao nhau theo hash. Có 5.833 ảnh dùng định danh dự phòng, nên chưa chứng minh tuyệt đối mọi lá vật lý đã tách biệt. pHash/dHash không phát hiện cặp gần trùng ở ngưỡng đã thử cũng không chứng minh hết leakage.

**Test đã được xem trước.** Baseline đầu tiên train bằng train/validation. Các vòng sau đã dùng lại test để kiểm tra hồi quy, nên cần công bố việc tái sử dụng. Lượt GPU mới bị loại bằng validation; không điều chỉnh tiếp dựa vào test. Muốn đánh giá độc lập hơn cần tập ảnh ngoài đồng mới.

**Confidence và calibration.** Softmax là phân bố dự đoán của model, không phải xác nhận bệnh. Temperature được fit bằng validation; ECE test giảm từ 0,00839 xuống 0,00565. 97,50% accepted accuracy chỉ tính trên 90,87% ảnh test vượt ngưỡng, không áp dụng tự động cho ảnh ngoài đồng.

**Đo latency.** Benchmark mới gồm HTTP multipart, validation, preprocessing, inference, tra cứu JSON và response trên localhost. Có 80 request ở mỗi mức tải sau 10 warmup. Không gộp số này với phép đo chỉ chạy model trong process, và không hứa cùng độ trễ qua Internet.

**Tư vấn thuốc.** Nguồn nước ngoài được dùng để đối chiếu triệu chứng và canh tác. Chưa có xác nhận nhãn thuốc địa phương, liều, sản phẩm và PHI. API trả mảng thuốc/công thức sinh học rỗng, kể cả khi confidence cao. Không khẳng định dữ liệu đã được chuyên gia nông học duyệt.

**Ảnh ngoài miền.** Model luôn phân phối xác suất trên 11 lớp. Nó có thể dự đoán tự tin với đồ vật hoặc cây khác. Ngưỡng 75% không thay thế bộ phát hiện ảnh lạ; đây là giới hạn cần công bố.

**Phân công.** Giải thích theo vai trò trong Action Plan và mã/báo cáo thực tế. Không đọc các ghi chú đóng góp trong bảng nhóm như nhận xét cá nhân đã ký xác nhận.
