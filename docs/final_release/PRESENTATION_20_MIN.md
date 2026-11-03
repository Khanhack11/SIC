# Kịch bản báo cáo 20 phút

Ngày dự kiến: 04–05/11/2026. Deck `SIC_Final_Presentation_20min.pptx` có 14 trang; speaker notes chứa thời lượng, gợi ý nói và nguồn cho từng trang. Mở slide cùng `http://127.0.0.1:8000/app` trước khi trình bày.

Cập nhật ngày 03/11: deck và video giữ kết quả baseline ngày 31/10. Báo cáo Word và hồ sơ V3 đã cập nhật; dùng [bảng so sánh mới](mobile_field/continuation_20261103/README.md) cùng ba HTML Archify khi nói phần mở rộng. `run_field_demo.cmd` mở V3 thử nghiệm ở `http://127.0.0.1:8003/app`, chưa thay model mặc định.

| Phút | Slide | Nội dung |
|---|---:|---|
| 0:00–0:30 | 1 | Tên đề tài, Group 5, mục đích hỗ trợ nhận diện ảnh lá |
| 0:30–1:00 | 2 | Thứ tự trình bày và phần demo |
| 1:00–2:00 | 3 | Ba cây, 11 lớp, mục tiêu F1 và latency |
| 2:00–3:30 | 4 | Nguồn dữ liệu, loại trùng và split theo nhóm |
| 3:30–5:00 | 5 | Luồng ảnh, API, model, confidence gate và kho tri thức |
| 5:00–6:30 | 6 | Baseline, augmentation, loss, chọn epoch, calibration và lượt GPU |
| 6:30–8:00 | 7 | Test baseline và CNN đối chứng |
| 8:00–10:00 | 8 + bảng mới | Một phút V2 giảm recall; một phút V3 cải thiện thực địa nhưng chưa qua cổng triển khai |
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

Dành khoảng một phút trong phần này để chọn file từ máy, mở khung crop, áp dụng rồi khôi phục. Giải thích rằng khung do người dùng chọn, chưa tự tách lá. Đánh dấu có tổn thương và phân tích ảnh lá khỏe để cho thấy cảnh báo xung đột. Điểm 59,6% của ảnh khó thuộc V2 trên ảnh mẫu gốc; V3 hoặc vùng crop khác có thể cho điểm khác.

Chọn một file hỏng hoặc file trên 10 MB để minh họa validation. Nếu camera đã thử trước buổi trình bày, chụp một ảnh để cho thấy luồng capture và preview. Không dùng ảnh vật thể ngoài phạm vi để khẳng định model biết từ chối ảnh lạ: hệ thống hiện chưa có chức năng đó.

## Giải thích các câu hỏi kỹ thuật

**Lý do giữ baseline.** Validation macro F1 của ứng viên GPU tăng từ 94,48% lên 95,01%. Recall đốm lá xám giảm từ 93,51% xuống 84,42%, vượt giới hạn giảm hai điểm phần trăm đã đặt trước. Ứng viên bị loại ở validation và chưa đọc test để chọn tiếp. Tham chiếu `gpu_decision.json`.

**CNN đối chứng và MobileNetV2.** CNN nhỏ đạt test macro F1 75,47%; baseline đạt 93,76%. Đây là hai chiến lược với pretrained weights và budget khác nhau. Không kết luận chênh lệch chỉ do kiến trúc. CNN đối chứng chạy tám epoch, còn baseline dùng ImageNet và train classifier 59 epoch.

**Chống rò rỉ dữ liệu.** Audit loại 10 ảnh trùng pixel, giữ các nhóm lá đã biết trong một split và kiểm tra giao nhau theo hash. Có 5.833 ảnh dùng định danh dự phòng, nên chưa chứng minh tuyệt đối mọi lá vật lý đã tách biệt. pHash/dHash không phát hiện cặp gần trùng ở ngưỡng đã thử cũng không chứng minh hết leakage.

**Test đã được xem trước.** Baseline đầu tiên train bằng train/validation. Các vòng sau đã dùng lại test để kiểm tra hồi quy, nên cần công bố việc tái sử dụng. Lượt GPU mới bị loại bằng validation; không điều chỉnh tiếp dựa vào test. Muốn đánh giá độc lập hơn cần tập ảnh ngoài đồng mới.

**Confidence và calibration.** Softmax là phân bố dự đoán của model, không phải xác nhận bệnh. Temperature được fit bằng validation; ECE test giảm từ 0,00839 xuống 0,00565. 97,50% accepted accuracy chỉ tính trên 90,87% ảnh test vượt ngưỡng, không áp dụng tự động cho ảnh ngoài đồng.

**Đo latency.** Benchmark mới gồm HTTP multipart, validation, preprocessing, inference, tra cứu JSON và response trên localhost. Có 80 request ở mỗi mức tải sau 10 warmup. Không gộp số này với phép đo chỉ chạy model trong process, và không hứa cùng độ trễ qua Internet.

**Tư vấn thuốc.** Nguồn nước ngoài được dùng để đối chiếu triệu chứng và canh tác. Chưa có xác nhận nhãn thuốc địa phương, liều, sản phẩm và PHI. API trả mảng thuốc/công thức sinh học rỗng, kể cả khi confidence cao. Không khẳng định dữ liệu đã được chuyên gia nông học duyệt.

**Ảnh ngoài miền ở V2.** V2 luôn phân phối xác suất trên 11 lớp. Nó có thể dự đoán tự tin với đồ vật hoặc cây khác. Ngưỡng 75% không thay thế bộ phát hiện ảnh lạ; đây là giới hạn cần công bố.

**V3 mới và lý do chưa thay mặc định.** V3 có 12 đầu ra, gồm Unsupported. Sau hai epoch bổ sung, F1 validation CPU trên PlantDoc là 46,59%, so với V2 15,95%. Chỉ số này tính trên tám lớp trong phạm vi có mẫu. Trong 78/414 ảnh vượt cổng điểm, 45 dự đoán đúng, tức 57,69%; coverage 18,84%. Chưa đạt mức 85%/25% đã đặt trước. Không đo lại test cũ để chọn tiếp và chưa có kiểm định điện thoại thực địa với nhãn chuyên gia.

**Model học và crop.** Train so sánh dự đoán với nhãn rồi sửa trọng số. Khi nhận ảnh mới, model dùng trọng số đã lưu, không tự train thêm. Crop lấy vùng người dùng chọn; resize/letterbox chuẩn bị kích thước cho CNN. Đặc trưng qua đầu phân loại tạo logits, softmax/T tạo điểm, rồi hệ thống tra lời khuyên từ JSON. Có thể mở ba sơ đồ trong `Reports/SIC_Pipeline_20261103` để giải thích.

**Phân công.** Giải thích theo vai trò trong Action Plan và mã/báo cáo thực tế. Không đọc các ghi chú đóng góp trong bảng nhóm như nhận xét cá nhân đã ký xác nhận.
