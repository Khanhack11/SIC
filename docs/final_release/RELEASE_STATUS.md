# Bản bàn giao cho ngày 04 và 05/11/2026

Thời lượng trình bày: 20 phút. Source, báo cáo, slide, kịch bản và video demo đã hoàn thành. Gói ZIP có weights được bàn giao riêng trong Reports, không chứa Python environment hoặc raw dataset.

Frontend chạy chung FastAPI tại `/app`, có upload, camera, kiểm tra ảnh, chống gửi lặp, trạng thái mất kết nối/model chưa sẵn sàng và nguồn tham khảo cho 11 lớp. API ẩn liều thuốc và công thức sinh học chưa được duyệt tại địa phương.

Hai lượt train GPU RTX 2050 đã hoàn thành. CNN nhỏ đạt test macro F1 75,47%. MobileNetV2 fine-tune đạt validation macro F1 95,01%, nhưng recall Corn_Gray_leaf_spot giảm 9,09 điểm phần trăm, vượt giới hạn 2 điểm. Giữ baseline với test macro F1 93,76%; không đọc test để chọn tiếp ứng viên GPU.

Kiểm thử cuối: 25 unit/integration, 10 backend, 17 stress/edge và 13 tình huống browser đều đạt. Camera dùng thiết bị mô phỏng. Benchmark HTTP, log và hash được lưu cùng bản bàn giao. Báo cáo đã render và kiểm tra 11 trang. Slide có 14 trang theo mẫu Samsung, biểu đồ chỉnh sửa được và speaker notes. Video thao tác thật dài 38,56 giây, không có thuyết minh.

Trước buổi trình bày cần thử camera vật lý và máy chiếu. Chất lượng ảnh ngoài đồng, ảnh ngoài miền và hướng dẫn thuốc tại Việt Nam vẫn cần đánh giá riêng. Không tuyên bố đã có chuyên gia duyệt.

Người dùng yêu cầu tắt máy sau khi hoàn tất. Chỉ tắt sau khi lưu tài liệu, kiểm tra gói bàn giao và xác nhận push GitHub. Phiên không có API đọc hạn mức nên không có cơ chế tự tắt theo quota.
