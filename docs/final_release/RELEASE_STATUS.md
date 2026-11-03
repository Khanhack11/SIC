# Bản bàn giao cho ngày 04 và 05/11/2026

Thời lượng trình bày: 20 phút. Báo cáo Word cập nhật ngày 03/11, đã kiểm tra 11 trang và giữ mẫu Samsung. Kịch bản được bổ sung kết quả V3; slide 14 trang và video là bản lưu kết quả baseline ngày 31/10. Gói mới `Reports/SIC_mobile_field_20261103.zip` có source và weights, không có Python environment hoặc raw dataset.

Frontend chạy chung FastAPI tại `/app`, có upload, camera, kiểm tra ảnh, chống gửi lặp, trạng thái mất kết nối/model chưa sẵn sàng và nguồn tham khảo cho 11 lớp. API ẩn liều thuốc và công thức sinh học chưa được duyệt tại địa phương.

Hai lượt train GPU RTX 2050 đã hoàn thành. CNN nhỏ đạt test macro F1 75,47%. MobileNetV2 fine-tune đạt validation macro F1 95,01%, nhưng recall Corn_Gray_leaf_spot giảm 9,09 điểm phần trăm, vượt giới hạn 2 điểm. Giữ baseline với test macro F1 93,76%; không đọc test để chọn tiếp ứng viên GPU.

Kiểm thử mới: 38 unit/integration, 10 backend, 17 stress/edge, 24 kiểm tra frontend chính và ba hồi quy camera/crop đều đạt. Chromium và WebKit đã thử ở 360/390/412/844 px. Chưa thử điện thoại hoặc camera vật lý. [Hồ sơ mới](mobile_field/continuation_20261103/README.md).

V3 đã train và hoàn tất thêm hai epoch từ checkpoint trước. Validation CPU đạt F1 PlantDoc 46,59%, độ chính xác nhóm vượt cổng điểm 57,69% và coverage 18,84%, chưa đạt mức yêu cầu 85%/25%. `run_field_demo.cmd` chạy bản thử riêng tại port 8003; V2 vẫn là mặc định tại port 8000. Tập test cũ không được đo lại cho lượt tiếp tục.

Trước buổi trình bày cần thử camera vật lý và máy chiếu. Kết quả PlantDoc chưa thay cho kiểm định thực địa có nhãn chuyên gia; hướng dẫn thuốc tại Việt Nam vẫn cần duyệt riêng.
