# Các phase sửa hệ thống ảnh thực địa và điện thoại

Yêu cầu xác nhận ngày 03/11/2026: Android phổ thông và iPhone dùng web gọi API; model chạy trên server. Không cần cài model lên điện thoại.

## Phase 1. Rà soát và tái hiện

Đã rà soát kiến trúc backend, hợp đồng preprocessing, pipeline dữ liệu/huấn luyện, runtime ONNX, advisory, frontend và các bộ test. MobileNetV2 có 11 lớp, được train chủ yếu trên PlantVillage nền kiểm soát. Toàn ảnh được resize 224×224, không có detector/crop lá tự động và không có lớp ngoài phạm vi. UI gọi softmax là độ tin cậy, dễ bị hiểu thành độ chính xác.

Vùng lá trích từ screenshot người dùng vẫn bị phân loại Corn_healthy với score khoảng 99,99%. Bản trích không phải ảnh gốc, không có nhãn chuyên gia và không được dùng train hoặc tính accuracy. Có tổn thương nhìn thấy trong ảnh nhưng chưa đủ cơ sở gán loại bệnh hay sâu cụ thể. Giả thuyết ưu tiên: domain shift, tổn thương ngoài bộ nhãn, sau đó là mất thông tin do resize. Hash checkpoint và graph đang khớp.

## Phase 2. Input và khả năng đọc

Thực hiện crop thủ công bằng kéo vùng chọn, điều khiển tọa độ dùng được bằng bàn phím, khôi phục ảnh gốc và gửi đúng pixels đã xem trước. API nhận crop_box chuẩn hóa cho client di động. Không gọi crop thủ công là segmentation hay phát hiện lá tự động. Cỡ chữ nội dung kết quả ít nhất 18 px, tiêu đề lớn hơn, nút chạm tối thiểu 44 px, kiểm tra màn 360/390/412 px và landscape.

Điểm softmax được ghi rõ không phải tỷ lệ chẩn đoán đúng. Khi người dùng báo lá có tổn thương mà model dự đoán khỏe, chuyển sang yêu cầu kiểm tra. Lớp không có trong kho tri thức không được nhận advisory xác nhận. Không giảm score giả hoặc gán bệnh từ screenshot.

## Phase 3. Dữ liệu và model

Chọn MobileNetV3-Large ImageNet V2 làm ứng viên vì API chạy ở server, có thể ưu tiên biểu diễn tốt hơn bản Small. Đây là lựa chọn để kiểm thử, chưa phải tuyên bố model chính xác nhất. Torchvision công bố 0,22 GFLOPS và ImageNet top-1 75,274%; các số này không phải accuracy bệnh lá. Giữ baseline để đo đối chứng. Tải PlantDoc từ nguồn gốc, ghi revision và giấy phép, lọc file lỗi, kiểm tra nhãn, nhóm ảnh trùng trước khi chia. Kết hợp dữ liệu có nguồn và tập ngoài phạm vi; không dùng ảnh người dùng làm train.

Nguồn: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html ; https://github.com/pratikkayal/PlantDoc-Dataset ; https://arxiv.org/abs/1911.10317 . PlantDoc gồm ảnh thu thập trên Internet với nền phức tạp, cần ghi giới hạn nhãn và quyền ảnh gốc; không coi là bộ ảnh thực địa đã được chuyên gia của nhóm duyệt.

## Phase 4. Huấn luyện và kiểm định

Khóa test trước train. Chọn checkpoint và temperature bằng validation. Báo cáo riêng PlantVillage, PlantDoc và lớp từ chối; không cộng chung để che lỗi thực địa. Đánh giá baseline trên cùng ảnh để đo cải thiện. Kiểm tra recall healthy sai trên lá có bệnh, macro F1, confusion matrix, coverage/accepted accuracy, kích thước và latency CPU. Chỉ chuyển serving khi có bằng chứng phù hợp; checkpoint train mới không tự động thay baseline.

## Phase 5. Tích hợp và bàn giao

Chạy regression API/crop/đồng thời và browser với camera mô phỏng, thao tác chạm, crop, tải chậm, retry, màn nhỏ. API công khai qua HTTPS để camera web hoạt động trên điện thoại; localhost của điện thoại không trỏ tới máy server. Ghi rõ thiết bị mô phỏng so với điện thoại thật. Bàn giao test trong tests/, hồ sơ kết quả và hướng dẫn triển khai; không ghi đã kiểm tra iPhone/Android thật khi chưa có thiết bị.
