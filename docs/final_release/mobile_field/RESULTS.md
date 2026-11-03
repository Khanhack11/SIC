# Kết quả cập nhật ngày 03/11/2026

V3 mới cải thiện phân loại trên PlantDoc nhưng chưa đạt cổng triển khai. API vẫn dùng MobileNetV2 mặc định, kèm chính sách xem lại nhãn khỏe cho ảnh người dùng. V3 được lưu riêng để thử và nghiên cứu tiếp.

[Lượt tiếp tục ngày 03/11](continuation_20261103/README.md) đã hoàn tất thêm hai epoch, đo validation CPU 46,59% F1 PlantDoc và sửa lỗi camera/crop. Các bảng bên dưới giữ kết quả của checkpoint trước để đối chiếu, không phải kết quả của lượt tiếp tục.

## Dữ liệu và huấn luyện

Đã tải PlantDoc tại revision `5467f6012d78d1c446145d5f582da6096f852ae8`, xác minh 2.580 file gồm 2.578 ảnh và tài liệu nguồn. Sau kiểm tra và lọc, PlantDoc có 1.658 ảnh train, 414 validation và 217 test. Giữ nguyên split PlantVillage 8.043/1.731/1.720. `data_audit.json` ghi ánh xạ nhãn, ảnh loại và giới hạn dữ liệu.

Model dùng MobileNetV3-Large với trọng số ImageNet V2, đầu phân loại 960 → 256 → 12. Tiền xử lý letterbox giữ tỷ lệ, JPEG draft để giảm bộ nhớ, RGB và chuẩn hóa ImageNet được dùng nhất quán giữa train và serving.

Train bằng NVIDIA RTX 2050, học đầu phân loại rồi fine-tune ba khối cuối. Phiên làm việc bị ngắt giữa epoch fine-tune 3. Khôi phục checkpoint tốt nhất của các epoch hoàn tất, `finetune/2`, bỏ cập nhật chưa lưu. Không coi đây là đã hoàn tất cả bốn epoch dự kiến. Script `ml/recover_field_candidate.py` kiểm tra hash hợp đồng trước khi khôi phục và hiệu chỉnh T chỉ từ validation. Không khôi phục optimizer hoặc vị trí batch.

## Đối chứng cùng danh mục ảnh

| Macro F1 các lớp trong phạm vi có mẫu | Baseline V2 | Ứng viên V3 |
|---|---:|---:|
| PlantVillage validation | 94,48% | 97,96% |
| PlantDoc validation | 15,95% | 45,90% |
| PlantVillage test | 93,76% | 97,53% |
| PlantDoc test | 15,31% | 53,54% |

Các file `baseline_validation.json`, `candidate_validation.json`, `baseline_test.json`, `candidate_test.json` chứa confusion matrix và chỉ số từng lớp. Test được đo sau khi cố định checkpoint, T và cổng validation; không dùng để chỉnh model. PlantVillage test là split cũ được tái sử dụng để đối chứng, không phải bộ dữ liệu mới độc lập với lịch sử project.

PlantDoc validation chỉ có 8 lớp trong phạm vi có mẫu. Chỉ số bỏ lớp không có ground truth khỏi phép trung bình, vẫn tính lỗi khi dự đoán vào các đầu ra khác. Không coi đây là xác nhận đủ 11 lớp ngoài thực địa. Nhãn Internet chưa được chuyên gia tại địa phương kiểm tra; lớp gỉ ngô có giới hạn ánh xạ tác nhân. Unsupported chỉ đại diện các nhóm ảnh ngoài phạm vi đã thu thập.

## Quyết định triển khai

Theo `docs/FIELD_RELEASE_CRITERIA.json` đã chốt trước đánh giá, V3 qua tiêu chí tăng F1 và không làm giảm F1 PlantVillage. Trên 414 ảnh validation PlantDoc, 80 ảnh có điểm ≥75% với nhãn khác Unsupported, tương ứng coverage 19,32%. Có 44/80 dự đoán đúng, tức 55%. Hai kết quả dưới mức yêu cầu 25% và 85%, nên không thay baseline. Không chỉnh lại ngưỡng để làm cổng này đạt.

Không gặp dự đoán khỏe điểm cao trong 128 ảnh bệnh đã biết của validation PlantDoc. Số mẫu này nhỏ và thiếu các lớp; kết quả không chứng minh hết nguy cơ báo khỏe sai. API tiếp tục yêu cầu xem lại mọi nhãn khỏe trên ảnh người dùng.

Đo định tính trên vùng lá trích từ screenshot: V2 dự đoán Corn_healthy gần 100%; V3 nghiêng về Corn_Gray_leaf_spot ở 49,59%, dưới ngưỡng hiển thị thông tin bệnh. Chưa có ảnh gốc hoặc nhãn chuyên gia nên không kết luận V3 đã nhận đúng tổn thương. Không dùng screenshot để train hoặc tính accuracy.

## Web, điện thoại và tốc độ

Đã thêm crop thủ công, khôi phục ảnh, kiểm tra pixels preview/upload khớp nhau và phóng chữ kết quả. API có crop_box cho client khác. Đây chưa phải detector hay segmentation lá. Ảnh HEIC chưa được hỗ trợ; cần JPG/PNG.

Bộ kiểm tra ở `tests/results/20261103_110644_372541/summary.json` đạt cả unit/integration, backend, stress/edge, Chrome frontend và mobile browser. Bộ unit/integration có 34 test, backend 10 tình huống, stress/edge 17 tình huống, frontend 24 kiểm tra. Chromium và WebKit được kiểm tra ở 360/390/412/844 px, có thao tác chạm và camera mô phỏng. Chưa kiểm tra Android hoặc iPhone vật lý.

Weights V3 thật đã được kiểm tra riêng qua FastAPI TestClient: health, các ảnh demo, crop hợp lệ, ảnh hỏng, vùng crop quá nhỏ và chính sách nhãn khỏe. Đo CPU sau ba warmup, 30 lần tuần tự trên một JPEG cho median 28,37 ms, p95 80,13 ms. Có thể có nhiễu do các tiến trình đánh giá chạy cùng máy. Đây là decode + preprocessing + CNN, không phải thời gian mạng/điện thoại hoặc benchmark tải đồng thời. Xem `field_demo.json`.

## Sơ đồ để học và báo cáo

[Bản hướng dẫn và ba sơ đồ Archify](../../../.archify/architecture-sic-pipeline-20261103-114010/README.md) giải thích tải dữ liệu, crop, train, tensor, CNN, softmax và advisory. Mã nguồn Archify đã tải vào thư mục tools cục bộ, dùng gói CLI chính thức để tạo HTML độc lập. Validate, delivery và kiểm tra artifact đạt; browser-check của chính Archify lỗi đóng tiến trình trên Windows. Đã kiểm tra riêng bằng Playwright Chrome 1440/1920 px và xem ảnh chụp ba sơ đồ. Không ghi cổng browser-check của Archify là đạt.

## Việc cần dữ liệu mới

Thu ảnh điện thoại có nền phức tạp và đủ các lớp đang thiếu ở PlantDoc, gán nhãn với người có chuyên môn, bổ sung tổn thương sâu hại khi đã xác định phạm vi nhãn. Chốt bộ kiểm định thực địa mới theo nhóm cây/lần chụp trước lần train tiếp theo. Chỉ xây detector hoặc segmentation khi có nhãn hộp/mặt nạ và phép đo phù hợp; không gọi crop thủ công là tự động tìm lá.
