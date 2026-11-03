# Lượt tiếp tục ngày 03/11/2026

Đã hoàn tất thêm hai epoch fine-tune từ checkpoint V3 trước đó, sửa hai lỗi chọn ảnh/crop và chạy lại toàn bộ bộ test. V3 tiếp tục là bản thử nghiệm vì chưa đạt cổng kiểm định thực địa. V2 vẫn là mặc định của demo.

## Model mới được đo như thế nào

Run mới bắt đầu từ `mobilenetv3_field_20261103.pt`, dùng optimizer AdamW mới vì lượt cũ không giữ optimizer. Các khối mở để fine-tune, tiền xử lý và tốc độ học giữ như lượt trước. Chỉ train và validation được đưa vào DataLoader. Model được chọn bằng trung bình loss validation của hai miền, rồi hiệu chỉnh temperature bằng validation.

Đã chủ động yêu cầu lưu/dừng sau batch 294 (9.408 mẫu của epoch đầu), sau đó chạy lại đúng lệnh. Log xác nhận khôi phục tại `epoch 0, next_sample 9408` và hoàn tất hai epoch. Checkpoint tiến trình giữ model, optimizer, AMP scaler, thứ tự lấy mẫu, vị trí và RNG. Nếu tiến trình bị tắt đột ngột, chạy lại từ lần lưu gần nhất, tối đa khoảng 50 batch đã chạy cần làm lại. Không tuyên bố bảo đảm bitwise giữa mọi GPU/runtime.

| Validation, runtime CPU phục vụ | Baseline V2 | V3 trước | V3 tiếp tục |
|---|---:|---:|---:|
| F1 các lớp trong phạm vi có mẫu, PlantVillage | 94,48% | 97,96% | 97,52% |
| F1 các lớp trong phạm vi có mẫu, PlantDoc | 15,95% | 45,90% | 46,59% |
| Độ chính xác nhóm PlantDoc vượt cổng điểm | 12,28% | 55,00% | 57,69% |
| Coverage trên toàn bộ PlantDoc validation | 55,07% | 19,32% | 18,84% |

Cổng điểm là điểm cao nhất ≥0,75 và nhãn khác Unsupported, trước chính sách xem lại ảnh khỏe. V3 tiếp tục có 78/414 ảnh vượt cổng, trong đó 45 ảnh được dự đoán đúng. Độ chính xác 57,69% dưới yêu cầu 85%; coverage 18,84% dưới yêu cầu 25%. Các tiêu chí giữ nguyên trong `docs/FIELD_RELEASE_CRITERIA.json`. [Quyết định](release_decision.json) được tính bằng lệnh `ml.field_release`, không đọc test và không tự thay weights phục vụ.

PlantDoc validation chỉ có 8 lớp trong phạm vi có mẫu, nhiều lớp rất ít ảnh; nhãn từ Internet chưa được chuyên gia địa phương xác nhận. Lớp Unsupported chỉ bao phủ phần ngoài phạm vi đã thu thập. Các chỉ số không chứng minh đủ 11 lớp trên ảnh điện thoại. Tập test đã dùng ở lượt trước được giữ nguyên, không đánh giá lại hoặc chọn model theo test. Cần bộ ảnh thực địa độc lập trước khi xác nhận hiệu quả của lượt train tiếp theo.

Số đo trong log GPU là 46,64% F1 PlantDoc; runtime CPU cho 46,59%. Báo cáo dùng số CPU vì đây là runtime phục vụ. Chênh lệch nhỏ có thể đổi lớp ở trường hợp sát nhau; không coi hai runtime là bitwise tương đương.

Trên vùng lá trích từ screenshot người dùng, V3 tiếp tục nghiêng về Corn_Gray_leaf_spot ở 75,57%, tăng so với 49,59% của V3 trước. Chưa có nhãn chuyên gia cho ảnh này nên không coi đây là nhận đúng bệnh. Việc điểm tăng qua ngưỡng cũng không chứng minh lỗi phân tích đã được giải quyết. Giao diện V3 ghi rõ bản thử nghiệm và độ chính xác thực địa chưa được xác nhận.

## Lỗi demo đã sửa

Camera mã hóa ảnh bất đồng bộ. Trước sửa, callback chụp đến chậm thay thế file người dùng vừa chọn, hoặc xuất ảnh sau khi đã đóng camera. Test tái hiện được việc `Tomato_Early_blight.jpg` bị đổi thành `camera_leaf.jpg` trước khi gọi API. Callback hiện kiểm tra cả phiên camera và phiên chọn ảnh trước khi nhận kết quả.

Crop trước đây làm tròn vị trí và độ rộng riêng. Chọn nửa phải ảnh rộng 95 px tạo ảnh rộng 48 px và vượt mép nguồn một cột. Hiện tính hai mép trước rồi lấy hiệu, cho đúng 47 px. API và web dùng cùng quy tắc làm tròn nửa pixel lên trên; mép phải/dưới được chặn trong ảnh. Ảnh rộng 97 px được kiểm tra riêng để tránh khác biệt giữa làm tròn Python và JavaScript.

Crop vẫn do người dùng chọn khung. Chưa có detector hoặc segmentation tự tìm lá.

## Kiểm thử

[Tóm tắt regression](regression_summary.json) ghi 38 unit/integration test, 10 backend, 17 stress/edge, 24 kiểm tra frontend chính và 3 kiểm tra mới về camera/crop. Chromium và WebKit đạt ở viewport 360/390/412/844 px, có thao tác chạm. Đây là mô phỏng trên máy tính, chưa kiểm tra điện thoại hoặc camera vật lý.

Weights V3 mới được kiểm tra qua FastAPI với ảnh thật của bộ demo, crop hợp lệ, ảnh lỗi và vùng cắt quá nhỏ. [Kết quả](field_demo.json) có hash weights và phạm vi đo. Trên một JPEG, ba warmup rồi 30 lần gọi tuần tự cho median 20,24 ms và p95 28,02 ms, gồm decode, tiền xử lý và CNN. Không phải thời gian HTTP, mạng điện thoại, khởi động nguội hoặc tải đồng thời.

Kiểm tra HTTP trên server thử nghiệm xác nhận model V3 đã nạp và ảnh khỏe cần review. Lần gọi đầu có latency model 228,7 ms, cho thấy số warmup không đại diện cho khởi động nguội.

## Mở demo và chạy lại

`run_demo.cmd` dùng V2 tại port 8000. `run_field_demo.cmd` mở V3 thử nghiệm tại port 8003 để so sánh, cần weights trong gói `Reports/SIC_mobile_field_20261103.zip`. Điện thoại cùng mạng mở `http://IP_MAY_CHU:8003/app`; live camera trên trình duyệt điện thoại cần HTTPS, còn nút chọn ảnh/chụp bằng ứng dụng camera có thể dùng trên HTTP.

Lệnh train mới, chạy trong môi trường CUDA, đặt output mới để giữ kết quả cũ:

```powershell
.\.venv-gpu\Scripts\python.exe -m ml.continue_field --source models/mobilenetv3_field_continued_20261103.pt --output models/mobilenetv3_field_next.pt --epochs 2
```

Nếu bị ngắt, chạy lại chính lệnh của run đó để đọc `.progress.pt`. Nếu có marker `artifacts/PAUSE_REQUESTED`, giữ nguyên cho tới khi muốn tiếp tục. Mỗi lượt train mới cần lựa chọn lại cách kiểm định; không train và đo đi đo lại trên test cũ.

Để tính cổng validation cho một checkpoint mới:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate_field --weights models/mobilenetv3_field_next.pt --output artifacts/next_validation.json
.\.venv\Scripts\python.exe -m ml.field_release --candidate artifacts/next_validation.json --output artifacts/next_release.json
```

Chạy `run_tests.cmd` để kiểm tra phần mềm, không dùng bộ test phần mềm làm số đo độ chính xác bệnh lá.
