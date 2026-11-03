# Crop Disease Advisory System

Hệ thống nhận diện ảnh lá cà chua, khoai tây và ngô, sau đó trả thông tin chăm sóc tham khảo. MobileNetV2 phân loại 11 lớp gồm 8 bệnh và 3 lớp lá khỏe. FastAPI phục vụ cả frontend tại `/app` và API.

Luồng điện thoại và ảnh thực địa được ghi trong [các phase](docs/MOBILE_FIELD_PHASES.md). Frontend có crop thủ công và chữ kết quả lớn hơn. Ảnh người dùng mặc định là ảnh thực địa: dự đoán lớp khỏe vẫn yêu cầu kiểm tra lại. Điểm dự đoán không phải độ chính xác chẩn đoán. MobileNetV3-Large có thêm lớp ngoài phạm vi là ứng viên riêng, chỉ nạp khi đặt `SIC_FIELD_MODEL_PATH`; không tự thay checkpoint baseline.

Để hiểu model hoạt động, mở [sơ đồ hệ thống bằng Archify](.archify/architecture-sic-pipeline-20261103-114010/he-thong.html), [pipeline train](.archify/architecture-sic-pipeline-20261103-114010/huan-luyen.html), [pipeline suy luận](.archify/architecture-sic-pipeline-20261103-114010/suy-luan.html) và [giải thích từng bước](.archify/architecture-sic-pipeline-20261103-114010/README.md). V3 mới đã train và đánh giá, nhưng chưa qua tiêu chí triển khai; xem [kết quả và giới hạn](docs/final_release/mobile_field/RESULTS.md).

Samsung Innovation Campus, Group 5. Chuẩn bị báo cáo **04–05/11/2026**, thời lượng 20 phút.

| Thành viên | Phạm vi công việc |
|---|---|
| Đinh Gia Bảo | Điều phối, phát triển và đánh giá model |
| Trần Đặng Công Tâm | Dữ liệu, preprocessing, huấn luyện và runtime |
| Nguyễn Quốc Khánh | Backend và kho tri thức |
| Ma Hồng Hải | Frontend và đối chiếu tài liệu nông học |

## Chạy demo trên Windows

Gói bàn giao `SIC_demo_20261104.zip` có source, ảnh demo và weights baseline. Git chỉ giữ code và hồ sơ kết quả; raw data, môi trường Python và weights được bàn giao riêng.

Môi trường đã kiểm tra: Windows 11, Python 3.13.9. Chạy trong thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Mở **http://127.0.0.1:8000/app**. Khi `.venv` đã có, có thể chạy `run_demo.cmd`. Giữ cửa sổ server mở trong buổi demo. Swagger: http://127.0.0.1:8000/docs.

Để dùng điện thoại cùng mạng, chạy `run_mobile_demo.cmd`, lấy IPv4 máy server bằng `ipconfig`, rồi mở `http://IP_SERVER:8000/app` trên điện thoại. Không dùng `127.0.0.1` của điện thoại để gọi máy server. Camera web trực tiếp cần HTTPS; HTTP dùng chọn file/camera gốc của thiết bị. Chưa kiểm tra điện thoại vật lý, NAT hoặc cấu hình firewall của từng mạng.

Model cần `models/mobilenetv2_crop_disease.pt` và graph tương ứng `models/mobilenetv2_crop_disease.onnx`. ONNX là mặc định khi graph và dependency phù hợp; còn lại dùng TorchScript. Khi không có checkpoint hợp lệ, health báo `MODEL_NOT_READY`, diagnose trả HTTP 503 và giao diện khóa nút phân tích. Graph phải khớp hash checkpoint.

## Sử dụng và phạm vi tư vấn

Chọn JPG/PNG hoặc bấm Chụp ảnh. File tối đa 10 MB, mỗi cạnh ít nhất 32 px, không quá 20 megapixel. Camera trực tiếp cần localhost/HTTPS và quyền camera. Thiết bị không hỗ trợ sẽ chuyển sang trình chọn ảnh/camera của thiết bị.

Ngưỡng giao diện cố định 75%. Confidence thấp sẽ chỉ trả hướng dẫn chụp lại. Khi đạt ngưỡng, app hiển thị triệu chứng, canh tác và nguồn tài liệu. `custom_threshold` vẫn có trong API để kiểm thử.

Mỗi record có nguồn và phạm vi đối chiếu với UMN Extension, Cornell CALS hoặc UMD Extension. Chưa có kiểm định chuyên gia đối với nhãn thuốc tại Việt Nam. API ẩn liều thuốc và công thức sinh học chưa duyệt ở mọi endpoint, kể cả khi confidence cao. Dữ liệu thuốc gốc chỉ được giữ để đối chiếu.

Ảnh thật trong `sample_images/demo` được chọn từ validation để minh họa thao tác; manifest ghi nguồn, nhãn, hash và confidence quan sát. Không dùng các ảnh này để kết luận chất lượng model.

## Model và hiệu năng

| Thí nghiệm | Validation macro F1 | Test macro F1 | Quyết định |
|---|---:|---:|---|
| MobileNetV2 backbone frozen | 94,48% | 93,76% | Giữ để phục vụ |
| CNN nhỏ, 8 epoch GPU | 75,87% | 75,47% | Đối chứng |
| MobileNetV2 fine-tune GPU, chọn epoch 5 | 95,01% | Không đọc test ứng viên | Loại vì recall một lớp giảm quá giới hạn |
| Fine-tune 03/11, chọn epoch 3, validation CPU | 94.64% | Không đọc test ứng viên | Chưa đạt gate tăng 0,5 điểm F1 |

Recall `Corn_Gray_leaf_spot` của ứng viên giảm 9,09 điểm phần trăm. Giới hạn phát hành đã đặt là không giảm quá 2 điểm ở bất kỳ lớp nào, nên giữ baseline và graph mặc định.

Baseline đạt accuracy 94,19% trên 1.720 ảnh test. Temperature fit bằng validation. Benchmark HTTP mới dùng ONNX FP32, 2 threads, 2 slots, 80 request mỗi mức tải:

| Clients | Mean | P95 |
|---:|---:|---:|
| 1 | 11,07 ms | 13,63 ms |
| 4 | 28,91 ms | 32,49 ms |
| 8 | 55,51 ms | 62,70 ms |

Đo gồm upload multipart, kiểm tra ảnh, preprocessing, inference, tư vấn và response trên localhost. Không xem đây là bảo đảm độ trễ Internet. [Hồ sơ bản phát hành](docs/final_release/README.md).

## Kiểm thử

Toàn bộ file test nằm trong [tests](tests/README.md). Mở `tests/run_tests.cmd` để chạy cả browser hoặc chạy `python tests/run_all.py` bằng môi trường `.venv`. Log và screenshot lưu theo từng lần chạy trong `tests/results/`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/test_backend.py
.\.venv\Scripts\python.exe tests/stress_and_edge_test.py
```

Hồ sơ frontend có kiểm tra tự động và screenshot desktop/mobile. Camera mô phỏng xác minh luồng phần mềm; cần thử camera vật lý trước buổi trình bày.

## Huấn luyện

Pipeline ứng viên mới ghép PlantDoc với split PlantVillage đã audit, giữ test riêng và có lớp `Unsupported`. Xem [audit dữ liệu](docs/final_release/mobile_field/data_audit.json) và [tiêu chí phát hành](docs/FIELD_RELEASE_CRITERIA.json). Với `.venv` đã cài, tải và chuẩn bị:

```powershell
.\.venv\Scripts\python.exe -m ml.download_plantdoc
.\.venv\Scripts\python.exe -m ml.prepare_field
.\.venv-gpu\Scripts\python.exe -m ml.train_field
```

Pretrained ImageNet V2 của Torchvision được tải khi train lần đầu. Model mới dùng letterbox 224 để giữ tỷ lệ; JPEG lớn được giải mã giảm kích thước trước khi tạo tensor. Học classifier từ đặc trưng cache, sau đó mở ba block cuối để fine-tune. Chọn checkpoint/temperature trên validation; không dùng test để chọn tiếp. Checkpoint `.resume.pt` của pipeline này chỉ giữ best weights và lịch sử, chưa khôi phục optimizer/vị trí batch. Model cũ dùng pipeline bên dưới.

Pipeline dùng PlantVillage ghim revision, chia 8.043/1.731/1.720 ảnh train/validation/test. [Hướng dẫn dữ liệu](docs/data_optimization/README.md).

```powershell
python -m ml.download_dataset
python -m ml.prepare_dataset
python -m ml.train
```

Prepare/train từ chối ghi đè đầu ra có sẵn. Train GPU dùng môi trường riêng:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-gpu\Scripts\python.exe -m ml.train_gpu --architecture small_cnn --epochs 8 --batch-size 32 --lr .001 --output models/small_cnn_new.pt
.\.venv-gpu\Scripts\python.exe -m ml.train_gpu --architecture mobilenet_v2 --epochs 6 --batch-size 16 --lr .00001 --output models/mobilenetv2_gpu_new.pt
```

Checkpoint mới lưu riêng. Trainer có resume và marker `artifacts/PAUSE_REQUESTED`, dừng có lưu ở ranh giới batch. Không xóa marker cũ khi chưa muốn tiếp tục. Chỉ đánh giá test sau khi khóa cấu hình bằng validation.

## Tài liệu và giới hạn

- [API và kho tri thức](docs/API_AND_DATABASE_SPECIFICATION.md)
- [Báo cáo và kết quả mới](docs/final_release/README.md)
- [Kịch bản trình bày 20 phút](docs/final_release/PRESENTATION_20_MIN.md)

PlantVillage có nền kiểm soát. Chưa đo độ chính xác ngoài đồng hoặc hiệu quả giảm thiệt hại mùa vụ. Một phần ảnh dùng định danh lá dự phòng nên audit không chứng minh tuyệt đối mọi lá vật lý đã tách biệt. Test đã được sử dụng ở các vòng trước. Model không phát hiện ảnh ngoài miền; cây khác hoặc đồ vật vẫn có thể nhận confidence cao.
