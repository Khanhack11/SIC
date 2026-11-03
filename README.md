# Crop Disease Advisory System

Hệ thống nhận diện ảnh lá cà chua, khoai tây và ngô, sau đó trả thông tin chăm sóc tham khảo. MobileNetV2 phân loại 11 lớp gồm 8 bệnh và 3 lớp lá khỏe. FastAPI phục vụ cả frontend tại `/app` và API.

Phạm vi ưu tiên hiện tại: chọn ảnh có sẵn từ máy → xem trước → crop nếu cần → phân tích → đọc kết quả. Chụp bằng điện thoại được để sang giai đoạn sau; chức năng camera vẫn có trong mục “Chụp ảnh (chức năng bổ sung)”, mặc định thu gọn. Chọn file không yêu cầu quyền camera.

Luồng điện thoại và ảnh thực địa được ghi trong [các phase](docs/MOBILE_FIELD_PHASES.md). Frontend có crop thủ công và chữ kết quả lớn hơn. Ảnh người dùng mặc định là ảnh thực địa: dự đoán lớp khỏe vẫn yêu cầu kiểm tra lại. Điểm dự đoán không phải độ chính xác chẩn đoán. MobileNetV3-Large có thêm lớp ngoài phạm vi là ứng viên riêng, chỉ nạp khi đặt `SIC_FIELD_MODEL_PATH`; không tự thay checkpoint baseline.

Để hiểu model hoạt động, mở [sơ đồ hệ thống bằng Archify](.archify/architecture-sic-pipeline-20261103-114010/he-thong.html), [pipeline train](.archify/architecture-sic-pipeline-20261103-114010/huan-luyen.html), [pipeline suy luận](.archify/architecture-sic-pipeline-20261103-114010/suy-luan.html) và [giải thích từng bước](.archify/architecture-sic-pipeline-20261103-114010/README.md). V3 mới đã train và đánh giá, nhưng chưa qua tiêu chí triển khai; xem [kết quả và giới hạn](docs/final_release/mobile_field/RESULTS.md).

Lượt tiếp tục ngày 03/11 đã hoàn tất thêm hai epoch và sửa lỗi camera ghi đè ảnh mới, crop sát mép ảnh rộng lẻ. [Kết quả lượt tiếp tục](docs/final_release/mobile_field/continuation_20261103/README.md). Mở `run_field_demo.cmd` để thử V3 riêng tại port 8003; V2 mặc định vẫn ở port 8000. Weights nằm trong gói `Reports/SIC_mobile_field_20261103.zip`, không lưu trong Git.

[Đối chiếu ViLLite và pipeline áp dụng cho SIC](docs/final_release/villite_adaptation/README.md): V3 có thêm runtime ONNX FP32 tùy chọn, kiểm tra hợp đồng/checksum và đối chứng trên validation trước khi đo HTTP. `run_field_demo.cmd onnx` dùng graph đã export, vẫn là bản V3 thử nghiệm, không đổi cổng kiểm định bệnh lá.

Samsung Innovation Campus, Group 5. Chuẩn bị báo cáo **04–05/11/2026**, thời lượng 20 phút.

## Tìm mã nguồn và tài liệu

| Thư mục | Nội dung |
|---|---|
| [frontend](frontend/README.md) | HTML, JavaScript upload, crop và kết quả |
| backend | FastAPI, inference, preprocessing và tư vấn |
| [ml](ml/README.md) | Tải dữ liệu, train, đánh giá, export và đóng gói |
| [data](data/README.md) | Kho tri thức, ảnh gốc và manifest đã xử lý |
| [models](models/README.md) | Trọng số đang dùng và checkpoint thí nghiệm |
| [tests](tests/README.md) | Toàn bộ test, fixture và kết quả mỗi lần chạy |
| [docs](docs/README.md) | Báo cáo, số liệu và kịch bản demo |
| [notebooks](notebooks/README.md) | Notebook khám phá backend |
| [scripts](scripts/README.md) | Kiểm kê thư mục và dung lượng |
| artifacts | Cache, log, công cụ và file sinh ra trên máy |

[Bản đồ và kết quả sắp xếp thư mục](docs/project_structure/README.md) ghi thay đổi đường dẫn cùng hồ sơ kiểm tra. Mở web bằng `run_demo.cmd`; URL `/app` và các lệnh `python -m ml.*` vẫn giữ.

| Thành viên | Phạm vi công việc |
|---|---|
| Đinh Gia Bảo | Điều phối, phát triển và đánh giá model |
| Trần Đặng Công Tâm | Dữ liệu, preprocessing, huấn luyện và runtime |
| Nguyễn Quốc Khánh | Backend và kho tri thức |
| Ma Hồng Hải | Frontend và đối chiếu tài liệu nông học |

## Chạy demo trên Windows

Gói bàn giao `SIC_mobile_field_20261103.zip` có source, ảnh demo và weights. Git chỉ giữ code và hồ sơ kết quả; raw data, môi trường Python và weights được bàn giao riêng.

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

Chọn JPG/PNG từ máy, xem ảnh và chọn vùng lá nếu cần, rồi bấm Phân tích ảnh. File tối đa 10 MB, mỗi cạnh ít nhất 32 px, không quá 20 megapixel. Phần camera bổ sung chỉ mở khi người dùng chọn; đóng mục này sẽ dừng camera. Camera trực tiếp cần localhost/HTTPS và quyền camera.

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

[Danh mục và phân tích dữ liệu ngày 03/11](docs/datasets/20261103/README.md) tập hợp PlantVillage 11 lớp, PlantDoc Classification/Detection, PlantWild_v2 và PlantSeg trong `data/raw/dataset_hub_20261103`. Đã quét 35.947 bản ghi ảnh, tách 639 ứng viên mới để duyệt nhãn và đánh dấu các trường hợp cần xử lý hộp/mask. Toàn bộ ảnh PlantSeg trùng pixel với ảnh PlantWild_v2; không dùng hai bộ như hai nguồn đánh giá độc lập.

```powershell
.\.venv\Scripts\python.exe -m ml.download_recommended_datasets
.\.venv\Scripts\python.exe -m ml.analyze_recommended_datasets
```

Bản PlantDoc mới bảo toàn tên khác chữ hoa/thường và ký tự Windows bằng `extraction_map.json`. Snapshot PlantDoc cũ có 6 Git blob bị ghi trùng tên trên Windows. Đợt train tiếp theo cần nối manifest đã duyệt với đường dẫn mới; `ml.prepare_field` hiện chưa đọc ánh xạ này. Chưa train lại hoặc thay model phục vụ từ danh mục bổ sung.

Pipeline thí nghiệm trước ghép PlantDoc với split PlantVillage đã audit, giữ test riêng và có lớp `Unsupported`. Xem [audit dữ liệu của thí nghiệm](docs/final_release/mobile_field/data_audit.json) và [tiêu chí phát hành](docs/FIELD_RELEASE_CRITERIA.json). Các lệnh dưới đây mô tả pipeline cũ, chưa dùng danh mục bổ sung:

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
- [12 nguồn dữ liệu bổ sung và quy trình gán nhãn](docs/research/DATASET_SOURCES_20261103.md)
- [Lượt train V3 với dữ liệu bổ sung ngày 03/11](docs/final_release/data_expansion_20261103/README.md)
- [Lượt tiếp tục train phối hợp Antigravity](docs/final_release/antigravity_continuation_20261103/RESULTS.md)
- [Kịch bản trình bày 20 phút](docs/final_release/PRESENTATION_20_MIN.md)

PlantVillage có nền kiểm soát. Chưa đo độ chính xác ngoài đồng hoặc hiệu quả giảm thiệt hại mùa vụ. Một phần ảnh dùng định danh lá dự phòng nên audit không chứng minh tuyệt đối mọi lá vật lý đã tách biệt. Test đã được sử dụng ở các vòng trước. Model không phát hiện ảnh ngoài miền; cây khác hoặc đồ vật vẫn có thể nhận confidence cao.
