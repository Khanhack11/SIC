# Phần Data và Optimization của Trần Đặng Công Tâm

Vòng triển khai mới: [tối ưu HTTP và ONNX ngày 17/10/2026](OPTIMIZATION_20261017.md). Cấu hình mặc định hiện dùng 2 threads, 2 inference đồng thời, ONNX nếu có graph và dependency; khi chưa có ONNX dùng TorchScript.

Đây là bộ công cụ để chuẩn bị dữ liệu, huấn luyện một baseline MobileNetV2 trên CPU, đánh giá chất lượng và đo độ trễ trước khi bàn giao cho ML Engineer và Backend Developer. Vai trò trong action plan là Data & Optimization Engineer. Baseline được bổ sung vì nhóm bắt đầu chưa có checkpoint.

## Sản phẩm bàn giao

| Công việc | Mã nguồn | Đầu ra |
|---|---|---|
| Tải dữ liệu có kiểm tra hash và phiên bản | `ml/download_dataset.py` | Ảnh RGB và `data/raw/plantvillage/source.json` |
| Kiểm tra ảnh, loại trùng và chia theo nhóm lá | `ml/prepare_dataset.py` | CSV train/validation/test, nhãn, audit và danh sách loại bỏ |
| Tiền xử lý và augmentation | `backend/preprocessing.py`, `ml/dataset.py` | Tensor RGB 224×224 chuẩn hóa ImageNet |
| Baseline transfer learning | `ml/train.py` | Checkpoint có metadata nhãn và temperature, lịch sử huấn luyện |
| Đánh giá trên test riêng | `ml/evaluate.py` | Accuracy, macro precision/recall/F1, confusion matrix, NLL/ECE |
| Tối ưu inference và đo tải | `backend/inference_engine.py`, `ml/benchmark.py` | So sánh eager/TorchScript, số CPU thread, độ trễ và throughput |

## Nguồn dữ liệu và phạm vi nhãn

Dữ liệu dùng [PlantVillage của nhóm tác giả](https://github.com/spMohanty/PlantVillage-Dataset), biến thể `raw/color`, phiên bản `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Downloader chọn tối đa 1.500 ảnh mỗi lớp theo seed 42, giữ lớp ít ảnh ở kích thước gốc; tổng chọn 11.504 ảnh. Mỗi file và leaf-map được kiểm tra theo Git blob SHA-1 từ tree của commit. `source.json` lưu danh sách file đã chọn và hash để kiểm tra lại. Lần chạy lại xác minh file hiện có trước khi tải tiếp.

[README của nguồn](https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/README.md) và [leaf-map](https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/leaf-map.json) là căn cứ cho việc nhóm các ảnh cùng lá. Tài liệu nguồn cũng có [script đọc metadata nhóm lá](https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/plant_village.py). Nguồn này có split 80/20; project tự tạo split 70/15/15 theo action plan, giữ nhóm lá tách biệt.

`backend.config.CLASS_NAMES` là hợp đồng thứ tự nhãn giữa dữ liệu, model và API. Không sử dụng thứ tự thư mục do ImageFolder tự suy ra. `ml/labels.py` ánh xạ rõ tên thư mục gốc về 11 nhãn đang có trong backend và knowledge base:

| Index | Nhãn API | Thư mục PlantVillage |
|---|---|---|
| 0 | Corn_Common_rust | Corn_(maize)___Common_rust_ |
| 1 | Corn_Gray_leaf_spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| 2 | Corn_Northern_Leaf_Blight | Corn_(maize)___Northern_Leaf_Blight |
| 3 | Corn_healthy | Corn_(maize)___healthy |
| 4 | Potato_Early_blight | Potato___Early_blight |
| 5 | Potato_Late_blight | Potato___Late_blight |
| 6 | Potato_healthy | Potato___healthy |
| 7 | Tomato_Early_blight | Tomato___Early_blight |
| 8 | Tomato_Late_blight | Tomato___Late_blight |
| 9 | Tomato_Septoria_leaf_spot | Tomato___Septoria_leaf_spot |
| 10 | Tomato_healthy | Tomato___healthy |

Plan mô tả nhóm bệnh ở mức khái quát; phạm vi triển khai giữ 11 lớp đang được API hỗ trợ. Nhóm cần ghi phạm vi này vào báo cáo cuối kỳ.

## Chạy từ đầu

Chạy lệnh tại thư mục gốc SIC. Cần Python, các thư viện trong `requirements.txt`, kết nối mạng khi tải ảnh hoặc weights ImageNet. Chạy trên CPU được; không cần tài khoản Kaggle.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -u -m ml.download_dataset --workers 16
python -u -m ml.prepare_dataset
$env:TORCH_HOME = "$PWD/artifacts/torch"
python -u -m ml.train --threads 4 --batch-size 64 --views 2 --epochs 80 --patience 12
python -u -m ml.evaluate
python -u -m ml.benchmark --threads 1 2 4 8 --runtimes eager torchscript --concurrency 1 4
python -m ml.handoff
```

Không chạy lại prepare vào output đã có; dùng `--output data/processed/version2` để tạo phiên bản khác. Các lệnh train/evaluate/benchmark có `--data` để chọn manifest mới, và `--source` để trỏ đến ảnh khi di chuyển thư mục project. Train từ chối ghi đè checkpoint cuối cùng; chọn `--output models/baseline_v2.pt` cho một lần huấn luyện mới.

## Kiểm tra và chia tập

Pipeline kiểm tra nội dung JPEG/PNG, khả năng decode, cạnh tối thiểu 32px và tối đa 20 megapixel. Ảnh được đọc với EXIF orientation rồi chuyển RGB; SHA-256 của file và nội dung pixel được ghi vào manifest. Những ảnh pixel giống hệt nhau được loại trùng sau khi gộp các nhóm lá tương ứng. Nhóm có cùng lá hoặc pixel nhưng nhiều nhãn khác nhau bị đưa vào danh sách loại bỏ để nhóm kiểm tra thủ công.

Chia tập theo từng lớp, gán cả nhóm lá vào một tập, gần tỷ lệ 70/15/15 nhất bằng thuật toán greedy. Tỷ lệ thực tế có thể lệch do kích thước nhóm nguyên. Audit kiểm tra không giao nhau về leaf group, file hash và pixel hash. Mỗi lần đọc ảnh phục vụ train/evaluate, loader kiểm tra lại SHA-256 để phát hiện dữ liệu bị thay đổi sau khi chia tập.

Ảnh không tìm được leaf-map dùng identifier dự phòng và được đếm trong audit. Các kiểm tra kỹ thuật không xác nhận được nhãn sinh học đúng; ảnh gần giống nhưng không trùng pixel cũng có thể chưa bị phát hiện. Nhóm vẫn cần rà soát mẫu ảnh và nhãn trước khi dùng kết quả trong báo cáo.

## Pipeline và huấn luyện baseline

Train, validation, test và serving dùng chung `backend/preprocessing.py`: áp dụng EXIF orientation, RGB, resize 224×224, ToTensor và ImageNet mean/std. Chỉ train có lật ngang/dọc, xoay trong ±20° và brightness jitter 0,2. Validation, test và serving không có biến đổi ngẫu nhiên.

Baseline khởi tạo `MobileNet_V2_Weights.IMAGENET1K_V1`, đóng băng backbone và batch normalization, trích xuất hai phiên bản augmentation cố định mỗi ảnh train rồi cache feature. Đầu phân loại được huấn luyện bằng Adam, cosine annealing, cross-entropy có trọng số lớp và early stopping trên validation loss. Cách này giảm chi phí CPU; không tương đương fine-tune toàn bộ mạng. ML Engineer có thể dùng dữ liệu và hợp đồng đầu vào này để fine-tune sâu hơn ở giai đoạn tiếp theo.

Test không được đọc khi huấn luyện, chọn epoch, hiệu chỉnh temperature hoặc chọn cấu hình tốc độ. Temperature scaling chỉ dùng validation; việc hiệu chỉnh này không bảo đảm độ tin cậy trên ảnh thực địa. Checkpoint lưu class_names, input_size, mean/std, architecture, temperature và lịch sử huấn luyện. Backend kiểm tra các metadata này trước khi nhận weights. State_dict thuần từ phiên bản cũ vẫn đọc được nhưng không được đánh dấu calibrated.

## Phục hồi sau khi dừng

```powershell
New-Item -ItemType File -Path artifacts/PAUSE_REQUESTED -Force
```

Training dừng tại ranh giới batch trích xuất feature hoặc ranh giới epoch huấn luyện; cache shard và `.resume.pt` giữ tiến độ đã hoàn tất. Để tiếp tục, xóa đúng file marker này rồi chạy lại lệnh train với cùng tham số. Train khôi phục head, optimizer, scheduler và RNG từ resume checkpoint. Khi checkpoint cuối cùng đã tồn tại, dùng nó để evaluate/benchmark; không chạy lại train lên cùng output.

## Đo độ trễ và tích hợp backend

Benchmark chỉ dùng ảnh validation, bao phủ mỗi lớp một ảnh, có warmup rồi đo mean/p50/p95/max và throughput. Chạy cả concurrency 1 và 4 bằng client thread để mô phỏng tải trong process. Thời gian gồm kiểm tra file, decode, tiền xử lý, inference và chuyển xác suất; không gồm HTTP/network hay truy xuất tư vấn. Vì vậy chưa thể dùng số này làm độ trễ end-to-end của web app.

So sánh eager và TorchScript với nhiều số CPU thread. Kiểm tra parity trên 11 ảnh validation, yêu cầu nhãn giống nhau và chênh xác suất đã làm tròn không quá 0,0002. Parity ở tập mẫu này chưa thay thế kiểm định đầy đủ khi thay runtime triển khai. Với `recommended_serial_setting` trong benchmark.json, đặt cấu hình trước khi chạy server:

```powershell
$env:SIC_TORCH_THREADS = "4"  # thay bằng kết quả benchmark của máy triển khai
$env:SIC_INFERENCE_RUNTIME = "eager"  # hoặc torchscript nếu đã kiểm định
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Tối ưu serving gồm tái sử dụng transform, `torch.inference_mode()`, bố trí tensor channels-last trên CPU, một lần chuyển xác suất về CPU và tùy chọn freeze TorchScript. Feature extraction cũng dùng channels-last; cache có metadata bố trí tensor để không trộn các lượt chạy. Khi checkpoint thiếu hoặc không tương thích, health báo `MODEL_NOT_READY` và diagnose trả 503 để tránh đưa ra kết quả từ trọng số ngẫu nhiên.

## Kiểm tra mã nguồn và báo cáo

```powershell
python -m unittest discover -s tests -v
python test_backend.py
python stress_and_edge_test.py
```

Kiểm tra mới bao phủ tính tái lập, leaf-disjoint split, ảnh hỏng/trùng, hợp đồng nhãn, train-serving parity và trường hợp thiếu model. Fixture tổng hợp dùng để kiểm tra logic; chỉ các báo cáo tạo từ bộ ảnh thật mới được dùng làm kết quả chất lượng.

Kết quả chạy thực tế sẽ nằm trong `results/`; trạng thái và các lệnh tiếp tục nằm trong `PROGRESS.md`. Ảnh, feature cache và weights được giữ local, bỏ qua bởi Git để tránh đưa hàng nghìn ảnh và binary lớn vào source repo. Khi bàn giao cho nhóm, chuyển riêng bộ ảnh/weights hoặc để nhóm chạy các lệnh tái lập. Không coi kiểm thử API pass là bằng chứng đạt macro F1 ≥85%; chỉ dùng test report của model đã huấn luyện.

`python -m ml.handoff` kiểm tra cùng checkpoint và manifest giữa các báo cáo, sao chép manifest cùng audit vào bộ bàn giao, tạo `REPORT.md` và lưu phiên bản thư viện. `python -m ml.preview` tạo contact sheet mẫu train trong `artifacts/` để nhóm xem ảnh và nhãn nguồn; thao tác này không thay thế kiểm định nhãn bởi chuyên gia.
