# Kiểm kê dữ liệu ngày 03/11/2026

Audit ghi nhận 40.219 record ảnh từ 10 nguồn, 40.217 record đọc được. Đây là số record nguồn, có ảnh trùng giữa các bộ, không phải số ảnh độc lập dùng train. Dữ liệu gốc và metadata nằm trong `data/raw/dataset_hub_20261103/`.

| Nguồn | Record ảnh | Đọc được | Kiểu nhãn/giới hạn |
|---|---:|---:|---|
| maize_field_bd | 994 | 994 | Thư mục và CSV hộp; tọa độ hộp không khớp bản ảnh resized |
| plantdoc_classification | 2578 | 2578 | Thư mục, 28 lớp; ảnh Internet/nền phức tạp |
| plantdoc_detection | 2603 | 2603 | CSV/XML, 8.921 hộp; crop chỉ lấy ảnh cha train |
| plantseg | 7774 | 7774 | Mask tổn thương; ảnh trùng PlantWild, chưa train segmentation |
| plantvillage | 11504 | 11504 | Thư mục, 11 lớp SIC; nền kiểm soát |
| plantwild_v2 | 11488 | 11486 | Thư mục, 115 lớp; cần kiểm nhãn và trùng nguồn |
| potato_field_indonesia | 1096 | 1096 | Thư mục; nhãn Fungi mơ hồ bị loại khỏi nhãn bệnh cụ thể |
| potato_holeta_v2 | 430 | 430 | Thư mục Healthy/LateBlight; đã có trong train đa nguồn |
| potato_late_blight_iitkgp | 1561 | 1561 | Thiếu ground truth từng ảnh; chưa đưa vào supervised train |
| tomato_field_pakistan | 191 | 191 | Thư mục; chọn ảnh tên camera, không tự xem đó là chứng nhận nguồn ảnh |

[Summary audit nguyên bản](summary.json) và [kiểm chứng tải](download_verification.json) ghi kích thước, nhãn, trạng thái và thông tin hash. Metadata nguồn/DOI/license được lưu ở `data/raw/dataset_hub_20261103/<source>/source.json` và `sources/`. Hash xác nhận tính toàn vẹn file theo bằng chứng có sẵn, không xác nhận bệnh đúng.

## Dữ liệu thực sự dùng train

`field_mobile_v2/records.json` có 20.902 record: 16.225 train, 2.740 validation, 1.937 test. Bổ sung 3.160 crop PlantDoc train và 3.959 ảnh đa nguồn (3.364 train, 595 validation); giữ các holdout cũ. Bản khóa này đã được sử dụng cho candidate hiện tại, không tái tạo hoặc sửa tại chỗ.

`prepare_field_v2` lọc hash và ảnh gần giống theo ngưỡng đã chọn, giữ nhóm ảnh cùng split. Heuristic hash chưa chứng minh không còn trùng cùng lá/phiên chụp. Nhãn thư mục và ánh xạ thuật ngữ cần người có chuyên môn xem lại; hiện chưa có quy trình chuyên gia duyệt đầy đủ.

Không tự chuyển ảnh chưa có nhãn thành healthy/Unsupported. Corn blight chung chung và Potato Fungi bị loại khỏi nhãn bệnh cụ thể. Mask PlantSeg chưa tạo thành một model crop/segment đang chạy. Web chỉ crop thủ công.

[Nhãn thật và class index](../../DATA_LABELING.md), [thống kê nhãn hiện tại](../../benchmarks/label_summary.json), [benchmark candidate hiện tại](../../BENCHMARKS.md). Các báo cáo thí nghiệm cũ có weights/calibration khác; không trộn điểm của chúng vào kết quả hiện tại.

## Mã tái hiện

`ml/download_recommended_datasets.py`, `ml/download_additional_field_datasets.py` tải dữ liệu; `ml/analyze_recommended_datasets.py` kiểm kê và ánh xạ nhãn. `ml/prepare_field_v2.py` tạo phiên bản manifest mới; từ chối ghi đè thư mục đã có. Nên xem source metadata/license và duyệt nhãn trước khi train phiên bản mới.

[Lệnh train và đánh giá](../../../ml/README.md). Lượt train mới lưu trong `artifacts/training_runs/`, không sinh hàng loạt weights trong `models/`.
