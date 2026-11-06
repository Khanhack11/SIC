# Hồ sơ Kiểm định Đa nguồn (10 Bộ dữ liệu Thực địa & Phòng thí nghiệm) và Danh mục Nguồn Uy tín Quốc tế

Báo cáo này ghi lại kết quả thu thập, xác thực mã băm nguyên bản (Git blob SHA-1, SHA-256, MD5, ZIP CRC32), phân tích quy cách gán nhãn (`Annotation Schema`), kiểm tra trùng lặp đa nguồn (`pHash` / `dHash` / `EXIF-oriented RGB SHA-256`) và phân tích chất lượng thực tế trên **10 bộ dữ liệu đã tải về máy (`40.219` bản ghi ảnh)** cùng danh mục các nguồn học thuật uy tín quốc tế cho 3 loại cây mục tiêu (**Ngô / Corn**, **Khoai tây / Potato**, **Cà chua / Tomato**).

---

## 1. Tổng quan 10 Bộ dữ liệu Đã Tải và Kiểm định tại `data/raw/dataset_hub_20261103`

| # | Tên định danh (`dataset`) | Nguồn học thuật / DOI / Kho lưu trữ | Giấy phép | Số ảnh thô | Ảnh hợp lệ | Dung lượng | Quy cách gán nhãn (`Annotation`) | Đặc điểm môi trường & Thiết bị |
|---:|---|---|---|---:|---:|---:|---|---|
| 1 | `plantvillage` | Mohanty et al. (2016), *Front. Plant Sci.* ([GitHub](https://github.com/spMohanty/PlantVillage-Dataset)) | CC0-1.0 | 11.504 | 11.504 | 186,6 MB | Phân loại mức ảnh (`Image-level folder`), 11 lớp mục tiêu | Phòng thí nghiệm (`256×256`), 1 lá tách rời trên nền xám/đen đồng nhất |
| 2 | `plantdoc_classification` | Singh et al. (2020), *ACM CoDS COMAD* ([GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)) | CC-BY-4.0 | 2.578 | 2.578 | 996,3 MB | Phân loại mức ảnh (`28` lớp thư mục, chia sẵn `train`/`test`) | Ảnh thực địa & Internet (`115×69` đến `6000×6000`), nhiều lá, nền đất/trời phức tạp |
| 3 | `plantdoc_detection` | Singh et al. (2020), *ACM CoDS COMAD* ([GitHub](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)) | CC-BY-4.0 | 2.603 | 2.603 | 1.005,8 MB | `8.921` hộp giới hạn (`Bounding Boxes`) trong CSV + Pascal VOC XML (`29` lớp) | Trùng `2.488` ảnh gốc với `plantdoc_classification`; cung cấp tọa độ từng lá đơn lẻ |
| 4 | `plantwild_v2` | Wei et al. (2024), *ECCV / MVP Lab* ([HuggingFace](https://huggingface.co/datasets/uqtwei2/PlantWild)) | CC-BY-NC-ND-4.0 | 11.488 | 11.486 | 1.657,9 MB | Phân loại mức ảnh (`115` lớp bệnh thực vật tự nhiên + `prompts.json`) | Ảnh thực địa đa dạng (`65×52` đến `6000×5408`), không có lớp lá khỏe (`Healthy`) |
| 5 | `plantseg` | Wei et al. (2024), *Zenodo Record 17719108* ([DOI: 10.5281/zenodo.17719108](https://doi.org/10.5281/zenodo.17719108)) | CC-BY-NC-4.0 | 7.774 | 7.774 | 1.026,6 MB | Mặt nạ phân vùng tổn thương (`8-bit PNG` + `COCO JSON` + `Metadata.csv`, `115` lớp) | `100%` ảnh gốc trùng pixel với `plantwild_v2`; mặt nạ chỉ bao vết bệnh (trung vị `13,86%` diện tích ảnh) |
| 6 | `potato_holeta_v2` | Holeta Agricultural Research Center, Ethiopia ([DOI: 10.17632/v4w72bsts5.2](https://doi.org/10.17632/v4w72bsts5.2)) | CC-BY-4.0 | 430 | 430 | 5,3 MB | Phân loại mức ảnh (`Local/Healthy`: 363, `Local/LateBlight`: 67), chuyên gia bệnh cây hỗ trợ gán nhãn | Ảnh lá khoai tây chụp ngoài đồng ruộng bằng điện thoại di động, đã chuẩn hóa `256×256` |
| 7 | `potato_field_indonesia` | Shabrina et al. (2024), *Data in Brief* 52:109955 ([DOI: 10.17632/ptz377bwb8.1](https://doi.org/10.17632/ptz377bwb8.1)) | CC-BY-4.0 | 1.096 (từ 3.076 file gốc) | 1.096 | 285,3 MB | Phân loại mức ảnh (`7` lớp: `Phytophthora` 347, `Healthy` 201, `Nematode` 68, và 120 ảnh/lớp cho `Fungi`, `Pest`, `Bacteria`, `Virus`) | Ảnh smartphone thực địa (`1500×1500`) tại nông trại khoai tây Wonosobo, Indonesia dưới nắng/mưa tự nhiên |
| 8 | `potato_late_blight_iitkgp` | IIT Kharagpur, *Cogent Food & Agriculture* (2026) ([DOI: 10.5281/zenodo.22059910](https://doi.org/10.5281/zenodo.22059910)) | CC-BY-4.0 | 1.561 | 1.561 | 146,4 MB | Kho ảnh thực địa khoai tây (`640×640`) + `14` bảng CSV thống kê thực nghiệm; **không công bố nhãn từng ảnh** | Ảnh luống khoai tây thực địa tại Ấn Độ chứa hỗn hợp ~22.216 lá (`38,2% Healthy`, `33,4% ILB`, `28,4% ALB`) |
| 9 | `maize_field_bd` | Multi-Format Maize Leaf Dataset, Bangladesh ([DOI: 10.17632/sb4wwjs8bz.1](https://doi.org/10.17632/sb4wwjs8bz.1)) | CC-BY-4.0 | 994 (+ 7 file CSV) | 994 | 423,6 MB | Kết hợp phân loại thư mục (`7` lớp bệnh ngô) + `7` file CSV hộp giới hạn (`x, y, width, height`) | Ảnh chụp điện thoại ngoài ruộng ngô tại Bangladesh (`~1080×1916`), giữ nguyên tỷ lệ khung hình |
| 10 | `tomato_field_pakistan` | Sukkur IBA University, Pakistan ([DOI: 10.17632/3mbnb82mxd.2](https://doi.org/10.17632/3mbnb82mxd.2)) | CC-BY-4.0 | 191 (lọc từ 7.200 file ZIP) | 191 | 474,3 MB | Phân loại mức ảnh (`6` lớp cà chua: `Early_Blight`, `Late_Blight`, `Septoria_Leaf_Spot`, `Healthy`, `Yellow_Leaf_Curl_Virus`, `Bacterial_Spot`) | Ảnh gốc chụp trực tiếp bằng camera điện thoại (`IMG_20251102_*`, `172345*`, độ phân giải lên tới `4080×3060`) |

---

## 2. Phân tích Chuyên sâu: Quy cách Gán dữ liệu (`Annotation`) & Phát hiện Pháp y Dữ liệu (`Forensic Audit`)

### 2.1. Phân loại mức ảnh (`Image-Level Folder Labels`) và Rủi ro Nhãn Mơ hồ / Sai lệch
1. **Nhãn mơ hồ về loài tác nhân gây bệnh**:
   - Trong `plantdoc_classification` và `maize_field_bd`, thư mục `Corn leaf blight` / `Corn_Leaf_Blight` không phân biệt rõ giữa cháy lá lớn do *Exserohilum turcicum* (`Corn_Northern_Leaf_Blight`) và cháy lá nhỏ do *Bipolaris maydis* (`Southern Corn Leaf Blight`). Hệ thống kiểm định tự động gắn cờ `ambiguous_corn_blight` và **loại khỏi tập huấn luyện trực tiếp** thay vì gán bừa vào `Corn_Northern_Leaf_Blight`.
   - Trong `potato_field_indonesia`, lớp `Phytophthora` (347 ảnh) tương ứng chính xác với tác nhân *Phytophthora infestans* (`Potato_Late_blight`), và `Healthy` (201 ảnh) tương ứng với `Potato_healthy`. Ngược lại, lớp `Fungi` (748 ảnh trên kho gốc) bao gồm nhiều bệnh nấm lá khoai tây khác nhau ngoài *Alternaria solani*, nên được gắn cờ `ambiguous_broad_fungal_label` và không gán tự động vào `Potato_Early_blight`.
2. **Phát hiện quan trọng trên `potato_late_blight_iitkgp` (Zenodo `22059910`)**:
   - Mặc dù tiêu đề bài báo là *"Stage-Wise Field-Scale Potato Late Blight Detection"*, kiểm tra pháp y tệp `DATA_PROVENANCE.md` và `13_dataset_statistics_manuscript_reported.csv` cho thấy `1.561` ảnh trong `archive/data/` là ảnh chụp tán lá ngoài ruộng chứa đồng thời cả lá khỏe (`38,2%`), mốc sương giai đoạn đầu (`ILB: 33,4%`) và giai đoạn nặng (`ALB: 28,4%`), và nhóm tác giả **không đính kèm tệp nhãn YOLO/COCO cho từng ảnh** trong bản lưu trữ Zenodo.
   - Nếu gán toàn bộ `1.561` ảnh này vào `Potato_Late_blight` sẽ gây nhiễm bẩn nhãn nghiêm trọng (`Label Contamination`). Do đó, bộ dữ liệu này được định danh rõ là `unlabeled_corpus_no_per_image_ground_truth` để phục vụ kiểm thử miền (`Domain Benchmark`) chứ không đưa vào nhánh giám sát trực tiếp.
3. **Phát hiện rò rỉ dữ liệu tăng cường (`Offline Augmentation Leakage`) và sao chép chéo trên `tomato_field_pakistan` (Mendeley `3mbnb82mxd.2`)**:
   - Kho lưu trữ gốc tuyên bố có `7.200` ảnh chia sẵn `train`/`val`/`test`. Tuy nhiên, phân tích bảng mục lục trung tâm ZIP (`sources/tomato_field_pakistan_zip_index.json`) phát hiện `6.370` ảnh là các bản nhân bản ngoại tuyến (`_aug_15aug53.jpg`) được tạo **trước khi chia tách** `train`/`val`/`test` (gây rò rỉ dữ liệu nghiêm trọng nếu dùng split gốc).
   - Hơn nữa, trong `830` ảnh thuộc thư mục `Dataset (raw)`, có tới `267` ảnh trùng tên file với `PlantDoc` và ảnh tải từ Internet. Ngay cả sau khi lọc riêng `191` ảnh có định danh camera (`IMG_20251102_*`, `172345*`, `IMG20240812*`), bộ kiểm định pixel `RGB SHA-256` tiếp tục phát hiện **10 ảnh trùng chính xác từng pixel với `PlantDoc`** và **1 ảnh trùng với `PlantWild_v2`**, đã được tự động loại bỏ hoàn toàn để bảo vệ tính độc lập của tập đánh giá.
4. **Phát hiện trùng lặp đa biểu diễn và ảnh chụp liên tiếp (`Burst-Shot Leakage`) trên `maize_field_bd` (Mendeley `sb4wwjs8bz.1`)**:
   - Kho lưu trữ gốc chứa `5.789` file vì tải lên 4 bản sao của cùng ~1.000 ảnh chụp (bản gốc full-res, bản Detection full-res, bản giữ tỷ lệ `~1080×1916`, và bản ép vuông `224×224` kèm ảnh nhân bản `aug_0_IMG`).
   - Ngay cả khi chỉ lấy duy nhất 1 phiên bản giữ tỷ lệ (`994` ảnh JPEG), thuật toán ghép nhóm `pHash <= 4 & dHash <= 4` phát hiện **34 cặp trùng pixel tuyệt đối** và **2.935 cặp gần giống nhau** do người thu thập chụp liên tiếp nhiều tấm của cùng một chiếc lá ngô đặt trên cùng một tấm ván gỗ! Nhờ gom nhóm theo đồ thị `HashIndex`, toàn bộ các ảnh chụp cùng một lá được gộp về duy nhất 1 ảnh đại diện (`154` nhóm độc lập sau khi lọc), ngăn chặn triệt để hiện tượng rò rỉ giữa `train` và `validation`.

### 2.2. Gán nhãn Hộp giới hạn (`Bounding-Box Annotations`)
- **`plantdoc_detection`**:
  - Gồm `8.921` hộp trong `train_labels.csv` và `test_labels.csv` (`xmin, ymin, xmax, ymax`).
  - Có `8.813` hộp hợp lệ về mặt hình học; `108` hộp bị lỗi (11 hộp trỏ tới file ảnh bị thiếu, 88 hộp nằm trên ảnh có cờ xoay `EXIF != 1`, 21 hộp khai báo sai `width/height` so với ảnh thực, 11 hộp tràn ra ngoài biên ảnh).
  - **Ứng dụng vào huấn luyện (`field_mobile_v2`)**: Từ các ảnh cha nằm trong `PlantDoc/train` (tuyệt đối không chạm vào `PlantDoc/validation` hay `PlantDoc/test`), hệ thống đã trích xuất **3.160 khung cắt lá đơn (`crops`)** đạt chuẩn hình học (`valid_geometry`, `exif_orientation == 1`, kích thước tối thiểu `>= 64px`, mở rộng lề ngữ cảnh `8%`), gán cùng mã nhóm (`group`) với ảnh cha để huấn luyện mô hình nhận diện lá đã cắt khung giống quy trình cắt ảnh trên giao diện web.
- **`maize_field_bd`**:
  - Gồm `7` file CSV theo định dạng `label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, image_width, image_height` với tổng cộng `742` hộp (`741` hộp hợp lệ hình học).
  - Phát hiện quan trọng: tệp `Common Rust.csv` trên Mendeley v1 chỉ có dòng tiêu đề (`0` hộp dữ liệu), và tọa độ hộp được gán trên ảnh gốc `3000×4000` (hoặc `3472×4624`) trong khi thư mục ảnh phân loại đã được thu nhỏ về `~1080×1916`.

### 2.3. Gán nhãn Mặt nạ Phân vùng (`Segmentation Masks`)
- **`plantseg`**:
  - `7.774` mặt nạ PNG kênh đơn (`0 = nền`, `Metadata.Index + 1 = vùng bệnh`), trong đó `7.745` mặt nạ hợp lệ hoàn toàn và `29` mặt nạ có lỗi hướng EXIF hoặc mã lớp.
  - Diện tích vùng mặt nạ có trung vị chỉ `13,86%` diện tích ảnh (phân vị thứ 25 là `5,94%`) vì chỉ tô màu vết bệnh chứ không bao toàn bộ phiến lá.

---

## 3. Danh mục Các Nguồn Dữ liệu Uy tín Khác trên Thế giới (Đã Khảo sát & Đối chiếu)

Ngoài 10 bộ dữ liệu đã tải và kiểm định trực tiếp ở trên, dưới đây là danh mục các bộ dữ liệu học thuật đáng tin cậy khác cùng kết quả thẩm định nguồn gốc:

1. **Cornell University — Northern Leaf Blight (NLB) Field Dataset (Wiesner-Hanks et al., 2018)**:
   - **Công bố**: *The Plant Phenome Journal* (Wiley) / OSF ([DOI: 10.17605/OSF.IO/4F89K](https://doi.org/10.17605/OSF.IO/4F89K)).
   - **Nội dung**: `18.222` ảnh thực địa ngô chụp bằng cầm tay, gắn trên cần cẩu và UAV tại New York, kèm `105.705` nhãn đường thẳng/đa giác dọc trục vết bệnh *Exserohilum turcicum* (`Corn_Northern_Leaf_Blight`).
2. **CD&S Maize Disease Dataset — Purdue / GitHub (`cd&s`)**:
   - **Nội dung**: Ảnh thực địa ngô cho 3 bệnh chính: `Northern Leaf Blight`, `Gray Leaf Spot`, `Northern Leaf Spot`.
3. **Potato Leaf Disease Dataset in Uncontrolled Environment (Cảnh báo trùng lặp)**:
   - Một số bản phát hành lại trên Kaggle/HuggingFace (`Spitblaze/Corn_or_Maize_Leaf_Disease_Dataset` 4.188 ảnh, `AlvaroVasquezAI/Maize_Leaf_Diseases`) đã được kiểm tra và xác nhận là **bản gộp đổi tên từ `PlantVillage` + `PlantDoc`**. Tuyệt đối không tải các bản đóng gói lại này làm dữ liệu thực địa mới để tránh rò rỉ dữ liệu vào tập kiểm thử.

---

## 4. Tập Dữ liệu Huấn luyện & Đánh giá Mở rộng `data/processed/field_mobile_v2`

Tập dữ liệu `field_mobile_v2` (`20.902` bản ghi, mã băm SHA-256: `4678faca1433bbb79d31185cdb6600994e29b777c9fbe8ed23bfb58fe92f5f24`) được xây dựng bởi [`ml/prepare_field_v2.py`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/ml/prepare_field_v2.py) với các cam kết bảo toàn nghiêm ngặt:
- **100% bất biến tập đánh giá cũ**: Giữ nguyên toàn bộ `PlantVillage/validation` (`1.731`), `PlantVillage/test` (`1.720`), `PlantDoc/validation` (`414`) và `PlantDoc/test` (`217`) từ `field_mobile_v1`.
- **0 rò rỉ dữ liệu (`Zero Leakage`)**: Loại bỏ mọi ảnh mới có trùng lặp byte SHA-256, trùng pixel RGB sau xoay EXIF, hoặc trùng cảm nhận thị giác (`pHash <= 4` và `dHash <= 4`) với bất kỳ ảnh `validation` hoặc `test` nào của `field_mobile_v1` (đã loại `50` ảnh trùng holdout, `164` ảnh trùng train cũ, `426` ảnh trùng nội bộ nguồn mới và `41` ảnh xung đột nhãn).
- **Bổ sung đầy đủ cả 11 lớp thực địa + `Unsupported`**:
  - `PlantDoc/train`: Tăng từ `1.658` lên `4.818` bản ghi nhờ bổ sung `3.160` khung cắt lá đơn (`plantdoc_detection_crop`) từ chính các ảnh cha trong `PlantDoc/train`.
  - `MultiSourceField/train` (`3.364` ảnh) và `MultiSourceField/validation` (`595` ảnh): Bao phủ toàn bộ 11 lớp mục tiêu ngoài đồng ruộng (bao gồm cả 3 lớp trước đây `PlantDoc` bị khuyết hoặc thiếu hụt là `Corn_Northern_Leaf_Blight`, `Corn_healthy` và `Potato_healthy`) cùng `Unsupported` thực địa.
