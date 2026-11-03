# Phân tích dữ liệu SIC ngày 03/11/2026

Đã tập hợp 5 nguồn vào `data/raw/dataset_hub_20261103`. Các gói tải mới có tổng dung lượng 4.68 GB. PlantVillage được sao chép từ bản đã có, sau khi đối chiếu Git blob. Tổng dung lượng thư mục gồm ZIP, bản giải nén và kết quả phân tích khoảng 9,75 GB.

## Phạm vi và kiểm tra nguồn

| Nguồn | Số ảnh thô | Đọc ảnh đạt kiểm tra | Mục đích |
|---|---:|---:|---|
| PlantVillage, tập con 11 lớp | 11,504 | 11,504 | Phân loại; 11.504 ảnh thô, có giới hạn 1.500 ảnh/lớp |
| PlantDoc Classification | 2,578 | 2,578 | Phân loại theo nhãn thư mục gốc |
| PlantWild_v2 | 11,488 | 11,486 | Phân loại ảnh có nền thực tế |
| PlantDoc Detection | 2,603 | 2,603 | Tìm vùng lá bằng hộp; không gán một nhãn bệnh cho toàn ảnh nhiều đối tượng |
| PlantSeg | 7,774 | 7,774 | Phân vùng tổn thương, không phải mặt nạ toàn bộ lá |

Số ảnh trên đếm riêng từng nguồn, chưa phải số ảnh độc lập sau loại trùng. PlantVillage là tập con 11 lớp của project, không phải toàn bộ bộ dữ liệu công bố. Hai ảnh PlantWild_v2 trên 30 megapixel được giữ nguyên bản và tách khỏi bước giải mã để giới hạn RAM; không kết luận chúng là ảnh hỏng. Ngưỡng này phục vụ audit, không thay giới hạn upload 20 megapixel của web.

Các phiên bản nguồn:

- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, 11.504 Git blob ảnh đã đối chiếu.
- [PlantDoc Classification](https://github.com/pratikkayal/PlantDoc-Dataset), revision `5467f6012d78d1c446145d5f582da6096f852ae8`, toàn bộ 2.581 file trong ZIP khớp Git tree.
- [PlantDoc Detection](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset), revision `4730a233a555b30ee98e0879c63ad25d82407455`, toàn bộ 5.200 file khớp Git tree.
- [PlantWild_v2](https://tqwei05.github.io/PlantWild/), tải từ tài khoản tác giả [uqtwei2/PlantWild](https://huggingface.co/datasets/uqtwei2/PlantWild), revision `527a72eb8f00c95e41698bb09d982c7d4625dce3`. ZIP khớp SHA-256 LFS.
- [PlantSeg](https://github.com/tqwei05/PlantSeg), [Zenodo 17719108](https://doi.org/10.5281/zenodo.17719108). ZIP khớp MD5 nguồn công bố, có lưu thêm SHA-256 và đã kiểm tra CRC khi giải nén.

PlantWild công bố CC-BY-NC-ND-4.0, PlantSeg công bố CC-BY-NC-4.0, hai repo PlantDoc công bố CC-BY-4.0. Giữ dữ liệu gốc và thông tin nguồn trong thư mục local; Git chỉ lưu code, số liệu audit và tài liệu. License theo từng ảnh của PlantSeg nằm trong Metadata.csv.

## Dữ liệu phù hợp với 11 lớp SIC

PlantWild_v2 có 115 lớp, nhưng chỉ 7 nhãn bệnh khớp trực tiếp theo tên với hệ thống. Tổng số ảnh ở các lớp này là 960, trước kiểm tra chất lượng và loại trùng.

| Nhãn project | Ảnh thô PlantWild_v2 |
|---|---:|
| Corn_Gray_leaf_spot | 107 |
| Corn_Northern_Leaf_Blight | 130 |
| Potato_Early_blight | 126 |
| Potato_Late_blight | 117 |
| Tomato_Early_blight | 187 |
| Tomato_Late_blight | 163 |
| Tomato_Septoria_leaf_spot | 130 |

177 ảnh có nhãn `corn rust` được giữ ở trạng thái cần kiểm tra loài bệnh, chưa tự gán chắc chắn vào `Corn_Common_rust`. Nhãn `Corn leaf blight` của PlantDoc vẫn mơ hồ; không tự đổi thành Northern Leaf Blight. Nhãn chung như `Tomato leaf` cần kiểm tra trước khi xem là lá khỏe.

PlantWild_v2 không có lớp lá khỏe. Vẫn cần ảnh khỏe thực tế của ngô, khoai tây và cà chua. Những ảnh lá thủng hoặc tổn thương ngoài 11 lớp cần được gán nhãn sau khi đối chiếu chuyên môn, hoặc giữ làm trường hợp chưa xác định. Tên bệnh khớp không bảo đảm từng ảnh đúng nhãn hay chỉ có lá; cần loại ảnh quả, thân và ảnh nhiều bệnh không phù hợp bài toán phân loại một lá.

## Ảnh trùng và bảo vệ tập đánh giá

Đã tìm thấy 10,132 nhóm trùng file, 10,113 nhóm trùng pixel RGB sau xoay EXIF và 9,651 nhóm khi tính cả cặp gần giống. Có 47 nhóm chứa nhãn đích xung đột, cần xem lại.

| Hai nguồn và cách phát hiện | Số cạnh ghép nhóm |
|---|---:|
| plantdoc_classification / plantdoc_detection: exact_oriented_rgb_pixels | 2,488 |
| plantdoc_classification / plantdoc_detection: perceptual_review_candidate | 112 |
| plantdoc_classification / plantseg: perceptual_review_candidate | 120 |
| plantdoc_classification / plantwild_v2: exact_oriented_rgb_pixels | 90 |
| plantdoc_classification / plantwild_v2: perceptual_review_candidate | 160 |
| plantdoc_detection / plantseg: perceptual_review_candidate | 121 |
| plantdoc_detection / plantwild_v2: exact_oriented_rgb_pixels | 90 |
| plantdoc_detection / plantwild_v2: perceptual_review_candidate | 162 |
| plantseg / plantwild_v2: exact_oriented_rgb_pixels | 7,774 |
| plantseg / plantwild_v2: perceptual_review_candidate | 1,026 |
| plantvillage / plantwild_v2: perceptual_review_candidate | 3 |

Các số trên là cạnh trong đồ thị ghép nhóm, không phải số ảnh mới hoặc toàn bộ cặp có thể có. Trùng pixel dùng một ảnh đại diện nối với các ảnh cùng nhóm. Tìm gần giống dùng pHash và dHash sau chuẩn hóa hướng; có thể bỏ sót ảnh crop khác hoặc góc chụp khác của cùng lá. Các cặp gần giống cần xem trực tiếp.

Toàn bộ 7.774 ảnh PlantSeg có pixel trùng ảnh trong PlantWild_v2. Giá trị bổ sung của PlantSeg là nhãn vùng tổn thương. Không cộng số ảnh này thành mẫu phân loại mới, không dùng hai bộ như hai nguồn train/test độc lập. Các ảnh trùng hoặc gần giống với validation/test hiện có, hoặc với split đánh giá chính thức của các nguồn mới, được đánh dấu bảo vệ trong danh sách ứng viên. Manifest train/validation/test cũ vẫn khớp checksum trong audit ban đầu.

Danh sách ứng viên PlantWild_v2 hiện có các trạng thái:

| Trạng thái | Số ảnh |
|---|---:|
| `needs_visual_label_review` | 639 |
| `exclude_overlap_with_protected_holdout` | 227 |
| `exclude_overlap_with_existing_train` | 40 |
| `review_conflicting_labels` | 51 |
| `exclude_duplicate_within_new_source` | 2 |
| `review_rust_species` | 177 |
| `review_image_quality` | 1 |

`needs_visual_label_review` nghĩa là ứng viên qua bộ lọc tự động, vẫn cần kiểm tra ảnh và nhãn trước khi train. Chưa chia split mới, chưa tự thêm các ảnh này vào trainer.

639 ứng viên này được phân bố như sau:

| Lớp cần duyệt | Ảnh |
|---|---:|
| Corn_Gray_leaf_spot | 70 |
| Corn_Northern_Leaf_Blight | 99 |
| Potato_Early_blight | 91 |
| Potato_Late_blight | 80 |
| Tomato_Early_blight | 107 |
| Tomato_Late_blight | 118 |
| Tomato_Septoria_leaf_spot | 74 |

## Hộp crop và mặt nạ tổn thương

PlantDoc Detection có 8,921 dòng hộp trong CSV, thuộc 29 nhãn. 8.813 dòng qua kiểm tra hình học; 108 dòng cần xử lý. Trong nhóm cần xử lý có 11 dòng tham chiếu ảnh thiếu, 88 dòng cần đổi tọa độ theo hướng EXIF, 21 dòng khai báo sai kích thước và 11 dòng hộp vượt biên hoặc không hợp lệ. Một dòng có thể chứa nhiều vấn đề. Có 25 ảnh không có dòng hộp CSV tương ứng. Xem `detection_boxes.csv` và `review/detection_flagged_108.csv`. Hộp có tọa độ hợp lệ vẫn cần xem có bao đúng lá hay không.

PlantSeg có 7.774 cặp ảnh và mặt nạ, trong đó 7.745 cặp qua kiểm tra. Có 17 cặp cần xử lý hướng EXIF (8 cặp trong số này còn khác kích thước ảnh và mask), 12 mask có mã lớp khác mã dự kiến. Các trường hợp này nằm trong `review/segmentation_flagged_29.csv`, cần duyệt trước khi xác định cách sửa. Quy ước PNG dùng `0` cho nền và `Metadata.Index + 1` cho vùng bệnh; COCO dùng `Metadata.Index` làm `category_id`. Ba file COCO đều để trống `categories`. Đã xuất `plantseg_categories.json` để ánh xạ lại khi chuẩn bị train. Giữ nguyên JSON gốc.

Mặt nạ PlantSeg đánh dấu vùng tổn thương. Muốn crop toàn bộ lá, cần detector vùng lá hoặc bộ mặt nạ đường viền lá phù hợp. Dùng khung bao tổn thương làm khung toàn bộ lá có thể bỏ mất ngữ cảnh.

## Sửa việc lưu tên file trên Windows

Bản PlantDoc cũ trong `data/raw/plantdoc_snapshot` có 6 Git blob không khớp do các cặp tên chỉ khác chữ hoa/thường. Các file bị ảnh hưởng trong lần kiểm tra:

- `train/Apple rust leaf/car1.jpg`
- `train/Blueberry leaf/Blueberry-Leaf.jpg`
- `train/Blueberry leaf/Blueberry-leaves.jpg`
- `train/Corn leaf blight/Northern-Corn-Leaf-Blight.jpg`
- `train/Peach leaf/peach-leaf.jpg`
- `train/Potato leaf early blight/potato-blight-phytophora-infestans-close-up-of-infected-leaf-showing-A60HXN.jpg`

Đã tải lại repo vào thư mục mới, giữ riêng từng nội dung bằng `extraction_map.json`. Bản Classification có 623 đường dẫn được chuẩn hóa, bản Detection có 1.034 đường dẫn, gồm tên Windows không hợp lệ, đường dẫn dài và trùng hoa/thường. Không sửa dữ liệu gốc, checkpoint hoặc kết quả thí nghiệm cũ. Đợt train tiếp theo phải đọc đường dẫn từ danh sách mới, không tiếp tục sử dụng đường dẫn snapshot cũ cho dữ liệu mới.

## Các phase tiếp theo

1. Đã hoàn tất tải nguồn, kiểm tra tính toàn vẹn, thống kê nhãn và đánh dấu ảnh trùng.
2. Duyệt ứng viên phân loại và các nhóm nhãn xung đột. Bổ sung ảnh khỏe thực tế, ảnh ngoài phạm vi và một bộ kiểm thử upload độc lập. Người duyệt nhãn cần hiểu bệnh cây.
3. Khóa nhãn và nhóm ảnh trước khi tạo split mới. Giữ mỗi nhóm cùng một split; bảo vệ test cũ. Pipeline chuẩn bị hiện có chưa đọc `extraction_map.json` của danh mục mới, cần nối bằng manifest đã duyệt.
4. Train lại MobileNetV3-Large bằng GPU, giữ preprocessing letterbox 224 đồng nhất với API. Chọn checkpoint và ngưỡng bằng validation, đo từng lớp rồi mới đánh giá bộ test đã khóa. Chưa chạy train mới trong bước tải và audit này.
5. Dùng hộp lá đã duyệt để phát triển tự tìm vùng lá. Đánh giá crop riêng trước khi ghép với phân loại bệnh; chỉ thêm phân vùng tổn thương khi cần hiển thị vị trí bệnh.

## Chạy lại và hồ sơ

Chạy từ thư mục SIC:

```powershell
.\.venv\Scripts\python.exe -m ml.download_recommended_datasets
.\.venv\Scripts\python.exe -m ml.analyze_recommended_datasets
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_dataset_a*.py' -v
```

Các bài kiểm tra mới: 10/10 pass với fixture nhân tạo, không gọi mạng và không đo độ chính xác model. Đã chạy audit trên toàn bộ 35.947 bản ghi ảnh thuộc 5 nguồn; đây không phải số ảnh độc lập sau loại trùng. Không train mới và không thay model phục vụ.

Thư mục local `analysis` có `image_inventory.csv`, `classification_candidates.csv`, `detection_boxes.csv`, `segmentation_masks.csv`, `duplicate_pairs.csv`, `duplicate_groups.csv`, ảnh xem mẫu và JSON chi tiết. Danh sách ứng viên dùng đường dẫn tuyệt đối trên máy hiện tại; cần tạo lại nếu chuyển thư mục sang máy khác.

`analysis/review` tách sẵn 639 ứng viên phân loại, 51 ảnh có nhóm nhãn xung đột, 177 ảnh corn rust, 108 dòng hộp và 29 cặp ảnh/mask cần xem lại. Duyệt đầy đủ nội dung ảnh và đối chiếu nhãn chuyên môn trước khi chuyển thành tập train. Audit hiện tại kiểm tra tính toàn vẹn, nhãn danh mục và dữ liệu trùng; chưa kiểm định chẩn đoán cho toàn bộ ảnh.
