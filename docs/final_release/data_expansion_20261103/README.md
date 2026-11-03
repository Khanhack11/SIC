# Bổ sung dữ liệu và train SIC ngày 03/11/2026

Thí nghiệm `expanded_20261103` tiếp tục MobileNetV3-Large từ `mobilenetv3_field_continued_20261103.pt`. Model phân loại ảnh lá vào 11 lớp SIC và một lớp Unsupported. Thí nghiệm này chưa huấn luyện detector hoặc segmentation để tự tìm/cắt lá.

[Kết quả đã hoàn tất](RESULTS.md): chọn epoch 3 sau bốn epoch train. F1 validation bổ sung tăng từ 50,21% lên 65,17%, nhưng F1 PlantDoc giảm từ 46,59% xuống 45,83% và F1 external Holeta giảm còn 29,94%. Không thay model web vì chưa đạt tiêu chí phát hành.

Danh mục 12 nguồn mới, nguồn phát hành và cách gán nhãn nằm trong [báo cáo khảo sát](../../research/DATASET_SOURCES_20261103.md). Các bộ có trên máy không mặc nhiên được dùng train.

## Dữ liệu đã khóa

| Thành phần | Train | Validation | Test lịch sử | External test |
|---|---:|---:|---:|---:|
| PlantVillage cũ | 8.043 | 1.731 | 1.720 | 0 |
| PlantDoc cũ | 1.658 | 414 | 217 | 0 |
| PlantWild_v2 bổ sung | 413 | 103 | 0 | 0 |
| Ngô Healthy Ecuador | 224 | 0 | 0 | 0 |
| Khoai tây Holeta | 0 | 0 | 0 | 418 |
| Tổng | 10.338 | 2.248 | 1.937 | 418 |

Đã xem bố cục 871 ảnh ứng viên qua 14 contact sheet. Loại 123 ảnh ghép khung, sơ đồ, chữ chú thích lớn, chủ thể quả/thân/củ hoặc không đủ rõ để giữ. Đây là bước xem ảnh bằng AI; nhãn bệnh còn lại vẫn là nhãn nguồn, chưa có chuyên gia của nhóm xác nhận lại.

Ecuador tải 232 ảnh Healthy và kiểm từng SHA-256 do nhà phát hành cung cấp. Loại tám ảnh trùng/gần giống, giữ 224 ảnh chỉ cho train do thiếu ID lá/phiên chụp. Không lấy một tập random từ nguồn này để tuyên bố accuracy ngoài đồng.

Holeta công bố 426 ảnh nhưng RAR thực tế có 430 JPG: 363 Healthy, 67 Late Blight. Sau loại 12 ảnh trùng/gần giống còn 351 Healthy và 67 Late Blight. Cả nguồn được giữ làm external test cho cặp parent/expanded; model `field_mobile_v2` khác đã học Holeta nên không thuộc phép so sánh độc lập này. Nguồn chỉ có hai lớp khoai tây và mất cân bằng, cần đọc macro F1 và recall từng lớp cùng accuracy.

Hash SHA-256 của `records.json`:

```text
b6eb04f51252f2107bd5b726d8e02d691f8ccffe2bcbc9a1ce34e6cbcb0b798a
```

Holdout cũ giữ nguyên path, hash, nhãn và split. Kiểm trùng dùng SHA-256, hash pixel theo hướng ảnh và cặp pHash/dHash. Không phát hiện trùng bằng các bộ lọc này chưa chứng minh mọi lá vật lý hoặc phiên chụp đã tách biệt. Các benchmark cũ đã được dùng ở những vòng trước.

## Cách train và đánh giá

Ảnh được đọc từ manifest và kiểm hash, xử lý hướng EXIF, chuyển RGB, letterbox 224 × 224 rồi chuẩn hóa ImageNet. Train có augmentation theo transform dùng chung. Pipeline phục vụ dùng cùng hợp đồng tiền xử lý; thao tác crop thủ công của người dùng là bước riêng trước khi gửi ảnh vào model.

Lượt train chạy CUDA trên NVIDIA RTX 2050, bốn epoch, batch 32, seed 42. Khởi tạo từ trọng số parent, mở ba block cuối của backbone và classifier. AdamW dùng learning rate backbone 0,00001 và head 0,0001, label smoothing 0,05, mixed precision và giới hạn gradient norm 5. Sampler cân bằng domain/lớp và lấy mẫu có hoàn lại; một epoch có 10.338 lượt lấy mẫu, không bảo đảm mỗi ảnh xuất hiện đúng một lần.

Checkpoint được chọn bằng trung bình validation cross-entropy của từng domain, kể cả FieldSupplement. Parent cũng là một ứng viên giữ lại nếu các epoch mới không giảm loss. Sau đó fit temperature bằng validation. Không chọn checkpoint theo external test. Confidence sau calibration vẫn là điểm xác suất của model, không phải tỷ lệ chẩn đoán đúng của riêng ảnh đó.

Đánh giá CPU so sánh baseline V2, parent V3 và expanded trên cùng manifest validation. Tiêu chí phát hành kiểm gain F1 PlantDoc, mức giảm F1 PlantVillage, accuracy/coverage ở ngưỡng 75% và tỷ lệ ảnh bệnh bị trả khỏe với confidence cao. Trọng số phục vụ không tự thay đổi. Khóa cấu hình và hash checkpoint trước khi chạy một lượt external test cho mỗi model parent/expanded; không điều chỉnh tiếp từ kết quả external.

## Lệnh tái hiện

Chạy từ thư mục SIC. Đầu ra có sẵn được giữ lại; chọn tên mới khi thực hiện một thí nghiệm khác.

```powershell
.venv/Scripts/python.exe -m ml.download_corn_ecuador
.venv/Scripts/python.exe -m ml.prepare_expanded_field --triage artifacts/data_expansion_20261103/triage/excluded.json --inventory artifacts/data_expansion_20261103/input_inventory.csv
.venv-gpu/Scripts/python.exe -u -m ml.continue_field --source models/mobilenetv3_field_continued_20261103.pt --data data/processed/field_expanded_v2_20261103 --output models/mobilenetv3_expanded_20261103.pt --epochs 4 --batch-size 32 --expanded-data
.venv/Scripts/python.exe -m ml.evaluate_field --data data/processed/field_expanded_v2_20261103 --weights models/mobilenetv3_expanded_20261103.pt --split validation --output artifacts/new_candidate_validation.json
```

Chuẩn bị manifest cần hub dữ liệu gốc và parent manifest. `input_inventory.csv` là bản inventory năm nguồn đã đóng băng trước khi các audit khác được cập nhật. RAR Holeta có URL tải và SHA-256 trong [source.json](sources/potato_holeta_v2/source.json); giải nén giữ thư mục `original/Local/Healthy` và `original/Local/Late Blight`. Metadata và danh sách file Ecuador nằm trong `sources/`. Không trộn manifest của các thí nghiệm.

## Hồ sơ

- [Audit split và ảnh bị loại](data_audit.json), [manifest 1.158 ảnh bổ sung/giữ ngoài](supplement_manifest.csv).
- [Phạm vi duyệt ảnh](layout_triage_review.json), [123 ảnh loại theo bố cục](layout_triage_excluded.json).
- File trọng số và trạng thái phục hồi lưu local trong `models/`; ảnh raw lưu trong `data/raw/dataset_hub_20261103/`, không đưa vào Git.

## Giới hạn cần nêu khi báo cáo

Ảnh lá thủng hoặc bị sâu ăn không có lớp nguyên nhân riêng trong 11 nhãn SIC. Thêm ảnh bệnh công khai chưa đủ để bảo đảm model nhận diện đúng nhóm này; IP102 có nhãn côn trùng, không cung cấp trực tiếp nhãn nguyên nhân của vết thủng trên lá. Kết quả healthy cần đối chiếu triệu chứng thực tế và quy tắc review của ứng dụng.

External Holeta chỉ đo hai lớp khoai tây trong một nguồn công khai. Không dùng kết quả đó để tuyên bố độ chính xác cho toàn bộ 11 lớp, mọi điện thoại hay dữ liệu thực địa tại Việt Nam. Bước crop tự động vẫn cần bộ annotation và mô hình riêng được đánh giá theo hộp/mask.
