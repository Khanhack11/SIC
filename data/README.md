# Dữ liệu của hệ thống

| Vị trí | Nội dung | Cách sử dụng |
|---|---|---|
| advisory_knowledge_base.json | Thông tin bệnh, chăm sóc và nguồn tham khảo | Backend đọc khi khởi động |
| raw/dataset_hub_20261103/README.md | Danh mục chung 5 nguồn đã kiểm tra | Bắt đầu từ báo cáo và analysis/review |
| raw/plantvillage | Snapshot tập con 11 lớp của thí nghiệm cũ | Giữ để tái hiện manifest hiện có |
| raw/plantdoc_snapshot | Snapshot PlantDoc cũ có 6 xung đột tên trên Windows | Đợt train mới dùng bản có extraction_map trong danh mục chung |
| raw/plantdoc | Dữ liệu từ giai đoạn chuẩn bị trước | Giữ làm lịch sử; không tự gộp vào train |
| processed/plantvillage | Manifest và audit PlantVillage | Split đã khóa, không sửa trực tiếp |
| processed/field_mobile_v1 | Manifest ghép dữ liệu của thí nghiệm V3 hiện có | Ghi đường dẫn, hash, nhãn, domain và split |

Raw/processed không nằm trong Git. Các đường dẫn ảnh trong manifest hiện có phụ thuộc vị trí project trên máy này, vì vậy giữ nguyên thư mục dữ liệu sau khi sắp xếp code. Khi chuyển máy, cần chuẩn bị manifest phù hợp rồi kiểm tra lại hash.

639 ảnh ứng viên mới vẫn cần duyệt nhãn; chưa được đưa vào trainer. PlantSeg cung cấp mask tổn thương trên ảnh trùng PlantWild_v2. Xem [báo cáo dữ liệu](../docs/datasets/20261103/README.md).
