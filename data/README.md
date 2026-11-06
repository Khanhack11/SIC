# Dữ liệu và nhãn

Ảnh nguồn ở `raw/`. Nhãn chuẩn hóa và split nằm trong manifest `processed/`; trainer đọc nhãn mục tiêu từ manifest, không tự dùng dự đoán của model làm nhãn thật.

| Vị trí | Vai trò |
|---|---|
| `advisory_knowledge_base.json` | Thông tin chăm sóc tham khảo |
| `raw/dataset_hub_20261103/` | Nguồn tải thêm, metadata và audit |
| `processed/plantvillage/{train,validation,test}.csv` | Nhãn V2: path, label, class_index, SHA, nhóm |
| `processed/field_mobile_v2/records.json` | V3 đa nguồn: 16.225 train, 2.740 validation, 1.937 test lịch sử |
| `processed/field_expanded_v2_20261103/` | Thí nghiệm riêng giữ Holeta làm external; không trộn với v2 đã học Holeta |
| `processed/field_mobile_v1/` | Manifest lịch sử của parent |

Xuất bảng thật bằng `python -m ml.inspect_labels --output artifacts/label_review/current.csv`. [Ánh xạ nhãn và đưa vào loss](../docs/DATA_LABELING.md).

Dữ liệu bổ sung đã được dùng trong các lượt train đa nguồn. Nhãn vẫn chủ yếu là nhãn nhà phát hành; chưa có chuyên gia của nhóm duyệt lại toàn bộ. Raw/processed không nằm trong Git. Manifest hiện có chứa đường dẫn Windows tuyệt đối và SHA đã khóa; không sửa trực tiếp nhãn/split. Ảnh demo thuộc validation, dùng minh họa thao tác.

## Lượt bổ sung nguồn khác

`processed/field_alternative_v3_20261103` có 15.458 train, 2.740 validation và 1.937 test. 2.829 ảnh train bổ sung từ CCMT Ghana, BARI và Ecuador; validation/test giữ nguyên từ parent. Nhãn nguồn không tự trở thành nhãn đã được chuyên gia của nhóm duyệt. [Nguồn, cách lọc và kết quả lượt train](../docs/final_release/alternative_sources_20261103/README.md).
