# Pipeline học máy

Chạy lệnh từ thư mục SIC với dạng `python -m ml.<tên_module>`.

| Giai đoạn | Module |
|---|---|
| Tải dữ liệu mới và audit | download_recommended_datasets, analyze_recommended_datasets |
| Bổ sung dữ liệu có giữ holdout parent | download_corn_ecuador, prepare_expanded_field |
| Tái hiện dữ liệu thí nghiệm cũ | download_dataset, download_plantdoc, prepare_dataset, prepare_field |
| Train V2 và CNN đối chứng | train, train_gpu |
| Thử tối ưu V2 | optimize_head, finetune_projection |
| Train V3, tiếp tục và khôi phục | train_field, continue_field, recover_field_candidate |
| Đánh giá, kiểm tra phát hành | evaluate, evaluate_field, field_release |
| Export và đối chiếu runtime | export_onnx, export_field_onnx, runtime_validation, validate_field_runtime |
| Đo hiệu năng | benchmark, api_benchmark |
| Ảnh mẫu và bàn giao | preview, prepare_demo, handoff, package_demo |
| Hàm dùng chung | dataset, labels, runtime |

`train_field` hiện đọc `data/processed/field_mobile_v1/records.json`, có bước học classifier rồi fine-tune các block cuối của MobileNetV3-Large. `continue_field` lưu optimizer, scaler, thứ tự batch và RNG để tiếp tục đúng trạng thái.

`continue_field` có thể chọn số block mở, learning rate riêng cho bốn block cuối, số lượt lấy mẫu mỗi epoch, ngưỡng F1 PlantVillage để giữ checkpoint và calibration cân bằng domain. Các tham số được lưu trong cấu hình recovery; đổi cấu hình hoặc mã sẽ từ chối khôi phục checkpoint cũ. Xem [lượt phối hợp Antigravity](../docs/final_release/antigravity_continuation_20261103/README.md).

Danh mục mới nằm ở `data/raw/dataset_hub_20261103`. Cần duyệt nhãn và tạo manifest mới trước khi đưa vào trainer. Metrics và validation loss hỗ trợ các domain có trong manifest; tiêu chí phát hành vẫn kiểm PlantVillage và PlantDoc theo cấu hình đã khóa. Giữ riêng test, không chọn checkpoint hoặc điều chỉnh ngưỡng bằng test.

Thí nghiệm bổ sung ngày 03/11 dùng `prepare_expanded_field` để giữ holdout cũ, thêm PlantWild_v2 đã xem bố cục và ngô khỏe Ecuador, dành Holeta làm `external_test`. `continue_field --expanded-data` kiểm hash manifest parent, giữ nguyên holdout và chặn nhóm ảnh train trùng holdout. Xem [nguồn dữ liệu và cách gán nhãn](../docs/research/DATASET_SOURCES_20261103.md). Các ảnh nhãn nguồn chưa được chuyên gia của nhóm xác nhận lại.

Môi trường `.venv-gpu` dùng để train bằng CUDA; `.venv` dùng phục vụ API và chạy kiểm thử. Chạy `python -m ml.package_demo` tạo gói trong `artifacts/packages/`, có manifest kiểm tra SHA-256.
