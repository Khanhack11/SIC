# Pipeline học máy

Chạy lệnh từ thư mục SIC với dạng `python -m ml.<tên_module>`.

| Giai đoạn | Module |
|---|---|
| Tải dữ liệu mới và audit | download_recommended_datasets, analyze_recommended_datasets |
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

Danh mục mới nằm ở `data/raw/dataset_hub_20261103`. Cần duyệt nhãn và tạo manifest mới trước khi đưa vào trainer. Các hàm chọn checkpoint/validation hiện còn gắn với domain PlantVillage và PlantDoc; cần cập nhật khi thêm domain mới. Giữ riêng test, không chọn checkpoint hoặc điều chỉnh ngưỡng bằng test.

Môi trường `.venv-gpu` dùng để train bằng CUDA; `.venv` dùng phục vụ API và chạy kiểm thử. Chạy `python -m ml.package_demo` tạo gói trong `artifacts/packages/`, có manifest kiểm tra SHA-256.
