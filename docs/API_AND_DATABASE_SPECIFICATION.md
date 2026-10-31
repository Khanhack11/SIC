# API và kho tri thức SIC

FastAPI phục vụ giao diện `/app`, script `/frontend.js` và ảnh `/demo`. `/` giữ metadata JSON để tương thích client cũ. Hai subsystem nhận diện và tư vấn nối bằng class ID trong `backend/config.py`.

| Method | Path | Nội dung |
|---|---|---|
| GET | `/api/health` | Readiness, runtime, device, calibration và ngưỡng |
| POST | `/api/diagnose` | Upload ảnh và trả kết quả tổng hợp |
| GET | `/api/crops` | Ba cây và các điều kiện được hỗ trợ |
| GET | `/api/diseases` | Tóm tắt 11 lớp |
| GET | `/api/diseases/{disease_id}` | Record đã áp dụng chính sách kiểm định thuốc |
| GET | `/docs` | OpenAPI và thử request |

## Request và xử lý

`POST /api/diagnose` nhận multipart field `file`, JPEG/PNG tối đa 10 MB. Ảnh cần mỗi cạnh ít nhất 32 px, không quá 20 megapixel. Backend xác minh nội dung bằng Pillow, xử lý EXIF, chuyển RGB, resize 224×224 và normalize ImageNet.

`custom_threshold` tùy chọn trong 0–1, mặc định 0,75 và riêng cho từng request. Giao diện giữ ngưỡng 0,75. CPU work chạy trong threadpool với hai slot mặc định.

| HTTP | Trường hợp |
|---:|---|
| 200 | `CONFIRMED` hoặc `UNCERTAIN` |
| 400 | Ảnh rỗng, hỏng, sai định dạng/kích thước |
| 404 | Không có disease ID |
| 422 | Thiếu upload hoặc threshold ngoài khoảng |
| 503 | Checkpoint/runtime chưa hợp lệ |
| 500 | Lỗi nội bộ, thông báo chung; chi tiết chỉ ghi log |

`CONFIRMED` là tên trạng thái tương thích cũ, nghĩa là vượt ngưỡng hiển thị, không xác nhận bệnh ngoài thực địa. Payload có `prediction`, `advisory_details`, `message`, `image_metadata`, `remediation_notice`, `regulatory_disclaimer`.

`prediction.latency_ms` đo tiền xử lý và suy luận. Header `Server-Timing` có `inference` và `processing`. Thời gian browser đo còn có upload, mạng và hàng đợi.

Khi `UNCERTAIN`, `advisory_details` chỉ có nhận định sơ bộ và hướng dẫn chụp lại. Ngưỡng không phải bộ phát hiện ảnh ngoài miền.

## Tri thức và kiểm định

`data/advisory_knowledge_base.json` phiên bản 1.1.0 giữ 11 record. Có tên cây/bệnh, tác nhân, triệu chứng, điều kiện phát triển, biện pháp canh tác và lưu ý. `sources` chứa title, publisher, URL, scope. `review` ghi phương pháp, ngày và phạm vi đối chiếu.

Nguồn đã đối chiếu:

- [Early blight in tomato and potato, UMN](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato)
- [Late blight, UMN](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight)
- [Common rust on corn, UMN](https://extension.umn.edu/agriculture/crop-production/corn/common-rust-on-corn)
- [Northern corn leaf blight, UMN](https://extension.umn.edu/agriculture/crop-production/corn/northern-corn-leaf-blight)
- [Gray Leaf Spot, Cornell](https://cals.cornell.edu/field-crops/corn/diseases-of-corn/gray-leaf-spot)
- [Septoria Leaf Spot of Tomatoes, UMD](https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes)

Chưa có xác nhận chuyên gia hoặc kiểm định nhãn thuốc tại Việt Nam. `review.chemical_status` là `pending_local_review`, `expert_approved` là false. Endpoint chi tiết và chẩn đoán dùng cùng bộ lọc: trả bản sao record, thay `chemical_controls` và `organic_biological_controls` bằng mảng rỗng, kèm `treatment_notice`. Dữ liệu gốc được giữ để đối chiếu, không phải hướng dẫn đã duyệt.

Chỉ trạng thái `verified_local_label` mới cho phép mở hướng dẫn thuốc. Không tự đổi trạng thái khi chỉ có nguồn nước ngoài. Cần người kiểm tra, sản phẩm và nồng độ, cây/bệnh áp dụng, liều, PHI và căn cứ nhãn địa phương.

## Benchmark

[Hồ sơ HTTP mới](final_release/api_benchmark.json): ONNX FP32, 2 threads, 2 slots, 80 request/mức tải và 10 warmup. Mean/p95 là 11,07/13,63 ms ở một client, 28,91/32,49 ms ở bốn client, 55,51/62,70 ms ở tám client. Không dùng các số này thay cho phép đo Internet hoặc workload khác.
