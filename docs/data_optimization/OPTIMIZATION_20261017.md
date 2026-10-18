# Tối ưu triển khai ngày 17 tháng 10 năm 2026

Chọn ONNX Runtime FP32 với 2 CPU threads và tối đa 2 inference đồng thời trên máy hiện tại. Lựa chọn dựa trên validation và phép đo HTTP localhost; INT8 bị loại vì chất lượng giảm. Kết quả xác nhận ở `results/optimization_20261017/api_benchmark_confirmed.json`.

## Chất lượng

Trên đủ 1.731 ảnh validation, eager, TorchScript và ONNX FP32 cùng macro F1 94,4801%, nhãn dự đoán trùng nhau 100%. Chênh logits lớn nhất của ONNX FP32 so với eager khoảng 0,0000344. INT8 chỉ đạt macro F1 90,7411% và agreement 92,0277%; không sử dụng để triển khai.

Checkpoint gốc được giữ nguyên. Chỉ validation dùng để hiệu chỉnh lượng tử hóa, so sánh runtime và chọn cấu hình. Kết quả test gốc của checkpoint là macro F1 93,7632%; không dùng test để chọn runtime.

## HTTP localhost

Benchmark khởi chạy uvicorn bằng subprocess do công cụ sở hữu, gửi multipart upload thật, nhận cả tư vấn và JSON. Thay đổi ngưỡng 0,1 và 0,99 xen kẽ giữa các request để kiểm tra isolation. Khởi động server và warmup không nằm trong số liệu latency. Mỗi worker client gửi tuần tự, có nhiều worker chạy đồng thời; latency gồm hàng chờ ở server nhưng không gồm chờ trong executor của client.

Kết quả khảo sát ONNX FP32, 2 threads, 2 inference slots, 80 request mỗi mức tải:

| Client đồng thời | Mean HTTP ms | P95 HTTP ms | Request mỗi giây |
|---:|---:|---:|---:|
| 1 | 10,64 | 11,50 | 92,21 |
| 4 | 27,56 | 32,13 | 141,68 |
| 8 | 54,58 | 63,38 | 139,83 |

Vòng khảo sát có một phần chạy cùng kiểm thử; chỉ dùng vòng xác nhận sau kiểm thử để kết luận hiệu năng cuối. Các cấu hình runtime/thread/slot còn lại nằm trong `api_benchmark.json`. Không suy rộng số đo trên máy này thành cam kết cho mọi máy triển khai hay mạng ngoài localhost.

Vòng xác nhận độc lập, 150 request mỗi mức tải trên phiên bản code cuối:

| Runtime | Client | Mean HTTP ms | P95 HTTP ms |
|---|---:|---:|---:|
| ONNX FP32 | 1 | 19,28 | 35,63 |
| ONNX FP32 | 4 | 45,15 | 54,39 |
| ONNX FP32 | 8 | 81,31 | 93,96 |
| TorchScript | 1 | 19,67 | 37,65 |
| TorchScript | 4 | 47,37 | 64,28 |
| TorchScript | 8 | 100,28 | 139,35 |

Cả 14 unit/integration checks, 10 kiểm thử backend và 17 kiểm thử stress đã pass trên cấu hình cuối. Các log và snapshot phiên bản thư viện nằm cùng thư mục kết quả. Thời gian giữa các lượt đo có dao động; lựa chọn dựa trên cả bảo toàn nhãn và hiệu năng dưới tải.

## Những thay đổi phục vụ tải đồng thời

- CPU inference chạy ngoài event loop. CapacityLimiter tạo trong ASGI lifespan để request chờ không chiếm hết worker thread; semaphore cũng giới hạn CPU cho môi trường gọi không chạy lifespan.
- Ngưỡng tin cậy là tham số riêng của request, không sửa `advisory_engine.threshold` dùng chung.
- Đọc upload vào bộ nhớ tối đa giới hạn file cộng 1 byte, rồi kiểm tra kích thước trước decode.
- Header `Server-Timing` cung cấp thời gian inference và xử lý phía server để phân biệt với latency HTTP.
- ONNX kiểm tra input contract, thứ tự nhãn, temperature và fingerprint checkpoint. Graph cũ khác checkpoint sẽ không được chấp nhận.
- ORT sử dụng CPU provider, execution tuần tự, tối ưu graph, số thread tường minh và tắt spinning nhằm hạn chế chiếm CPU khi chờ.

Các lựa chọn thread và quantization tham khảo [tài liệu ONNX Runtime về thread](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) và [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Chạy trên máy khác

```powershell
python -m pip install -r requirements-onnx.txt
python -m ml.export_onnx
$env:SIC_INFERENCE_RUNTIME = "onnx"
$env:SIC_TORCH_THREADS = "2"
$env:SIC_INFERENCE_CONCURRENCY = "2"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Cần chuẩn bị checkpoint theo hướng dẫn Data & Optimization trước khi export. Ảnh, checkpoint, graph ONNX và cache vẫn được giữ local. Mặc định dùng ONNX nếu graph và dependency có sẵn, còn lại dùng TorchScript; có thể ghi đè bằng các biến môi trường trên. Khi cập nhật checkpoint, xuất graph mới với output khác, xác nhận rồi thay graph triển khai; công cụ export từ chối ghi đè graph có sẵn.

## Kiểm tra dữ liệu bổ sung

Perceptual audit quét 11.494 ảnh, chuẩn hóa xoay/lật, tìm cặp khác split bằng DCT pHash distance ≤2 và dHash distance ≤4. Không có cặp bị gắn cờ trong lượt này. Đây là kiểm tra bảo thủ, không chứng minh các lá vật lý đã tách biệt; 5.833 ảnh dùng identifier dự phòng vẫn cần ghi là giới hạn của bộ dữ liệu. Công cụ không thay đổi manifest hay tự loại ảnh dựa trên perceptual hash.

Kết quả và cách tái lập:

```powershell
python -m ml.runtime_validation
python -m ml.near_duplicate_audit
python -m ml.api_benchmark --runtimes onnx torchscript --threads 2 --slots 2 --runs 150
python -m unittest discover -s tests -v
```

Theo dõi Limits mỗi 5 phút đã được bật theo yêu cầu người dùng. Khi một cửa sổ còn không quá 5%, lưu tiến độ, đặt marker pause, dừng job của chat và tắt máy không ép đóng ứng dụng bằng `/f`.
