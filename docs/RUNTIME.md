# CPU x86-64, GPU NVIDIA và file model

Máy hiện tại dùng Intel Core i5-12450H, 8 core/12 logical processors, Windows 64-bit và RTX 2050 4 GB. `AMD64` trong Python là tên kiến trúc x86-64, không chỉ thương hiệu CPU. GPU NVIDIA có kiến trúc và tập lệnh riêng, được gọi qua CUDA.

## Thành phần thực thi

| Thành phần | Nội dung | Bộ thực thi |
|---|---|---|
| `backend/inference_engine.py` | Tạo V2, nạp weights, gọi forward | Python + PyTorch |
| `backend/mobile_model.py`, `field_inference.py` | Tạo V3, gọi forward | Python + PyTorch |
| `.pt` | Tensor trọng số, nhãn, preprocessing/calibration | PyTorch nạp vào CPU/CUDA |
| `.onnx` | Graph và trọng số export | ONNX Runtime CPUExecutionProvider hiện tại |
| `.pyd`, `.dll` trong môi trường | Mã native cho toán tử | CPU x86-64 hoặc CUDA kernel NVIDIA |

`model.to(device)` đặt weights lên thiết bị; tensor ảnh cũng được chuyển lên cùng thiết bị rồi gọi `model(images)`. Cấu hình GPU dùng `.venv-gpu`, `SIC_DEVICE=cuda`, `SIC_INFERENCE_RUNTIME=eager`. CPU dùng `.venv`, ONNX CPU, hai thread mặc định.

Các binary thực thi nằm ở `torch._C...win_amd64.pyd`, `torch/lib/torch_cpu.dll`, `torch/lib/torch_cuda.dll` và binding native ONNX Runtime. [JSON benchmark](benchmarks/deployment_20261103/) ghi đường dẫn thật, CUDA, compute capability, phiên bản Torch và kết quả chạy.

## Lệnh

```powershell
./run_demo.cmd
./run_gpu_demo.cmd
./run_field_demo.cmd
./run_field_demo.cmd gpu
./run_field_demo.cmd onnx
```

Mỗi dòng là một lựa chọn riêng. Baseline dùng cổng 8000; candidate dùng 8003. GPU launcher báo lỗi nếu không có CUDA. ONNX profile hiện tại là CPU, chưa cấu hình CUDA Execution Provider.

`GET /api/health` báo `device: cuda` và runtime `eager` khi chạy GPU; CPU ONNX báo `cpu` và `onnx`. Web hiển thị CPU/GPU máy chủ. Trình duyệt điện thoại chỉ gửi ảnh đến API.

## Đo lại

```powershell
.venv/Scripts/python.exe -m ml.benchmark_deployment --device cpu --runtime onnx --output artifacts/new_cpu_onnx.json
.venv/Scripts/python.exe -m ml.benchmark_deployment --device cpu --runtime eager --output artifacts/new_cpu_pytorch.json
.venv-gpu/Scripts/python.exe -m ml.benchmark_deployment --device cuda --runtime eager --output artifacts/new_gpu_cuda.json
```

Mỗi lệnh đo hai model. CUDA được đồng bộ trước/sau đo; có warmup, batch 1/4 và pipeline ảnh riêng. Không dùng các ảnh demo để tính accuracy. [Kết quả và phạm vi](BENCHMARKS.md).
