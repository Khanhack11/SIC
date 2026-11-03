@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gpu\Scripts\python.exe" (
  echo Missing .venv-gpu. See README.md for the CUDA environment.
  exit /b 1
)
".venv-gpu\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 exit /b 1
set "SIC_DEVICE=cuda"
set "SIC_INFERENCE_RUNTIME=eager"
set "SIC_FIELD_MODEL_PATH="
set "SIC_FIELD_ONNX_PATH="
echo MobileNetV2 .pt on NVIDIA CUDA. Open http://127.0.0.1:8000/app
".venv-gpu\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
endlocal
