@echo off
setlocal
cd /d "%~dp0"
set "SIC_FIELD_MODEL_PATH=%~dp0models\candidate\mobilenetv3_field.pt"
set "SIC_FIELD_ONNX_PATH="
set "SIC_DEVICE=cpu"
set "SIC_INFERENCE_RUNTIME=eager"
set "SIC_PYTHON=.venv\Scripts\python.exe"
if /i "%~1"=="gpu" (
  set "SIC_DEVICE=cuda"
  set "SIC_PYTHON=.venv-gpu\Scripts\python.exe"
)
if not exist "%SIC_PYTHON%" (
  echo Missing %SIC_PYTHON%. Follow README.md to install the selected environment.
  exit /b 1
)
if /i "%~1"=="onnx" set "SIC_FIELD_ONNX_PATH=%~dp0models\candidate\mobilenetv3_field.onnx"
if /i "%~1"=="onnx" if not exist "%SIC_FIELD_ONNX_PATH%" (
  echo Export the field ONNX graph first. See docs\RUNTIME.md.
  pause
  exit /b 1
)
if not exist "%SIC_FIELD_MODEL_PATH%" (
  echo Missing candidate weights. Restore models from the current demo package in Reports.
  pause
  exit /b 1
)
echo Experimental MobileNetV3. Field validation release criteria were not met.
if defined SIC_FIELD_ONNX_PATH (echo Runtime: ONNX FP32.) else (echo Runtime: PyTorch eager.)
echo Open http://127.0.0.1:8003/app or http://SERVER_IPV4:8003/app on the same network.
echo Baseline demo uses port 8000. HTTPS is required for live browser camera on a phone.
"%SIC_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8003
if errorlevel 1 pause
endlocal
