@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow README.md to install the demo environment.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000/app in your browser.
set "SIC_DEVICE=cpu"
set "SIC_INFERENCE_RUNTIME=onnx"
set "SIC_FIELD_MODEL_PATH="
set "SIC_FIELD_ONNX_PATH="
echo Keep this window open during the demo. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
endlocal
