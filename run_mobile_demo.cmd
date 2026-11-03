@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Follow README.md to install .venv first.
  pause
  exit /b 1
)
echo Phone and server must use the same network.
set "SIC_DEVICE=cpu"
set "SIC_INFERENCE_RUNTIME=onnx"
set "SIC_FIELD_MODEL_PATH="
set "SIC_FIELD_ONNX_PATH="
echo Open http://SERVER_IPV4:8000/app on the phone. See ipconfig for the server IPv4.
echo HTTPS is needed for browser live camera; native camera/file capture is available on HTTP.
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
if errorlevel 1 pause
endlocal
