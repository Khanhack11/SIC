@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Follow README.md to install .venv first.
  pause
  exit /b 1
)
set "SIC_FIELD_MODEL_PATH=%~dp0models\mobilenetv3_field_continued_20261103.pt"
if not exist "%SIC_FIELD_MODEL_PATH%" (
  echo Experimental weights are in Reports\SIC_mobile_field_20261103.zip.
  pause
  exit /b 1
)
echo Experimental MobileNetV3 for comparison. Field validation release criteria were not met.
echo Open http://127.0.0.1:8003/app or http://SERVER_IPV4:8003/app on the same network.
echo Baseline demo uses port 8000. HTTPS is required for live browser camera on a phone.
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8003
if errorlevel 1 pause
endlocal
