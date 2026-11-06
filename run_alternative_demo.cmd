@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install the CPU environment from README.md first.
  exit /b 1
)
if not exist "docs\final_release\alternative_sources_20261103\runtime_parity.json" (
  echo Model runtime validation is not ready. See the alternative sources training report.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import hashlib,json; from pathlib import Path; r=json.loads(Path('docs/final_release/alternative_sources_20261103/runtime_parity.json').read_text()); p=Path('artifacts/training_runs/alternative_sources_20261103/model'); assert r['quality_parity_passed'], 'ONNX parity failed'; assert hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest()==r['weights_sha256'], 'Weights differ from validated model'; assert hashlib.sha256(p.with_suffix('.onnx').read_bytes()).hexdigest()==r['onnx_sha256'], 'ONNX differs from validated graph'"
if errorlevel 1 exit /b 1
set "SIC_DEVICE=cpu"
set "SIC_INFERENCE_RUNTIME=onnx"
set "SIC_FIELD_MODEL_PATH=%~dp0artifacts\training_runs\alternative_sources_20261103\model.pt"
set "SIC_FIELD_ONNX_PATH=%~dp0artifacts\training_runs\alternative_sources_20261103\model.onnx"
echo Experimental model trained with alternative sources. Field release criteria were NOT met.
echo Open http://127.0.0.1:8004/app . Previous candidate remains on port 8003.
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8004
endlocal
