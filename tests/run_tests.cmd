@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tests/run_all.py --web --mobile
set "test_result=%errorlevel%"
pause
exit /b %test_result%
