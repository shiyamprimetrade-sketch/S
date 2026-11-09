@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 or newer from python.org, then try again.
  echo This application runs locally and does not need your ChatGPT account.
  pause
  exit /b 1
)
py -3 -c "import tkinter,sqlite3;print('S prerequisites: ready')"
if errorlevel 1 (
  echo Python is present, but required standard components are unavailable.
  pause
  exit /b 1
)
start "" powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765'"
echo S local server starting. Keep this window open while using S.
py -3 -m continuum.cli serve --port 8765
if errorlevel 1 (
  echo S exited with an error. Check the message above.
  pause
)
