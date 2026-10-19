@echo off
cd /d "%~dp0"
title Railway Infrastructure Inspection System
echo ========================================================
echo   Railway Infrastructure Inspection Suite
echo ========================================================
echo [1/2] Checking Python environment...
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at: %~dp0venv\Scripts\python.exe
    pause
    exit /b 1
)
echo [2/2] Launching Inspection GUI dashboard...
echo Loading deep learning models (PyTorch, Torchvision, Ultralytics)...
echo.
"%~dp0venv\Scripts\python.exe" -u "%~dp0gui_app.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
