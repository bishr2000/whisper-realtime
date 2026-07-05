@echo off
echo ============================================
echo   Whisper Translator - Real-time Setup
echo ============================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)

REM Create venv if missing
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing core dependencies...
pip install --upgrade pip
pip install faster-whisper fastapi "uvicorn[standard]" python-multipart deep-translator jinja2 websockets numpy soundfile

echo.
echo Installing noise cancellation...
pip install noisereduce scipy

echo.
echo Installing PyTorch with CUDA support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo ============================================
echo   Starting server on http://localhost:8000
echo   Press Ctrl+C to stop
echo ============================================
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
