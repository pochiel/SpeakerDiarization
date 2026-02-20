@echo off
echo ============================================
echo  Speaker Diarization System - Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/3] Installing PyTorch (CPU-only)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [3/3] Setup .env file...
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example
    echo Please edit .env and set your HF_TOKEN.
) else (
    echo .env already exists, skipping.
)

echo.
echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit .env and set HF_TOKEN=your_token
echo   2. Accept model terms on Hugging Face (see README.md)
echo   3. Run: python transcribe.py your_audio.wav
echo.
pause
