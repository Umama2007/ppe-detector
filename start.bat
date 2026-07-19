@echo off
REM start.bat - One-click setup + launch for Ice Cream Detection app (Windows)
REM Usage: double-click this file, or run it from Command Prompt.

echo Ice Cream Detection - Setup and Launch
echo ========================================

REM 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not on PATH. Please install Python 3.10+ from python.org and re-run this script.
    pause
    exit /b 1
)
echo Found Python:
python --version

REM 2. Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment ...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

REM 3. Activate virtual environment
call .venv\Scripts\activate.bat

REM 4. Install dependencies
echo Installing required packages, this may take a few minutes the first time...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo All dependencies installed.

REM 5. Launch Streamlit app on localhost
echo Starting app at http://localhost:8501 ...
streamlit run app.py --server.port 8501 --server.address localhost

pause
