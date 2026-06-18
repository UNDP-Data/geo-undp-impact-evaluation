@echo off
title UNDP Early Recovery Impact Dashboard
cd /d "%~dp0"

echo =========================================
echo   UNDP Early Recovery Impact Dashboard
echo =========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    mshta "javascript:var sh=new ActiveXObject('WScript.Shell'); sh.Popup('Python is required to run this dashboard.\n\nClick OK to open the download page. During install, CHECK the box \"Add Python to PATH\". Then double-click this launcher again.',0,'UNDP Dashboard - Setup Needed',48); close();"
    start https://www.python.org/downloads/
    exit /b
)

if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

if not exist ".venv" (
    echo First-time setup: preparing the application.
    echo This takes a few minutes. Please wait...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    echo Setup complete.
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Starting the dashboard. It will open in your browser.
echo To STOP the app, close this window.
echo.
streamlit run app.py
