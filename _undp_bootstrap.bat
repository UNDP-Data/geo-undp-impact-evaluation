@echo off
cd /d "%~dp0"
title UNDP Dashboard

REM --- detect an existing Python (do NOT force a reinstall) ---
set "PY="
where py        >nul 2>nul && set "PY=py"
if not defined PY ( where python  >nul 2>nul && set "PY=python" )
if not defined PY ( where python3 >nul 2>nul && set "PY=python3" )
if not defined PY (
  mshta "javascript:alert('Python 3.10 or newer is required.\n\nInstall it from https://www.python.org/downloads/ and tick \"Add python.exe to PATH\" during setup, then double-click the launcher again.');close();"
  exit /b 1
)

REM --- create the virtual environment only if it is missing ---
if not exist ".venv\Scripts\python.exe" (
  echo Setting up for first use. This takes a few minutes...
  %PY% -m venv .venv
)

REM --- install dependencies ONLY once (sentinel file) ---
if not exist ".venv\.deps_installed" (
  echo Installing components, please wait...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    mshta "javascript:alert('Setup failed while installing components. Please check your internet connection and try again.');close();"
    exit /b 1
  )
  echo done > ".venv\.deps_installed"
)

REM --- run the app (opens the browser automatically) ---
".venv\Scripts\python.exe" -m streamlit run app.py
