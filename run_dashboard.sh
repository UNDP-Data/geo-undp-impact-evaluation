#!/bin/bash
# UNDP Early Recovery Dashboard - macOS bootstrap (called by the .app)
cd "$(dirname "$0")" || exit 1

# --- detect an existing Python (do NOT force a reinstall) ---
PY=""
for c in python3 python /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  osascript -e 'display dialog "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ then open the app again." buttons {"OK"} with icon caution'
  exit 1
fi

# --- create venv only if missing ---
if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi

# --- install deps ONLY once (sentinel) ---
if [ ! -f ".venv/.deps_installed" ]; then
  osascript -e 'display notification "Setting up for first use, this takes a few minutes." with title "UNDP Dashboard"'
  ".venv/bin/python" -m pip install --upgrade pip
  ".venv/bin/python" -m pip install -r requirements.txt && touch ".venv/.deps_installed"
fi

# --- run (opens browser automatically) ---
".venv/bin/python" -m streamlit run app.py
