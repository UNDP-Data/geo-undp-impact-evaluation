#!/bin/bash
# UNDP Early Recovery Dashboard - macOS bootstrap (invoked by UNDP Dashboard.app)
cd "$(dirname "$0")" || exit 1

CRED_DIR="$HOME/.undp_dashboard"
HOME_ENV="$CRED_DIR/.env"
SENTINEL=".venv/.deps_installed"

# --- Seed credentials from sibling pilot_credentials.txt (gitignored, private) ---
if [ -f "pilot_credentials.txt" ] && [ ! -f "$HOME_ENV" ]; then
  mkdir -p "$CRED_DIR"
  grep -v '^#' pilot_credentials.txt | grep -v '^[[:space:]]*$' > "$HOME_ENV"
fi

# --- Detect an existing Python (do NOT force a reinstall) ---
PY=""
for c in python3 python /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  osascript -e 'display dialog "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ then open the app again." buttons {"OK"} with icon caution'
  exit 1
fi

# --- First-run setup only (skip if already installed) ---
if [ ! -x ".venv/bin/python" ]; then
  osascript -e 'display notification "Setting up for first use (a few minutes)..." with title "UNDP Dashboard"'
  "$PY" -m venv .venv
fi
if [ ! -f "$SENTINEL" ]; then
  ".venv/bin/python" -m pip install --upgrade pip
  if ".venv/bin/python" -m pip install -r requirements.txt; then
    touch "$SENTINEL"
  else
    osascript -e 'display dialog "Setup failed while installing components. Check your internet connection and open the app again." buttons {"OK"} with icon caution'
    exit 1
  fi
fi

# --- Run (browser opens automatically) ---
".venv/bin/python" -m streamlit run app.py
