#!/bin/bash
# Double-click to start. First time: right-click -> Open -> Open (to bypass the warning).
cd "$(dirname "$0")" || exit 1

# Seed credentials from pilot_credentials.txt (if present) into a stable home location
CRED_DIR="$HOME/.undp_dashboard"; HOME_ENV="$CRED_DIR/.env"
if [ -f "pilot_credentials.txt" ] && [ ! -f "$HOME_ENV" ]; then
  mkdir -p "$CRED_DIR"
  grep -v '^#' pilot_credentials.txt | grep -v '^[[:space:]]*$' > "$HOME_ENV"
fi

# Find Python
PY=""
for c in python3 python /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  osascript -e 'display dialog "Python is needed once. Opening the download page. Install it, then double-click this file again." buttons {"OK"}'
  open "https://www.python.org/downloads/"; exit 1
fi

# Set up once, then reuse
[ -x ".venv/bin/python" ] || "$PY" -m venv .venv
if [ ! -f ".venv/.ready" ]; then
  echo "Setting up (a few minutes the first time)..."
  ".venv/bin/python" -m pip install --upgrade pip >/dev/null
  ".venv/bin/python" -m pip install -r requirements.txt && touch ".venv/.ready"
fi

# Run (opens the browser)
".venv/bin/python" -m streamlit run app.py
