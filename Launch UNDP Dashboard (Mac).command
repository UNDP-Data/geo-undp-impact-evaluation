#!/bin/bash
cd "$(dirname "$0")"
clear
echo "========================================="
echo "  UNDP Early Recovery Impact Dashboard"
echo "========================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    echo "Opening the download page. Install Python, then run this again."
    open "https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Skip Streamlit's first-run email prompt
mkdir -p ~/.streamlit
[ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml

if [ ! -d ".venv" ]; then
    echo "First-time setup: preparing the application."
    echo "This takes a few minutes. Please wait..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip >/dev/null 2>&1
    pip install -r requirements.txt
    echo "Setup complete."
else
    source .venv/bin/activate
fi

echo ""
echo "Starting the dashboard. It will open in your browser."
echo "To STOP the app, close this window."
echo ""
streamlit run app.py
