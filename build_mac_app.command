#!/bin/bash
# Run this ONCE on a Mac (double-click) to create "UNDP Dashboard.app" with its icon.
cd "$(dirname "$0")" || exit 1

osacompile -o "UNDP Dashboard.app" "UNDP Dashboard.applescript" || {
  osascript -e 'display dialog "Could not build the app. Make sure UNDP Dashboard.applescript is in this folder." buttons {"OK"} with icon caution'
  exit 1
}

# Install the custom icon (convert icon.png -> .icns using built-in macOS tools)
if [ -f "icon.png" ]; then
  sips -s format icns "icon.png" --out "UNDP Dashboard.app/Contents/Resources/applet.icns" >/dev/null 2>&1
  # nudge Finder to refresh the icon
  touch "UNDP Dashboard.app"
fi

osascript -e 'display dialog "UNDP Dashboard.app created. Double-click it to launch (no Terminal)." buttons {"OK"}'
