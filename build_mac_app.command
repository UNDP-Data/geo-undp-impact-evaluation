#!/bin/bash
# Run this ONCE on a Mac (double-click it) to create "UNDP Dashboard.app".
cd "$(dirname "$0")" || exit 1
osacompile -o "UNDP Dashboard.app" "UNDP Dashboard.applescript" && \
osascript -e 'display dialog "UNDP Dashboard.app created. Double-click it to launch (no Terminal)." buttons {"OK"}'
