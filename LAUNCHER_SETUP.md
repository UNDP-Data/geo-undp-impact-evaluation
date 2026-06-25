# UNDP Dashboard — Launcher Setup

These launchers fix three problems with the old ones:
- they detect Python that is **already installed** (no forced reinstall),
- they **skip installation** after the first run (instant re-open),
- they run **without a Terminal/console window**.

Put all launcher files in the **same folder as `app.py`** (the repo root,
e.g. `undp-impact-evaluation-main`).

## Windows
1. Copy `Launch UNDP Dashboard.vbs` and `_undp_bootstrap.bat` into the repo folder.
2. Double-click **`Launch UNDP Dashboard.vbs`**.
   - First run: a small window shows setup progress, then the dashboard opens in the browser.
   - Later runs: silent, no console window — opens straight to the dashboard.
3. (Optional) Right-click the .vbs → Send to → Desktop (create shortcut) for a one-click icon.

Prerequisite: Python 3.10+ installed with "Add to PATH" ticked. If it is missing,
the launcher shows a message with the download link.

## macOS
1. Copy `run_dashboard.sh` and `UNDP Dashboard.applescript` into the repo folder.
2. Compile the AppleScript into a no-Terminal app (run once in Terminal, from the repo folder):
   ```
   osacompile -o "UNDP Dashboard.app" "UNDP Dashboard.applescript"
   ```
3. Double-click **`UNDP Dashboard.app`**. No Terminal window appears; the browser opens
   when ready (first run installs once; later runs are instant).
   - Logs go to `/tmp/undp_dashboard.log` if you ever need to troubleshoot.
4. (Optional) Drag `UNDP Dashboard.app` to the Dock for one-click access.

Note: keep the .app **inside the repo folder** — it locates `app.py` as its parent folder.

## How "skip reinstall" works
A marker file `.venv/.deps_installed` is written after the first successful install.
While it exists, the launcher goes straight to running the app. Delete it to force a
clean reinstall (e.g. after updating `requirements.txt`).
