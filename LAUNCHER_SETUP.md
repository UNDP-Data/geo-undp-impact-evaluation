# UNDP Dashboard — Download & Run (no terminal)

Goal: two downloadable launchers (Windows and Mac) that set up on first use and then
run the app with **no terminal window**. The browser opens automatically.

All launcher files live in the **repo root, next to `app.py`**.

## Windows (single file)
- File: **`UNDP Dashboard.vbs`**
- Double-click it. First run shows a short "setting up" dialog, then the dashboard
  opens in the browser. Later runs are instant and silent — no console window at all.
- Prerequisite: Python 3.10+ (installer link is shown automatically if missing).
- Optional: right-click → Send to → Desktop (create shortcut) for a one-click icon.

## Mac (build once, then a real .app)
1. Double-click **`build_mac_app.command`** once. It creates **`UNDP Dashboard.app`**.
   (If macOS blocks it: right-click → Open → Open.)
2. From then on, double-click **`UNDP Dashboard.app`** — no Terminal window appears.
3. Optional: drag the .app to the Dock.
- Keep the .app inside the repo folder (it finds `app.py` as its parent folder).
- Logs: `/tmp/undp_dashboard.log` for troubleshooting.

## Credentials (pilot)
So testers don't each have to paste keys:
1. Copy `pilot_credentials.txt.EXAMPLE` to **`pilot_credentials.txt`** and fill in your keys.
2. Include `pilot_credentials.txt` **only** in the private ZIP you send to testers.
3. On first run, the launcher copies those keys to `~/.undp_dashboard/.env`, where the
   app reads them. They persist across restarts.

**SECURITY — important:**
- `pilot_credentials.txt` is **gitignored** and must **never** be committed (the repo is public).
- When UNDP provides their own keys, replace the file. Each tester's machine keeps its own
  copy in their home folder; nothing is shared.

## How "skip reinstall" works
After the first successful setup, `.venv/.deps_installed` is created. While it exists the
launcher goes straight to running the app. Delete it to force a clean reinstall.
