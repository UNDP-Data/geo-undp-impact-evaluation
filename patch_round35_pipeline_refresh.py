#!/usr/bin/env python3
"""Round 35 (pipeline). Run from the repo root.
- Downloads no longer skip forever once a file exists: satellite (45 days),
  Black Marble (30), conflict (30), prices (60) and projects (90) refresh
  when their newest data is older than that, keeping the old file if the
  refresh fails or comes back truncated.
- The study window extends itself to today whenever it ends in the past,
  so every download reaches the present."""
import sys
p = "run_pipeline.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

# ── study window to today ──
rep('''    cfg = load_cfg(args.config)
    code = code_for(cfg, args.config)''',
'''    cfg = load_cfg(args.config)
    code = code_for(cfg, args.config)

    # Monitoring data should run to the present: if the study window ends in the
    # past, extend it to today and save, so every download reaches now.
    try:
        from datetime import date as _date
        _end = str(cfg.get("study_period_end", ""))[:10]
        _today = _date.today().isoformat()
        if _end and _end < _today:
            cfg["study_period_end"] = _today
            import yaml as _yaml
            with open(args.config, "w") as _fw:
                _yaml.dump(cfg, _fw, default_flow_style=False, sort_keys=False)
            print(f"  ↻ Study window extended to today ({_end} -> {_today})", flush=True)
    except Exception as _e:
        print(f"  ⚠ Could not extend the study window: {_e}", flush=True)''')

# ── generic refresh helper, inserted before the first stage ──
rep('''    def boundaries():''',
'''    def _latest_days(path, date_based=True):
        """Days since the newest observation in the file (file age as fallback)."""
        import time as _t
        if date_based:
            try:
                import pandas as pd
                df = pd.read_csv(path)
                col = next((c for c in df.columns if c.lower() in
                            ("period_start", "date", "event_date", "week", "month", "period")), None)
                if col is None:
                    col = next((c for c in df.columns if "date" in c.lower()), None)
                if col is not None:
                    last = pd.to_datetime(df[col], errors="coerce").max()
                    if pd.notna(last):
                        return (pd.Timestamp.now() - last).days
            except Exception:
                pass
        return (_t.time() - path.stat().st_mtime) / 86400

    def refresh(path, max_age, module, label, extra=None, optional=False, date_based=True):
        """Download if the file is missing or stale; never lose the old file."""
        import shutil
        if path.exists():
            age = _latest_days(path, date_based)
            if age <= max_age:
                print(f"  ✓ {label} up to date ({file_rows(path)} rows, latest {age:.0f} days ago) — skipping", flush=True)
                return 0
            print(f"  ↻ {label} data is {age:.0f} days old — refreshing", flush=True)
        bak = path.with_name(path.name + ".bak")
        before, prev_rows = (path.stat().st_mtime, file_rows(path)) if path.exists() else (None, 0)
        if path.exists():
            shutil.copy2(path, bak)
        try:
            rc = run_module(module, args.config, extra)
        except Exception:
            rc = 1
        ok = (path.exists() and file_rows(path) > 0
              and (before is None or path.stat().st_mtime != before)
              and file_rows(path) >= 0.8 * prev_rows)
        if not ok:
            if bak.exists():
                shutil.copy2(bak, path)
                print(f"  ⚠ {label} refresh failed or came back incomplete — kept the previous data", flush=True)
                return 0
            print(f"  ⚠ {label} download failed", flush=True)
            return 0 if optional else (rc or 1)
        print(f"  ✓ {label} refreshed ({file_rows(path)} rows)", flush=True)
        return 0

    def boundaries():''')

rep('''    def gee():
        f = data_dir / f"gee_panel_{code}_centroid.csv"
        if f.exists():
            print(f"  ✓ GEE data exists ({file_rows(f)} rows) — skipping", flush=True); return 0
        return run_module("pipeline.gee_export_debug", args.config)''',
'''    def gee():
        return refresh(data_dir / f"gee_panel_{code}_centroid.csv", 45,
                       "pipeline.gee_export_debug", "Satellite")''')

rep('''    def bm():
        f = ext / "blackmarble_ntl.csv"
        if f.exists():
            print(f"  ✓ Black Marble exists ({file_rows(f)} rows) — skipping", flush=True); return 0
        try:
            run_module("pipeline.blackmarble_download", args.config)
        except Exception:
            print("  ⚠ Black Marble failed — continuing without it", flush=True)
        return 0   # experimental stage: never counts as an issue''',
'''    def bm():
        return refresh(ext / "blackmarble_ntl.csv", 30, "pipeline.blackmarble_download",
                       "Black Marble", optional=True)''')

rep('''    def acled():
        f = ext / "acled_conflict.csv"
        if f.exists():
            print(f"  ✓ ACLED exists ({file_rows(f)} events) — skipping", flush=True); return 0
        email = os.environ.get("ACLED_EMAIL"); key = os.environ.get("ACLED_KEY") or os.environ.get("ACLED_PASSWORD")
        if email and key:
            return run_module("pipeline.acled_download", args.config, ["--email", email, "--password", key])
        print("  ⚠ ACLED credentials not set (ACLED_EMAIL / ACLED_KEY) — skipping", flush=True); return 0''',
'''    def acled():
        email = os.environ.get("ACLED_EMAIL"); key = os.environ.get("ACLED_KEY") or os.environ.get("ACLED_PASSWORD")
        if not (email and key):
            print("  ⚠ ACLED credentials not set (ACLED_EMAIL / ACLED_KEY) — skipping", flush=True); return 0
        return refresh(ext / "acled_conflict.csv", 30, "pipeline.acled_download", "Conflict events (ACLED)",
                       extra=["--email", email, "--password", key])''')

rep('''    def wfp():
        f = ext / "wfp_prices.csv"
        if f.exists():
            print(f"  ✓ WFP prices exist ({file_rows(f)} rows) — skipping", flush=True); return 0
        try:
            run_module("pipeline.wfp_download", args.config)
        except Exception:
            print("  ⚠ WFP download failed — continuing without prices", flush=True)
        return 0   # optional stage: never counts as an issue''',
'''    def wfp():
        return refresh(ext / "wfp_prices.csv", 60, "pipeline.wfp_download", "Food prices (WFP)", optional=True)''')

rep('''    def undp():
        f = ext / "undp_projects.csv"
        if f.exists() and file_rows(f) > 0:
            print(f"  ✓ UNDP projects exist ({file_rows(f)}) — skipping", flush=True); return 0
        return run_module("pipeline.undp_projects", args.config)''',
'''    def undp():
        return refresh(ext / "undp_projects.csv", 90, "pipeline.undp_projects", "UNDP projects", date_based=False)''')

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched run_pipeline.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
