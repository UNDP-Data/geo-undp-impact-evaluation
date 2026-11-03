#!/usr/bin/env python3
"""Round 33: Google Trends refresh. Run from the repo root.
- The gt stage re-downloads when the data is more than 30 days old
  (checked from the dates inside the file; file age as a fallback),
  instead of skipping forever once a file exists.
- If Google refuses the request (common from servers), the previous data is
  kept and the stage reports it without failing the run.
- Running the Trends stage on its own (the app's Trends button) also redraws
  the Google Trends charts."""
import sys
p = "run_pipeline.py"; s = open(p, encoding="utf-8").read(); orig = s
old = '''    def gt():
        f = ext / "google_trends.csv"
        if f.exists():
            print(f"  ✓ Google Trends exist ({file_rows(f)} records) — skipping", flush=True); return 0
        try:
            return run_module("pipeline.google_trends", args.config)
        except Exception:
            print("  ⚠ Google Trends failed — skipping", flush=True); return 0'''
new = '''    def gt():
        import shutil, time as _t
        f = ext / "google_trends.csv"
        MAX_AGE_DAYS = 30

        def _age_days(path):
            # Age of the newest observation in the file; file age as a fallback.
            try:
                import pandas as pd
                df = pd.read_csv(path)
                col = next((c for c in df.columns
                            if "date" in c.lower() or c.lower() in ("week", "month", "period", "period_start")), None)
                if col is not None:
                    last = pd.to_datetime(df[col], errors="coerce").max()
                    if pd.notna(last):
                        return (pd.Timestamp.now() - last).days
            except Exception:
                pass
            return (_t.time() - path.stat().st_mtime) / 86400

        if f.exists():
            age = _age_days(f)
            if age <= MAX_AGE_DAYS:
                print(f"  ✓ Google Trends up to date ({file_rows(f)} records, latest {age:.0f} days ago) — skipping", flush=True)
                return 0
            print(f"  ↻ Google Trends data is {age:.0f} days old — refreshing", flush=True)
        bak = ext / "google_trends.csv.bak"
        before = f.stat().st_mtime if f.exists() else None
        if f.exists():
            shutil.copy2(f, bak)
        try:
            run_module("pipeline.google_trends", args.config)
        except Exception:
            pass
        refreshed = f.exists() and file_rows(f) > 0 and (before is None or f.stat().st_mtime != before)
        if not refreshed:
            if bak.exists():
                shutil.copy2(bak, f)
                print("  ⚠ Google Trends refresh failed (Google often blocks automated requests from servers) "
                      "— kept the previous data", flush=True)
            else:
                print("  ⚠ Google Trends download failed — no data for this country yet", flush=True)
            return 0   # optional source: never counts as a pipeline issue
        print(f"  ✓ Google Trends refreshed ({file_rows(f)} records)", flush=True)
        if stage == "gt":
            trends()   # the Trends button also redraws the charts
        return 0'''
if s.count(old) != 1:
    sys.exit(f"ABORT: gt() block found {s.count(old)}x (expected 1)")
s = s.replace(old, new)
compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched run_pipeline.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
