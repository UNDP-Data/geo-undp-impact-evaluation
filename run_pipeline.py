#!/usr/bin/env python3
"""
Cross-platform pipeline runner — replaces the Makefile so the app works on
Windows and Mac with no `make` and no shell scripting.

Usage:
  python run_pipeline.py --config config/ven.yaml --stage all
  python run_pipeline.py --config config/ven.yaml --stage gee
Stages: boundaries gee acled gt undp preprocess visuals trends scorecard all
"""
import argparse, os, sys, subprocess
from pathlib import Path

# Map country_name (or code) to the short code used in data/ and outputs/
NAME_TO_CODE = {"gaza": "pse", "myanmar": "mmr", "afghanistan": "afg", "ukraine": "ukr"}

def load_cfg(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)

def code_for(cfg, config_path):
    cc = (cfg.get("country_code") or "").lower()
    if cc:
        return cc
    stem = Path(config_path).stem.lower()
    return NAME_TO_CODE.get(stem, stem)

def py():
    # Use the SAME interpreter running this script (the app's venv) — works on all OSes.
    return sys.executable

def run_module(mod, config_path, extra=None):
    cmd = [py(), "-m", mod, "--config", config_path] + (extra or [])
    print(f">>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode

def file_rows(p):
    try:
        import pandas as pd
        return len(pd.read_csv(p))
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    code = code_for(cfg, args.config)
    data_dir = Path(f"data/{code}")
    ext = data_dir / "external"
    out_dir = Path(f"outputs/{code}")
    (ext).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = args.stage.lower()
    rc_total = 0

    def boundaries():
        try:
            bf = cfg["boundaries"]["file_path"]
            if Path(bf).exists():
                print(f"  ✓ Boundaries exist: {bf} — skipping", flush=True); return 0
        except Exception:
            pass
        return run_module("pipeline.download_boundaries", args.config)

    def gee():
        f = data_dir / f"gee_panel_{code}_centroid.csv"
        if f.exists():
            print(f"  ✓ GEE data exists ({file_rows(f)} rows) — skipping", flush=True); return 0
        return run_module("pipeline.gee_export_debug", args.config)

    def bm():
        f = ext / "blackmarble_ntl.csv"
        if f.exists():
            print(f"  ✓ Black Marble exists ({file_rows(f)} rows) — skipping", flush=True); return 0
        try:
            run_module("pipeline.blackmarble_download", args.config)
        except Exception:
            print("  ⚠ Black Marble failed — continuing without it", flush=True)
        return 0   # experimental stage: never counts as an issue

    def acled():
        f = ext / "acled_conflict.csv"
        if f.exists():
            print(f"  ✓ ACLED exists ({file_rows(f)} events) — skipping", flush=True); return 0
        email = os.environ.get("ACLED_EMAIL"); key = os.environ.get("ACLED_KEY") or os.environ.get("ACLED_PASSWORD")
        if email and key:
            return run_module("pipeline.acled_download", args.config, ["--email", email, "--password", key])
        print("  ⚠ ACLED credentials not set (ACLED_EMAIL / ACLED_KEY) — skipping", flush=True); return 0

    def wfp():
        f = ext / "wfp_prices.csv"
        if f.exists():
            print(f"  ✓ WFP prices exist ({file_rows(f)} rows) — skipping", flush=True); return 0
        try:
            run_module("pipeline.wfp_download", args.config)
        except Exception:
            print("  ⚠ WFP download failed — continuing without prices", flush=True)
        return 0   # optional stage: never counts as an issue

    def gt():
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
        return 0

    def undp():
        f = ext / "undp_projects.csv"
        if f.exists() and file_rows(f) > 0:
            print(f"  ✓ UNDP projects exist ({file_rows(f)}) — skipping", flush=True); return 0
        return run_module("pipeline.undp_projects", args.config)

    def preprocess():
        return run_module("pipeline.orchestrator", args.config, ["--stage", "preprocess"])

    def visuals():
        rc = run_module("pipeline.visualizations", args.config)
        if (ext / "undp_projects.csv").exists():
            try: run_module("pipeline.undp_timeline", args.config)
            except Exception: print("  ⚠ UNDP timeline failed", flush=True)
        return rc

    def trends():
        if (ext / "google_trends.csv").exists():
            return run_module("pipeline.google_trends_plots", args.config)
        print("  ⚠ No Google Trends data — skipping plots", flush=True); return 0

    def scorecard():
        cmd = [py(), "pipeline/recovery_scorecard.py", "--config", args.config]
        print(f">>> {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd).returncode

    stages = {
        "boundaries": boundaries, "gee": gee, "bm": bm, "acled": acled, "wfp": wfp, "gt": gt,
        "undp": undp, "preprocess": preprocess, "visuals": visuals,
        "trends": trends, "scorecard": scorecard,
    }

    if stage == "all":
        order = ["boundaries", "gee", "bm", "acled", "wfp", "gt", "undp", "preprocess", "visuals", "trends", "scorecard"]
        for st_name in order:
            print(f"\n=== {st_name.upper()} ===", flush=True)
            rc = stages[st_name]()
            rc_total += (rc or 0)
    elif stage in stages:
        rc_total = stages[stage]() or 0
    else:
        print(f"Unknown stage: {stage}", flush=True); sys.exit(2)

    print(f"\n{'='*48}\nPipeline finished (stage={stage}). Issues: {rc_total}\n{'='*48}", flush=True)
    sys.exit(0 if rc_total == 0 else 1)

if __name__ == "__main__":
    main()
