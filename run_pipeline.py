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

    def acled():
        f = ext / "acled_conflict.csv"
        if f.exists():
            print(f"  ✓ ACLED exists ({file_rows(f)} events) — skipping", flush=True); return 0
        email = os.environ.get("ACLED_EMAIL"); key = os.environ.get("ACLED_KEY") or os.environ.get("ACLED_PASSWORD")
        if email and key:
            return run_module("pipeline.acled_download", args.config, ["--email", email, "--password", key])
        print("  ⚠ ACLED credentials not set (ACLED_EMAIL / ACLED_KEY) — skipping", flush=True); return 0

    def gt():
        f = ext / "google_trends.csv"
        if f.exists():
            print(f"  ✓ Google Trends exist ({file_rows(f)} records) — skipping", flush=True); return 0
        try:
            return run_module("pipeline.google_trends", args.config)
        except Exception:
            print("  ⚠ Google Trends failed — skipping", flush=True); return 0

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
        "boundaries": boundaries, "gee": gee, "acled": acled, "gt": gt,
        "undp": undp, "preprocess": preprocess, "visuals": visuals,
        "trends": trends, "scorecard": scorecard,
    }

    if stage == "all":
        order = ["boundaries", "gee", "acled", "gt", "undp", "preprocess", "visuals", "trends", "scorecard"]
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
