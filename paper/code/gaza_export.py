#!/usr/bin/env python3
"""
Export the Gaza estimation panel that analysis/pse/01_did_comprehensive.py
builds in memory, using that script's OWN functions (no re-implementation),
so the sample matches the paper exactly. Then part3_diagnostics.py can test
it under PPML.

Usage (from repo root, venv active):
    python3 gaza_export.py --inner 500 --outer 1500
Writes: outputs/pse/estimation_panel_<inner>_<outer>.csv
"""
import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--inner", type=int, default=500, help="inner donut radius, metres")
ap.add_argument("--outer", type=int, default=1500, help="outer donut radius, metres")
ap.add_argument("--indicator", default="nightlights",
                help="name of the raw-radiance indicator inside the built dataset")
a = ap.parse_args()

src = Path("analysis/pse/01_did_comprehensive.py")
if not src.exists():
    sys.exit(f"not found: {src} (run from the repo root)")
spec = importlib.util.spec_from_file_location("gaza_did", src)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    sys.exit(f"could not import your script: {e}\n"
             "If it runs analysis at import time, wrap that part in "
             "`if __name__ == '__main__':` and retry.")

for fn in ("load_sites_with_projects", "load_acled", "find_rasters", "build_analysis_dataset"):
    if not hasattr(mod, fn):
        sys.exit(f"function {fn} not found in your script; functions present: "
                 f"{[n for n in dir(mod) if callable(getattr(mod, n)) and not n.startswith('_')][:30]}")

print("Loading sites, ACLED and rasters via the paper's functions...")
sites = mod.load_sites_with_projects()
acled = mod.load_acled()
rasters = mod.find_rasters()
panel_controls = mod.load_district_panel() if hasattr(mod, "load_district_panel") else None
print(f"  sites: {0 if sites is None else len(sites)} | rasters: {len(rasters) if rasters is not None else 0}")

print(f"Building dataset for donut {a.inner}-{a.outer} m (this is the slow raster step)...")
df = mod.build_analysis_dataset(sites, acled, rasters, a.inner, a.outer, panel_controls)
print(f"  records: {len(df)} | columns: {list(df.columns)}")

ind_col = next((c for c in df.columns if c.lower() in ("indicator", "outcome", "variable")), None)
if a.indicator == "all" and ind_col:
    df["date"] = pd.to_datetime(df["date"])
    df["period_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    keep = [v for v in df[ind_col].unique() if not str(v).lower().startswith("log")]
    wide = (df[df[ind_col].isin(keep)]
            .pivot_table(index=["site_id", "period_start"], columns=ind_col,
                         values="value", aggfunc="mean").reset_index())
    meta = (df.groupby(["site_id", "period_start"], as_index=False)
              .agg(group=("group", "first"), project_start=("project_start", "first"),
                   district=("district", "first"), intervention_type=("intervention_type", "first"),
                   site_conflict_events=("site_conflict_events", "mean")))
    out = wide.merge(meta, on=["site_id", "period_start"])
    out["treated"] = (out["group"].astype(str) == "Treated").astype(int)
    out = out.rename(columns={"site_id": "unit_id", "project_start": "treat_date"}).drop(columns="group")
    dest = Path(f"outputs/pse/estimation_panel_{a.inner}_{a.outer}_all.csv")
    out.to_csv(dest, index=False)
    print(f"\nSaved {dest}: {out.unit_id.nunique()} sites, outcomes: {keep}")
    sys.exit(0)
# keep the raw-radiance indicator only (the script also stores a log version)
if ind_col:
    avail = df[ind_col].unique().tolist()
    match = [v for v in avail if a.indicator.lower() in str(v).lower() and "log" not in str(v).lower()]
    if not match:
        sys.exit(f"indicator '{a.indicator}' not found. Available: {avail}")
    df = df[df[ind_col] == match[0]].copy()
    print(f"  using indicator: {match[0]}")

df["date"] = pd.to_datetime(df["date"])
df["period_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["treated"] = (df["group"].astype(str) == "Treated").astype(int)
tcol = next((c for c in df.columns if "treat" in c.lower() and "date" in c.lower()), None)
if tcol is None:
    tcol = next((c for c in df.columns if c.lower() in ("start_date", "project_start", "intervention_date")), None)
out = pd.DataFrame({
    "unit_id": df["site_id"].astype(str),
    "period_start": df["period_start"],
    "nightlights": df["value"].astype(float),
    "treated": df["treated"],
    "treat_date": pd.to_datetime(df[tcol], errors="coerce") if tcol else pd.NaT,
    "district": df["district"] if "district" in df.columns else "",
    "intervention_type": df["intervention_type"] if "intervention_type" in df.columns else "",
    "site_conflict_events": df["site_conflict_events"] if "site_conflict_events" in df.columns else 0,
})
out = out.groupby(["unit_id", "period_start"], as_index=False).agg(
    nightlights=("nightlights", "mean"), treated=("treated", "first"),
    treat_date=("treat_date", "first"), district=("district", "first"),
    intervention_type=("intervention_type", "first"),
    site_conflict_events=("site_conflict_events", "mean"))
dest = Path(f"outputs/pse/estimation_panel_{a.inner}_{a.outer}.csv")
out.to_csv(dest, index=False)
print(f"\nSaved {dest}: {out.unit_id.nunique()} sites "
      f"({out.loc[out.treated==1,'unit_id'].nunique()} treated), "
      f"{out.period_start.min().date()} to {out.period_start.max().date()}")
print(f"treatment-date column used: {tcol}" if tcol else
      "WARNING: no per-site treatment date column found; run diagnostics with --event-date only")
