#!/usr/bin/env python3
"""Stage the country content the dashboard reads, within GitHub's size limits.
Run from the repo root with the venv active (needs geopandas, yaml).
- keeps: config/*.yaml; data/<code>/boundaries/<code>_admin{level,1}.geojson;
  data/<code>/external/{undp_projects,undp_sites*,wfp_prices,blackmarble_ntl,wb_indicators}.*;
  outputs/<code>/** (panel, figures, results, showcase)
- drops: Earth Engine caches, raw ACLED/mobility exports, rasters, anything > 50 MB
- simplifies boundary files > 25 MB in place (display resolution)"""
import glob, os, subprocess, sys
from pathlib import Path
import yaml

MAX_MB, SIMPLIFY_MB = 50, 25
def mb(p): return os.path.getsize(p) / 1e6

def simplify(path):
    import geopandas as gpd
    for tol in (0.0005, 0.001, 0.002, 0.005):
        g = gpd.read_file(path)
        g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
        try:
            g.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=5)
        except Exception:
            g.to_file(path, driver="GeoJSON")
        print(f"   simplified {path} (tol {tol}) -> {mb(path):.1f} MB")
        if mb(path) < MAX_MB * 0.8:
            return True
    return mb(path) < MAX_MB

staged, skipped = [], []
def add(p):
    if not os.path.exists(p): return
    if mb(p) > MAX_MB:
        skipped.append(f"{p} ({mb(p):.0f} MB)"); return
    staged.append(p)

for cfg in sorted(glob.glob("config/*.yaml")):
    c = yaml.safe_load(open(cfg)) or {}
    code = str(c.get("country_code", "")).lower()
    if not code: continue
    add(cfg)
    level = str(c.get("admin_level", c.get("adm_level", 2))).lower().replace("adm", "")
    for lvl in {level, "1"}:
        for p in glob.glob(f"data/{code}/boundaries/{code}_admin{lvl}.geojson") + glob.glob(f"data/{code}/boundaries/*admin{lvl}*.geojson"):
            if os.path.islink(p): continue
            if mb(p) > SIMPLIFY_MB and not simplify(p):
                skipped.append(f"{p} (still {mb(p):.0f} MB after simplification)"); continue
            add(p)
    for pat in ("undp_projects.csv", "undp_sites*.csv", "undp_sites*.geojson", "undp_sites_digitized.csv",
                "wfp_prices.csv", "blackmarble_ntl.csv", "wb_indicators.csv"):
        for p in glob.glob(f"data/{code}/external/{pat}"): add(p)
    for p in glob.glob(f"outputs/{code}/**/*", recursive=True):
        if os.path.isfile(p) and not p.endswith((".tif", ".DS_Store", ".lock")): add(p)

subprocess.run(["git", "reset", "-q"])                      # start from a clean index
for i in range(0, len(staged), 200):
    subprocess.run(["git", "add", "-f", "--"] + staged[i:i+200], check=True)
print(f"\nStaged {len(staged)} files for {len(glob.glob('config/*.yaml'))} configs.")
if skipped:
    print("Left out (too large for GitHub):"); [print("   " + s) for s in skipped]
big = [p for p in staged if mb(p) > MAX_MB]
if big: sys.exit(f"ABORT: oversized files slipped through: {big}")
out = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True).stdout.strip().splitlines()
print(out[-1] if out else "nothing staged")
print("Next: git commit -m 'Publish country content for deployment' && git push")
