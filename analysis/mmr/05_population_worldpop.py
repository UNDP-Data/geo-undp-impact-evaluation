#!/usr/bin/env python3
"""
analysis/mmr/05_population_worldpop.py
=======================================
Satellite-verified population estimate for Myanmar's treated townships,
mirroring Gaza's WorldPop buffer method (analysis/pse/archive/07_population_estimation.py)
so that both pilots use the same population source and logic.

PURPOSE
-------
Replaces the unverified agency-reported "~3.5M beneficiaries" in the BCR
denominator logic with a WorldPop-based resident population of treated
townships, per buffer specification. 2020 vintage = pre-shock resident
population (deliberate: matches pre-shock GDP-per-capita logic; post-quake
displacement makes current population unknowable).

DATA REQUIREMENT (download once, ~20 MB @100m for Myanmar is large; 1km ok too):
  Preferred: WorldPop Constrained 2020 UN-adjusted 100m, Myanmar
             hub.worldpop.org -> Population Counts -> Constrained Individual
             countries 2020 UN adjusted (100m) -> Myanmar
             file: mmr_ppp_2020_UNadj_constrained.tif
  Also accepted: mmr_ppp_2020_UNadj.tif (unconstrained UN-adjusted, HDX:
             https://data.humdata.org/dataset/worldpop-population-counts-for-myanmar)
  Place the .tif in: data/mmr/population/

CITATION (for the paper):
  WorldPop (www.worldpop.org - School of Geography and Environmental Science,
  University of Southampton). 2020. Constrained individual countries 2020
  UN adjusted (100m resolution) - Myanmar. CC BY 4.0.
  UN adjustment: 2019 Revision of World Population Prospects (UNPD).

OUTPUTS
-------
  outputs/mmr/results/population_worldpop.csv   (per buffer spec)
  outputs/mmr/results/population_estimates.json (mirrors Gaza's
      beneficiary_estimates.json, incl. recomputed BCR grid)

Usage:  .venv/bin/python analysis/mmr/05_population_worldpop.py
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import date

import rasterio
from rasterio.mask import mask as rio_mask

# =============================================================================
# CONFIGURATION — paths identical in spirit to 02_pretrends_verified.py
# =============================================================================
DATA_DIR = Path("data/mmr")
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"
POPULATION_DIR = DATA_DIR / "population"

RESULTS_DIR = Path("outputs/mmr/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Same six specs as the DiD (did_buffer_results.csv), incl. underpowered ones
BUFFER_SPECS = [
    (500,   1500,  "0.5km-1.5km", "ultra_tight"),
    (1000,  3000,  "1km-3km",     "tight"),
    (2000,  5000,  "2km-5km",     "moderate"),
    (3000,  8000,  "3km-8km",     "baseline"),
    (5000, 15000,  "5km-15km",    "wide"),
    (8000, 20000,  "8km-20km",    "very_wide"),
]

# Valuation parameters (mirror roi_evidence/CALCULATIONS.md)
ELASTICITIES = {"low": 0.15, "medium": 0.3, "high": 0.5}
DID_PCT_RANGE = (31.9, 53.2)          # exact powered range, Table 3
GDP_PER_CAPITA = 1122                  # WDI, latest pre-shock year (Myanmar)
COST_BASE = 16_783_198                 # full active ER portfolio (undp_projects.csv)
AGENCY_REPORTED_N = 3_500_000          # for comparison only — unverified


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_worldpop_raster():
    if not POPULATION_DIR.exists():
        return None
    tifs = sorted(POPULATION_DIR.glob("mmr_ppp*UNadj*.tif")) or sorted(POPULATION_DIR.glob("*.tif"))
    return tifs[0] if tifs else None


def township_population(gdf, raster_path):
    """Sum WorldPop pixels within each township polygon. Returns {unit_id: pop}."""
    pops = {}
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for _, row in gdf.iterrows():
            try:
                out, _ = rio_mask(src, [row.geometry], crop=True, filled=True,
                                  nodata=nodata if nodata is not None else -99999)
                arr = out[0].astype("float64")
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                arr = np.where(arr < 0, np.nan, arr)  # belt & braces
                pops[row["_uid"]] = float(np.nansum(arr))
            except Exception:
                pops[row["_uid"]] = np.nan
    return pops


def main():
    print("=" * 70)
    print(" MYANMAR: WORLDPOP POPULATION OF TREATED TOWNSHIPS (Gaza-consistent)")
    print("=" * 70)

    # --- Boundaries (same id detection as fixed 02_pretrends_verified.py) ---
    gdf = gpd.read_file(BOUNDARIES_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower() or c == 'shapeID'), None)
    if id_col is None:
        print("  ❌ No pcode/shapeID column in boundaries"); return
    gdf["_uid"] = gdf[id_col]
    cent = gdf.geometry.centroid
    gdf["center_lat"], gdf["center_lon"] = cent.y, cent.x
    print(f"  ✓ Boundaries: {len(gdf)} townships (id: {id_col})")

    # --- UNDP sites ---
    sites = pd.read_csv(UNDP_SITES_FILE)
    lat_col = next((c for c in sites.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in sites.columns if 'lon' in c.lower()), None)
    sites = sites.rename(columns={lat_col: 'lat', lon_col: 'lon'}).dropna(subset=['lat', 'lon'])
    print(f"  ✓ UNDP sites: {len(sites)}")

    # --- Distances: township centroid -> nearest site ---
    slat, slon = sites['lat'].values, sites['lon'].values
    gdf["min_dist"] = [
        float(np.min(haversine_m(r.center_lat, r.center_lon, slat, slon)))
        for r in gdf.itertuples()
    ]

    # --- WorldPop raster ---
    raster = find_worldpop_raster()
    if raster is None:
        print(f"\n  ❌ No WorldPop .tif found in {POPULATION_DIR}/")
        print("     Download mmr_ppp_2020_UNadj_constrained.tif from hub.worldpop.org")
        print("     (or mmr_ppp_2020_UNadj.tif from HDX) and place it there.")
        return
    print(f"  ✓ WorldPop raster: {raster.name}")

    print("\n--- Extracting township populations (one-time, all townships) ---")
    pops = township_population(gdf, raster)
    gdf["worldpop_2020"] = gdf["_uid"].map(pops)
    total_ok = gdf["worldpop_2020"].notna().sum()
    print(f"  ✓ Extracted for {total_ok}/{len(gdf)} townships "
          f"(national sum in file extent: {gdf['worldpop_2020'].sum():,.0f})")

    # --- Per-spec treated population ---
    rows = []
    for inner_m, outer_m, label, key in BUFFER_SPECS:
        treated = gdf[gdf["min_dist"] <= inner_m]
        control = gdf[(gdf["min_dist"] > inner_m) & (gdf["min_dist"] <= outer_m)]
        tp = float(treated["worldpop_2020"].sum())
        rows.append({
            "spec_label": label, "spec_key": key,
            "inner_m": inner_m, "outer_m": outer_m,
            "n_treated_townships": len(treated),
            "n_control_townships": len(control),
            "treated_population_worldpop2020": round(tp),
            "control_population_worldpop2020": round(float(control["worldpop_2020"].sum())),
        })
        print(f"  {label:>12}: {len(treated):>3} treated townships, "
              f"pop {tp:,.0f}")

    res = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "population_worldpop.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved {out_csv}")

    # --- Recomputed BCR grid (satellite-verified N vs agency-reported N) ---
    lo_pct, hi_pct = DID_PCT_RANGE
    grid = {}
    for _, r in res.iterrows():
        N = r["treated_population_worldpop2020"]
        entry = {"N_worldpop": int(N), "n_treated_townships": int(r["n_treated_townships"])}
        for name, eps in ELASTICITIES.items():
            gain_lo = eps * (lo_pct/100) * N * GDP_PER_CAPITA
            gain_hi = eps * (hi_pct/100) * N * GDP_PER_CAPITA
            entry[f"gdp_gain_usd_{name}"] = [round(gain_lo), round(gain_hi)]
            entry[f"bcr_{name}"] = [round(gain_lo/COST_BASE, 1), round(gain_hi/COST_BASE, 1)]
        grid[r["spec_label"]] = entry

    payload = {
        "method": "WorldPop zonal sum over treated township polygons "
                  "(consistent with Gaza's WorldPop buffer method)",
        "raster_file": raster.name,
        "raster_vintage": "2020 (pre-shock resident population, deliberate)",
        "citation": ("WorldPop (www.worldpop.org - School of Geography and "
                     "Environmental Science, University of Southampton). 2020. "
                     "Individual countries 2020 UN adjusted (100m) - Myanmar. CC BY 4.0. "
                     "UN adjustment: WPP 2019 (UNPD)."),
        "access_date": str(date.today()),
        "valuation_params": {
            "did_pct_range_exact": list(DID_PCT_RANGE),
            "elasticities": ELASTICITIES,
            "gdp_per_capita_usd": GDP_PER_CAPITA,
            "cost_base_usd": COST_BASE,
            "agency_reported_N_for_comparison": AGENCY_REPORTED_N,
        },
        "per_spec": grid,
    }
    out_json = RESULTS_DIR / "population_estimates.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ Saved {out_json}")

    print("\n" + "=" * 70)
    print(" HEADLINE (widest powered spec, 8km-20km, medium elasticity 0.3):")
    w = grid.get("8km-20km", {})
    if w:
        print(f"   Satellite-verified treated population: {w['N_worldpop']:,}")
        print(f"   GDP gain: ${w['gdp_gain_usd_medium'][0]:,} - ${w['gdp_gain_usd_medium'][1]:,}")
        print(f"   BCR: {w['bcr_medium'][0]}:1 - {w['bcr_medium'][1]}:1")
        print(f"   (Agency-reported 3.5M gave 21-39:1 — compare and report both.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
