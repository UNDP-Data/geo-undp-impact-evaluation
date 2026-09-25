#!/usr/bin/env python3
"""
Black Marble daily nighttime lights (NASA VNP46A2) — pipeline stage.

Pulls the daily, moonlight/atmosphere-corrected NTL product from Google
Earth Engine (NASA/VIIRS/002/VNP46A2), quality-filters it
(Mandatory_Quality_Flag == 0), aggregates to monthly means per admin unit,
and writes data/{code}/external/blackmarble_ntl.csv with columns:
    unit_id, period_start, bm_nightlights

Why this exists: the standard monthly VIIRS composite lags ~3-4 months.
VNP46A2 lags roughly one week, giving near-current nightlight coverage
after a fresh crisis. It is kept as a SEPARATE column (bm_nightlights),
never spliced into the 'nightlights' series — the two products are not
radiometrically interchangeable (validated r≈0.95 cross-sectionally on
Venezuela, 2026-08, but treated as distinct indicators by design).

Resilience: this stage must NEVER break the pipeline. Any failure prints
a warning and exits 0.

Usage:
    python -m pipeline.blackmarble_download --config config/ven.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def init_gee():
    """Service-account auth, handling both file paths and raw JSON strings."""
    import ee

    _gee_project = os.environ.get("GEE_PROJECT", "") or None
    _sa_env = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    _creds = None

    # 1. Handle raw JSON string
    if _sa_env.startswith("{"):
        _sa_email = json.loads(_sa_env).get("client_email")
        _creds = ee.ServiceAccountCredentials(_sa_email, key_data=_sa_env)

    # 2. Handle file path (with fallbacks)
    else:
        _sa_path = _sa_env
        if not _sa_path:
            for _cand in [
                "gee_service_account.json",
                os.path.join(os.path.expanduser("~"), ".undp_dashboard", "gee_service_account.json")
            ]:
                if os.path.exists(_cand):
                    _sa_path = _cand
                    break

        if _sa_path and os.path.exists(_sa_path):
            with open(_sa_path) as _f:
                _sa_email = json.load(_f).get("client_email")
            _creds = ee.ServiceAccountCredentials(_sa_email, _sa_path)

    # 3. Initialize
    if _creds:
        if _gee_project:
            ee.Initialize(_creds, project=_gee_project)
        else:
            ee.Initialize(_creds)
        print("✓ GEE Initialized (service account)")
    else:
        ee.Initialize()
        print("✓ GEE Initialized (default credentials)")

    return ee


BAND = "Gap_Filled_DNB_BRDF_Corrected_NTL"
QF = "Mandatory_Quality_Flag"
COLLECTION = "NASA/VIIRS/002/VNP46A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    code = cfg.get("country_code", "xxx").lower()
    name = cfg.get("country_name", code)
    start = cfg.get("study_period_start")
    b = cfg.get("boundaries", {})
    bpath = Path(b.get("file_path", ""))
    id_col = b.get("id_column", "shapeID")
    name_col = b.get("name_column", "shapeName")

    out_dir = Path(f"data/{code}/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "blackmarble_ntl.csv"

    print("=" * 60)
    print(f"BLACK MARBLE DAILY NTL (VNP46A2): {name}")
    print("=" * 60)

    if not bpath.exists():
        print(f"  ⚠ Boundaries not found ({bpath}) — skipping Black Marble stage")
        return 0

    ee = init_gee()
    import geopandas as gpd

    gdf = gpd.read_file(bpath)[[id_col, name_col, "geometry"]]
    print(f"  Admin units: {len(gdf)}")

    # end = today; start = config study start (daily archive begins 2012)
    end = pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    print(f"  Window: {start_ts.date()} to {end.date()}")

    feats = [ee.Feature(ee.Geometry(geom.__geo_interface__), {"uid": str(uid)})
             for uid, geom in zip(gdf[id_col], gdf.geometry.simplify(0.01))]
    fc = ee.FeatureCollection(feats)

    coll = ee.ImageCollection(COLLECTION).filterDate(str(start_ts.date()),
                                                     str((end + pd.Timedelta(days=1)).date()))
    n = coll.size().getInfo()
    if n == 0:
        print("  ⚠ No VNP46A2 images in window — skipping")
        return 0
    latest = ee.Date(coll.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"  Daily images: {n} | most recent: {latest}")


    rows = []
    months = pd.period_range(start_ts, end, freq="M")
    for m in months:
        s = str(m.start_time.date())
        e = str((m.end_time + pd.Timedelta(days=1)).date())
        msub = coll.filterDate(s, e)
        try:
            if msub.size().getInfo() == 0:
                continue

            def qmask(img):
                return img.select(BAND).updateMask(img.select(QF).eq(0))

            comp = msub.map(qmask).mean()
            stats = comp.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                       scale=500).getInfo()
            got = 0
            for f in stats["features"]:
                v = f["properties"].get("mean")
                if v is not None:
                    rows.append({"unit_id": f["properties"]["uid"],
                                 "period_start": f"{m}-01",
                                 "bm_nightlights": v})
                    got += 1
            print(f"  > {m}... ✓ ({got} units)", flush=True)
        except Exception as e2:
            print(f"  > {m}... ⚠ {e2}", flush=True)

    if not rows:
        print("  ⚠ No data retrieved — skipping write")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("=" * 60)
    print(f"✓ Saved: {out_file} ({len(df)} rows, "
          f"{df.period_start.min()[:7]} to {df.period_start.max()[:7]})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Experimental stage: never break the pipeline.
        print(f"  ⚠ Black Marble stage failed ({e}) — continuing without it")
        sys.exit(0)
