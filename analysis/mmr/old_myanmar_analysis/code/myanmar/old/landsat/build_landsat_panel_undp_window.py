#!/usr/bin/env python3
"""
Build Landsat-based township x period panel for the UNDP study window.

Uses:
  - undp_study_window_townships.csv (admin3_id = admin3 row index, plus treatment flag)
  - mmr_admin3.geojson

For each township in the study window and each period:
  - Query Planetary Computer STAC (Landsat Collection 2 Level-2).
  - Restrict to Landsat-8 and Landsat-9 only.
  - Filter by scene-level cloud cover (eo:cloud_cover < max_cloud).
  - Cap the number of scenes per township x period (max_items_per_cell).
  - Load scenes with odc.stac.load clipped to the township polygon.
  - Apply QA_PIXEL-based cloud/shadow mask.
  - Compute NDVI, NDBI, NBR, NDMI.
  - Median over time → mean over space.
  - Save one row per township × period.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
import odc.stac


# =====================================================================
# PERIOD DEFINITIONS
# =====================================================================

@dataclass
class Period:
    code: str
    label: str
    start: str
    end: str


PERIODS: List[Period] = [
    Period("2024Q4", "Oct–Dec 2024 (extra pre)", "2024-10-01", "2024-12-31"),
    Period("PRE2025", "Jan–Feb 2025 (pre)", "2025-01-01", "2025-02-28"),
    Period("POST1", "Apr–Jun 2025 (immediate post)", "2025-04-01", "2025-06-30"),
    Period("POST2", "Jul–Sep 2025 (recovery)", "2025-07-01", "2025-09-30"),
]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_study_window_townships(csv_path: Path) -> pd.DataFrame:
    print(f"📁 Loading study-window townships from: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"admin3_id", "admin3_name", "undp_treated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Study-window CSV missing required columns: {missing}")
    print(f"   ✓ Loaded {len(df)} townships")
    return df


def load_admin3(boundaries_path: Path) -> gpd.GeoDataFrame:
    print(f"📁 Loading admin3 boundaries from: {boundaries_path}")
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        raise ValueError("Admin3 file has no CRS.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    print(f"   ✓ Loaded {len(gdf)} polygons in EPSG:4326")
    return gdf


def create_clear_mask(qa_pixel: xr.DataArray) -> xr.DataArray:
    """Conservative QA_PIXEL-based clear mask."""
    qa = qa_pixel.astype("uint16")
    bad = (
        (qa & 1)  |   # fill
        (qa & 2)  |   # dilated cloud
        (qa & 4)  |   # cirrus
        (qa & 8)  |   # cloud
        (qa & 16) |   # cloud shadow
        (qa & 32)     # snow
    ) != 0
    return ~bad


def compute_indices(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute NDVI, NDBI, NBR, NDMI from Landsat C2 L2 bands:
      - red    : 'red'
      - nir    : 'nir08'
      - swir1  : 'swir16'
      - swir2  : 'swir22'
      - qa     : 'qa_pixel'
    """
    required = {"red", "nir08", "swir16", "swir22", "qa_pixel"}
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing bands: {missing}")

    red = ds["red"].astype("float32")
    nir = ds["nir08"].astype("float32")
    swir1 = ds["swir16"].astype("float32")
    swir2 = ds["swir22"].astype("float32")
    qa = ds["qa_pixel"]

    clear = create_clear_mask(qa)

    def safe(num, den):
        x = num / den
        return x.where(np.isfinite(x))

    ndvi = safe(nir - red, nir + red).where(clear)
    ndbi = safe(swir1 - nir, swir1 + nir).where(clear)
    nbr  = safe(nir - swir2, nir + swir2).where(clear)
    ndmi = safe(nir - swir1, nir + swir1).where(clear)

    return xr.Dataset({
        "NDVI": ndvi,
        "NDBI": ndbi,
        "NBR":  nbr,
        "NDMI": ndmi,
    })


def summarize_valid_fraction(med: xr.Dataset) -> float:
    ndvi = med["NDVI"]
    vals = ndvi.values
    if vals.size == 0:
        return 0.0
    valid = np.isfinite(vals)
    return float(valid.sum()) / vals.size


def spatial_mean_indices(med: xr.Dataset) -> dict:
    ndvi = med["NDVI"]
    dims = [d for d in ndvi.dims if d not in ("time", "band")]
    if not dims:
        return {
            "ndvi_mean": float(ndvi.values),
            "ndbi_mean": float(med["NDBI"].values),
            "nbr_mean":  float(med["NBR"].values),
            "ndmi_mean": float(med["NDMI"].values),
        }
    return {
        "ndvi_mean": float(med["NDVI"].mean(dim=dims, skipna=True).values),
        "ndbi_mean": float(med["NDBI"].mean(dim=dims, skipna=True).values),
        "nbr_mean":  float(med["NBR"].mean(dim=dims, skipna=True).values),
        "ndmi_mean": float(med["NDMI"].mean(dim=dims, skipna=True).values),
    }


def process_township_period(
    admin3_id: str,
    admin3_geom,
    period: Period,
    catalog: Client,
    resolution_deg: float,
    max_cloud: float,
    max_items_per_cell: int,
):
    print(f"      → STAC search for {admin3_id}, period={period.code}")
    search = catalog.search(
        collections=[COLLECTION],
        intersects=mapping(admin3_geom),
        datetime=f"{period.start}/{period.end}",
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )

    items = list(search.items())
    if len(items) == 0:
        print("         No scenes found.")
        return dict(
            n_scenes_used=0,
            valid_fraction=0.0,
            ndvi_mean=np.nan,
            ndbi_mean=np.nan,
            nbr_mean=np.nan,
            ndmi_mean=np.nan,
        )

    items_sorted = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items_kept = items_sorted[:max_items_per_cell]
    signed_items = [pc.sign(it) for it in items_kept]

    print("         Loading via odc.stac.load...")
    ds = odc.stac.load(
        signed_items,
        bands=["red", "nir08", "swir16", "swir22", "qa_pixel"],
        geopolygon=admin3_geom,
        groupby="solar_day",
        resolution=resolution_deg,
        chunks={"time": 1, "x": 256, "y": 256},
    )

    if ds.time.size == 0:
        print("         Empty dataset after load.")
        return dict(
            n_scenes_used=0,
            valid_fraction=0.0,
            ndvi_mean=np.nan,
            ndbi_mean=np.nan,
            nbr_mean=np.nan,
            ndmi_mean=np.nan,
        )

    indices = compute_indices(ds)
    med = indices.median(dim="time", skipna=True)

    valid_fraction = summarize_valid_fraction(med)
    means = spatial_mean_indices(med)
    means["n_scenes_used"] = len(items_kept)
    means["valid_fraction"] = valid_fraction
    return means


# =====================================================================
# MAIN
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build township x period Landsat panel for UNDP study window (index-based IDs)."
    )
    p.add_argument("--study-window-csv", required=True)
    p.add_argument("--admin3-boundaries", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--resolution-deg", type=float, default=0.02)
    p.add_argument("--max-cloud", type=float, default=20.0)
    p.add_argument("--max-items-per-cell", type=int, default=30)
    p.add_argument("--limit-n-admin3", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    df_study = load_study_window_townships(Path(args.study_window_csv))
    gdf_admin3 = load_admin3(Path(args.admin3_boundaries))

    # Use admin3 row index as ID, to match 'admin3_id' in CSV
    gdf_admin3 = gdf_admin3.copy()
    gdf_admin3["admin3_id"] = gdf_admin3.index.astype(str)

    # Harmonise ID types
    df_study["admin3_id"] = df_study["admin3_id"].astype(str)
    study_ids = set(df_study["admin3_id"])

    gdf_sub = gdf_admin3[gdf_admin3["admin3_id"].isin(study_ids)].copy()
    print(f"   ✓ Matched {len(gdf_sub)} polygons to study-window IDs.")

    # Merge in admin3_name + undp_treated from the CSV
    merged = gdf_sub.merge(
        df_study[["admin3_id", "admin3_name", "undp_treated"]],
        on="admin3_id",
        how="left",
    )

    if args.limit_n_admin3 is not None:
        print(f"   ⚠️ Limiting to {args.limit_n_admin3} townships for testing.")
        merged = merged.iloc[: args.limit_n_admin3].copy()

    catalog = Client.open(STAC_URL)
    rows = []
    total_cells = len(merged) * len(PERIODS)
    counter = 0

    print(f"\n🚀 Starting Landsat panel extraction for {len(merged)} townships...")
    print(f"   Total cells = {total_cells}")

    for _, row in merged.iterrows():
        admin3_id = row["admin3_id"]
        admin3_name = row["admin3_name"]
        treated = int(row["undp_treated"])
        geom = row.geometry

        if geom is None or geom.is_empty:
            print(f"   ⚠️ Skipping {admin3_id} (empty geometry).")
            continue

        print(f"\n🧩 Township {admin3_id} | {admin3_name} | treated={treated}")

        for period in PERIODS:
            counter += 1
            print(f"   → [{counter}/{total_cells}] {period.code}")

            stats = process_township_period(
                admin3_id,
                geom,
                period,
                catalog,
                args.resolution_deg,
                args.max_cloud,
                args.max_items_per_cell,
            )

            rows.append({
                "admin3_id": admin3_id,
                "admin3_name": admin3_name,
                "undp_treated": treated,
                "period_code": period.code,
                "period_label": period.label,
                "period_start": period.start,
                "period_end": period.end,
                "ndvi_mean": stats["ndvi_mean"],
                "ndbi_mean": stats["ndbi_mean"],
                "nbr_mean": stats["nbr_mean"],
                "ndmi_mean": stats["ndmi_mean"],
                "n_scenes_used": stats["n_scenes_used"],
                "valid_fraction": stats["valid_fraction"],
            })

    df_out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"\n✅ DONE — wrote {len(df_out)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
