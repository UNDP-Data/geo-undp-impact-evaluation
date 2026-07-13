#!/usr/bin/env python3
"""
Build Landsat-based township x period panel for Myanmar earthquake analysis.

For each admin3 polygon and each period, this script:
  - Queries Planetary Computer STAC (Landsat Collection 2 Level-2).
  - Restricts to Landsat-8 and Landsat-9 only.
  - Filters by scene-level cloud cover (eo:cloud_cover < max_cloud).
  - Caps the number of scenes per admin3 x period (max_items_per_cell).
  - Loads scenes with odc.stac.load clipped to the township polygon.
  - Applies QA_PIXEL-based cloud/shadow mask.
  - Computes NDVI, NDBI, NBR, NDMI.
  - Takes the median over time (per pixel), then mean over space (per township).
  - Records summary stats in a tidy CSV.

Output columns (per row):
  admin3_row, admin3_id, admin3_name,
  period_code, period_label, period_start, period_end,
  ndvi_mean, ndbi_mean, nbr_mean, ndmi_mean,
  n_scenes_used, valid_fraction
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer as pc
import odc.stac


# ---------------------------------------------------------------------
# Period definitions
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_admin3(boundaries_path: Path) -> gpd.GeoDataFrame:
    print(f"📁 Loading admin3 boundaries from: {boundaries_path}")
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        raise ValueError("Boundaries file has no CRS; please define one (likely EPSG:4326).")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    print(f"   ✓ Loaded {len(gdf)} polygons in EPSG:4326")
    return gdf


def create_clear_mask(qa_pixel: xr.DataArray) -> xr.DataArray:
    """
    Conservative QA_PIXEL-based clear mask.

    Marks as bad: fill, dilated cloud, cirrus, cloud, cloud shadow, snow (bits 0–5).
    """
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
        raise ValueError(f"Dataset missing required bands: {missing}")

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
    nbr = safe(nir - swir2, nir + swir2).where(clear)
    ndmi = safe(nir - swir1, nir + swir1).where(clear)

    out = xr.Dataset(
        {
            "NDVI": ndvi,
            "NDBI": ndbi,
            "NBR": nbr,
            "NDMI": ndmi,
        }
    )
    return out


def spatial_mean_indices(med: xr.Dataset) -> dict:
    """
    Take mean over spatial dimensions of the median composite for each index.

    Returns a dict:
      {
        "ndvi_mean": float or np.nan,
        "ndbi_mean": ...,
        "nbr_mean": ...,
        "ndmi_mean": ...
      }
    """
    if "NDVI" not in med.data_vars:
        return dict(ndvi_mean=np.nan, ndbi_mean=np.nan, nbr_mean=np.nan, ndmi_mean=np.nan)

    # Determine spatial dimensions (usually y, x)
    ndvi = med["NDVI"]
    spatial_dims = [d for d in ndvi.dims if d not in ("time", "band")]

    if not spatial_dims:
        # No spatial dimension? Unusual, but handle gracefully.
        return dict(
            ndvi_mean=float(ndvi.values),
            ndbi_mean=float(med["NDBI"].values),
            nbr_mean=float(med["NBR"].values),
            ndmi_mean=float(med["NDMI"].values),
        )

    ndvi_mean = float(ndvi.mean(dim=spatial_dims, skipna=True).values)
    ndbi_mean = float(med["NDBI"].mean(dim=spatial_dims, skipna=True).values)
    nbr_mean = float(med["NBR"].mean(dim=spatial_dims, skipna=True).values)
    ndmi_mean = float(med["NDMI"].mean(dim=spatial_dims, skipna=True).values)

    return dict(
        ndvi_mean=ndvi_mean,
        ndbi_mean=ndbi_mean,
        nbr_mean=nbr_mean,
        ndmi_mean=ndmi_mean,
    )


def summarize_valid_fraction(med: xr.Dataset) -> float:
    """
    Share of pixels with a finite NDVI value in the median composite.
    """
    ndvi = med["NDVI"]
    vals = ndvi.values
    if vals.size == 0:
        return 0.0
    valid = np.isfinite(vals)
    return float(valid.sum()) / float(valid.size)


def process_admin3_period(
    admin3_idx: int,
    admin3_geom,
    period: Period,
    catalog: Client,
    resolution_deg: float,
    max_cloud: float,
    max_items_per_cell: int,
) -> dict:
    """
    Query STAC & compute indices for a single admin3 polygon and one period.
    """
    geom_geojson = mapping(admin3_geom)

    search = catalog.search(
        collections=[COLLECTION],
        intersects=geom_geojson,
        datetime=f"{period.start}/{period.end}",
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )

    items = list(search.items())
    n_items = len(items)

    if n_items == 0:
        return {
            "n_scenes_used": 0,
            "valid_fraction": 0.0,
            "ndvi_mean": np.nan,
            "ndbi_mean": np.nan,
            "nbr_mean": np.nan,
            "ndmi_mean": np.nan,
        }

    # Sort by cloud cover and keep the best scenes
    # (lower cloud cover first)
    try:
        items_sorted = sorted(
            items,
            key=lambda it: it.properties.get("eo:cloud_cover", 100.0)
        )
    except Exception:
        items_sorted = items

    items_kept = items_sorted[:max_items_per_cell]
    n_used = len(items_kept)

    # Sign each item for Planetary Computer access
    signed_items = [pc.sign(it) for it in items_kept]

    ds = odc.stac.load(
        signed_items,
        bands=["red", "nir08", "swir16", "swir22", "qa_pixel"],
        geopolygon=admin3_geom,
        groupby="solar_day",
        resolution=resolution_deg,
        chunks={"time": 1, "x": 256, "y": 256},
    )

    if ds.time.size == 0:
        return {
            "n_scenes_used": 0,
            "valid_fraction": 0.0,
            "ndvi_mean": np.nan,
            "ndbi_mean": np.nan,
            "nbr_mean": np.nan,
            "ndmi_mean": np.nan,
        }

    indices = compute_indices(ds)
    med = indices.median(dim="time", skipna=True)

    valid_fraction = summarize_valid_fraction(med)
    means = spatial_mean_indices(med)
    means["n_scenes_used"] = n_used
    means["valid_fraction"] = valid_fraction
    return means


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build township x period Landsat panel for Myanmar"
    )
    p.add_argument(
        "--boundaries",
        required=True,
        help="Path to Myanmar admin3 boundaries (GeoJSON/Shapefile).",
    )
    p.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file.",
    )
    p.add_argument(
        "--admin3-id-col",
        default=None,
        help="Column name for admin3 ID (optional). If omitted, row index is used.",
    )
    p.add_argument(
        "--admin3-name-col",
        default=None,
        help="Column name for admin3 name (optional).",
    )
    p.add_argument(
        "--resolution-deg",
        type=float,
        default=0.02,
        help="Spatial resolution in degrees (default 0.02 ≈ ~2 km).",
    )
    p.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum allowable scene-level cloud cover percentage.",
    )
    p.add_argument(
        "--max-items-per-cell",
        type=int,
        default=80,
        help="Maximum number of scenes per admin3 x period.",
    )
    p.add_argument(
        "--limit-n-admin3",
        type=int,
        default=None,
        help="If set, only process the first N admin3 polygons (useful for testing).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    boundaries_path = Path(args.boundaries)
    output_csv = Path(args.output_csv)

    gdf = load_admin3(boundaries_path)

    # Optional: subset for testing
    if args.limit_n_admin3 is not None:
        gdf = gdf.iloc[: args.limit_n_admin3].copy()
        print(f"   ⚠️ Limiting to first {len(gdf)} admin3 polygons for this run.")

    admin3_id_col: Optional[str] = args.admin3_id_col if args.admin3_id_col else None
    admin3_name_col: Optional[str] = args.admin3_name_col if args.admin3_name_col else None

    catalog = Client.open(STAC_URL)

    rows = []
    total_cells = len(gdf) * len(PERIODS)
    cell_counter = 0

    print("\n🚀 Starting admin3 x period panel construction...")
    print(f"   Total cells to process: {total_cells}")

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            print(f"   ⚠️ Skipping admin3 row {idx} due to empty geometry.")
            continue

        admin3_id = row[admin3_id_col] if admin3_id_col and admin3_id_col in row else idx
        admin3_name = row[admin3_name_col] if admin3_name_col and admin3_name_col in row else None

        print(f"\n🧩 Admin3 idx={idx}, id={admin3_id}, name={admin3_name}")

        for period in PERIODS:
            cell_counter += 1
            print(f"   → [{cell_counter}/{total_cells}] Period {period.code}: {period.label}")

            stats = process_admin3_period(
                admin3_idx=idx,
                admin3_geom=geom,
                period=period,
                catalog=catalog,
                resolution_deg=args.resolution_deg,
                max_cloud=args.max_cloud,
                max_items_per_cell=args.max_items_per_cell,
            )

            row_out = {
                "admin3_row": int(idx),
                "admin3_id": admin3_id,
                "admin3_name": admin3_name,
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
            }
            rows.append(row_out)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Done. Wrote {len(df)} rows to: {output_csv}")


if __name__ == "__main__":
    main()
