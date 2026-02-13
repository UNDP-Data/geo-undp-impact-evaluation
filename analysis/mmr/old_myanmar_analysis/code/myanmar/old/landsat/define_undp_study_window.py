#!/usr/bin/env python3
"""
Define a "central Myanmar study window" around UNDP interventions.

Steps:
  1. Read UNDP interventions Excel file.
  2. Extract unique treated townships from a township column (e.g. 'L3. Township').
  3. Read Myanmar admin3 boundaries.
  4. Match treated township names to admin3 polygons.
  5. Compute convex hull of treated township centroids and buffer it in km.
  6. Use the buffered hull as the "study window" geometry.
  7. Select all admin3 polygons that intersect the study window.
  8. Flag those that are treated (UNDP=1) vs controls (UNDP=0).
  9. Save:
       - a CSV with township IDs, names, and treatment flag
       - a GeoJSON with polygons for the study window townships.

You can then:
  - use this CSV to run Landsat only on these ~100 townships,
  - use the GeoJSON to visualize the window or to clip rasters.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def parse_args():
    p = argparse.ArgumentParser(
        description="Define study window around UNDP-treated townships in Myanmar."
    )
    p.add_argument(
        "--undp-excel",
        required=True,
        help="Path to UNDP interventions Excel file "
             "(e.g. Myanmar_Early_Recovery_Needs_Assessment_*.xlsx).",
    )
    p.add_argument(
        "--undp-township-col",
        default="L3. Township",
        help="Column in the UNDP Excel with township names (default: 'L3. Township').",
    )
    p.add_argument(
        "--admin3-boundaries",
        required=True,
        help="Path to Myanmar admin3 boundaries (GeoJSON/Shapefile).",
    )
    p.add_argument(
        "--admin3-name-col",
        required=True,
        help="Column in the admin3 file with township names "
             "that correspond (as closely as possible) to the UNDP township names.",
    )
    p.add_argument(
        "--admin3-id-col",
        default=None,
        help="Optional ID column for admin3 (e.g. 'ADM3_PCODE'). If omitted, row index is used.",
    )
    p.add_argument(
        "--buffer-km",
        type=float,
        default=50.0,
        help="Buffer radius around the convex hull of treated centroids, in km "
             "(default: 50 km).",
    )
    p.add_argument(
        "--output-townships-csv",
        required=True,
        help="Path for CSV output with selected townships and treatment flag.",
    )
    p.add_argument(
        "--output-townships-geojson",
        required=True,
        help="Path for GeoJSON output with selected township polygons and treatment flag.",
    )
    return p.parse_args()


def load_undp_treated_townships(excel_path: Path, township_col: str):
    print(f"📁 Loading UNDP Excel from: {excel_path}")
    df = pd.read_excel(excel_path)
    if township_col not in df.columns:
        raise ValueError(f"Column '{township_col}' not found in UNDP Excel.")

    treated_townships = (
        df[township_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    treated_townships = np.sort(treated_townships)
    print(f"   ✓ Found {len(treated_townships)} unique treated townships in '{township_col}'.")
    return treated_townships


def load_admin3(boundaries_path: Path, name_col: str):
    print(f"📁 Loading admin3 boundaries from: {boundaries_path}")
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        raise ValueError("Admin3 file has no CRS; please define one (likely EPSG:4326).")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    if name_col not in gdf.columns:
        raise ValueError(f"Column '{name_col}' not found in admin3 file.")

    print(f"   ✓ Loaded {len(gdf)} polygons in EPSG:4326")
    return gdf


def km_to_deg(km: float) -> float:
    """
    Rough conversion: 1 degree ≈ 111 km.
    Good enough for defining a buffer around a study area.
    """
    return km / 111.0


def main():
    args = parse_args()
    undp_excel = Path(args.undp_excel)
    admin3_path = Path(args.admin3_boundaries)
    out_csv = Path(args.output_townships_csv)
    out_geojson = Path(args.output_townships_geojson)

    treated_names = load_undp_treated_townships(
        undp_excel,
        township_col=args.undp_township_col,
    )

    gdf_admin3 = load_admin3(
        admin3_path,
        name_col=args.admin3_name_col,
    )

    # Normalise name column a bit
    gdf_admin3["_name_clean"] = (
        gdf_admin3[args.admin3_name_col]
        .astype(str)
        .str.strip()
    )

    # Match treated townships to admin3 polygons
    gdf_treated = gdf_admin3[gdf_admin3["_name_clean"].isin(treated_names)].copy()
    print(f"   ✓ Matched {len(gdf_treated)} admin3 polygons to the {len(treated_names)} treated names.")
    unmatched = set(treated_names) - set(gdf_treated["_name_clean"].unique())
    if unmatched:
        print("   ⚠️ WARNING: Some UNDP township names didn't match admin3 boundaries:")
        for n in sorted(unmatched):
            print(f"      - {n}")
        print("     You may need to adjust admin3_name_col or manually check spellings.")

    if gdf_treated.empty:
        raise RuntimeError("No treated admin3 polygons matched UNDP township names. Aborting.")

    # Compute centroids and convex hull of treated townships
    print("📐 Computing convex hull of treated township centroids...")
    centroids = gdf_treated.geometry.centroid
    hull = centroids.unary_union.convex_hull

    # Buffer hull in km (converted to degrees)
    buffer_deg = km_to_deg(args.buffer_km)
    print(f"   ✓ Buffering convex hull by {args.buffer_km} km ≈ {buffer_deg:.3f} degrees")
    study_geom = hull.buffer(buffer_deg)

    # Select all admin3 polygons that intersect the buffered hull
    print("🧩 Selecting admin3 polygons inside the study window...")
    gdf_admin3["in_window"] = gdf_admin3.geometry.intersects(study_geom)
    gdf_window = gdf_admin3[gdf_admin3["in_window"]].copy()
    print(f"   ✓ Selected {len(gdf_window)} admin3 polygons in the study window.")

    # Flag treatment within study window
    gdf_window["_name_clean"] = gdf_window["_name_clean"].astype(str)
    gdf_window["undp_treated"] = gdf_window["_name_clean"].isin(treated_names).astype(int)

    n_treated_window = int(gdf_window["undp_treated"].sum())
    n_control_window = len(gdf_window) - n_treated_window
    print(f"   • Treated in window: {n_treated_window}")
    print(f"   • Control in window: {n_control_window}")

    # Build output table
    id_col = args.admin3_id_col if args.admin3_id_col and args.admin3_id_col in gdf_window.columns else None
    if id_col is None:
        gdf_window["admin3_id"] = gdf_window.index.astype(str)
    else:
        gdf_window["admin3_id"] = gdf_window[id_col].astype(str)

    gdf_window["admin3_name"] = gdf_window[args.admin3_name_col].astype(str)

    df_out = gdf_window[
        ["admin3_id", "admin3_name", "undp_treated"]
    ].copy()

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"💾 Saved township list with treatment flag to: {out_csv}")

    # Save GeoJSON with geometry
    gdf_out = gdf_window[
        ["admin3_id", "admin3_name", "undp_treated", "geometry"]
    ].copy()
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(out_geojson, driver="GeoJSON")
    print(f"💾 Saved study-window admin3 polygons to: {out_geojson}")

    print("\n✅ Done. You can now:")
    print("   1) Use the CSV to run Landsat only on these townships.")
    print("   2) Use the GeoJSON to visualise the band and choose controls.")


if __name__ == "__main__":
    main()
