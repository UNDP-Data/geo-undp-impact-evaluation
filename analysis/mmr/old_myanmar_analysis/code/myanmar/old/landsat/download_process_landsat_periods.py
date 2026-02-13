#!/usr/bin/env python3
"""
Optimized multi-period Landsat compositing for Myanmar earthquake analysis.

Key features
------------
- Uses Planetary Computer STAC.
- Landsat Collection 2 Level-2, but ONLY platforms: landsat-8 and landsat-9.
- Filters for low cloud cover (default max_cloud=20).
- Caps items per tile (default max_items_per_tile=200) to avoid RAM kills.
- Splits Myanmar into 3 vertical tiles (west / centre / east) to reduce load.
- For each period and tile:
    * loads scenes
    * applies QA cloud/shadow mask
    * computes NDVI, NDBI, NBR, NDMI
    * takes median over time
- Merges tile composites into a full-country mosaic per period.
- Outputs 4-band GeoTIFFs (NDVI, NDBI, NBR, NDMI).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr

from shapely.geometry import box
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


PERIODS = [
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

def load_roi_geometry(boundaries_path: Path):
    print(f"📁 Loading ROI boundaries from: {boundaries_path}")
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        raise ValueError("Boundaries file has no CRS; please define one (likely EPSG:4326).")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    # unary_union is deprecated but still works; fine for now
    roi = gdf.unary_union
    print("   ✓ ROI loaded and unified in EPSG:4326")
    return roi


def make_tiles(roi_geometry, n_tiles: int = 3):
    """
    Split the ROI bounding box into n_tiles vertical strips (west → east) and
    intersect each with the ROI geometry. Returns a list of
    (tile_id, tile_geometry, tile_bbox).
    """
    minx, miny, maxx, maxy = roi_geometry.bounds
    width = (maxx - minx) / n_tiles
    tiles = []

    print(f"\n🧩 Creating {n_tiles} vertical tiles over ROI bbox: "
          f"({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

    for i in range(n_tiles):
        t_minx = minx + i * width
        t_maxx = minx + (i + 1) * width
        tile_box = box(t_minx, miny, t_maxx, maxy)
        tile_geom = roi_geometry.intersection(tile_box)

        if tile_geom.is_empty:
            print(f"   • Tile {i+1} has empty intersection with ROI; skipping.")
            continue

        t_bbox = tile_geom.bounds
        tiles.append((i + 1, tile_geom, t_bbox))
        print(f"   • Tile {i+1}: bbox={t_bbox}")

    return tiles


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


def median_composite(indices: xr.Dataset) -> xr.Dataset:
    """
    Take median over time for each index.
    """
    return indices.median(dim="time", skipna=True)


def write_period_geotiff(indices_med: xr.Dataset, out_path: Path):
    """
    Write mosaic to a 4-band GeoTIFF: NDVI, NDBI, NBR, NDMI.
    """
    import rioxarray

    arr = xr.concat(
        [
            indices_med["NDVI"],
            indices_med["NDBI"],
            indices_med["NBR"],
            indices_med["NDMI"],
        ],
        dim="band",
    )
    arr = arr.assign_coords(band=[1, 2, 3, 4])
    arr.rio.write_crs(4326, inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.rio.to_raster(out_path, compress="deflate")
    print(f"   💾 Saved composite to: {out_path}")


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def build_period_composite(
    period: Period,
    tiles,
    output_dir: Path,
    resolution_deg: float,
    max_cloud: float,
    max_items_per_tile: int,
    overwrite: bool,
):
    """
    For a given period, process each tile, then mosaic tiles.
    """
    out_path = output_dir / f"LANDSAT_{period.code}_{period.start}_to_{period.end}_NDVI_NDBI_NBR_NDMI.tif"

    if out_path.exists() and not overwrite:
        print(f"\n🟡 Skipping {period.code} (file already exists): {out_path.name}")
        return

    print("\n" + "=" * 80)
    print(f"🛰  Processing period: {period.code}  |  {period.label}")
    print(f"    Date range: {period.start} → {period.end}")
    print("=" * 80)

    catalog = Client.open(STAC_URL)

    tile_mosaics = []

    for tile_id, tile_geom, t_bbox in tiles:
        t_minx, t_miny, t_maxx, t_maxy = t_bbox

        print(f"\n   → Tile {tile_id}: querying STAC for bbox={t_bbox}")
        search = catalog.search(
            collections=[COLLECTION],
            bbox=[t_minx, t_miny, t_maxx, t_maxy],
            datetime=f"{period.start}/{period.end}",
            query={
                "eo:cloud_cover": {"lt": max_cloud},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )

        items = list(search.items())
        print(f"      • Found {len(items)} items before capping")

        if len(items) == 0:
            print("      ⚠️ No items for this tile; skipping tile.")
            continue

        if len(items) > max_items_per_tile:
            print(f"      ⚠️ Too many items; capping at {max_items_per_tile}")
            items = items[:max_items_per_tile]

        print("      • Signing items...")
        items = [pc.sign(item) for item in items]

        print("      ⏳ Loading data with odc.stac.load for this tile...")
        ds = odc.stac.load(
            items,
            bands=["red", "nir08", "swir16", "swir22", "qa_pixel"],
            geopolygon=tile_geom,
            groupby="solar_day",
            resolution=resolution_deg,
            chunks={"time": 1, "x": 1024, "y": 1024},
        )

        if ds.time.size == 0:
            print("      ⚠️ Loaded dataset has zero time dimension; skipping tile.")
            continue

        print(f"      • Loaded {ds.time.size} observation days for Tile {tile_id}")

        indices = compute_indices(ds)
        med = median_composite(indices)

        # Log coverage
        valid_ndvi = np.isfinite(med["NDVI"].values)
        frac_valid = valid_ndvi.sum() / valid_ndvi.size if valid_ndvi.size > 0 else 0.0
        print(f"      • Tile {tile_id} valid NDVI fraction: {frac_valid:.3f}")

        tile_mosaics.append(med)

    if not tile_mosaics:
        print("   ⚠️ No tiles produced a valid composite for this period. Skipping period.")
        return

    print("\n   🔗 Combining tiles into country mosaic...")
    # Combine by coordinates (x, y), merging tiles
    full_med = xr.combine_by_coords(tile_mosaics, combine_attrs="drop_conflicts")

    valid_ndvi_full = np.isfinite(full_med["NDVI"].values)
    frac_valid_full = valid_ndvi_full.sum() / valid_ndvi_full.size if valid_ndvi_full.size > 0 else 0.0
    print(f"   ℹ️ Full mosaic valid NDVI fraction: {frac_valid_full:.3f}")

    write_period_geotiff(full_med, out_path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Optimized tiled Landsat compositing for Myanmar")
    p.add_argument(
        "--boundaries",
        required=True,
        help="Path to Myanmar admin boundaries (GeoJSON/Shapefile). Used to define ROI.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output period composites (GeoTIFFs).",
    )
    p.add_argument(
        "--resolution-deg",
        type=float,
        default=0.02,
        help="Output resolution in degrees (default 0.02 ≈ ~2 km).",
    )
    p.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum allowable scene-level cloud cover percentage.",
    )
    p.add_argument(
        "--max-items-per-tile",
        type=int,
        default=200,
        help="Maximum number of STAC items to load per tile per period.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they exist.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    boundaries_path = Path(args.boundaries)
    output_dir = Path(args.output_dir)

    roi_geom = load_roi_geometry(boundaries_path)
    tiles = make_tiles(roi_geom, n_tiles=3)

    print("\n🚀 Starting tiled compositing pipeline...")
    print(f"   Output dir: {output_dir}")
    print(f"   Resolution: {args.resolution_deg} degrees")
    print(f"   Max scene cloud cover: {args.max_cloud}%")
    print(f"   Max items per tile: {args.max_items_per_tile}")

    for period in PERIODS:
        build_period_composite(
            period=period,
            tiles=tiles,
            output_dir=output_dir,
            resolution_deg=args.resolution_deg,
            max_cloud=args.max_cloud,
            max_items_per_tile=args.max_items_per_tile,
            overwrite=args.overwrite,
        )

    print("\n✅ All periods processed.")


if __name__ == "__main__":
    main()
