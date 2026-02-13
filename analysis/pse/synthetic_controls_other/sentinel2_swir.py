#!/usr/bin/env python3
"""
pipeline/sentinel2_swir.py
==========================
Download Sentinel-2 SWIR bands and calculate BSI/NDBI indices.

These indices are better for debris/built-up area detection than VH/VV:
- BSI (Bare Soil Index): Measures exposed soil after debris clearing
- NDBI (Normalized Difference Built-up Index): Measures built-up density

Formulas:
    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

Sentinel-2 Bands:
    Band 2 (Blue): 10m
    Band 4 (Red): 10m  
    Band 8 (NIR): 10m
    Band 11 (SWIR1): 20m → needs resampling to 10m
    Band 12 (SWIR2): 20m

Usage:
    python -m pipeline.sentinel2_swir --region gaza
    python -m pipeline.sentinel2_swir --region gaza --start 2023-01-01 --end 2025-12-31
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import planetary_computer as pc
    import pystac_client
    HAS_PC = True
except ImportError:
    HAS_PC = False
    print("⚠ planetary_computer not installed. Run: pip install planetary-computer pystac-client")

try:
    import rioxarray
    import xarray as xr
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False

try:
    import stackstac
    HAS_STACKSTAC = True
except ImportError:
    HAS_STACKSTAC = False

try:
    from odc.stac import load as odc_load
    HAS_ODC = True
except ImportError:
    HAS_ODC = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "pse" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region definitions
REGIONS = {
    'gaza': {
        'bbox': [34.2, 31.2, 34.6, 31.6],
        'name': 'Gaza Strip',
        'admin_ids': ['PS0201', 'PS0202', 'PS0203', 'PS0204', 'PS0205'],
    },
    'west_bank': {
        'bbox': [34.8, 31.3, 35.6, 32.6],
        'name': 'West Bank',
    },
}

# Governorate boundaries (approximate polygons for zonal stats)
GAZA_GOVERNORATES = {
    'PS0201': {'name': 'North Gaza', 'lat_range': (31.50, 31.60), 'lon_range': (34.45, 34.55)},
    'PS0202': {'name': 'Gaza City', 'lat_range': (31.45, 31.55), 'lon_range': (34.40, 34.50)},
    'PS0203': {'name': 'Deir al-Balah', 'lat_range': (31.38, 31.48), 'lon_range': (34.30, 34.42)},
    'PS0204': {'name': 'Khan Younis', 'lat_range': (31.30, 31.40), 'lon_range': (34.25, 34.38)},
    'PS0205': {'name': 'Rafah', 'lat_range': (31.20, 31.32), 'lon_range': (34.20, 34.30)},
}

# Sentinel-2 bands needed
S2_BANDS = ['B02', 'B04', 'B08', 'B11', 'B12', 'SCL']  # Blue, Red, NIR, SWIR1, SWIR2, Scene Classification


def search_sentinel2(bbox, start_date, end_date, max_cloud=30):
    """Search for Sentinel-2 L2A scenes."""
    if not HAS_PC:
        raise ImportError("planetary_computer required")
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    
    items = list(search.items())
    print(f"  Found {len(items)} Sentinel-2 scenes")
    
    return items


def calculate_indices(blue, red, nir, swir1, swir2):
    """
    Calculate spectral indices from Sentinel-2 bands.
    
    Returns dict with: ndvi, ndbi, bsi, ndwi, ndbai
    """
    # Avoid division by zero
    eps = 1e-10
    
    # NDVI (Normalized Difference Vegetation Index)
    ndvi = (nir - red) / (nir + red + eps)
    
    # NDBI (Normalized Difference Built-up Index)
    # Higher values = more built-up area
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    
    # BSI (Bare Soil Index)
    # Higher values = more exposed soil (cleared debris)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
    
    # NDWI (Normalized Difference Water Index)
    ndwi = (nir - swir1) / (nir + swir1 + eps)
    
    # NDBaI (Normalized Difference Bareness Index) - alternative for bare soil
    ndbai = (swir1 - swir2) / (swir1 + swir2 + eps)
    
    # UI (Urban Index)
    ui = (swir2 - nir) / (swir2 + nir + eps)
    
    return {
        'ndvi': ndvi,
        'ndbi': ndbi,
        'bsi': bsi,
        'ndwi': ndwi,
        'ndbai': ndbai,
        'ui': ui,
    }


def process_scene_stackstac(item, bbox, resolution=20):
    """Process a single Sentinel-2 scene using stackstac (more reliable)."""
    if not HAS_STACKSTAC:
        raise ImportError("stackstac required: pip install stackstac")
    
    try:
        # Sign the item
        signed_item = pc.sign(item)
        
        # Load with stackstac - handles mixed resolutions better
        data = stackstac.stack(
            [signed_item],
            assets=['B02', 'B04', 'B08', 'B11', 'B12', 'SCL'],
            bounds_latlon=bbox,
            resolution=resolution,
            epsg=4326,
            resampling=1,  # bilinear for reflectance bands
        ).compute()
        
        # Extract bands (scale to reflectance)
        blue = data.sel(band='B02').values.astype(float) / 10000
        red = data.sel(band='B04').values.astype(float) / 10000
        nir = data.sel(band='B08').values.astype(float) / 10000
        swir1 = data.sel(band='B11').values.astype(float) / 10000
        swir2 = data.sel(band='B12').values.astype(float) / 10000
        scl = data.sel(band='SCL').values
        
        # Cloud mask
        cloud_mask = np.isin(scl, [3, 8, 9, 10])
        
        # Calculate indices
        indices = calculate_indices(blue, red, nir, swir1, swir2)
        
        # Apply cloud mask
        for key in indices:
            indices[key] = np.where(cloud_mask, np.nan, indices[key])
        
        return indices, data.coords
        
    except Exception as e:
        raise RuntimeError(f"stackstac load failed: {e}")


def process_scene_odc(item, bbox, resolution=20):
    """Process a single Sentinel-2 scene using odc-stac."""
    if not HAS_ODC:
        raise ImportError("odc-stac required: pip install odc-stac")
    
    try:
        # Load bands - need to sign the item first
        signed_item = pc.sign(item)
        
        data = odc_load(
            [signed_item],
            bands=S2_BANDS,
            bbox=bbox,
            resolution=resolution,
            crs="EPSG:4326",
            chunks={},  # Load into memory
        )
        
        # Extract bands (scale to reflectance)
        blue = data['B02'].values.astype(float) / 10000
        red = data['B04'].values.astype(float) / 10000
        nir = data['B08'].values.astype(float) / 10000
        swir1 = data['B11'].values.astype(float) / 10000
        swir2 = data['B12'].values.astype(float) / 10000
        scl = data['SCL'].values
        
        # Cloud mask (SCL: 3=shadow, 8=cloud_medium, 9=cloud_high, 10=cirrus)
        cloud_mask = np.isin(scl, [3, 8, 9, 10])
        
        # Calculate indices
        indices = calculate_indices(blue, red, nir, swir1, swir2)
        
        # Apply cloud mask
        for key in indices:
            indices[key] = np.where(cloud_mask, np.nan, indices[key])
        
        return indices, data.coords
        
    except Exception as e:
        raise RuntimeError(f"ODC load failed: {e}")


def compute_zonal_stats(indices, governorates, coords):
    """Compute zonal statistics per governorate."""
    results = []
    
    # Get lat/lon from coords
    if 'latitude' in coords:
        lats = coords['latitude'].values
        lons = coords['longitude'].values
    elif 'y' in coords:
        lats = coords['y'].values
        lons = coords['x'].values
    else:
        print("  ⚠ Could not determine coordinates")
        return results
    
    for admin_id, gov_info in governorates.items():
        lat_min, lat_max = gov_info['lat_range']
        lon_min, lon_max = gov_info['lon_range']
        
        # Create mask for this governorate
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        
        # Handle 2D vs 1D coordinates
        if lats.ndim == 1:
            # Create 2D mask from 1D coords
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                continue
            lat_slice = slice(lat_idx.min(), lat_idx.max() + 1)
            lon_slice = slice(lon_idx.min(), lon_idx.max() + 1)
        else:
            # 2D coordinates
            mask = lat_mask & lon_mask
        
        row = {'unit_id': admin_id, 'admin_name': gov_info['name']}
        
        for idx_name, idx_data in indices.items():
            if idx_data.ndim == 3:
                # Time dimension present
                if lats.ndim == 1:
                    subset = idx_data[:, lat_slice, lon_slice]
                else:
                    subset = idx_data[:, mask]
            else:
                if lats.ndim == 1:
                    subset = idx_data[lat_slice, lon_slice]
                else:
                    subset = idx_data[mask]
            
            row[idx_name] = np.nanmean(subset)
            row[f'{idx_name}_std'] = np.nanstd(subset)
        
        results.append(row)
    
    return results


def process_monthly_composites(items, bbox, governorates, start_date, end_date):
    """Process scenes into monthly composites with zonal stats."""
    
    # Group items by month
    items_by_month = {}
    for item in items:
        dt = pd.to_datetime(item.datetime)
        month_key = dt.strftime('%Y-%m')
        if month_key not in items_by_month:
            items_by_month[month_key] = []
        items_by_month[month_key].append(item)
    
    print(f"  Processing {len(items_by_month)} months...")
    
    # Determine which method to use
    if HAS_STACKSTAC:
        process_func = process_scene_stackstac
        print("  Using: stackstac")
    elif HAS_ODC:
        process_func = process_scene_odc
        print("  Using: odc-stac")
    else:
        print("  ⚠ No processing library available!")
        print("  Install: pip install stackstac --break-system-packages")
        return pd.DataFrame()
    
    all_results = []
    errors = []
    
    for month, month_items in sorted(items_by_month.items()):
        print(f"    {month}: {len(month_items)} scenes...", end=' ', flush=True)
        
        # Process first good scene for this month
        last_error = None
        for item in month_items[:3]:  # Try up to 3 scenes
            try:
                indices, coords = process_func(item, bbox)
                
                # Compute zonal stats
                stats = compute_zonal_stats(indices, governorates, coords)
                
                if stats:
                    for row in stats:
                        row['period_start'] = pd.Timestamp(f"{month}-01")
                    all_results.extend(stats)
                    print("✓")
                    break
                    
            except Exception as e:
                last_error = str(e)
                continue
        else:
            print(f"⚠ failed")
            if last_error:
                errors.append(f"{month}: {last_error}")
    
    # Print first few errors for debugging
    if errors and not all_results:
        print("\n  Sample errors:")
        for err in errors[:3]:
            print(f"    {err[:150]}...")
    
    return pd.DataFrame(all_results)


def download_swir_simple(region_name, start_date, end_date):
    """
    Simplified download using direct band calculation.
    Falls back to this if ODC not available.
    """
    region = REGIONS[region_name]
    bbox = region['bbox']
    
    print(f"\n--- Searching Sentinel-2 for {region['name']} ---")
    items = search_sentinel2(bbox, start_date, end_date)
    
    if not items:
        print("  No scenes found")
        return None
    
    if HAS_ODC:
        print("\n--- Processing with ODC-STAC ---")
        df = process_monthly_composites(
            items, bbox, GAZA_GOVERNORATES, start_date, end_date
        )
    else:
        print("  ⚠ ODC-STAC not available, using fallback method")
        # Fallback: just return scene metadata for manual processing
        df = pd.DataFrame([{
            'datetime': item.datetime,
            'cloud_cover': item.properties.get('eo:cloud_cover'),
            'id': item.id,
        } for item in items])
    
    return df


def merge_with_panel(swir_df, panel_path):
    """Merge SWIR data with existing panel data."""
    if not Path(panel_path).exists():
        print(f"  ⚠ Panel file not found: {panel_path}")
        return swir_df
    
    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    # Merge on unit_id and period_start
    swir_df['period_start'] = pd.to_datetime(swir_df['period_start'])
    
    merged = panel.merge(
        swir_df[['unit_id', 'period_start', 'ndbi', 'bsi', 'ndbai', 'ui']],
        on=['unit_id', 'period_start'],
        how='left'
    )
    
    print(f"  Merged: {merged['ndbi'].notna().sum()} SWIR observations")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Download Sentinel-2 SWIR and calculate indices')
    parser.add_argument('--region', default='gaza', choices=['gaza', 'west_bank'])
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--merge-panel', action='store_true', help='Merge with existing panel data')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SENTINEL-2 SWIR DOWNLOAD & INDEX CALCULATION")
    print("=" * 70)
    print(f"""
    Indices to calculate:
    - BSI (Bare Soil Index): Debris/exposed soil detection
    - NDBI (Built-up Index): Urban/built-up area density
    - NDBaI (Bareness Index): Alternative bare soil measure
    - UI (Urban Index): Urban area detection
    """)
    
    if not HAS_PC:
        print("\n⚠ ERROR: planetary_computer not installed")
        print("  Run: pip install planetary-computer pystac-client odc-stac --break-system-packages")
        return
    
    # Download and process
    df = download_swir_simple(args.region, args.start, args.end)
    
    if df is None or len(df) == 0:
        print("\n⚠ No data retrieved")
        return
    
    # Save raw SWIR data
    output_path = OUTPUT_DIR / f'swir_indices_{args.region}.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    # Optionally merge with panel
    if args.merge_panel:
        panel_path = Path("outputs/pse/panel_data.csv")
        merged = merge_with_panel(df, panel_path)
        merged.to_csv(panel_path, index=False)
        print(f"✓ Updated: {panel_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'ndbi' in df.columns:
        print(f"\n  Records: {len(df)}")
        print(f"  Date range: {df['period_start'].min()} to {df['period_start'].max()}")
        print(f"\n  Index statistics:")
        for idx in ['ndbi', 'bsi', 'ndbai', 'ui']:
            if idx in df.columns:
                print(f"    {idx.upper()}: mean={df[idx].mean():.4f}, std={df[idx].std():.4f}")
    
    print(f"\n{'='*70}")
    print("✓ SWIR PROCESSING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
