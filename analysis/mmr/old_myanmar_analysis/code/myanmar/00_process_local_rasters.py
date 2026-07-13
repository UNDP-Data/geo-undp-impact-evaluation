#!/usr/bin/env python3
"""
00_process_local_rasters.py
===========================
Extracts Zonal Statistics from Local VIIRS TIFs for Townships.
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from datetime import datetime
import re

# --- CONFIGURATION ---
VIIRS_DIR = "data/myanmar/viirs_25"
BOUNDARIES_FILE = "data/myanmar/myanmar_boundaries/mmr_admin3.geojson"
OUTPUT_DIR = "outputs/myanmar"

def parse_date_from_filename(filename):
    """
    Extracts date from filenames like:
    SVDNB_npp_20250101-20250131_global_vcmslcfg_v10_c202502181200.avg_rade9h.tif
    """
    match = re.search(r'(\d{8})-\d{8}', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None

def main():
    print("--- Starting Local Raster Processing ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Boundaries
    print(f"Loading boundaries from {BOUNDARIES_FILE}...")
    gdf = gpd.read_file(BOUNDARIES_FILE)
    
    # --- FIX: Explicitly check for 'adm3_pcode' ---
    id_col = None
    # Priority list based on your GeoJSON logs
    candidates = ['adm3_pcode', 'admin3_id', 'TS_PCODE', 'township_id', 'id']
    
    for col in candidates:
        if col in gdf.columns:
            id_col = col
            break
    
    if not id_col:
        print(f"Available columns: {gdf.columns.tolist()}")
        raise ValueError("Could not find Township ID column (adm3_pcode, etc) in GeoJSON.")
    
    print(f"Using ID column: {id_col}")

    # 2. Find TIF Files
    tif_files = sorted(glob.glob(os.path.join(VIIRS_DIR, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {VIIRS_DIR}")
    
    print(f"Found {len(tif_files)} VIIRS files.")

    results = []

    # 3. Process Each Month
    for tif in tif_files:
        dt = parse_date_from_filename(os.path.basename(tif))
        if not dt:
            print(f"Skipping could not parse date: {tif}")
            continue
            
        month_str = dt.strftime('%Y-%m')
        print(f"Processing {month_str}...")
        
        # Run Zonal Statistics
        stats = zonal_stats(gdf, tif, stats="mean")
        
        # Combine with IDs
        for idx, stat in enumerate(stats):
            # Handle cases where raster might not cover the polygon (stat is None)
            val = stat['mean'] if stat and stat['mean'] is not None else 0
            
            results.append({
                'township_id': gdf.iloc[idx][id_col],
                # Try to get a name column for reference
                'township_name': gdf.iloc[idx].get('adm3_name', gdf.iloc[idx].get('admin3_nam', 'Unknown')),
                'period': month_str,
                'period_start': dt.strftime('%Y-%m-01'),
                'nightlights': val
            })

    # 4. Save to CSV
    df_local = pd.DataFrame(results)
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, "local_nightlights_panel.csv")
    df_local.to_csv(output_file, index=False)
    print(f"\nSuccess! Local nightlights saved to: {output_file}")
    print(df_local.head())

if __name__ == "__main__":
    main()
