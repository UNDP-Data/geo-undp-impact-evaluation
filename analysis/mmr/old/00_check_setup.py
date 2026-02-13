#!/usr/bin/env python3
"""
analysis/mmr/00_check_setup.py
==============================
Verify Myanmar data setup before running pipeline.

Usage:
    python analysis/mmr/00_check_setup.py
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import yaml

def main():
    print("=" * 70)
    print(" MYANMAR SETUP CHECK")
    print("=" * 70)
    
    data_dir = Path("data/mmr")
    config_path = Path("config/myanmar.yaml")
    
    issues = []
    
    # 1. Check config
    print("\n--- Config ---")
    if config_path.exists():
        print(f"  ✓ {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"    Study period: {config.get('study_period_start')} to {config.get('study_period_end')}")
        print(f"    Earthquake: {config.get('events', {}).get('disaster_date')}")
    else:
        print(f"  ✗ {config_path} - MISSING")
        issues.append("Copy config/myanmar.yaml to your project")
    
    # 2. Check boundaries
    print("\n--- Boundaries ---")
    boundary_paths = [
        data_dir / "myanmar_boundaries" / "mmr_admin3.geojson",
        data_dir / "myanmar_boundaries" / "mmr_admbnda_adm3_mimu_250201.shp",
        data_dir / "boundaries" / "mmr_admin3.geojson",
    ]
    
    boundary_found = None
    for bp in boundary_paths:
        if bp.exists():
            boundary_found = bp
            break
    
    if boundary_found:
        print(f"  ✓ {boundary_found}")
        try:
            gdf = gpd.read_file(boundary_found)
            print(f"    {len(gdf)} admin units")
            print(f"    Columns: {list(gdf.columns)[:8]}")
            
            # Detect ID/name columns
            id_cols = [c for c in gdf.columns if 'pcode' in c.lower() or 'code' in c.lower() or 'id' in c.lower()]
            name_cols = [c for c in gdf.columns if 'name' in c.lower() or c in ['TS', 'ST']]
            
            if id_cols:
                print(f"    Likely ID column: {id_cols[0]}")
            if name_cols:
                print(f"    Likely name column: {name_cols[0]}")
                
        except Exception as e:
            print(f"    ⚠ Error reading: {e}")
    else:
        print(f"  ✗ Boundaries not found")
        issues.append("Add admin boundaries to data/mmr/myanmar_boundaries/")
    
    # 3. Check UNDP data
    print("\n--- UNDP Data ---")
    undp_dir = data_dir / "undp"
    
    if undp_dir.exists():
        files = list(undp_dir.glob("*"))
        print(f"  ✓ {undp_dir} ({len(files)} files)")
        for f in files:
            print(f"    - {f.name}")
        
        # Check project_data.xlsx
        excel_path = undp_dir / "project_data.xlsx"
        if excel_path.exists():
            try:
                df = pd.read_excel(excel_path, nrows=5)
                print(f"\n  project_data.xlsx columns:")
                for col in df.columns:
                    print(f"    - {col}")
                
                # Check for GPS columns
                gps_cols = [c for c in df.columns if 'gps' in c.lower() or 'lat' in c.lower() or 'lon' in c.lower()]
                if gps_cols:
                    print(f"\n  ✓ GPS columns found: {gps_cols}")
                else:
                    print(f"\n  ⚠ No GPS columns found")
                    issues.append("Check GPS column names in project_data.xlsx")
            except Exception as e:
                print(f"  ⚠ Error reading Excel: {e}")
        
        # Check GeoJSON
        geojson_path = undp_dir / "undp_interventions.geojson"
        if geojson_path.exists():
            try:
                gdf = gpd.read_file(geojson_path)
                print(f"\n  ✓ undp_interventions.geojson: {len(gdf)} features")
            except Exception as e:
                print(f"  ⚠ Error reading GeoJSON: {e}")
    else:
        print(f"  ✗ {undp_dir} - MISSING")
        issues.append("Add UNDP data to data/mmr/undp/")
    
    # 4. Check existing raster data
    print("\n--- Raster Data ---")
    raster_dirs = [
        ("VIIRS 2023", data_dir / "viirs_23"),
        ("VIIRS 2025", data_dir / "viirs_25"),
        ("Landsat", data_dir / "landsat"),
    ]
    
    for name, rdir in raster_dirs:
        if rdir.exists():
            tifs = list(rdir.glob("*.tif"))
            print(f"  ✓ {name}: {len(tifs)} TIF files")
        else:
            print(f"  - {name}: not found (will download via GEE)")
    
    # 5. Check external data
    print("\n--- External Data ---")
    external_dir = data_dir / "external"
    external_files = [
        "acled_myanmar.csv",
        "wfp_prices.csv", 
        "google_trends.csv",
        "undp_sites.csv",  # Created by parse_undp_sites.py
    ]
    
    for fname in external_files:
        fpath = external_dir / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                print(f"  ✓ {fname}: {len(df)} rows")
            except:
                print(f"  ✓ {fname}: exists")
        else:
            print(f"  - {fname}: not found")
    
    # 6. Check GEE panel data
    print("\n--- GEE Panel Data ---")
    gee_files = list(data_dir.glob("gee_panel_*.csv"))
    if gee_files:
        for gf in gee_files:
            df = pd.read_csv(gf, nrows=5)
            print(f"  ✓ {gf.name}")
    else:
        print("  - No GEE panel data yet (run: make gee COUNTRY=myanmar)")
    
    # Summary
    print("\n" + "=" * 70)
    if issues:
        print(" ISSUES TO FIX:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print(" ✓ Setup looks good!")
    
    print("\n NEXT STEPS:")
    print("   1. python -m pipeline.parse_undp_sites --config config/myanmar.yaml")
    print("   2. make gee COUNTRY=myanmar")
    print("   3. make run COUNTRY=myanmar")
    print("=" * 70)


if __name__ == "__main__":
    main()
