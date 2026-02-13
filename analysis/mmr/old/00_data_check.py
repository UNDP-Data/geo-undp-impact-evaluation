#!/usr/bin/env python3
"""
analysis/myanmar/00_data_check.py
=================================
Scans Myanmar data directories and reports what's available for the
site-level DiD analysis (Gaza-style approach).

Run this first to understand what data you have and what's missing.

Usage:
    python analysis/myanmar/00_data_check.py
"""

import os
import glob
from pathlib import Path
from datetime import datetime
import re

# =============================================================================
# CONFIGURATION - ADJUST THESE PATHS AS NEEDED
# =============================================================================
BASE_DIR = Path("data/myanmar")
OUTPUT_DIR = Path("outputs/myanmar")

# Expected data locations
EXPECTED_PATHS = {
    'boundaries': BASE_DIR / "myanmar_boundaries",
    'viirs': BASE_DIR / "viirs_25",
    'rasters': BASE_DIR / "rasters",  # For SAR, NDVI, etc.
    'external': BASE_DIR / "external",
    'sites': BASE_DIR / "external" / "undp_sites.csv",
    'projects': BASE_DIR / "external" / "undp_projects.csv",
    'acled': BASE_DIR / "external" / "acled_myanmar.csv",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_path(path, description):
    """Check if path exists and return status."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    return exists, f"  {status} {description}: {path}"


def scan_rasters(directory, pattern="*.tif"):
    """Scan directory for raster files and extract dates."""
    if not directory.exists():
        return []
    
    files = list(directory.glob(pattern))
    results = []
    
    for f in sorted(files):
        # Try to extract date from filename
        # Common patterns: YYYYMM, YYYY-MM, YYYYMMDD-YYYYMMDD
        date_match = re.search(r'(\d{8})-\d{8}|(\d{6})|(\d{4}-\d{2})', f.stem)
        
        if date_match:
            date_str = date_match.group(1) or date_match.group(2) or date_match.group(3)
            try:
                if len(date_str) == 8:
                    dt = datetime.strptime(date_str, '%Y%m%d')
                elif len(date_str) == 6:
                    dt = datetime.strptime(date_str, '%Y%m')
                else:
                    dt = datetime.strptime(date_str.replace('-', ''), '%Y%m')
                results.append((f, dt))
            except:
                results.append((f, None))
        else:
            results.append((f, None))
    
    return results


def parse_csv_columns(filepath):
    """Read first few rows of CSV to get columns."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, nrows=5)
        return list(df.columns)
    except:
        return None


# =============================================================================
# MAIN CHECK
# =============================================================================

def main():
    print("=" * 80)
    print(" MYANMAR DATA CHECK")
    print(" Scanning for available data for site-level DiD analysis")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. DIRECTORY STRUCTURE
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 1. DIRECTORY STRUCTURE")
    print("─" * 80)
    
    for name, path in EXPECTED_PATHS.items():
        if path.is_dir():
            exists, msg = check_path(path, name.upper())
            print(msg)
            if exists:
                # Count files
                n_files = len(list(path.glob("*")))
                print(f"      └─ {n_files} files/folders")
        elif path.suffix:  # It's a file
            exists, msg = check_path(path, name.upper())
            print(msg)
    
    # -------------------------------------------------------------------------
    # 2. BOUNDARY FILES
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 2. BOUNDARY FILES")
    print("─" * 80)
    
    boundary_dir = EXPECTED_PATHS['boundaries']
    if boundary_dir.exists():
        for ext in ['*.geojson', '*.shp', '*.gpkg']:
            files = list(boundary_dir.glob(ext))
            for f in files:
                print(f"  ✓ {f.name}")
                # Try to read and get info
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(f)
                    print(f"      └─ {len(gdf)} features, columns: {list(gdf.columns)[:5]}...")
                except Exception as e:
                    print(f"      └─ Could not read: {e}")
    else:
        print("  ✗ Boundary directory not found")
    
    # -------------------------------------------------------------------------
    # 3. RASTER DATA (VIIRS)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 3. VIIRS NIGHTLIGHTS")
    print("─" * 80)
    
    viirs_dir = EXPECTED_PATHS['viirs']
    if viirs_dir.exists():
        rasters = scan_rasters(viirs_dir)
        print(f"  Found {len(rasters)} VIIRS rasters")
        
        if rasters:
            # Show date range
            dates = [r[1] for r in rasters if r[1]]
            if dates:
                print(f"  Date range: {min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')}")
            
            # Show first few files
            print("  Sample files:")
            for f, dt in rasters[:3]:
                dt_str = dt.strftime('%Y-%m') if dt else 'unknown date'
                print(f"    - {f.name} ({dt_str})")
            if len(rasters) > 3:
                print(f"    ... and {len(rasters)-3} more")
    else:
        print("  ✗ VIIRS directory not found")
        print("  → Create: data/myanmar/viirs_25/")
    
    # -------------------------------------------------------------------------
    # 4. OTHER RASTERS (SAR, NDVI, etc.)
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 4. OTHER RASTERS (SAR, NDVI, BSI, NDBI)")
    print("─" * 80)
    
    raster_dir = EXPECTED_PATHS['rasters']
    if raster_dir.exists():
        rasters = scan_rasters(raster_dir)
        
        # Group by indicator
        by_indicator = {}
        for f, dt in rasters:
            # Try to identify indicator from filename
            name_lower = f.stem.lower()
            if 'sar' in name_lower or 'vh' in name_lower or 'vv' in name_lower:
                ind = 'SAR'
            elif 'ndvi' in name_lower:
                ind = 'NDVI'
            elif 'ndbi' in name_lower:
                ind = 'NDBI'
            elif 'bsi' in name_lower:
                ind = 'BSI'
            elif 'viirs' in name_lower or 'nightlight' in name_lower:
                ind = 'VIIRS'
            else:
                ind = 'Other'
            
            if ind not in by_indicator:
                by_indicator[ind] = []
            by_indicator[ind].append((f, dt))
        
        for ind, files in sorted(by_indicator.items()):
            dates = [r[1] for r in files if r[1]]
            date_range = f"{min(dates).strftime('%Y-%m')} to {max(dates).strftime('%Y-%m')}" if dates else "unknown"
            print(f"  ✓ {ind}: {len(files)} files ({date_range})")
    else:
        print("  ✗ Raster directory not found")
        print("  → Create: data/myanmar/rasters/")
        print("  → Will need SAR, NDVI, NDBI, BSI for full analysis (like Gaza)")
    
    # -------------------------------------------------------------------------
    # 5. UNDP SITES
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 5. UNDP INTERVENTION SITES")
    print("─" * 80)
    
    sites_file = EXPECTED_PATHS['sites']
    if sites_file.exists():
        cols = parse_csv_columns(sites_file)
        if cols:
            print(f"  ✓ Sites file found: {sites_file}")
            print(f"      Columns: {cols}")
            
            # Check for required columns
            required = ['latitude', 'longitude', 'site_id']
            missing = [c for c in required if c.lower() not in [x.lower() for x in cols]]
            if missing:
                print(f"      ⚠ Missing required columns: {missing}")
            else:
                print("      ✓ Has required columns (latitude, longitude, site_id)")
    else:
        print("  ✗ Sites file not found")
        print(f"  → Create: {sites_file}")
        print("  → Required columns: site_id, latitude, longitude, intervention_type")
        print("")
        print("  ALTERNATIVE: If you have township-level intervention data,")
        print("  we can create a script to generate site coordinates from:")
        print("    - Township centroids")
        print("    - Random points within affected townships")
        print("    - Building footprints (if available)")
    
    # -------------------------------------------------------------------------
    # 6. UNDP PROJECTS
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 6. UNDP PROJECTS (for staggered treatment timing)")
    print("─" * 80)
    
    projects_file = EXPECTED_PATHS['projects']
    if projects_file.exists():
        cols = parse_csv_columns(projects_file)
        if cols:
            print(f"  ✓ Projects file found")
            print(f"      Columns: {cols}")
    else:
        print("  ✗ Projects file not found (optional)")
        print(f"  → If available, create: {projects_file}")
        print("  → Should have: project_id, start_date, category/type")
        print("  → Without this, we'll use earthquake date as single treatment time")
    
    # -------------------------------------------------------------------------
    # 7. ACLED CONFLICT DATA
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" 7. ACLED CONFLICT DATA")
    print("─" * 80)
    
    acled_file = EXPECTED_PATHS['acled']
    if acled_file.exists():
        cols = parse_csv_columns(acled_file)
        if cols:
            print(f"  ✓ ACLED file found")
            print(f"      Columns: {cols[:8]}...")
            
            # Count rows
            try:
                import pandas as pd
                n = len(pd.read_csv(acled_file))
                print(f"      Events: {n:,}")
            except:
                pass
    else:
        print("  ✗ ACLED file not found (optional)")
        print(f"  → If using conflict controls, create: {acled_file}")
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" SUMMARY: READINESS FOR SITE-LEVEL DiD")
    print("=" * 80)
    
    # Check readiness
    has_viirs = EXPECTED_PATHS['viirs'].exists() and len(list(EXPECTED_PATHS['viirs'].glob("*.tif"))) > 0
    has_boundaries = EXPECTED_PATHS['boundaries'].exists()
    has_sites = EXPECTED_PATHS['sites'].exists()
    
    print(f"""
  REQUIRED:
  {'✓' if has_viirs else '✗'} VIIRS Nightlights
  {'✓' if has_boundaries else '✗'} Boundary files
  {'✓' if has_sites else '✗'} UNDP site coordinates

  OPTIONAL (enhance analysis):
  {'✓' if EXPECTED_PATHS['rasters'].exists() else '✗'} SAR/NDVI/BSI rasters
  {'✓' if EXPECTED_PATHS['projects'].exists() else '✗'} Project timing data
  {'✓' if EXPECTED_PATHS['acled'].exists() else '✗'} Conflict data
    """)
    
    if has_viirs and has_boundaries:
        if has_sites:
            print("  ★ READY for full site-level analysis (like Gaza)")
        else:
            print("  ◐ PARTIAL: Need to create UNDP site coordinates")
            print("     Options:")
            print("     1. Digitize from project documents/maps")
            print("     2. Use township centroids as proxy")
            print("     3. Generate from building footprints")
    else:
        print("  ✗ NOT READY: Missing critical data")
        print("     Please ensure VIIRS rasters and boundary files exist")
    
    # -------------------------------------------------------------------------
    # NEXT STEPS
    # -------------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(" NEXT STEPS")
    print("─" * 80)
    print("""
  1. Run this script from your data_analysis directory:
     python analysis/myanmar/00_data_check.py

  2. Once you confirm what data you have, I'll create:
     - 01_myanmar_site_preparation.py (generate/load sites + controls)
     - 02_myanmar_did_comprehensive.py (main DiD analysis)  
     - 03_myanmar_pretrends_calendar.py (pre-trends plots)
     - 04_myanmar_economic_impact.py (ROI calculations)

  3. Key information I need from you:
     - Earthquake date (for the event timing)
     - Affected region (townships/states)
     - UNDP intervention types (Shelter, WASH, etc.)
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
