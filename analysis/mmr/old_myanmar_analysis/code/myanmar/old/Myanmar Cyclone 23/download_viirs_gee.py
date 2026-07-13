#!/usr/bin/env python3
"""
VIIRS Downloader Using Google Earth Engine - THE EASIEST WAY
=============================================================

This downloads VIIRS data directly from Google's servers.
NO manual downloading needed!

Requirements:
    pip install earthengine-api geemap

Setup (one-time):
    1. Sign up: https://earthengine.google.com/signup/
    2. Run: earthengine authenticate
    3. Follow the browser authentication

Usage:
    python download_viirs_gee.py
"""

import ee
import geemap
from pathlib import Path
from datetime import datetime, timedelta
import time


def setup_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("✓ Google Earth Engine initialized")
        return True
    except:
        print("✗ GEE not authenticated")
        print("\nRun this command first:")
        print("  earthengine authenticate")
        print("\nThen re-run this script")
        return False


def get_myanmar_bbox():
    """Get bounding box for Myanmar (Rakhine focus)."""
    # Myanmar full bbox
    # Rakhine State: (92.0, 19.0, 94.5, 21.5)
    # Myanmar full: (92.0, 9.0, 101.5, 28.5)
    return ee.Geometry.Rectangle([92.0, 9.0, 101.5, 28.5])


def download_viirs_month(year, month, output_dir, bbox):
    """
    Download VIIRS data for a specific month.
    """
    yyyymm = f"{year}{month:02d}"
    output_file = output_dir / f"VIIRS_{yyyymm}_gee.tif"
    
    # Check if already exists
    if output_file.exists():
        print(f"  ✓ Already exists: {output_file.name}")
        return True
    
    # Date range
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    
    print(f"\n[{year}-{month:02d}]")
    print(f"  Fetching from Google Earth Engine...")
    
    try:
        # Get VIIRS DNB monthly composites
        viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(bbox) \
                  .first()
        
        if viirs is None:
            print(f"  ⚠ No data available for {year}-{month:02d}")
            return False
        
        # Select the average radiance band
        viirs = viirs.select('avg_rad').clip(bbox)
        
        # Download
        print(f"  Downloading...")
        geemap.ee_export_image(
            viirs,
            filename=str(output_file),
            scale=463.83,  # VIIRS native resolution
            region=bbox,
            file_per_band=False
        )
        
        print(f"  ✓ Downloaded: {output_file.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("="*80)
    print("  VIIRS DOWNLOADER - GOOGLE EARTH ENGINE")
    print("  The Easy Way!")
    print("="*80)
    
    # Setup
    if not setup_gee():
        return
    
    # Create output directory
    output_dir = Path("data/viirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Get Myanmar bbox
    bbox = get_myanmar_bbox()
    
    # Define months to download
    print("\nWhat time period do you want?")
    print("  1. Just 2023 (12 files, test fast)")
    print("  2. Full 2021-2024 (46 files, complete dataset)")
    print("  3. Custom range")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        years_months = [(2023, m) for m in range(1, 13)]
    elif choice == '2':
        years_months = []
        for year in [2021, 2022, 2023]:
            years_months.extend([(year, m) for m in range(1, 13)])
        years_months.extend([(2024, m) for m in range(1, 11)])
    else:
        start = input("Start (YYYY-MM): ")
        end = input("End (YYYY-MM): ")
        
        start_year, start_month = map(int, start.split('-'))
        end_year, end_month = map(int, end.split('-'))
        
        years_months = []
        current_year, current_month = start_year, start_month
        
        while (current_year, current_month) <= (end_year, end_month):
            years_months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
    
    print(f"\nWill download {len(years_months)} months")
    print("This uses Google's servers - much faster than manual download!")
    
    input("\nPress Enter to start...")
    
    # Download
    downloaded = 0
    failed = 0
    
    start_time = time.time()
    
    for year, month in years_months:
        if download_viirs_month(year, month, output_dir, bbox):
            downloaded += 1
        else:
            failed += 1
        
        # Small delay to avoid rate limits
        time.sleep(1)
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(years_months)}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    
    # Check files
    tif_count = len(list(output_dir.glob("*.tif")))
    print(f"\n  Current .tif files: {tif_count}")
    
    if tif_count > 0:
        print("\n✓ Ready for analysis!")
        print("\nNext steps:")
        print("  python download_myanmar_simple.py")
        print("  python 3_visualize_mocha_nightlights.py \\")
        print("    --viirs-dir ./data/viirs \\")
        print("    --boundaries-dir ./data/boundaries/processed \\")
        print("    --output-dir ./outputs/mocha_analysis")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf you see authentication errors:")
        print("  1. Run: earthengine authenticate")
        print("  2. Follow browser instructions")
        print("  3. Re-run this script")
