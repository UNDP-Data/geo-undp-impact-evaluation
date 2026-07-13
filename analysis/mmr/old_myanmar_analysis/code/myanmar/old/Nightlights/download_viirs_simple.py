#!/usr/bin/env python3
"""
Simple Automated VIIRS Downloader - NO WEB SCRAPING
===================================================

This version tries common filename patterns directly without parsing HTML.

Requirements:
    pip install requests tqdm

Usage:
    python download_viirs_simple.py
"""

import os
import requests
from pathlib import Path
from datetime import datetime
import gzip
import shutil
from tqdm import tqdm


BASE_URL = "https://eogdata.mines.edu/nighttime_light/monthly/v10"


def get_months_list(start="2021-01", end="2024-10"):
    """Generate list of (year, month) tuples."""
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    
    months = []
    current = start_dt
    while current <= end_dt:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return months


def try_download(url, output_path):
    """Try to download a file."""
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            # File exists, now download it
            response = requests.get(url, stream=True, timeout=60)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, 
                         desc=output_path.name[:30]) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
    except:
        return False
    
    return False


def extract_gz(gz_path):
    """Extract a .gz file."""
    tif_path = gz_path.parent / gz_path.stem
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(tif_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()  # Delete .gz
        return True
    except:
        return False


def main():
    print("="*80)
    print("  VIIRS NIGHTLIGHTS DOWNLOADER (SIMPLE VERSION)")
    print("="*80)
    
    # Setup
    output_dir = Path("data/viirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get months to download
    months = get_months_list("2021-01", "2024-10")
    print(f"\nWill try to download {len(months)} months")
    print("This may take 2-4 hours for ~70 GB\n")
    
    # Download each month
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, (year, month) in enumerate(months, 1):
        yyyymm = f"{year}{month:02d}"
        print(f"\n[{i}/{len(months)}] {year}-{month:02d}")
        
        # Check if already have it
        existing = list(output_dir.glob(f"*{yyyymm}*.tif"))
        if existing:
            print(f"  ✓ Already exists: {existing[0].name}")
            skipped += 1
            continue
        
        # Try common filename patterns
        # Pattern: VIIRS_YYYYMM_vcm-orm-ntl_v10_cYYYYMMDDHHMM_global_vcmslcfg.tif.gz
        
        # Generate some likely timestamps
        timestamps = []
        
        # For current month (YYYYMM), processing usually happens next month
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year = year + 1
        
        # Try first week of next month
        for day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            timestamps.append(f"c{next_year}{next_month:02d}{day:02d}0000")
            timestamps.append(f"c{next_year}{next_month:02d}{day:02d}1200")
        
        # Try URLs
        found = False
        for ts in timestamps:
            filename = f"VIIRS_{yyyymm}_vcm-orm-ntl_v10_{ts}_global_vcmslcfg.tif.gz"
            url = f"{BASE_URL}/{year}/{filename}"
            output_file = output_dir / filename
            
            print(f"  Trying: {ts}...", end='')
            
            if try_download(url, output_file):
                print(" ✓")
                print(f"  Downloaded: {filename}")
                
                # Extract
                print(f"  Extracting...")
                if extract_gz(output_file):
                    print(f"  ✓ Extracted")
                    downloaded += 1
                    found = True
                    break
                else:
                    print(f"  ✗ Extraction failed")
            else:
                print(" ✗")
        
        if not found:
            print(f"  ✗ Could not find file for {year}-{month:02d}")
            print(f"  Manual URL: {BASE_URL}/{year}/")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(months)}")
    
    tif_count = len(list(output_dir.glob("*.tif")))
    print(f"\n  Current .tif files: {tif_count}")
    
    if tif_count > 0:
        print("\n✓ Ready for analysis!")
        print("\nNext: python download_myanmar_simple.py")
    else:
        print("\n⚠ No files downloaded")
        print("You may need to download manually from:")
        print(f"  {BASE_URL}/")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
