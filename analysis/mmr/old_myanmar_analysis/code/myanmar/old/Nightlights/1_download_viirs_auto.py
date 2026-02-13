#!/usr/bin/env python3
"""
Automated VIIRS Nightlights Downloader for Cyclone Mocha Analysis
=================================================================

Automatically downloads VIIRS monthly nightlight data from NOAA servers.
This version scrapes the actual file listings to get exact filenames.

Requirements:
    pip install requests beautifulsoup4 tqdm

Usage:
    python 1_download_viirs_auto.py --output-dir ./data/viirs --start 2021-01 --end 2024-10
"""

import os
import argparse
import requests
from pathlib import Path
from datetime import datetime
import gzip
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup


BASE_URL = "https://eogdata.mines.edu/nighttime_light/monthly/v10"


def get_month_list(start_year_month, end_year_month):
    """Generate list of year-month tuples."""
    start = datetime.strptime(start_year_month, "%Y-%m")
    end = datetime.strptime(end_year_month, "%Y-%m")
    
    months = []
    current = start
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return months


def find_viirs_filename(year, month):
    """
    Scrape the NOAA directory to find the exact VIIRS filename for a year/month.
    
    Returns:
        str: filename or None if not found
    """
    year_url = f"{BASE_URL}/{year}/"
    yyyymm = f"{year}{month:02d}"
    
    try:
        response = requests.get(year_url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML to find links
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for links containing our pattern
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if yyyymm in href and 'vcmslcfg.tif.gz' in href:
                return href
        
        return None
    
    except Exception as e:
        print(f"  ⚠ Error finding filename for {year}-{month:02d}: {e}")
        return None


def download_file(url, output_path, chunk_size=8192):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"\n  ✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False


def extract_gz_file(gz_path, output_path):
    """Extract .gz file."""
    try:
        print(f"  Extracting {gz_path.name}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automatically download VIIRS nightlights"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/viirs",
        help="Output directory"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01",
        help="Start year-month (YYYY-MM)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2024-10",
        help="End year-month (YYYY-MM)"
    )
    
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract .gz files after download"
    )
    
    parser.add_argument(
        "--keep-gz",
        action="store_true",
        help="Keep .gz files after extraction"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already downloaded files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  AUTOMATED VIIRS NIGHTLIGHTS DOWNLOADER")
    print("  Cyclone Mocha Analysis - Myanmar")
    print("="*80)
    print(f"\nPeriod: {args.start} to {args.end}")
    print(f"Output: {output_dir.absolute()}")
    
    # Generate month list
    months = get_month_list(args.start, args.end)
    print(f"\nWill download {len(months)} months")
    
    # Download files
    print("\n" + "="*80)
    print("DOWNLOADING VIIRS FILES")
    print("="*80)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, (year, month) in enumerate(months, 1):
        print(f"\n[{i}/{len(months)}] {year}-{month:02d}")
        
        # Check if already downloaded
        yyyymm = f"{year}{month:02d}"
        existing_tif = list(output_dir.glob(f"*{yyyymm}*.tif"))
        existing_gz = list(output_dir.glob(f"*{yyyymm}*.tif.gz"))
        
        if args.skip_existing and (existing_tif or existing_gz):
            if existing_tif:
                print(f"  ✓ Already exists (extracted): {existing_tif[0].name}")
            else:
                print(f"  ✓ Already exists (compressed): {existing_gz[0].name}")
            skipped += 1
            continue
        
        # Find exact filename
        print(f"  Finding filename...")
        filename = find_viirs_filename(year, month)
        
        if not filename:
            print(f"  ✗ Could not find file for {year}-{month:02d}")
            failed += 1
            continue
        
        print(f"  Found: {filename}")
        
        # Download
        url = f"{BASE_URL}/{year}/{filename}"
        output_file = output_dir / filename
        
        print(f"  Downloading...")
        if download_file(url, output_file):
            print(f"  ✓ Downloaded: {output_file.name}")
            downloaded += 1
            
            # Extract if requested
            if args.extract:
                tif_file = output_dir / output_file.stem  # Remove .gz
                if extract_gz_file(output_file, tif_file):
                    print(f"  ✓ Extracted: {tif_file.name}")
                    
                    # Delete .gz if not keeping
                    if not args.keep_gz:
                        output_file.unlink()
                        print(f"  ✓ Removed .gz file")
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"\n  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(months)}")
    
    # Check what we have now
    tif_files = list(output_dir.glob("*.tif"))
    gz_files = list(output_dir.glob("*.tif.gz"))
    
    print(f"\n  Current files in {output_dir.name}:")
    print(f"    .tif files: {len(tif_files)}")
    print(f"    .tif.gz files: {len(gz_files)}")
    
    # Next steps
    if len(tif_files) > 0:
        print("\n" + "="*80)
        print("READY TO PROCEED")
        print("="*80)
        print("\n✓ You have .tif files ready for analysis")
        print("\nNext step: Run script 2 to get Myanmar boundaries")
        print("  python 2_get_myanmar_boundaries.py --output-dir ./data/boundaries")
    elif len(gz_files) > 0:
        print("\n" + "="*80)
        print("EXTRACTION NEEDED")
        print("="*80)
        print("\n⚠ You have .gz files but need to extract them")
        print("\nRun this script again with --extract:")
        print(f"  python {Path(__file__).name} --extract --output-dir {args.output_dir}")
    else:
        print("\n⚠ No files downloaded. Check errors above.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
