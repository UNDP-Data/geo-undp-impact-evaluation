#!/usr/bin/env python3
"""
Download VIIRS from NOAA ArcGIS ImageServer
===========================================

FREE - NO REGISTRATION NEEDED!

Uses NOAA's ArcGIS service to export VIIRS data directly as GeoTIFF.
This is the easiest method with no authentication required.

Requirements:
    pip install requests tqdm

Usage:
    python download_viirs_noaa.py
"""

import requests
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm


# NOAA NCEI ImageServer endpoints (FREE, no auth needed!)
MONTHLY_AVERAGE = "https://gis.ngdc.noaa.gov/arcgis/rest/services/VIIRS_DNB_Monthly_Composite/ImageServer/exportImage"

# Myanmar bounding box (covers whole country including Rakhine)
MYANMAR_BBOX = "92.0,9.0,101.5,28.5"  # lon_min,lat_min,lon_max,lat_max


def download_month_from_arcgis(year, month, output_dir, bbox=MYANMAR_BBOX):
    """
    Download VIIRS monthly composite for Myanmar from NOAA ArcGIS service.
    
    Parameters:
        year: int
        month: int (1-12)
        output_dir: Path
        bbox: str (lon_min,lat_min,lon_max,lat_max)
    
    Returns:
        bool: Success
    """
    yyyymm = f"{year}{month:02d}"
    output_file = output_dir / f"VIIRS_{yyyymm}_noaa.tif"
    
    # Check if exists
    if output_file.exists():
        print(f"  ✓ Already exists: {output_file.name}")
        return True
    
    # Create time range (start and end of month)
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # Convert to milliseconds since epoch (ArcGIS format)
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    print(f"\n[{year}-{month:02d}]")
    print(f"  Requesting from NOAA ArcGIS...")
    
    # ArcGIS export parameters
    params = {
        'bbox': bbox,
        'bboxSR': '4326',  # WGS84
        'size': '4096,4096',  # Image size (adjust for quality vs file size)
        'imageSR': '4326',
        'format': 'tiff',
        'pixelType': 'F32',  # Float32 for radiance values
        'noData': '-999',
        'interpolation': 'RSP_BilinearInterpolation',
        'compression': 'LZ77',
        'compressionQuality': 100,
        'time': f'{start_ms},{end_ms}',  # Time range in milliseconds
        'f': 'image'
    }
    
    try:
        # Request the image
        print(f"  Downloading (~500 KB for Myanmar bbox)...")
        response = requests.get(MONTHLY_AVERAGE, params=params, stream=True, timeout=120)
        response.raise_for_status()
        
        # Check if we got an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type and 'tiff' not in content_type:
            print(f"  ⚠ Unexpected response type: {content_type}")
            print(f"  This might mean no data for {year}-{month:02d}")
            return False
        
        # Save
        total_size = int(response.headers.get('content-length', 0))
        with open(output_file, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc="  ") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                f.write(response.content)
        
        file_size = output_file.stat().st_size / (1024**2)
        print(f"  ✓ Downloaded: {output_file.name} ({file_size:.1f} MB)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Download failed: {e}")
        if output_file.exists():
            output_file.unlink()  # Clean up partial file
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if output_file.exists():
            output_file.unlink()
        return False


def main():
    print("="*80)
    print("  VIIRS DOWNLOADER - NOAA ArcGIS ImageServer")
    print("  FREE - NO REGISTRATION REQUIRED!")
    print("="*80)
    
    print("\nThis downloads VIIRS data directly from NOAA's public ArcGIS service.")
    print("Files are cropped to Myanmar, so they're much smaller (~500 KB vs 1.5 GB)!")
    
    # Setup
    output_dir = Path("data/viirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Myanmar bounding box: {MYANMAR_BBOX}")
    
    # Choose time period
    print("\nWhat time period do you want?")
    print("  1. Just 2023 (12 files, ~5 minutes)")
    print("  2. Full 2021-2024 (46 files, ~20 minutes)")
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
    print("Each file is ~500 KB (cropped to Myanmar)")
    print(f"Total: ~{len(years_months) * 0.5:.0f} MB")
    
    input("\nPress Enter to start...")
    
    # Download
    downloaded = 0
    failed = 0
    
    start_time = time.time()
    
    for year, month in years_months:
        if download_month_from_arcgis(year, month, output_dir):
            downloaded += 1
        else:
            failed += 1
        
        # Small delay to be nice to the server
        time.sleep(0.5)
    
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
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.tif")) / (1024**2)
    
    print(f"\n  Total .tif files: {tif_count}")
    print(f"  Total size: {total_size:.1f} MB")
    
    if tif_count > 0:
        print("\n✓ Ready for analysis!")
        print("\nNext steps:")
        print("  python download_myanmar_simple.py")
        print("  python 3_visualize_mocha_nightlights.py \\")
        print("    --viirs-dir ./data/viirs \\")
        print("    --boundaries-dir ./data/boundaries/processed \\")
        print("    --output-dir ./outputs/mocha_analysis")
    else:
        print("\n⚠ No files downloaded successfully")
        print("The ArcGIS service might be down or data might not be available")
        print("for the requested time period")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
