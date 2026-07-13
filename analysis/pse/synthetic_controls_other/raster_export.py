#!/usr/bin/env python3
"""
pipeline/raster_export.py
=========================
Export satellite rasters from Google Earth Engine.

Datasets available:
- viirs: Nighttime lights (economic activity proxy)
- ndvi: Vegetation index
- ndbi: Built-up index (urban/construction proxy)
- sar: Sentinel-1 SAR (VV and VH backscatter)
- sar_vh: SAR VH only (better for structure detection)

Two modes:
1. Direct download (default): Downloads GeoTIFFs directly to project folder
2. Drive export: Exports to Google Drive (for large files)

Usage:
    python -m pipeline.raster_export --config config/gaza.yaml
    python -m pipeline.raster_export --config config/gaza.yaml --datasets viirs ndvi ndbi
    python -m pipeline.raster_export --config config/gaza.yaml --months 2023-09 2024-09 2025-03
"""

import ee
import yaml
import argparse
import requests
import zipfile
import io
import os
from pathlib import Path
from datetime import datetime
import geopandas as gpd


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("✓ GEE initialized")
    except Exception:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize()
        print("✓ GEE authenticated and initialized")


def get_viirs_monthly(year: int, month: int) -> ee.Image:
    """
    Get VIIRS nightlight composite for a month.
    Tries monthly composite first, falls back to daily if unavailable.
    """
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    # Try 1: Monthly composite (faster, but ~2-3 month lag)
    monthly = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
        .filterDate(start_date, end_date) \
        .select('avg_rad')
    
    # Check if monthly data exists
    monthly_count = monthly.size().getInfo()
    
    if monthly_count > 0:
        return monthly.mean()
    
    # Try 2: Daily VIIRS (VNP46A2) - more recent data available
    # This is NASA's daily product, we composite it to monthly
    print(f"    (Using daily VIIRS composite for {year}-{month:02d})")
    daily = ee.ImageCollection("NASA/VIIRS/002/VNP46A2") \
        .filterDate(start_date, end_date) \
        .select('DNB_BRDF_Corrected_NTL') \
        .mean()
    
    return daily



def get_modis_ndvi(year: int, month: int) -> ee.Image:
    """Get MODIS NDVI composite."""
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    modis = ee.ImageCollection("MODIS/061/MOD13A2") \
        .filterDate(start_date, end_date) \
        .select('NDVI') \
        .mean() \
        .multiply(0.0001)  # Scale factor
    
    return modis


def get_sentinel2_ndbi(year: int, month: int) -> ee.Image:
    """
    Get Sentinel-2 NDBI (Normalized Difference Built-up Index).
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Higher values indicate more built-up/urban areas.
    Useful for detecting construction, debris, and urban changes.
    """
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    # Sentinel-2 with cloud masking
    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .map(mask_clouds) \
        .select(['B8', 'B11', 'B12'])  # NIR, SWIR1, SWIR2
    
    # Calculate median composite
    composite = s2.median()
    
    # Calculate NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)
    ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI')
    
    return ndbi


def get_sentinel2_bsi(year: int, month: int) -> ee.Image:
    """
    Get Sentinel-2 BSI (Bare Soil Index).
    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    Higher values indicate bare soil, rubble, or debris.
    """
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .map(mask_clouds) \
        .select(['B2', 'B4', 'B8', 'B11'])  # Blue, Red, NIR, SWIR1
    
    composite = s2.median()
    
    # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    bsi = composite.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': composite.select('B11'),
            'RED': composite.select('B4'),
            'NIR': composite.select('B8'),
            'BLUE': composite.select('B2')
        }
    ).rename('BSI')
    
    return bsi


def get_sentinel1_sar(year: int, month: int) -> ee.Image:
    """Get Sentinel-1 SAR composite (VV and VH)."""
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH']) \
        .mean()
    
    return s1


def get_sentinel1_vh(year: int, month: int) -> ee.Image:
    """
    Get Sentinel-1 VH band only.
    VH polarization is better for detecting structures and buildings.
    Changes in VH can indicate building damage/destruction.
    """
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select('VH') \
        .mean()
    
    return s1


def download_image_direct(image: ee.Image, region: ee.Geometry, filename: str, 
                          output_dir: Path, scale: int = 500) -> bool:
    """Download image directly using ee.data.getDownloadId()."""
    try:
        # Get download URL
        url = image.getDownloadURL({
            'scale': scale,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        # Download
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Check if it's a zip file (GEE sometimes returns zipped tiffs)
        if response.content[:2] == b'PK':  # ZIP magic number
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Extract the first tif file
                for name in zf.namelist():
                    if name.endswith('.tif'):
                        with open(output_dir / filename, 'wb') as f:
                            f.write(zf.read(name))
                        return True
        else:
            # Direct GeoTIFF
            with open(output_dir / filename, 'wb') as f:
                f.write(response.content)
            return True
            
    except Exception as e:
        print(f"Download error: {e}")
        return False
    
    return False


def download_with_geemap(image: ee.Image, region: ee.Geometry, filename: str,
                         output_dir: Path, scale: int = 500) -> bool:
    """Download using geemap (fallback method)."""
    try:
        import geemap
        
        output_path = str(output_dir / filename)
        geemap.ee_export_image(
            image,
            filename=output_path,
            scale=scale,
            region=region,
            crs='EPSG:4326',
            file_per_band=False
        )
        return Path(output_path).exists()
    except ImportError:
        print("  geemap not installed. Install with: pip install geemap")
        return False
    except Exception as e:
        print(f"  geemap error: {e}")
        return False


def export_to_drive(image: ee.Image, region: ee.Geometry, description: str, 
                    folder: str, scale: int = 500):
    """Export raster to Google Drive (for large files)."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    task.start()
    return task


def main():
    parser = argparse.ArgumentParser(description='Export satellite rasters from GEE')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--months', '-m', nargs='+', help='Specific months to export (YYYY-MM format)')
    parser.add_argument('--monthly-range', '-r', 
                        help='Date range for monthly time series (format: YYYY-MM:YYYY-MM, e.g., 2023-01:2025-10)')
    parser.add_argument('--datasets', '-d', nargs='+', 
                        default=['viirs', 'ndvi', 'ndbi', 'sar_vh'],
                        choices=['viirs', 'ndvi', 'ndbi', 'bsi', 'sar', 'sar_vh'],
                        help='Datasets to export')
    parser.add_argument('--scale', '-s', type=int, default=100, 
                        help='Resolution in meters (default: 100 for Sentinel-2)')
    parser.add_argument('--drive', action='store_true', help='Export to Google Drive instead of direct download')
    parser.add_argument('--folder', '-f', help='Google Drive folder name (only with --drive)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').lower()
    
    print(f"\n{'='*60}")
    print(f"GEE RASTER EXPORT: {country_name}")
    print(f"{'='*60}")
    
    # Initialize GEE
    initialize_gee()
    
    # Output directory
    output_dir = Path(f"data/{country_code}/rasters")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get boundaries and create region
    boundary_path = config['boundaries']['file_path']
    print(f"Loading boundaries from: {boundary_path}")
    
    gdf = gpd.read_file(boundary_path)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Add buffer for context
    buffer = 0.05  # degrees (~5km)
    region = ee.Geometry.Rectangle([
        bounds[0] - buffer, 
        bounds[1] - buffer, 
        bounds[2] + buffer, 
        bounds[3] + buffer
    ])
    
    print(f"Region bounds: {bounds}")
    
    # Determine months to export
    if args.months:
        months = [tuple(map(int, m.split('-'))) for m in args.months]
    elif args.monthly_range:
        # Parse range like "2023-01:2025-10" for full time series
        start_str, end_str = args.monthly_range.split(':')
        start_y, start_m = map(int, start_str.split('-'))
        end_y, end_m = map(int, end_str.split('-'))
        
        months = []
        y, m = start_y, start_m
        while (y, m) <= (end_y, end_m):
            months.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        print(f"\nGenerating monthly time series: {start_str} to {end_str} ({len(months)} months)")
    else:
        # Default: key months based on disaster date
        disaster = config['events'].get('disaster_date', '2023-10-07')
        disaster_dt = datetime.strptime(disaster, '%Y-%m-%d')
        
        # Pre-disaster, post-disaster, latest
        pre_month = disaster_dt.month - 1 if disaster_dt.month > 1 else 12
        pre_year = disaster_dt.year if disaster_dt.month > 1 else disaster_dt.year - 1
        
        post_month = disaster_dt.month + 1 if disaster_dt.month < 12 else 1
        post_year = disaster_dt.year if disaster_dt.month < 12 else disaster_dt.year + 1
        
        months = [
            (pre_year, pre_month),   # 1 month before disaster
            (post_year, post_month), # 1 month after disaster
            (2025, 10),              # Latest available (Oct 2025)
        ]
    
    print(f"\nDatasets: {', '.join(args.datasets)}")
    print(f"Months: {[f'{y}-{m:02d}' for y, m in months]}")
    print(f"Scale: {args.scale}m")
    print(f"Mode: {'Google Drive' if args.drive else 'Direct Download'}")
    
    # Dataset info
    dataset_info = {
        'viirs': ('Nightlights', get_viirs_monthly, 500),
        'ndvi': ('NDVI', get_modis_ndvi, 250),
        'ndbi': ('Built-up Index', get_sentinel2_ndbi, 100),
        'bsi': ('Bare Soil Index', get_sentinel2_bsi, 100),
        'sar': ('SAR VV+VH', get_sentinel1_sar, 100),
        'sar_vh': ('SAR VH', get_sentinel1_vh, 100),
    }
    
    # Process each month and dataset
    success_count = 0
    tasks = []
    
    for year, month in months:
        month_str = f"{year}{month:02d}"
        print(f"\n--- {year}-{month:02d} ---")
        
        for dataset in args.datasets:
            info = dataset_info.get(dataset)
            if not info:
                continue
                
            name, get_func, default_scale = info
            scale = args.scale if args.scale != 100 else default_scale
            filename = f"{dataset}_{country_code}_{month_str}.tif"
            
            # Check if already exists
            if (output_dir / filename).exists() and not args.drive:
                print(f"  ✓ {filename} (already exists)")
                success_count += 1
                continue
            
            print(f"  Downloading {name}...", end=" ", flush=True)
            
            try:
                # Get the image
                img = get_func(year, month)
                
                # Clip to region
                img = img.clip(region)
                
                if args.drive:
                    # Export to Google Drive
                    drive_folder = args.folder or f"gee_exports_{country_code}"
                    task = export_to_drive(img, region, filename.replace('.tif', ''), 
                                          drive_folder, scale)
                    tasks.append(task)
                    print(f"✓ Task started → {drive_folder}/")
                else:
                    # Direct download
                    success = download_image_direct(img, region, filename, output_dir, scale)
                    
                    if not success:
                        # Try geemap as fallback
                        print("trying geemap...", end=" ", flush=True)
                        success = download_with_geemap(img, region, filename, output_dir, scale)
                    
                    if success:
                        print(f"✓ {filename}")
                        success_count += 1
                    else:
                        print(f"✗ Failed")
                        
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    if args.drive:
        print(f"✓ {len(tasks)} export tasks started")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Check GEE Task Manager: https://code.earthengine.google.com/tasks")
        print(f"2. Download from Google Drive")
        print(f"3. Save to: {output_dir}/")
        print(f"4. Run: python -m pipeline.raster_process --config {args.config}")
    else:
        print(f"✓ {success_count} files downloaded to {output_dir}")
        print(f"{'='*60}")
        print(f"\nNext step:")
        print(f"  python -m pipeline.raster_process --config {args.config}")


if __name__ == "__main__":
    main()
