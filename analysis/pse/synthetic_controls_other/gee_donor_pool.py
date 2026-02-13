#!/usr/bin/env python3
"""
pipeline/gee_donor_pool.py
==========================
Download satellite data for synthetic control donor pool.

Donor countries:
- Jordan (JOR) - Stable, similar climate, no active conflict
- Lebanon (LBN) - Nearby, some economic issues but useful
- Egypt-Sinai (EGY) - Geographically similar to Gaza

These will be used to construct a "synthetic Gaza" for causal inference.

Usage:
    python -m pipeline.gee_donor_pool --countries jordan lebanon egypt
    python -m pipeline.gee_donor_pool --countries jordan --start 2022-01 --end 2025-12
"""

import ee
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import argparse
import requests
import json
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Country configurations for donor pool
DONOR_CONFIGS = {
    'jordan': {
        'name': 'Jordan',
        'iso3': 'JOR',
        'admin_level': 1,  # Governorates
        'regions_of_interest': None,  # All regions
    },
    'lebanon': {
        'name': 'Lebanon',
        'iso3': 'LBN', 
        'admin_level': 1,  # Governorates
        'regions_of_interest': None,
    },
    'egypt': {
        'name': 'Egypt (Sinai)',
        'iso3': 'EGY',
        'admin_level': 1,
        'regions_of_interest': ['North Sinai', 'South Sinai'],  # Only Sinai
    },
    'tunisia': {
        'name': 'Tunisia',
        'iso3': 'TUN',
        'admin_level': 1,
        'regions_of_interest': None,
    },
    'morocco': {
        'name': 'Morocco',
        'iso3': 'MAR',
        'admin_level': 1,
        'regions_of_interest': None,
    },
}

OUTPUT_DIR = Path("data/synthetic_control")


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


def get_admin_boundaries(iso3: str, admin_level: int = 1, regions: list = None):
    """
    Get administrative boundaries from FAO GAUL or Natural Earth.
    """
    # Use FAO GAUL dataset in GEE
    if admin_level == 1:
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")
    else:
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
    
    # Filter by country
    country_fc = gaul.filter(ee.Filter.eq('ADM0_CODE', get_gaul_code(iso3)))
    
    # Filter by specific regions if provided
    if regions:
        region_filter = ee.Filter.inList('ADM1_NAME', regions)
        country_fc = country_fc.filter(region_filter)
    
    return country_fc


def get_gaul_code(iso3: str) -> int:
    """Map ISO3 to FAO GAUL country code."""
    gaul_codes = {
        'JOR': 130,   # Jordan
        'LBN': 141,   # Lebanon
        'EGY': 40765, # Egypt
        'TUN': 248,   # Tunisia
        'MAR': 169,   # Morocco
        'PSE': 91,    # Palestine (for reference)
    }
    return gaul_codes.get(iso3, 0)


def compute_monthly_stats(admin_fc, roi, start_date: str, end_date: str) -> list:
    """
    Compute zonal statistics for satellite indicators.
    Same methodology as Gaza to ensure comparability.
    """
    results = []
    
    # Get feature info
    features = admin_fc.getInfo()['features']
    
    for feat in features:
        props = feat['properties']
        admin_id = props.get('ADM1_CODE', props.get('ADM2_CODE', 'unknown'))
        admin_name = props.get('ADM1_NAME', props.get('ADM2_NAME', 'unknown'))
        
        results.append({
            'admin_id': str(admin_id),
            'admin_name': admin_name,
            'nightlights': None,
            'ndvi': None,
            'rainfall': None,
            'vv': None,
            'vh': None,
        })
    
    # --- VIIRS Nightlights ---
    try:
        viirs_col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG") \
            .filterDate(start_date, end_date) \
            .filterBounds(roi) \
            .select('avg_rad')
        
        count = viirs_col.size().getInfo()
        
        if count == 0:
            # Fallback to daily
            viirs_col = ee.ImageCollection("NASA/VIIRS/002/VNP46A2") \
                .filterDate(start_date, end_date) \
                .filterBounds(roi) \
                .select('DNB_BRDF_Corrected_NTL')
            count = viirs_col.size().getInfo()
        
        if count > 0:
            viirs = viirs_col.mean()
            nl_stats = viirs.reduceRegions(
                collection=admin_fc,
                reducer=ee.Reducer.mean(),
                scale=500
            ).getInfo()
            
            for feat in nl_stats['features']:
                props = feat['properties']
                admin_id = str(props.get('ADM1_CODE', props.get('ADM2_CODE', 'unknown')))
                result = next((r for r in results if r['admin_id'] == admin_id), None)
                if result:
                    result['nightlights'] = props.get('mean')
    except:
        pass
    
    # --- MODIS NDVI ---
    try:
        modis_col = ee.ImageCollection("MODIS/061/MOD13A1") \
            .filterDate(start_date, end_date) \
            .filterBounds(roi) \
            .select('NDVI')
        
        count = modis_col.size().getInfo()
        
        if count > 0:
            modis = modis_col.mean().multiply(0.0001)
            ndvi_stats = modis.reduceRegions(
                collection=admin_fc,
                reducer=ee.Reducer.mean(),
                scale=500
            ).getInfo()
            
            for feat in ndvi_stats['features']:
                props = feat['properties']
                admin_id = str(props.get('ADM1_CODE', props.get('ADM2_CODE', 'unknown')))
                result = next((r for r in results if r['admin_id'] == admin_id), None)
                if result:
                    result['ndvi'] = props.get('mean')
    except:
        pass
    
    # --- CHIRPS Rainfall ---
    try:
        chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD") \
            .filterDate(start_date, end_date) \
            .filterBounds(roi) \
            .select('precipitation')
        
        count = chirps_col.size().getInfo()
        
        if count > 0:
            chirps = chirps_col.sum()
            rain_stats = chirps.reduceRegions(
                collection=admin_fc,
                reducer=ee.Reducer.mean(),
                scale=5000
            ).getInfo()
            
            for feat in rain_stats['features']:
                props = feat['properties']
                admin_id = str(props.get('ADM1_CODE', props.get('ADM2_CODE', 'unknown')))
                result = next((r for r in results if r['admin_id'] == admin_id), None)
                if result:
                    result['rainfall'] = props.get('mean')
    except:
        pass
    
    # --- Sentinel-1 SAR ---
    try:
        s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date) \
            .filterBounds(roi) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .select(['VV', 'VH'])
        
        count = s1_col.size().getInfo()
        
        if count > 0:
            s1 = s1_col.mean()
            sar_stats = s1.reduceRegions(
                collection=admin_fc,
                reducer=ee.Reducer.mean(),
                scale=100
            ).getInfo()
            
            for feat in sar_stats['features']:
                props = feat['properties']
                admin_id = str(props.get('ADM1_CODE', props.get('ADM2_CODE', 'unknown')))
                result = next((r for r in results if r['admin_id'] == admin_id), None)
                if result:
                    result['vv'] = props.get('VV')
                    result['vh'] = props.get('VH')
    except:
        pass
    
    # Check if any data was retrieved
    has_data = any(
        r['nightlights'] is not None or r['ndvi'] is not None 
        for r in results
    )
    
    return results if has_data else None


def export_donor_country(country_key: str, start_date: str, end_date: str):
    """Export satellite data for a donor country."""
    
    config = DONOR_CONFIGS.get(country_key)
    if not config:
        print(f"Unknown country: {country_key}")
        return None
    
    print(f"\n{'='*60}")
    print(f"EXPORTING: {config['name']}")
    print(f"{'='*60}")
    
    # Get admin boundaries
    admin_fc = get_admin_boundaries(
        config['iso3'], 
        config['admin_level'],
        config['regions_of_interest']
    )
    
    # Get region of interest (bounding box)
    roi = admin_fc.geometry().bounds()
    
    # Count features
    n_units = admin_fc.size().getInfo()
    print(f"Admin units: {n_units}")
    
    # Generate monthly data
    months = pd.date_range(start_date, end_date, freq='MS')
    print(f"Period: {start_date} to {end_date} ({len(months)} months)")
    
    all_results = []
    
    for month_start in months:
        month_end = month_start + pd.DateOffset(months=1)
        month_str = month_start.strftime('%Y-%m')
        print(f"  > {month_str}...", end=" ", flush=True)
        
        try:
            monthly_data = compute_monthly_stats(
                admin_fc, roi,
                month_start.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d')
            )
            
            if monthly_data:
                for record in monthly_data:
                    record['period_start'] = month_start.strftime('%Y-%m-01')
                    record['country'] = config['name']
                    record['iso3'] = config['iso3']
                    all_results.append(record)
                print(f"✓ ({len(monthly_data)} units)")
            else:
                print("no data")
                
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"error: {e}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save
        output_file = OUTPUT_DIR / f"donor_{country_key}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file} ({len(df)} rows)")
        
        return df
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Export donor pool data for synthetic control')
    parser.add_argument('--countries', '-c', nargs='+', 
                        default=['jordan', 'lebanon', 'egypt'],
                        choices=list(DONOR_CONFIGS.keys()),
                        help='Countries to export')
    parser.add_argument('--start', '-s', default='2022-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', default='2025-12-31',
                        help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SYNTHETIC CONTROL DONOR POOL EXPORT")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize GEE
    initialize_gee()
    
    # Export each country
    all_dfs = []
    
    for country in args.countries:
        df = export_donor_country(country, args.start, args.end)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all donor data
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_file = OUTPUT_DIR / "donor_pool_combined.csv"
        combined.to_csv(combined_file, index=False)
        print(f"\n✓ Combined donor pool: {combined_file} ({len(combined)} rows)")
    
    print("\n" + "=" * 60)
    print("✓ DONOR POOL EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
