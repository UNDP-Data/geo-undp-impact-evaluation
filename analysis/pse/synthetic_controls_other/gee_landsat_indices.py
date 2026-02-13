#!/usr/bin/env python3
"""
pipeline/gee_landsat_indices.py
===============================
Download NDBI (Built-up) and BSI (Bare Soil) indices from Landsat.

These indices require SWIR bands which aren't available in Sentinel-2 
at the same resolution, so we use Landsat 8/9.

NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
  - High values = built-up/urban areas
  - Low values = vegetation/water

BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
  - High values = bare soil/rock
  - Low values = vegetation/water

Usage:
    python -m pipeline.gee_landsat_indices
"""

import ee
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import time

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
BOUNDARIES_DIR = DATA_DIR / "pse" / "boundaries"
OUTPUT_FILE = DATA_DIR / "pse" / "raster" / "landsat_indices.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Time range
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Cloud threshold
CLOUD_THRESHOLD = 30

# Gaza and West Bank admin boundaries
GAZA_ADMIN = BOUNDARIES_DIR / "pse_admbnda_adm2_pcbs_ocha.shp"


def get_landsat_collection(start_date, end_date, geometry):
    """Get merged Landsat 8 and 9 collection with cloud masking."""
    
    def mask_clouds(image):
        """Mask clouds using QA_PIXEL band."""
        qa = image.select('QA_PIXEL')
        cloud_bit = 1 << 3  # Cloud bit
        shadow_bit = 1 << 4  # Cloud shadow bit
        
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(shadow_bit).eq(0))
        
        return image.updateMask(mask)
    
    def add_indices(image):
        """Add NDBI and BSI indices."""
        # Landsat 8/9 bands
        nir = image.select('SR_B5')  # NIR
        swir1 = image.select('SR_B6')  # SWIR1
        red = image.select('SR_B4')  # Red
        blue = image.select('SR_B2')  # Blue
        
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('ndbi')
        
        # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        bsi_num = swir1.add(red).subtract(nir.add(blue))
        bsi_den = swir1.add(red).add(nir).add(blue)
        bsi = bsi_num.divide(bsi_den).rename('bsi')
        
        # Also compute NDVI for comparison
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi_landsat')
        
        return image.addBands([ndbi, bsi, ndvi])
    
    # Landsat 8
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(geometry)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESHOLD))
          .map(mask_clouds)
          .map(add_indices))
    
    # Landsat 9
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterBounds(geometry)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESHOLD))
          .map(mask_clouds)
          .map(add_indices))
    
    return l8.merge(l9)


def extract_monthly_values(collection, geometry, admin_id, admin_name, start_date, end_date):
    """Extract monthly mean values for an admin unit."""
    
    results = []
    
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current < end:
        month_start = current.strftime("%Y-%m-%d")
        next_month = current + timedelta(days=32)
        month_end = next_month.replace(day=1).strftime("%Y-%m-%d")
        
        # Filter to month
        monthly = collection.filterDate(month_start, month_end)
        
        # Check if any images
        count = monthly.size().getInfo()
        
        if count > 0:
            # Compute monthly mean
            monthly_mean = monthly.select(['ndbi', 'bsi', 'ndvi_landsat']).mean()
            
            # Reduce over geometry
            try:
                stats = monthly_mean.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1e9
                ).getInfo()
                
                results.append({
                    'admin_id': admin_id,
                    'admin_name': admin_name,
                    'period_start': month_start,
                    'ndbi': stats.get('ndbi'),
                    'bsi': stats.get('bsi'),
                    'ndvi_landsat': stats.get('ndvi_landsat'),
                    'n_images': count
                })
            except Exception as e:
                print(f"    Error for {admin_id} {month_start}: {e}")
                results.append({
                    'admin_id': admin_id,
                    'admin_name': admin_name,
                    'period_start': month_start,
                    'ndbi': None,
                    'bsi': None,
                    'ndvi_landsat': None,
                    'n_images': 0
                })
        else:
            results.append({
                'admin_id': admin_id,
                'admin_name': admin_name,
                'period_start': month_start,
                'ndbi': None,
                'bsi': None,
                'ndvi_landsat': None,
                'n_images': 0
            })
        
        current = next_month.replace(day=1)
    
    return results


def main():
    print("\n" + "=" * 70)
    print("LANDSAT NDBI/BSI DOWNLOAD")
    print("=" * 70)
    
    # Load admin boundaries
    if not GAZA_ADMIN.exists():
        print(f"❌ Boundaries not found: {GAZA_ADMIN}")
        return
    
    gdf = gpd.read_file(GAZA_ADMIN)
    print(f"\nLoaded {len(gdf)} admin units")
    
    # Get admin ID column
    id_col = None
    for col in ['ADM2_PCODE', 'adm2_pcode', 'pcode', 'admin2Pcod']:
        if col in gdf.columns:
            id_col = col
            break
    
    name_col = None
    for col in ['ADM2_EN', 'adm2_en', 'name', 'admin2Name']:
        if col in gdf.columns:
            name_col = col
            break
    
    if id_col is None:
        print("❌ Could not find admin ID column")
        print(f"   Available: {gdf.columns.tolist()}")
        return
    
    print(f"  Admin ID column: {id_col}")
    print(f"  Admin name column: {name_col}")
    
    # Process each admin unit
    all_results = []
    
    for idx, row in gdf.iterrows():
        admin_id = row[id_col]
        admin_name = row[name_col] if name_col else admin_id
        
        print(f"\n[{idx+1}/{len(gdf)}] {admin_name} ({admin_id})")
        
        # Convert geometry to EE
        geom = row.geometry
        if geom.geom_type == 'MultiPolygon':
            coords = [list(p.exterior.coords) for p in geom.geoms]
            ee_geom = ee.Geometry.MultiPolygon(coords)
        else:
            coords = list(geom.exterior.coords)
            ee_geom = ee.Geometry.Polygon(coords)
        
        # Get Landsat collection
        collection = get_landsat_collection(START_DATE, END_DATE, ee_geom)
        
        # Extract monthly values
        results = extract_monthly_values(
            collection, ee_geom, admin_id, admin_name,
            START_DATE, END_DATE
        )
        
        all_results.extend(results)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Saved: {OUTPUT_FILE}")
        print(f"  {len(df)} rows")
        
        # Summary
        print("\nSummary statistics:")
        for col in ['ndbi', 'bsi', 'ndvi_landsat']:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid}/{len(df)} valid ({100*valid/len(df):.0f}%)")
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"{'='*70}")
    print("""
    Next steps:
    1. Merge with existing panel:
       python -c "
       import pandas as pd
       panel = pd.read_csv('outputs/pse/panel_data.csv')
       landsat = pd.read_csv('data/pse/raster/landsat_indices.csv')
       landsat['period_start'] = pd.to_datetime(landsat['period_start'])
       panel['period_start'] = pd.to_datetime(panel['period_start'])
       merged = panel.merge(landsat[['admin_id', 'period_start', 'ndbi', 'bsi']], 
                           left_on=['unit_id', 'period_start'],
                           right_on=['admin_id', 'period_start'],
                           how='left')
       merged.to_csv('outputs/pse/panel_data.csv', index=False)
       print('✓ Panel updated with NDBI and BSI')
       "
    
    2. Also add to donor pool (run similar for Jordan/Lebanon/Egypt)
    """)


if __name__ == "__main__":
    main()
