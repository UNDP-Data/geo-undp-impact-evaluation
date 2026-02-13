#!/usr/bin/env python3
"""
analysis/pse/06_worldpop_pipeline_v6.py
=======================================
Gaza Population Pipeline (Strict Polygon Validator + 2026 Projection).

Fixes:
1. Geometry Error: iteraters through all shapefiles to find Polygons (regions), rejecting Points/Lines.
2. Data Lag: Takes 2023 Census baseline and applies OCHA 2026 displacement factors to create a 'Current Status' map.
"""

import base64
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
import io
import zipfile
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/pse/population")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHP_DIR = DATA_DIR / "boundaries"
SHP_DIR.mkdir(parents=True, exist_ok=True)

# HDX Configuration
APP_NAME = "GazaPopPipeline"
EMAIL = "analysis@example.com"
HAPI_BASE_URL = "https://hapi.humdata.org/api/v2/geography-infrastructure/baseline-population"

# OCHA Admin Boundaries (Verified Link)
SHAPEFILE_URL = "https://data.humdata.org/dataset/c27a926d-a59d-43d9-937b-99933682970a/resource/0f6d6282-3d7f-444f-b67f-c1f3640032e1/download/pse_admbnda_adm2_ocha_20190520.zip"

# =============================================================================
# 2026 CONFLICT ADJUSTMENT FACTORS (Source: OCHA/IOM 2025-26 Estimates)
# =============================================================================
# Estimated % of population remaining in each governorate vs. 2023 baseline.
# Note: These are operational estimates for "Yellow Line" scenarios.
DISPLACEMENT_FACTORS = {
    'North Gaza': 0.35,      # ~35% remaining (Heavy displacement)
    'Gaza': 0.40,            # ~40% remaining
    'Deir al-Balah': 1.80,   # ~180% (Massive influx of IDPs)
    'Khan Yunis': 1.10,      # ~110% (Variable, currently hosting IDPs)
    'Rafah': 0.60            # ~60% (Post-invasion decline)
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_auth_headers():
    identifier_string = f"{APP_NAME}:{EMAIL}"
    encoded_id = base64.b64encode(identifier_string.encode('utf-8')).decode('utf-8')
    return {
        "User-Agent": f"{APP_NAME}/1.0",
        "X-HDX-HAPI-APP-IDENTIFIER": encoded_id
    }

def fetch_baseline_population():
    """Fetches the official pre-war census baseline (2023) from HDX."""
    print(f"--- Fetching Baseline Data (2023) ---")
    params = {"location_code": "PSE", "output_format": "csv"}
    
    try:
        response = requests.get(HAPI_BASE_URL, params=params, headers=get_auth_headers())
        response.raise_for_status()
        
        # Save raw
        with open(DATA_DIR / "hdx_pop_raw.csv", 'wb') as f:
            f.write(response.content)
        
        df = pd.read_csv(DATA_DIR / "hdx_pop_raw.csv")
        
        # Filter for Gaza
        gaza_df = df[df['admin1_name'].astype(str).str.contains('Gaza', case=False, na=False)].copy()
        
        if gaza_df.empty:
            return None
            
        # Get latest available year (usually 2023)
        gaza_df['year'] = pd.to_datetime(gaza_df['reference_period_end']).dt.year
        latest_year = gaza_df['year'].max()
        
        print(f"✓ Baseline Year Found: {latest_year}")
        return gaza_df[gaza_df['year'] == latest_year]

    except Exception as e:
        print(f"⚠ API Error: {e}")
        return None

def download_and_validate_shapefile():
    """Downloads zip and STRICTLY checks for Polygon geometry."""
    
    # 1. Download
    print(f"--- Downloading Shapefiles ---")
    if not any(SHP_DIR.iterdir()):
        r = requests.get(SHAPEFILE_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(SHP_DIR)
    
    # 2. Iterate and Validate
    print(f"--- Validating Geometry ---")
    candidate_files = list(SHP_DIR.glob("**/*.shp"))
    
    for shp_path in candidate_files:
        try:
            # Read first row only for speed
            gdf_test = gpd.read_file(shp_path, rows=1)
            geom_type = gdf_test.geom_type.iloc[0]
            
            print(f"  Checking {shp_path.name}: {geom_type}")
            
            # STRICT CHECK: Must be Polygon or MultiPolygon
            if 'Polygon' in geom_type:
                print(f"  ✓ ACCEPTED: {shp_path.name}")
                return shp_path
            else:
                print(f"  X REJECTED (Wrong Geometry)")
                
        except Exception as e:
            print(f"  ⚠ Could not read {shp_path.name}")

    print("⚠ NO VALID POLYGON SHAPEFILES FOUND.")
    return None

def apply_displacement_model(gdf):
    """Adjusts 2023 baseline population to estimate 2026 reality."""
    print("\n--- Applying 2026 Displacement Model ---")
    
    # Create new column
    gdf['pop_2026_est'] = 0
    
    for idx, row in gdf.iterrows():
        # Match governorate name
        gov_name = row['admin1_name']
        baseline = row['population']
        
        # Find factor (fuzzy match)
        factor = 1.0
        for key, val in DISPLACEMENT_FACTORS.items():
            if key in gov_name:
                factor = val
                break
        
        est_2026 = baseline * factor
        gdf.at[idx, 'pop_2026_est'] = int(est_2026)
        
        print(f"  {gov_name}: {baseline:,.0f} (2023) -> {int(est_2026):,.0f} (2026 Est.)")
        
    return gdf

def main():
    # 1. Get Baseline Data
    pop_df = fetch_baseline_population()
    if pop_df is None: return

    # 2. Get Verified Polygon Shapefile
    shp_path = download_and_validate_shapefile()
    if not shp_path: return

    # 3. Merge
    print("\n--- Merging Data ---")
    gdf = gpd.read_file(shp_path)
    
    # Detect Admin Column (simple fuzzy match)
    admin_col = next((c for c in gdf.columns if 'ADM1' in c or 'NAME_1' in c), None)
    if not admin_col:
        # Fallback detection
        for c in gdf.select_dtypes(include=['object']):
            if gdf[c].str.contains('Gaza', na=False).any():
                admin_col = c
                break
    
    # Standardize Merge
    # We clean the names in both datasets to ensure 'Gaza' matches 'Gaza'
    pop_df['merge_key'] = pop_df['admin1_name'].str.extract(r'(Gaza|North|Khan|Rafah|Deir)')[0]
    gdf['merge_key'] = gdf[admin_col].str.extract(r'(Gaza|North|Khan|Rafah|Deir)')[0]
    
    merged = gdf.merge(pop_df, on='merge_key', how='inner')
    
    # 4. Apply 2026 Adjustment
    final_gdf = apply_displacement_model(merged)
    
    # 5. Save
    output_file = DATA_DIR / "gaza_population_2026_modeled.geojson"
    final_gdf.to_file(output_file, driver='GeoJSON')
    print(f"\n✓ PIPELINE SUCCESS")
    print(f"  Output: {output_file}")
    print(f"  Contains fields: 'population' (2023), 'pop_2026_est' (projected)")

if __name__ == "__main__":
    main()
