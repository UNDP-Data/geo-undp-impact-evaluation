#!/usr/bin/env python3
"""
analysis/pse/03_did_final.py
============================
Final Heterogeneous DiD Analysis.

Updates:
1. Robust Name Parsing: Fixes the 'index out of range' error.
2. Open-Ended Treatment: Treatment stays 'On' indefinitely after start date.
3. Panel Integration: Merges rainfall/prices from panel_data.csv.

Usage:
    python analysis/pse/03_did_final.py --buffer 250
"""

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings
from datetime import datetime
import statsmodels.formula.api as smf
import re

# Geometry / Spatial Imports
try:
    from scipy.spatial import cKDTree
    import rasterio
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"

# Inputs
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"
ACLED_CSV = DATA_DIR / "pse" / "external" / "acled_conflict.csv"
PANEL_CSV = Path("outputs/pse/panel_data.csv")
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Map cleaned site types to broad categories if needed, 
# or use the raw types from the CSV if they are clean enough.
CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Social Cohesion': ['Governance & Rule of Law', 'Social Inclusion'],
    'Health': ['Health & Emergency']
}

# Controls to pull from the District Panel
DISTRICT_CONTROLS = ['rainfall', 'price_bread', 'conflict_events']

# ================= DATA LOADING =================

def load_acled():
    if not ACLED_CSV.exists(): return None
    df = pd.read_csv(ACLED_CSV)
    if 'admin1' in df.columns:
        df = df[df['admin1'].str.contains('Gaza', case=False, na=False)]
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['year_month'] = df['event_date'].dt.to_period('M')
    return df.dropna(subset=['latitude', 'longitude'])

def load_and_prep_sites():
    if not SITES_CSV.exists() or not BOUNDARIES_FILE.exists():
        print("⚠ Missing Sites CSV or Boundary GeoJSON.")
        return None

    sites = pd.read_csv(SITES_CSV)
    projects = pd.read_csv(PROJECTS_CSV)
    projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
    
    # Assign Start Dates based on Category Matching
    sites['project_start'] = pd.NaT
    for idx, site in sites.iterrows():
        # Clean type to match dictionary keys if needed
        site_type = site.get('intervention_type', '').strip()
        cats = CATEGORY_MAP.get(site_type, [])
        
        matches = projects[projects['ml_category'].isin(cats)]
        if not matches.empty:
            # Take earliest project as the conservative start date
            best = matches.sort_values('start_date').iloc[0]
            sites.at[idx, 'project_start'] = best['start_date']
    
    sites = sites.dropna(subset=['project_start'])

    # Spatial Join for District Name
    sites_gdf = gpd.GeoDataFrame(
        sites, 
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs="EPSG:4326"
    )
    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
    
    # Find name column (e.g. adm2_name, unit_name)
    name_col = next((c for c in boundaries.columns if 'name' in c.lower()), None)
    if not name_col:
        # Fallback
        sites_joined = sites_gdf.copy()
        sites_joined['district_name'] = 'Gaza'
    else:
        sites_joined = gpd.sjoin(sites_gdf, boundaries[[name_col, 'geometry']], how="left", predicate="within")
        sites_joined = sites_joined.rename(columns={name_col: 'district_name'})
        sites_joined['district_name'] = sites_joined['district_name'].fillna('Unknown')
    
    print(f"Sites loaded: {len(sites_joined)}")
    return sites_joined

def load_district_panel():
    if not PANEL_CSV.exists(): return None
    df = pd.read_csv(PANEL_CSV)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Clean name for matching
    df['unit_name_clean'] = df['unit_name'].str.lower().str.strip()
    
    # Select cols
    cols = ['unit_name_clean', 'period_start'] + [c for c in DISTRICT_CONTROLS if c in df.columns]
    return df[cols].drop_duplicates(subset=['unit_name_clean', 'period_start'])

# ================= ANALYSIS =================

def build_analysis_dataset(sites, panel_controls, acled, rasters, buffer):
    data_records = []
    
    # ACLED Setup
    acled_trees = {}
    if acled is not None:
        for period, group in acled.groupby('year_month'):
            ts = period.to_timestamp()
            coords = group[['longitude', 'latitude']].values
            acled_trees[ts] = cKDTree(coords)

    # Process Rasters
    for indicator, dates in rasters.items():
        print(f"Processing {indicator}...")
        
        for date, raster_path in dates.items():
            month_key = pd.Timestamp(year=date.year, month=date.month, day=1)
            tree = acled_trees.get(month_key)
            
            # Get District Controls for this date
            if panel_controls is not None:
                monthly_controls = panel_controls[panel_controls['period_start'] == date]
            else:
                monthly_controls = pd.DataFrame()

            with rasterio.open(raster_path) as src:
                band = src.read(1)
                
                for i, site in sites.iterrows():
                    try:
                        # 1. Pixel Extraction
                        r, c = src.index(site['longitude'], site['latitude'])
                        val = band[r, c]
                        
                        # 2. Site Conflict Buffer
                        site_events = 0
                        if tree:
                            deg_radius = buffer / 100000.0 # Approx deg conversion
                            site_events = len(tree.query_ball_point([site['longitude'], site['latitude']], deg_radius))
                        
                        # 3. District Merge
                        d_name = str(site['district_name']).lower().strip()
                        d_row = monthly_controls[monthly_controls['unit_name_clean'] == d_name]
                        
                        # 4. Record Creation
                        record = {
                            'site_id': site['site_id'],
                            'type': site['intervention_type'],
                            'district': site['district_name'],
                            'date': date,
                            'indicator': indicator,
                            'value': val,
                            # Treatment: 1 if AFTER project start. Stays 1 forever (Open-ended).
                            'post': 1 if date >= site['project_start'] else 0,
                            'site_conflict_events': site_events
                        }
                        
                        # Add panel controls
                        if not d_row.empty:
                            for ctrl in DISTRICT_CONTROLS:
                                if ctrl in d_row.columns:
                                    record[f"dist_{ctrl}"] = d_row.iloc[0][ctrl]
                        
                        data_records.append(record)

                    except Exception as e:
                        # Skip bad pixels but continue
                        continue

    return pd.DataFrame(data_records)

def run_heterogeneous_regression(df):
    results = []
    df['date_str'] = df['date'].dt.strftime('%Y-%m')
    
    # Fill NAs
    fill_cols = [c for c in df.columns if 'dist_' in c]
    df[fill_cols] = df[fill_cols].fillna(0)
    
    # Loop Outcomes
    for indicator in df['indicator'].unique():
        subset = df[df['indicator'] == indicator].copy()
        if len(subset) < 50: continue
        
        print(f"\n--- Model: {indicator.upper()} ---")
        
        # MODEL: Value ~ Post*Type + Controls + DistrictFE + TimeFE
        formula = (
            "value ~ post:C(type) + "
            "site_conflict_events + dist_rainfall + "
            "C(district) + C(date_str)"
        )
        
        try:
            model = smf.ols(formula, data=subset).fit(cov_type='HC1')
            
            print(f"  R2: {model.rsquared:.3f} | N: {int(model.nobs)}")
            print("  Significant Effects:")
            
            # Robust Parameter Extraction
            # We look for terms that are interactions containing 'post'
            interactions = [x for x in model.params.index if ':' in x and 'post' in x]
            
            for param in interactions:
                # Robust Parse: Extract anything between [ and ]
                # e.g. "post:C(type)[T.WASH]" -> "T.WASH"
                # e.g. "post:C(type)[WASH]" -> "WASH"
                try:
                    # Extract content inside brackets
                    raw_content = param.split("[")[1].split("]")[0]
                    # Remove "T." prefix if standard encoding used
                    clean_name = raw_content.replace("T.", "")
                except:
                    # Fallback if name format is weird
                    clean_name = param
                
                coef = model.params[param]
                pval = model.pvalues[param]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                
                if stars: # Only print significant ones to reduce noise
                    print(f"    {clean_name:<20}: {coef:.4f} {stars}")
                
                results.append({
                    'indicator': indicator,
                    'intervention': clean_name,
                    'coef': coef,
                    'pval': pval,
                    'stars': stars,
                    'R2': model.rsquared
                })
                
        except Exception as e:
            print(f"Fit failed: {e}")

    return pd.DataFrame(results)

def find_rasters():
    rasters = {}
    for f in RASTER_DIR.glob("*.tif"):
        parts = f.stem.split('_')
        if len(parts) >= 3 and 'sar' not in f.name:
            ind = parts[0]
            try:
                d = datetime.strptime(parts[-1], "%Y%m")
                if ind not in rasters: rasters[ind] = {}
                rasters[ind][d] = f
            except: pass
    return rasters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=int, default=250)
    args = parser.parse_args()

    print("="*60)
    print(" HETEROGENEOUS DiD (FINAL)")
    print("="*60)
    
    sites = load_and_prep_sites()
    panel_controls = load_district_panel()
    acled = load_acled()
    rasters = find_rasters()
    
    if sites is None: return

    print(f"\nBuilding dataset (Buffer={args.buffer}m)...")
    df = build_analysis_dataset(sites, panel_controls, acled, rasters, args.buffer)
    
    if df.empty:
        print("\n❌ Dataset is empty. Check input paths.")
        return
        
    print(f"Dataset created: {len(df)} obs")
    
    res = run_heterogeneous_regression(df)
    if not res.empty:
        res.to_csv(TABLES_DIR / "did_heterogeneous_results.csv", index=False)
        print(f"\n✓ Saved: {TABLES_DIR}/did_heterogeneous_results.csv")

if __name__ == "__main__":
    main()
