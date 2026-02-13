#!/usr/bin/env python3
"""
analysis/pse/04_did_event_study_by_type.py
==========================================
Event Study by Intervention Type (Strict Post-Oct 7 Filter).

UPDATES:
1. STRICT DATE FILTER: Only matches projects starting >= 2023-10-07.
2. SEPARATE PLOTS: Generates one plot per Intervention Type per Outcome.

Usage:
    python analysis/pse/04_did_event_study_by_type.py --buffer 250
"""

import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import rasterio
from scipy.spatial import cKDTree
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"
ACLED_CSV = DATA_DIR / "pse" / "external" / "acled_conflict.csv"
PANEL_CSV = Path("outputs/pse/panel_data.csv")
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "event_studies"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")

CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Health': ['Health & Emergency']
}

# ================= DATA LOADING =================

def load_data():
    print("Loading Data...")
    sites = pd.read_csv(SITES_CSV)
    projects = pd.read_csv(PROJECTS_CSV)
    projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
    
    # --- CRITICAL FIX: FILTER FOR POST-OCT 7 PROJECTS ONLY ---
    print(f"  Total Projects available: {len(projects)}")
    projects = projects[projects['start_date'] >= CONFLICT_START]
    print(f"  > Projects starting AFTER Oct 7, 2023: {len(projects)}")
    
    if len(projects) == 0:
        print("  ⚠ ERROR: No projects found after Oct 7. Check your CSV dates!")
        return None, None, None

    # Match Sites
    sites['project_start'] = pd.NaT
    for idx, site in sites.iterrows():
        site_type = site.get('intervention_type', '').strip()
        cats = CATEGORY_MAP.get(site_type, [])
        matches = projects[projects['ml_category'].isin(cats)]
        if not matches.empty:
            best = matches.sort_values('start_date').iloc[0]
            sites.at[idx, 'project_start'] = best['start_date']
    
    # Drop unmatched sites
    sites = sites.dropna(subset=['project_start'])
    print(f"  > Sites matched to Crisis Response projects: {len(sites)}")

    # Spatial Join
    sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.longitude, sites.latitude), crs="EPSG:4326")
    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("EPSG:4326")
    name_col = next((c for c in boundaries.columns if 'name' in c.lower()), None)
    
    if name_col:
        sites_joined = gpd.sjoin(sites_gdf, boundaries[[name_col, 'geometry']], how="left", predicate="within")
        sites_joined = sites_joined.rename(columns={name_col: 'district_name'})
        sites_joined['district_name'] = sites_joined['district_name'].fillna('Unknown')
    else:
        sites_joined['district_name'] = 'Gaza'
        
    # Controls
    panel = pd.read_csv(PANEL_CSV)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_name_clean'] = panel['unit_name'].str.lower().str.strip()
    
    # ACLED
    acled = None
    if ACLED_CSV.exists():
        acled = pd.read_csv(ACLED_CSV)
        acled['event_date'] = pd.to_datetime(acled['event_date'])
        acled['year_month'] = acled['event_date'].dt.to_period('M')

    return sites_joined, panel, acled

def build_event_dataset(sites, panel_controls, acled, buffer):
    print("\nExtracting Raster Data...")
    
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
            
    acled_trees = {}
    if acled is not None:
        for period, group in acled.groupby('year_month'):
            ts = period.to_timestamp()
            acled_trees[ts] = cKDTree(group[['longitude', 'latitude']].values)

    data_records = []
    
    for indicator, dates in rasters.items():
        for date, raster_path in dates.items():
            month_key = pd.Timestamp(year=date.year, month=date.month, day=1)
            tree = acled_trees.get(month_key)
            monthly_controls = panel_controls[panel_controls['period_start'] == date]
            
            with rasterio.open(raster_path) as src:
                band = src.read(1)
                for _, site in sites.iterrows():
                    try:
                        r, c = src.index(site['longitude'], site['latitude'])
                        val = band[r, c]
                        
                        # TIME CALC
                        diff = (date.year - site['project_start'].year) * 12 + (date.month - site['project_start'].month)
                        
                        # Binning (-6 to +12)
                        if diff < -6: rel_month = -7
                        elif diff > 12: rel_month = 13
                        else: rel_month = diff
                        
                        site_events = 0
                        if tree:
                            site_events = len(tree.query_ball_point([site['longitude'], site['latitude']], buffer/100000.0))
                            
                        d_name = str(site['district_name']).lower().strip()
                        d_row = monthly_controls[monthly_controls['unit_name_clean'] == d_name]
                        dist_rain = d_row['rainfall'].values[0] if 'rainfall' in d_row and not d_row.empty else 0
                        
                        data_records.append({
                            'site_id': site['site_id'],
                            'type': site['intervention_type'],
                            'district': site['district_name'],
                            'date': date,
                            'indicator': indicator,
                            'value': val,
                            'rel_month': rel_month,
                            'site_conflict': site_events,
                            'dist_rainfall': dist_rain
                        })
                    except: continue
                    
    return pd.DataFrame(data_records)

def plot_event_study(model, indicator, int_type, output_path):
    coefs, cis_lower, cis_upper, months = [], [], [], []
    plot_range = range(-7, 14)
    
    for t in plot_range:
        if t == -1: 
            coefs.append(0); cis_lower.append(0); cis_upper.append(0); months.append(t)
            continue
        
        name = f"C(rel_month, Treatment(reference=-1))[T.{t}]"
        if name in model.params:
            coefs.append(model.params[name])
            ci = model.conf_int().loc[name]
            cis_lower.append(ci[0])
            cis_upper.append(ci[1])
            months.append(t)
    
    if len(months) < 2: return

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axvline(-0.5, color='red', linestyle='--', label='Project Start')
    
    plt.errorbar(months, coefs, 
                 yerr=[np.array(coefs)-np.array(cis_lower), np.array(cis_upper)-np.array(coefs)], 
                 fmt='o-', color='#2c3e50', ecolor='#95a5a6', capsize=4, label='Estimate (95% CI)')
    
    plt.title(f'Event Study: {indicator.upper()} - {int_type}\n(Impact relative to start date)', fontweight='bold')
    plt.xlabel('Months Relative to Start (<-6: Pre-Bin | >12: Post-Bin)')
    plt.ylabel('Impact')
    
    labels = [str(m) for m in months]
    if -7 in months: labels[labels.index('-7')] = '<-6'
    if 13 in months: labels[labels.index('13')] = '>12'
    plt.xticks(months, labels)
    
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")

def main():
    print("="*60)
    print(" EVENT STUDY BY TYPE (STRICT POST-OCT 7)")
    print("="*60)
    
    sites, panel, acled = load_data()
    if sites is None or len(sites) == 0: return

    df = build_event_dataset(sites, panel, acled, buffer=250)
    
    if df.empty:
        print("Dataset empty.")
        return

    # Loop by Indicator AND Intervention Type
    for indicator in df['indicator'].unique():
        print(f"\nProcessing {indicator.upper()}...")
        
        for int_type in df['type'].unique():
            # Filter for specific type
            subset = df[(df['indicator'] == indicator) & (df['type'] == int_type)].copy()
            
            if len(subset) < 50:
                print(f"  > Skipping {int_type} (not enough data: {len(subset)})")
                continue
                
            formula = (
                "value ~ C(rel_month, Treatment(reference=-1)) + "
                "site_conflict + dist_rainfall + "
                "C(district) + C(date)"
            )
            
            try:
                model = smf.ols(formula, data=subset).fit(cov_type='HC1')
                out_file = FIGURES_DIR / f"event_study_{indicator}_{int_type}.png"
                plot_event_study(model, indicator, int_type, out_file)
            except Exception as e:
                print(f"  Failed {int_type}: {e}")

if __name__ == "__main__":
    main()
