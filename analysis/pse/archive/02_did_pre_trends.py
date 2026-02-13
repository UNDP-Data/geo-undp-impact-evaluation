#!/usr/bin/env python3
"""
analysis/pse/05_comparative_trends_fixed.py
===========================================
Comparative Trends Analysis (Fixed for Geopandas Versions).

Updates:
1. FIX: Replaced 'sindex.nearest' with direct distance check (Fixes TypeError).
2. STRICT FILTER: Only projects starting >= 2023-10-07.
3. PLOTS: Generates Treated vs. Control trends per intervention type.

Usage:
    python analysis/pse/05_comparative_trends_fixed.py
"""

import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio
from shapely.geometry import Point
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "comparative_trends"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CRITICAL: The start date of the conflict/response
CONFLICT_START_DATE = pd.Timestamp("2023-10-07")
CONFLICT_MARKER = datetime(2023, 10, 7)

CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Health': ['Health & Emergency']
}

# ================= DATA PREP =================

def generate_control_points(gdf_boundary, gdf_treated, n_controls=300):
    """Generate random points >1km from treated sites."""
    print(f"Generating {n_controls} Control Points...")
    control_points = []
    minx, miny, maxx, maxy = gdf_boundary.total_bounds
    
    attempts = 0
    
    while len(control_points) < n_controls and attempts < 50000:
        attempts += 1
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
        # 1. Must be inside Gaza boundaries
        if not gdf_boundary.contains(pnt).any():
            continue
            
        # 2. Must be > 1km from any Treated Site (approx 0.009 degrees)
        # FIX: Direct distance calculation (Safe for all Geopandas versions)
        # Note: warning about CRS is expected if using Lat/Lon, but fine for this approx check
        min_dist = gdf_treated.distance(pnt).min()
        
        if min_dist < 0.009: # Approx 1km
            continue
            
        control_points.append({
            'site_id': f'CONTROL_{len(control_points):03d}',
            'intervention_type': 'Control',
            'latitude': pnt.y,
            'longitude': pnt.x,
            'group': 'Control'
        })
        
    print(f"  > Created {len(control_points)} valid control points.")
    return pd.DataFrame(control_points)

def load_data():
    print("Loading Data...")
    sites = pd.read_csv(SITES_CSV)
    projects = pd.read_csv(PROJECTS_CSV)
    projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
    
    # --- STRICT FILTER FOR POST-OCT 7 PROJECTS ---
    print(f"  Total projects: {len(projects)}")
    projects = projects[projects['start_date'] >= CONFLICT_START_DATE]
    print(f"  > Projects starting AFTER Oct 7, 2023: {len(projects)}")
    
    if len(projects) == 0:
        print("  ⚠ ERROR: No recent projects found. Check CSV dates.")
        return None

    # Match Sites
    sites['project_start'] = pd.NaT
    for idx, site in sites.iterrows():
        site_type = site.get('intervention_type', '').strip()
        cats = CATEGORY_MAP.get(site_type, [])
        matches = projects[projects['ml_category'].isin(cats)]
        if not matches.empty:
            best = matches.sort_values('start_date').iloc[0]
            sites.at[idx, 'project_start'] = best['start_date']
    
    # Keep only sites matched to recent projects
    sites = sites.dropna(subset=['project_start'])
    sites['group'] = 'Treated'
    print(f"  > Sites matched to recent projects: {len(sites)}")
    
    if len(sites) == 0: return None

    # Load Boundaries & Create Controls
    bounds = gpd.read_file(BOUNDARIES_FILE)
    sites_gdf = gpd.GeoDataFrame(
        sites, geometry=gpd.points_from_xy(sites.longitude, sites.latitude), crs="EPSG:4326"
    )
    # Generate controls equal to the number of treated sites for balance
    controls_df = generate_control_points(bounds, sites_gdf, n_controls=len(sites))
    
    controls_df['project_start'] = pd.NaT 
    combined = pd.concat([sites, controls_df], ignore_index=True)
    return combined

def extract_time_series(sites):
    print("Extracting Time Series Data...")
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

    records = []
    for indicator, dates in rasters.items():
        # print(f"  > {indicator}...")
        for date, path in dates.items():
            with rasterio.open(path) as src:
                band = src.read(1)
                for _, site in sites.iterrows():
                    try:
                        r, c = src.index(site['longitude'], site['latitude'])
                        val = band[r, c]
                        records.append({
                            'group': site['group'],
                            'type': site['intervention_type'],
                            'date': date,
                            'indicator': indicator,
                            'value': val,
                            'project_start': site.get('project_start', pd.NaT)
                        })
                    except: continue
    return pd.DataFrame(records)

# ================= PLOTTING =================

def plot_comparative_trend(df, indicator, intervention_type):
    # Filter Data
    controls = df[df['group'] == 'Control']
    treated = df[df['type'] == intervention_type]
    
    if treated.empty or controls.empty: return

    # Aggregate by Month
    ctrl_trend = controls.groupby('date')['value'].mean()
    treat_trend = treated.groupby('date')['value'].mean()
    
    # Get Start Dates for Shading
    starts = treated['project_start'].dropna().unique()
    if len(starts) > 0:
        start_min = min(starts)
        median_start = pd.Timestamp(np.median(starts.astype(np.int64)))
    else:
        start_min, median_start = None, None

    # PLOT
    plt.figure(figsize=(12, 6))
    
    # Lines
    plt.plot(treat_trend.index, treat_trend.values, 
             color='#e74c3c', linewidth=2.5, marker='o', markersize=6, label=f'Treated: {intervention_type}')
    plt.plot(ctrl_trend.index, ctrl_trend.values, 
             color='#3498db', linewidth=2.5, linestyle='--', marker='x', markersize=6, label='Control: General Gaza (>1km away)')
    
    # Conflict Marker
    plt.axvline(CONFLICT_MARKER, color='black', linestyle=':', linewidth=2, label='Conflict Start (Oct 7)')
    
    # Intervention Shading
    if start_min:
        plt.axvspan(start_min, treat_trend.index.max(), 
                    color='#e74c3c', alpha=0.1, label='Intervention Period (Range)')
        plt.axvline(median_start, color='#e74c3c', linestyle='-', alpha=0.6, linewidth=1.5, label='Median Start Date')

    # Formatting
    plt.title(f'{indicator.upper()}: {intervention_type} Sites vs. Control\n(Comparative Trend Analysis)', fontweight='bold')
    plt.ylabel('Outcome Value (Mean)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    
    out_file = FIGURES_DIR / f"trend_{indicator}_{intervention_type}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {out_file.name}")

def main():
    print("="*60)
    print(" COMPARATIVE TRENDS (STRICT POST-OCT 7 FILTER)")
    print("="*60)
    
    sites_combined = load_data()
    if sites_combined is None: return

    df = extract_time_series(sites_combined)
    if df.empty:
        print("❌ No data extracted.")
        return

    # Loop through intervention types
    intervention_types = [t for t in df['type'].unique() if t != 'Control']
    indicators = df['indicator'].unique()
    
    print(f"\nGenerating plots for {len(intervention_types)} intervention types across {len(indicators)} indicators...")
    
    for indicator in indicators:
        for int_type in intervention_types:
            plot_comparative_trend(df[df['indicator'] == indicator], indicator, int_type)
            
    print(f"\nDone. Check {FIGURES_DIR}")

if __name__ == "__main__":
    main()
