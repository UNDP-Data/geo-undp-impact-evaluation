#!/usr/bin/env python3
"""
analysis/pse/02_did_pretrends.py
================================
Pre-Trends and Comparative Trends Analysis with Donut Controls.

Key Features:
1. DONUT CONTROL GROUP: Controls between 500m-1.5km from treatment sites
2. Clean, publication-ready visualizations
3. Statistical pre-trend tests
4. Fixed title formatting

Usage:
    python analysis/pse/02_did_pretrends.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import rasterio
    from shapely.geometry import Point
    from scipy.spatial import cKDTree
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "pse" / "rasters"
SITES_CSV = DATA_DIR / "pse" / "external" / "undp_sites_digitized.csv"
PROJECTS_CSV = DATA_DIR / "pse" / "external" / "undp_projects.csv"
BOUNDARIES_FILE = DATA_DIR / "pse" / "boundaries" / "gaza_admin2.geojson"

OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")
CONFLICT_MARKER = datetime(2023, 10, 7)

# =============================================================================
# DONUT CONTROL PARAMETERS
# =============================================================================
INNER_BUFFER_M = 500    # Minimum distance from treatment (avoid spillovers)
OUTER_BUFFER_M = 1500   # Maximum distance from treatment (ensure comparability)
DEG_PER_M = 1 / 100000  # Approximate conversion
INNER_BUFFER_DEG = INNER_BUFFER_M * DEG_PER_M
OUTER_BUFFER_DEG = OUTER_BUFFER_M * DEG_PER_M

CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Social Cohesion': ['Governance & Rule of Law', 'Social Inclusion'],
    'Other': ['Health & Emergency'],
    'Health': ['Health & Emergency']
}

# Outcome display names
OUTCOME_NAMES = {
    'viirs': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'sar_vh': 'SAR Backscatter (VH)',
    'sar_vv': 'SAR Backscatter (VV)',
}

# =============================================================================
# DONUT CONTROL GENERATION
# =============================================================================

def generate_donut_controls(gdf_boundary, gdf_treated, n_controls=None):
    """
    Generate control points in the "donut" zone around treatment sites.
    Controls are 500m-1.5km from the nearest treatment site.
    """
    if n_controls is None:
        n_controls = len(gdf_treated)
    
    print(f"  Generating {n_controls} donut controls ({INNER_BUFFER_M}m-{OUTER_BUFFER_M}m)...")
    
    # Build KD-tree for fast queries
    treated_coords = np.array([[p.x, p.y] for p in gdf_treated.geometry])
    tree = cKDTree(treated_coords)
    
    control_points = []
    minx, miny, maxx, maxy = gdf_boundary.total_bounds
    
    attempts = 0
    max_attempts = 200000
    
    while len(control_points) < n_controls and attempts < max_attempts:
        attempts += 1
        
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        
        pnt = Point(x, y)
        if not gdf_boundary.contains(pnt).any():
            continue
        
        dist, _ = tree.query([x, y])
        
        if dist < INNER_BUFFER_DEG or dist > OUTER_BUFFER_DEG:
            continue
        
        control_points.append({
            'site_id': f'CTRL_{len(control_points):04d}',
            'intervention_type': 'Control',
            'latitude': y,
            'longitude': x,
            'group': 'Control',
            'project_start': pd.NaT
        })
    
    print(f"    Created {len(control_points)} valid donut controls")
    return pd.DataFrame(control_points)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load treated sites and generate donut controls."""
    print("\n--- Loading Data ---")
    
    if not SITES_CSV.exists():
        print(f"  ⚠ Sites CSV not found: {SITES_CSV}")
        return None
    
    sites = pd.read_csv(SITES_CSV)
    
    # Load projects and match
    if PROJECTS_CSV.exists():
        projects = pd.read_csv(PROJECTS_CSV)
        projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
        
        projects = projects[projects['start_date'] >= CONFLICT_START]
        print(f"  Post-conflict projects: {len(projects)}")
        
        sites['project_start'] = pd.NaT
        for idx, site in sites.iterrows():
            site_type = site.get('intervention_type', '').strip()
            cats = CATEGORY_MAP.get(site_type, [])
            matches = projects[projects['ml_category'].isin(cats)]
            if not matches.empty:
                best = matches.sort_values('start_date').iloc[0]
                sites.at[idx, 'project_start'] = best['start_date']
        
        sites = sites.dropna(subset=['project_start'])
    
    sites['group'] = 'Treated'
    print(f"  Treated sites: {len(sites)}")
    
    if len(sites) == 0:
        return None
    
    # Load boundaries and create donut controls
    if BOUNDARIES_FILE.exists():
        bounds = gpd.read_file(BOUNDARIES_FILE)
        sites_gdf = gpd.GeoDataFrame(
            sites,
            geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
            crs="EPSG:4326"
        )
        
        controls_df = generate_donut_controls(bounds, sites_gdf, n_controls=len(sites))
        combined = pd.concat([sites, controls_df], ignore_index=True)
        return combined
    
    return sites


def find_all_rasters():
    """Find all raster files including SAR."""
    rasters = {}
    
    if not RASTER_DIR.exists():
        return rasters
    
    for f in RASTER_DIR.glob("*.tif"):
        parts = f.stem.split('_')
        
        if 'sar' in f.name.lower():
            if 'vh' in f.name.lower():
                ind = 'sar_vh'
            elif 'vv' in f.name.lower():
                ind = 'sar_vv'
            else:
                continue
        else:
            ind = parts[0].lower()
        
        for part in parts:
            try:
                if len(part) == 6 and part.isdigit():
                    d = datetime.strptime(part, "%Y%m")
                    if ind not in rasters:
                        rasters[ind] = {}
                    rasters[ind][d] = f
                    break
            except:
                continue
    
    return rasters


def extract_time_series(sites, rasters):
    """Extract raster values at all sites across time."""
    print("\n--- Extracting Time Series ---")
    
    records = []
    
    for indicator, dates in rasters.items():
        print(f"  Processing {indicator} ({len(dates)} dates)...")
        
        for date, path in dates.items():
            try:
                with rasterio.open(path) as src:
                    band = src.read(1)
                    
                    for _, site in sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            val = band[r, c]
                            
                            if val == src.nodata or np.isnan(val):
                                continue
                            
                            records.append({
                                'site_id': site['site_id'],
                                'group': site['group'],
                                'type': site['intervention_type'],
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                                'project_start': site.get('project_start', pd.NaT)
                            })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def test_parallel_trends(df, indicator, int_type):
    """
    Formal test of parallel pre-trends.
    H0: Pre-treatment trends are parallel between treated and control.
    """
    pre_data = df[(df['date'] < CONFLICT_START) & 
                  ((df['type'] == int_type) | (df['group'] == 'Control'))]
    
    if len(pre_data) < 20:
        return {'passed': None, 'pvalue': None, 'message': 'Insufficient data'}
    
    pre_data = pre_data.copy()
    pre_data['month_idx'] = (pre_data['date'].dt.year * 12 + pre_data['date'].dt.month)
    pre_data['is_treated'] = (pre_data['group'] == 'Treated').astype(int)
    
    try:
        model = smf.ols('value ~ month_idx * is_treated', data=pre_data).fit()
        
        interaction_term = 'month_idx:is_treated'
        if interaction_term in model.pvalues:
            pval = model.pvalues[interaction_term]
            coef = model.params[interaction_term]
            passed = pval > 0.10
            
            return {
                'passed': passed,
                'pvalue': pval,
                'coef': coef,
                'message': '✓ Parallel' if passed else '⚠ Non-parallel'
            }
    except:
        pass
    
    return {'passed': None, 'pvalue': None, 'message': 'Test failed'}


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparative_trend_improved(df, indicator, intervention_type, output_path):
    """Create improved comparative trend plot with donut controls."""
    controls = df[df['group'] == 'Control']
    treated = df[df['type'] == intervention_type]
    
    if treated.empty or controls.empty:
        return None, None
    
    # Aggregate by date with SE
    ctrl_agg = controls.groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    ctrl_agg['se'] = ctrl_agg['std'] / np.sqrt(ctrl_agg['count'])
    ctrl_agg['ci_lower'] = ctrl_agg['mean'] - 1.96 * ctrl_agg['se']
    ctrl_agg['ci_upper'] = ctrl_agg['mean'] + 1.96 * ctrl_agg['se']
    
    treat_agg = treated.groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    treat_agg['se'] = treat_agg['std'] / np.sqrt(treat_agg['count'])
    treat_agg['ci_lower'] = treat_agg['mean'] - 1.96 * treat_agg['se']
    treat_agg['ci_upper'] = treat_agg['mean'] + 1.96 * treat_agg['se']
    
    # Intervention start dates
    starts = treated['project_start'].dropna().unique()
    median_start = pd.Timestamp(np.median([s.value for s in starts])) if len(starts) > 0 else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_name = OUTCOME_NAMES.get(indicator, indicator.upper())
    
    # Shaded regions
    ax.axvspan(ctrl_agg['date'].min(), CONFLICT_MARKER, alpha=0.05, color='gray', label='Pre-Conflict')
    if median_start:
        ax.axvspan(CONFLICT_MARKER, median_start, alpha=0.1, color='red', label='Conflict Period')
        ax.axvspan(median_start, ctrl_agg['date'].max(), alpha=0.1, color='green', label='Intervention Period')
    
    # Control group with CI
    ax.fill_between(ctrl_agg['date'], ctrl_agg['ci_lower'], ctrl_agg['ci_upper'],
                    alpha=0.2, color='#3498db')
    ax.plot(ctrl_agg['date'], ctrl_agg['mean'],
            color='#3498db', linewidth=2.5, linestyle='--', marker='s', markersize=5,
            label=f'Donut Control ({INNER_BUFFER_M}m-{OUTER_BUFFER_M}m)')
    
    # Treatment group with CI
    ax.fill_between(treat_agg['date'], treat_agg['ci_lower'], treat_agg['ci_upper'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treat_agg['date'], treat_agg['mean'],
            color='#e74c3c', linewidth=2.5, marker='o', markersize=6,
            label=f'Treated: {intervention_type}')
    
    # Vertical lines
    ax.axvline(CONFLICT_MARKER, color='black', linestyle=':', linewidth=2, label='Conflict Start (Oct 7)')
    if median_start:
        ax.axvline(median_start, color='#27ae60', linestyle='-', linewidth=2, 
                   alpha=0.7, label='Median Intervention Start')
    
    # Run parallel trends test
    test_result = test_parallel_trends(df, indicator, intervention_type)
    
    # Build title with proper formatting
    pval = test_result.get("pvalue")
    if pval is not None:
        pval_str = f"(p={pval:.3f})"
    else:
        pval_str = ""
    
    ax.set_title(f'Parallel Trends: {outcome_name} — {intervention_type} vs Donut Control\n'
                 f'Pre-trends test: {test_result["message"]} {pval_str}',
                 fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    # Sample size annotation
    n_treat = len(treated['site_id'].unique())
    n_ctrl = len(controls['site_id'].unique())
    ax.annotate(f'N: {n_treat} treated, {n_ctrl} control sites',
               xy=(0.98, 0.02), xycoords='axes fraction',
               fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return test_result.get('message', ''), pval_str


def create_pretrends_summary_table(df, output_path):
    """Create summary table of pre-trends tests."""
    results = []
    
    indicators = [i for i in df['indicator'].unique()]
    int_types = [t for t in df['type'].unique() if t != 'Control']
    
    for indicator in indicators:
        ind_df = df[df['indicator'] == indicator]
        
        for int_type in int_types:
            test = test_parallel_trends(ind_df, indicator, int_type)
            
            results.append({
                'indicator': indicator,
                'intervention': int_type,
                'parallel_trends_passed': test['passed'],
                'pvalue': test['pvalue'],
                'message': test['message']
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n  Pre-Trends Test Summary (Donut Controls):")
    print("  " + "=" * 60)
    
    passed = results_df[results_df['parallel_trends_passed'] == True]
    failed = results_df[results_df['parallel_trends_passed'] == False]
    
    print(f"    ✓ Passed (parallel): {len(passed)}")
    print(f"    ⚠ Failed: {len(failed)}")
    
    if not failed.empty:
        print("\n    Non-parallel pre-trends:")
        for _, row in failed.head(10).iterrows():
            pval = row['pvalue']
            if pval is not None:
                print(f"      {row['indicator']} × {row['intervention']}: p={pval:.3f}")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" PRE-TRENDS ANALYSIS WITH DONUT CONTROLS")
    print(f" Control group: {INNER_BUFFER_M}m - {OUTER_BUFFER_M}m from treatment")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    sites = load_data()
    if sites is None:
        print("\n❌ Failed to load sites. Exiting.")
        return
    
    # Find rasters
    rasters = find_all_rasters()
    print(f"\n  Rasters found: {list(rasters.keys())}")
    
    # Extract time series
    df = extract_time_series(sites, rasters)
    
    if df.empty:
        print("\n❌ No data extracted. Exiting.")
        return
    
    n_treated = len(df[df['group'] == 'Treated']['site_id'].unique())
    n_control = len(df[df['group'] == 'Control']['site_id'].unique())
    print(f"\n  Dataset: {len(df):,} observations")
    print(f"  Sites: {n_treated} treated, {n_control} control")
    print(f"  Indicators: {df['indicator'].unique().tolist()}")
    
    # Generate plots
    print("\n--- Generating Comparative Trend Plots ---")
    
    indicators = df['indicator'].unique()
    int_types = [t for t in df['type'].unique() if t != 'Control']
    
    for indicator in indicators:
        print(f"\n  {indicator.upper()}:")
        ind_df = df[df['indicator'] == indicator]
        
        for int_type in int_types:
            clean_type = int_type.lower().replace(' ', '_')
            output_path = FIGURES_DIR / f"pretrends_{indicator}_{clean_type}.png"
            status, pval = plot_comparative_trend_improved(ind_df, indicator, int_type, output_path)
            if status:
                print(f"    ✓ {int_type}: {status} {pval}")
    
    # Summary table
    print("\n--- Pre-Trends Statistical Tests ---")
    create_pretrends_summary_table(df, TABLES_DIR / "pretrends_tests.csv")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    - {FIGURES_DIR}/ (trend plots)")
    print(f"    - {TABLES_DIR}/pretrends_tests.csv")


if __name__ == "__main__":
    main()
