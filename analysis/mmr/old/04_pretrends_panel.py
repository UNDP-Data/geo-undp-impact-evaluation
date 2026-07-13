#!/usr/bin/env python3
"""
analysis/mmr/04_pretrends_panel.py
==================================
Pre-Trends Analysis for Myanmar using PANEL DATA.

This script tests parallel trends assumption for the DiD analysis.
Uses township-level panel data (not pixel-level rasters like Gaza).

KEY DIFFERENCES FROM GAZA:
- Uses panel_data.csv (township-level aggregated data)
- Buffer specs are 500m-20km (not 200m-2km)
- Treatment = townships within X km of UNDP site centroids
- Pre-period = before earthquake (March 28, 2025)

The test checks whether treatment and control groups had similar trends
BEFORE the earthquake. If the interaction term (time × treated) is
not significant (p > 0.10), parallel trends assumption holds.

Usage:
    python analysis/mmr/04_pretrends_panel.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# Key date
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")

# Buffer specifications (meters) - ADAPTED FOR MYANMAR (larger than Gaza)
BUFFER_SPECS = [
    (1000, 3000, "1km-3km", "tight"),
    (2000, 5000, "2km-5km", "moderate"),
    (3000, 8000, "3km-8km", "baseline"),
    (5000, 15000, "5km-15km", "wide"),
    (8000, 20000, "8km-20km", "very_wide"),
]

# Outcomes to test
OUTCOMES = {
    'nightlights': 'VIIRS Nightlights',
    'log_nightlights': 'Log Nightlights',
    'ndvi': 'NDVI (Vegetation)',
    'vh': 'SAR VH (Structures)',
    'vv': 'SAR VV',
    'rainfall': 'Rainfall',
}


def load_panel_data():
    """Load panel data."""
    if not PANEL_FILE.exists():
        return None
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Create log nightlights if not exists
    if 'nightlights' in df.columns and 'log_nightlights' not in df.columns:
        df['log_nightlights'] = np.log(df['nightlights'] + 1)
    
    return df


def load_boundaries():
    """Load boundaries with centroids."""
    if not BOUNDARIES_FILE.exists():
        return None, None
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    gdf['centroid'] = gdf.geometry.centroid
    gdf['center_lon'] = gdf['centroid'].x
    gdf['center_lat'] = gdf['centroid'].y
    
    return gdf, id_col


def load_undp_sites():
    """Load UNDP sites."""
    if not UNDP_SITES_FILE.exists():
        return None
    
    df = pd.read_csv(UNDP_SITES_FILE)
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    
    if lat_col is None or lon_col is None:
        return None
    
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df = df.dropna(subset=['lat', 'lon'])
    
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calculate_min_distances(gdf, sites, id_col):
    """Calculate minimum distance from each township to nearest UNDP site."""
    distances = {}
    
    site_lats = sites['lat'].values
    site_lons = sites['lon'].values
    
    for _, row in gdf.iterrows():
        unit_id = row[id_col]
        unit_lat = row['center_lat']
        unit_lon = row['center_lon']
        
        dists = haversine_distance(unit_lat, unit_lon, site_lats, site_lons)
        distances[unit_id] = np.min(dists)
    
    return distances


def create_treatment_control_groups(panel, distances, inner_m, outer_m):
    """
    Create treatment and control groups based on distance buffers.
    
    Treatment: Townships within inner_m of nearest UNDP site
    Control: Townships between inner_m and outer_m
    """
    df = panel.copy()
    df['min_dist'] = df['unit_id'].map(distances)
    
    df['is_treated'] = (df['min_dist'] <= inner_m).astype(int)
    df['is_control'] = ((df['min_dist'] > inner_m) & (df['min_dist'] <= outer_m)).astype(int)
    
    # Keep only treated + control
    df = df[(df['is_treated'] == 1) | (df['is_control'] == 1)].copy()
    
    return df


def test_parallel_trends(df, outcome):
    """
    Test parallel trends in PRE-EARTHQUAKE period.
    
    Model: outcome ~ month_idx * is_treated
    
    If interaction term (month_idx:is_treated) is NOT significant (p > 0.10),
    parallel trends assumption holds.
    """
    # Filter to pre-earthquake only
    pre_data = df[df['period_start'] < EARTHQUAKE_DATE].copy()
    
    if outcome not in pre_data.columns:
        return None, None, None
    
    pre_data = pre_data.dropna(subset=[outcome])
    
    if len(pre_data) < 50:
        return None, None, None
    
    # Create month index
    pre_data['month_idx'] = pre_data['period_start'].dt.year * 12 + pre_data['period_start'].dt.month
    
    try:
        model = smf.ols(f'{outcome} ~ month_idx * is_treated', data=pre_data).fit()
        
        if 'month_idx:is_treated' in model.pvalues:
            pval = model.pvalues['month_idx:is_treated']
            coef = model.params['month_idx:is_treated']
            passed = pval > 0.10  # Pass if NOT significant
            return passed, pval, coef
    except Exception as e:
        print(f"    ⚠ Error: {e}")
    
    return None, None, None


def create_pretrends_plot(df, outcome, outcome_name, spec_label, passed, pval, output_path):
    """
    Create pre-trends plot showing treated vs control over time.
    """
    
    # Aggregate by month and group
    monthly = df.groupby(['period_start', 'is_treated']).agg({
        outcome: ['mean', 'std', 'count']
    }).reset_index()
    monthly.columns = ['month', 'is_treated', 'mean', 'std', 'count']
    monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
    
    treated = monthly[monthly['is_treated'] == 1].sort_values('month')
    control = monthly[monthly['is_treated'] == 0].sort_values('month')
    
    if len(treated) < 2 or len(control) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Shading
    xmin, xmax = monthly['month'].min(), monthly['month'].max()
    if xmin < EARTHQUAKE_DATE:
        ax.axvspan(xmin, min(EARTHQUAKE_DATE, xmax), alpha=0.1, color='green')
    if xmax > EARTHQUAKE_DATE:
        ax.axvspan(max(EARTHQUAKE_DATE, xmin), xmax, alpha=0.08, color='red')
    
    # Earthquake line
    ax.axvline(EARTHQUAKE_DATE, color='red', linewidth=2.5, linestyle='--', label='Earthquake (Mar 28, 2025)')
    
    # Treated
    ax.fill_between(treated['month'],
                    treated['mean'] - 1.96*treated['se'],
                    treated['mean'] + 1.96*treated['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treated['month'], treated['mean'], 'o-', color='#e74c3c',
            linewidth=2, markersize=6, label='Treated (Near UNDP sites)')
    
    # Control
    ax.fill_between(control['month'],
                    control['mean'] - 1.96*control['se'],
                    control['mean'] + 1.96*control['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(control['month'], control['mean'], 's--', color='#3498db',
            linewidth=2, markersize=6, label=f'Control ({spec_label})')
    
    # Pre-trends result
    if passed is not None:
        if passed:
            status = f"✓ Parallel Trends HOLD (p={pval:.3f})"
            color = '#27ae60'
        else:
            status = f"⚠ Parallel Trends FAIL (p={pval:.3f})"
            color = '#e74c3c'
        
        ax.annotate(status, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top', fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.95))
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Pre-Trends Analysis: {outcome_name}\nControl Group: {spec_label}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.text(0.15, 0.02, 'Pre-Earthquake', transform=ax.transAxes, fontsize=10,
           color='green', fontweight='bold', ha='center')
    ax.text(0.75, 0.02, 'Post-Earthquake', transform=ax.transAxes, fontsize=10,
           color='red', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_heatmap(all_results, output_dir):
    """Create summary heatmap of all pre-trends tests."""
    
    try:
        import seaborn as sns
    except ImportError:
        print("  ⚠ seaborn not installed, skipping heatmap")
        return
    
    specs = [s[2] for s in BUFFER_SPECS]
    outcomes_list = list(OUTCOMES.keys())
    
    matrix = np.full((len(outcomes_list), len(specs)), np.nan)
    annotations = np.empty((len(outcomes_list), len(specs)), dtype=object)
    
    for i, outcome in enumerate(outcomes_list):
        for j, spec_label in enumerate(specs):
            key = (spec_label, outcome)
            if key in all_results:
                passed, pval = all_results[key]
                if passed is not None:
                    matrix[i, j] = 1 if passed else 0
                    annotations[i, j] = f"p={pval:.2f}" if pval is not None else "N/A"
                else:
                    annotations[i, j] = "N/A"
            else:
                annotations[i, j] = "N/A"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(matrix, ax=ax, annot=annotations, fmt='', cmap=cmap,
                xticklabels=specs,
                yticklabels=[OUTCOMES[o] for o in outcomes_list],
                cbar_kws={'label': 'Pre-Trends Test (1=PASS, 0=FAIL)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Myanmar Pre-Trends Test Results\n(Green = PASS [p>0.10], Red = FAIL [p<0.10])',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Buffer Control Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "pretrends_summary_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: PRE-TRENDS ANALYSIS")
    print(" Oct 2023 to Latest Available Date")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading Data ---")
    
    panel = load_panel_data()
    if panel is None:
        print("  ⚠ Panel data not found")
        return
    
    gdf, id_col = load_boundaries()
    if gdf is None:
        print("  ⚠ Boundaries not found")
        return
    
    sites = load_undp_sites()
    if sites is None:
        print("  ⚠ UNDP sites not found")
        return
    
    print(f"  Treated sites: {len(sites)}")
    print(f"  Admin units: {len(gdf)}")
    print(f"  Panel: {len(panel)} rows, {panel['period_start'].min()} to {panel['period_start'].max()}")
    
    # Calculate distances
    distances = calculate_min_distances(gdf, sites, id_col)
    
    # Store all results
    all_results = {}
    
    # Process each buffer specification
    for inner_m, outer_m, spec_label, spec_key in BUFFER_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Create treatment/control groups
        df = create_treatment_control_groups(panel, distances, inner_m, outer_m)
        
        n_treated = df[df['is_treated'] == 1]['unit_id'].nunique()
        n_control = df[df['is_control'] == 1]['unit_id'].nunique()
        
        if n_treated < 5 or n_control < 5:
            print(f"  ⚠ Insufficient units (treated={n_treated}, control={n_control})")
            continue
        
        print(f"  Treated: {n_treated} townships")
        print(f"  Control: {n_control} townships")
        print(f"  Observations: {len(df):,}")
        
        # Test each outcome
        print("\n  Pre-trends results:")
        print(f"  {'Outcome':<30} {'Result':>10} {'p-value':>10}")
        print("  " + "-" * 50)
        
        for outcome, outcome_name in OUTCOMES.items():
            passed, pval, coef = test_parallel_trends(df, outcome)
            
            all_results[(spec_label, outcome)] = (passed, pval)
            
            status = "PASS ✓" if passed else ("FAIL ⚠" if passed is not None else "N/A")
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"  {outcome_name:<30} {status:>10} {pval_str:>10}")
            
            # Create plot
            create_pretrends_plot(df, outcome, outcome_name, spec_label, passed, pval,
                                 FIGURES_DIR / f"pretrends_{outcome}_{spec_key}.png")
        
        print(f"\n  ✓ Saved plots to {FIGURES_DIR}")
    
    # Summary heatmap
    print("\n--- Creating Summary ---")
    create_summary_heatmap(all_results, FIGURES_DIR)
    print(f"  ✓ Saved: pretrends_summary_heatmap.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print(" SUMMARY: Pre-Trends Pass Rates")
    print("=" * 70)
    
    print("\n  By Outcome:")
    for outcome, outcome_name in OUTCOMES.items():
        passed_count = sum(1 for (s, o), (p, _) in all_results.items() 
                         if o == outcome and p == True)
        total = sum(1 for (s, o), (p, _) in all_results.items() 
                   if o == outcome and p is not None)
        pct = passed_count / total * 100 if total > 0 else 0
        status = "★ CREDIBLE" if pct >= 50 else ""
        print(f"    {outcome_name:<30}: {passed_count}/{total} ({pct:.0f}%) {status}")
    
    print("\n  By Specification:")
    for inner_m, outer_m, spec_label, spec_key in BUFFER_SPECS:
        passed_count = sum(1 for (s, o), (p, _) in all_results.items() 
                         if s == spec_label and p == True)
        total = sum(1 for (s, o), (p, _) in all_results.items() 
                   if s == spec_label and p is not None)
        pct = passed_count / total * 100 if total > 0 else 0
        print(f"    {spec_label:<20}: {passed_count}/{total} ({pct:.0f}%)")
    
    print(f"\n  OUTPUT: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
