#!/usr/bin/env python3
"""
analysis/mmr/04_pretrends_verified.py
=====================================
PRE-TRENDS ANALYSIS FOR MYANMAR - VERIFIED PANEL-BASED APPROACH

This script:
1. Uses panel_data.csv (NOT raster extraction)
2. Shows DIFFERENT treated/control counts per buffer spec
3. Ensures no contamination (control units not near ANY UNDP site)

Key checks printed:
- Treated townships: Should be SMALL numbers (1-30) for tight buffers
- Control townships: Should vary by buffer spec
- Observations: Should vary proportionally

Usage:
    python analysis/mmr/04_pretrends_verified.py
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
# CONFIGURATION - MYANMAR SPECIFIC
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# Myanmar earthquake date
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")

# Buffer specifications (meters) - Myanmar scale
BUFFER_SPECS = [
    (1000, 3000, "1km-3km", "tight"),
    (2000, 5000, "2km-5km", "moderate"),
    (3000, 8000, "3km-8km", "baseline"),
    (5000, 15000, "5km-15km", "wide"),
    (8000, 20000, "8km-20km", "very_wide"),
]

# Outcomes in panel_data.csv
OUTCOMES = {
    'nightlights': 'VIIRS Nightlights',
    'log_nightlights': 'Log Nightlights',
    'ndvi': 'NDVI (Vegetation)',
    'vh': 'SAR VH (Structures)',
    'vv': 'SAR VV',
    'rainfall': 'Rainfall',
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between points."""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def main():
    print("=" * 70)
    print(" MYANMAR: PRE-TRENDS ANALYSIS (VERIFIED PANEL APPROACH)")
    print("=" * 70)
    print("\n⚠️  This script uses PANEL DATA, not raster extraction!")
    print("    You should see DIFFERENT counts for each buffer spec.\n")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("--- Loading Data ---")
    
    # 1. Panel data
    if not PANEL_FILE.exists():
        print(f"  ❌ Panel file not found: {PANEL_FILE}")
        return
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    if 'nightlights' in panel.columns and 'log_nightlights' not in panel.columns:
        panel['log_nightlights'] = np.log(panel['nightlights'] + 1)
    
    print(f"  ✓ Panel: {len(panel):,} rows, {panel['unit_id'].nunique()} townships")
    print(f"    Date range: {panel['period_start'].min().date()} to {panel['period_start'].max().date()}")
    
    # 2. Boundaries
    if not BOUNDARIES_FILE.exists():
        print(f"  ❌ Boundaries not found: {BOUNDARIES_FILE}")
        return
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    if id_col is None:
        print("  ❌ No pcode column found in boundaries")
        return
    
    # Add centroids
    gdf['centroid'] = gdf.geometry.centroid
    gdf['center_lon'] = gdf['centroid'].x
    gdf['center_lat'] = gdf['centroid'].y
    
    print(f"  ✓ Boundaries: {len(gdf)} townships")
    
    # 3. UNDP sites
    if not UNDP_SITES_FILE.exists():
        print(f"  ❌ UNDP sites not found: {UNDP_SITES_FILE}")
        return
    
    sites = pd.read_csv(UNDP_SITES_FILE)
    lat_col = next((c for c in sites.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in sites.columns if 'lon' in c.lower()), None)
    
    if lat_col is None or lon_col is None:
        print("  ❌ No lat/lon columns in sites file")
        return
    
    sites = sites.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    sites = sites.dropna(subset=['lat', 'lon'])
    
    print(f"  ✓ UNDP sites: {len(sites)} GPS coordinates")
    
    # =========================================================================
    # CALCULATE DISTANCES
    # =========================================================================
    print("\n--- Calculating Distances ---")
    
    site_lats = sites['lat'].values
    site_lons = sites['lon'].values
    
    distances = {}
    for _, row in gdf.iterrows():
        unit_id = row[id_col]
        unit_lat = row['center_lat']
        unit_lon = row['center_lon']
        
        # Distance to ALL sites, take minimum
        dists = haversine_distance(unit_lat, unit_lon, site_lats, site_lons)
        distances[unit_id] = np.min(dists)
    
    dist_values = list(distances.values())
    print(f"  Min distance: {min(dist_values)/1000:.1f} km")
    print(f"  Max distance: {max(dist_values)/1000:.1f} km")
    print(f"  Median: {np.median(dist_values)/1000:.1f} km")
    
    # Show distribution
    print("\n  Distance distribution (townships):")
    for threshold in [1, 2, 3, 5, 8, 10, 15, 20, 50]:
        n = sum(1 for d in dist_values if d <= threshold * 1000)
        print(f"    Within {threshold:>2} km: {n:>3} townships")
    
    # =========================================================================
    # PRE-TRENDS ANALYSIS PER BUFFER SPEC
    # =========================================================================
    all_results = {}
    
    for inner_m, outer_m, spec_label, spec_key in BUFFER_SPECS:
        print(f"\n{'='*70}")
        print(f" BUFFER: {spec_label}")
        print(f" Treatment: within {inner_m/1000:.1f} km | Control: {inner_m/1000:.1f}-{outer_m/1000:.1f} km")
        print("=" * 70)
        
        # Create treatment/control based on distance
        panel_copy = panel.copy()
        panel_copy['min_dist'] = panel_copy['unit_id'].map(distances)
        
        # Treatment: within inner_m of NEAREST site
        panel_copy['is_treated'] = (panel_copy['min_dist'] <= inner_m).astype(int)
        
        # Control: between inner_m and outer_m from NEAREST site
        # This means NOT within inner_m of ANY site (since we use min_dist)
        panel_copy['is_control'] = (
            (panel_copy['min_dist'] > inner_m) & 
            (panel_copy['min_dist'] <= outer_m)
        ).astype(int)
        
        # Keep only treated + control
        df = panel_copy[(panel_copy['is_treated'] == 1) | (panel_copy['is_control'] == 1)].copy()
        
        # Count units
        treated_units = df[df['is_treated'] == 1]['unit_id'].unique()
        control_units = df[df['is_control'] == 1]['unit_id'].unique()
        
        n_treated = len(treated_units)
        n_control = len(control_units)
        n_obs = len(df)
        
        print(f"\n  📊 SAMPLE SIZES (should vary by spec!):")
        print(f"     Treated townships:  {n_treated}")
        print(f"     Control townships:  {n_control}")
        print(f"     Total observations: {n_obs:,}")
        
        if n_treated < 3 or n_control < 3:
            print(f"  ⚠️ Insufficient sample (need at least 3 in each group)")
            continue
        
        # Verify no contamination
        treated_dists = [distances[u] for u in treated_units]
        control_dists = [distances[u] for u in control_units]
        
        print(f"\n  ✓ CONTAMINATION CHECK:")
        print(f"     Treated max distance: {max(treated_dists)/1000:.2f} km (should be ≤ {inner_m/1000:.1f})")
        print(f"     Control min distance: {min(control_dists)/1000:.2f} km (should be > {inner_m/1000:.1f})")
        print(f"     Control max distance: {max(control_dists)/1000:.2f} km (should be ≤ {outer_m/1000:.1f})")
        
        # Pre-trends test for each outcome
        print(f"\n  📈 PRE-TRENDS RESULTS:")
        print(f"  {'Outcome':<25} {'Result':>10} {'p-value':>10} {'Coef':>12}")
        print("  " + "-" * 60)
        
        for outcome, outcome_name in OUTCOMES.items():
            if outcome not in df.columns:
                print(f"  {outcome_name:<25} {'N/A':>10} {'N/A':>10}")
                continue
            
            # Filter to pre-earthquake
            pre_df = df[df['period_start'] < EARTHQUAKE_DATE].copy()
            pre_df = pre_df.dropna(subset=[outcome])
            
            if len(pre_df) < 50:
                print(f"  {outcome_name:<25} {'N/A':>10} {'N/A':>10}")
                continue
            
            # Create time index
            pre_df['month_idx'] = pre_df['period_start'].dt.year * 12 + pre_df['period_start'].dt.month
            
            try:
                # Test: outcome ~ time * treatment
                model = smf.ols(f'{outcome} ~ month_idx * is_treated', data=pre_df).fit()
                
                if 'month_idx:is_treated' in model.pvalues:
                    pval = model.pvalues['month_idx:is_treated']
                    coef = model.params['month_idx:is_treated']
                    passed = pval > 0.10
                    
                    status = "PASS ✓" if passed else "FAIL ⚠"
                    print(f"  {outcome_name:<25} {status:>10} {pval:>10.4f} {coef:>12.6f}")
                    
                    all_results[(spec_label, outcome)] = (passed, pval)
                    
                    # Create plot
                    create_pretrends_plot(df, outcome, outcome_name, spec_label, 
                                         passed, pval, n_treated, n_control,
                                         FIGURES_DIR / f"pretrends_{outcome}_{spec_key}.png")
                else:
                    print(f"  {outcome_name:<25} {'ERROR':>10}")
                    
            except Exception as e:
                print(f"  {outcome_name:<25} {'ERROR':>10} ({str(e)[:20]})")
        
        print(f"\n  ✓ Plots saved to {FIGURES_DIR}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    print("\n  By Outcome:")
    for outcome, outcome_name in OUTCOMES.items():
        passed = sum(1 for (s, o), (p, _) in all_results.items() if o == outcome and p)
        total = sum(1 for (s, o), (p, _) in all_results.items() if o == outcome and p is not None)
        if total > 0:
            pct = passed / total * 100
            status = "★" if pct >= 80 else ""
            print(f"    {outcome_name:<25}: {passed}/{total} PASS ({pct:.0f}%) {status}")
    
    print("\n  By Buffer Spec:")
    for inner_m, outer_m, spec_label, spec_key in BUFFER_SPECS:
        passed = sum(1 for (s, o), (p, _) in all_results.items() if s == spec_label and p)
        total = sum(1 for (s, o), (p, _) in all_results.items() if s == spec_label and p is not None)
        if total > 0:
            pct = passed / total * 100
            print(f"    {spec_label:<15}: {passed}/{total} PASS ({pct:.0f}%)")
    
    # Create summary heatmap
    create_summary_heatmap(all_results, FIGURES_DIR)
    
    print(f"\n  Output: {FIGURES_DIR}")
    print("=" * 70)


def create_pretrends_plot(df, outcome, outcome_name, spec_label, passed, pval, 
                          n_treated, n_control, output_path):
    """Create pre-trends plot."""
    
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
    ax.axvline(EARTHQUAKE_DATE, color='red', linewidth=2.5, linestyle='--', 
               label='Earthquake (Mar 28, 2025)')
    
    # Treated
    ax.fill_between(treated['month'],
                    treated['mean'] - 1.96*treated['se'],
                    treated['mean'] + 1.96*treated['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treated['month'], treated['mean'], 'o-', color='#e74c3c',
            linewidth=2, markersize=6, label=f'Treated (n={n_treated} townships)')
    
    # Control
    ax.fill_between(control['month'],
                    control['mean'] - 1.96*control['se'],
                    control['mean'] + 1.96*control['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(control['month'], control['mean'], 's--', color='#3498db',
            linewidth=2, markersize=6, label=f'Control (n={n_control} townships)')
    
    # Result annotation
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
    ax.set_title(f'Pre-Trends: {outcome_name}\nBuffer: {spec_label}',
                fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_heatmap(all_results, output_dir):
    """Create summary heatmap."""
    try:
        import seaborn as sns
    except ImportError:
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
                    annotations[i, j] = f"p={pval:.2f}"
                else:
                    annotations[i, j] = "N/A"
            else:
                annotations[i, j] = "N/A"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(matrix, ax=ax, annot=annotations, fmt='', cmap=cmap,
                xticklabels=specs,
                yticklabels=[OUTCOMES[o] for o in outcomes_list],
                cbar_kws={'label': 'Pre-Trends (1=PASS, 0=FAIL)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Myanmar Pre-Trends Summary\n(Green=PASS p>0.10, Red=FAIL p<0.10)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Buffer Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "pretrends_summary.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Summary heatmap saved")


if __name__ == "__main__":
    main()
