#!/usr/bin/env python3
"""
analysis/mmr/02_did_buffer_approach.py
======================================
Buffer-Based Difference-in-Differences Analysis (GAZA APPROACH)

This uses the SAME methodology as Gaza:
- Treatment = Areas within X meters of UNDP GPS coordinates
- Control = Donut buffer (e.g., areas 200-600m away from sites)
- Multiple buffer specifications for robustness

Buffer Specifications (adapted for Myanmar's more rural setting):
- 500m-1.5km (closest control ring)
- 1km-3km
- 2km-5km  
- 3km-8km
- 5km-15km
- 8km-20km (furthest control ring)

Treatment timing uses ACTUAL project start dates from undp_projects.csv.

Usage:
    python analysis/mmr/02_did_buffer_approach.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("⚠ linearmodels not installed. Run: pip install linearmodels --break-system-packages")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")

# Buffer specifications (inner_m, outer_m) - METERS not degrees!
# Adapted for Myanmar's more rural/dispersed setting (larger than Gaza's 200m-2km)
BUFFER_SPECS = [
    (500, 1500),      # 500m - 1.5km
    (1000, 3000),     # 1km - 3km
    (2000, 5000),     # 2km - 5km
    (3000, 8000),     # 3km - 8km
    (5000, 15000),    # 5km - 15km
    (8000, 20000),    # 8km - 20km
]

# Outcomes to analyze
OUTCOMES = [
    'nightlights', 'log_nightlights',
    'ndvi', 'vh', 'vv', 'rvi',
    'conflict_events', 'rainfall'
]


def load_panel_data():
    """Load panel data."""
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel not found: {PANEL_FILE}")
        return None
    
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Create log variables
    if 'nightlights' in df.columns and 'log_nightlights' not in df.columns:
        df['log_nightlights'] = np.log(df['nightlights'] + 1)
    
    return df


def load_boundaries():
    """Load boundaries with centroids for spatial matching."""
    if not BOUNDARIES_FILE.exists():
        return None, None
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get ID column
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    
    # Add centroids
    gdf['centroid'] = gdf.geometry.centroid
    gdf['center_lon'] = gdf['centroid'].x
    gdf['center_lat'] = gdf['centroid'].y
    
    return gdf, id_col


def load_undp_sites():
    """Load UNDP site GPS coordinates."""
    if not UNDP_SITES_FILE.exists():
        print(f"  ⚠ UNDP sites not found: {UNDP_SITES_FILE}")
        return None
    
    df = pd.read_csv(UNDP_SITES_FILE)
    
    # Get lat/lon columns
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    
    if lat_col is None or lon_col is None:
        print("  ⚠ Lat/Lon columns not found in sites file")
        return None
    
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df = df.dropna(subset=['lat', 'lon'])
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf_sites = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf_sites


def load_undp_projects():
    """Load UNDP projects with start dates for treatment timing."""
    if not UNDP_PROJECTS_FILE.exists():
        return None, EARTHQUAKE_DATE + pd.Timedelta(days=30)  # Default to 1 month after
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter to post-earthquake projects
    df = df[df['start_date'] >= EARTHQUAKE_DATE]
    
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    if len(df) == 0:
        return None, EARTHQUAKE_DATE + pd.Timedelta(days=30)
    
    # Get earliest intervention date
    earliest_date = df['start_date'].min()
    
    print(f"  UNDP Projects: {len(df)}")
    print(f"  Earliest intervention: {earliest_date.strftime('%Y-%m-%d')}")
    print(f"  Latest intervention: {df['start_date'].max().strftime('%Y-%m-%d')}")
    
    return df, earliest_date


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points."""
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_min_distance_to_sites(gdf_boundaries, gdf_sites, id_col):
    """
    Calculate minimum distance from each admin unit centroid to nearest UNDP site.
    Returns dict: {unit_id: min_distance_meters}
    """
    distances = {}
    
    site_lats = gdf_sites['lat'].values
    site_lons = gdf_sites['lon'].values
    
    for _, row in gdf_boundaries.iterrows():
        unit_id = row[id_col]
        unit_lat = row['center_lat']
        unit_lon = row['center_lon']
        
        # Calculate distance to all sites
        dists = haversine_distance(unit_lat, unit_lon, site_lats, site_lons)
        
        distances[unit_id] = np.min(dists)
    
    return distances


def create_buffer_treatment(panel, distances, inner_m, outer_m, intervention_date):
    """
    Create treatment panel for a specific buffer specification.
    
    Treatment: Units within inner_m of nearest UNDP site
    Control: Units between inner_m and outer_m from nearest site
    Excluded: Units beyond outer_m (too far to be valid control)
    
    Post: period_start >= intervention_date
    """
    df = panel.copy()
    
    # Add distance to each unit
    df['min_dist'] = df['unit_id'].map(distances)
    
    # Define treatment and control
    df['is_treated'] = (df['min_dist'] <= inner_m).astype(int)
    df['is_control'] = ((df['min_dist'] > inner_m) & (df['min_dist'] <= outer_m)).astype(int)
    
    # Keep only treated + control (exclude units beyond outer_m)
    df = df[(df['is_treated'] == 1) | (df['is_control'] == 1)].copy()
    
    # Post indicator
    df['post'] = (df['period_start'] >= intervention_date).astype(int)
    
    # Treatment interaction
    df['treated_x_post'] = df['is_treated'] * df['post']
    
    return df


def run_twfe_regression(df, outcome, spec_name, controls=None):
    """Run TWFE regression for buffer-based DiD."""
    
    if outcome not in df.columns:
        return None
    
    df_reg = df.dropna(subset=[outcome, 'treated_x_post']).copy()
    
    if len(df_reg) < 50:
        return None
    
    n_treated = df_reg[df_reg['is_treated'] == 1]['unit_id'].nunique()
    n_control = df_reg[df_reg['is_control'] == 1]['unit_id'].nunique()
    
    if n_treated == 0 or n_control == 0:
        return None
    
    # Build formula
    control_str = ' + '.join(controls) if controls else ''
    
    if control_str:
        formula = f"{outcome} ~ treated_x_post + {control_str} + EntityEffects + TimeEffects"
    else:
        formula = f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects"
    
    try:
        df_panel = df_reg.set_index(['unit_id', 'period_start'])
        
        mod = PanelOLS.from_formula(formula, data=df_panel, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coef = res.params['treated_x_post']
        se = res.std_errors['treated_x_post']
        pval = res.pvalues['treated_x_post']
        
        # Calculate % effect for log outcomes
        if outcome.startswith('log_'):
            pct_effect = (np.exp(coef) - 1) * 100
            pct_str = f"{pct_effect:+.1f}%"
        else:
            pct_str = ""
        
        return {
            'Outcome': outcome,
            'Buffer_Spec': spec_name,
            'Controls': control_str if control_str else 'None',
            'Coeff': coef,
            'Coeff_Fmt': f"{coef:.4f}",
            'SE': se,
            'SE_Fmt': f"({se:.4f})",
            'P_value': pval,
            'Stars': '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else '')),
            'Pct_Effect': pct_str,
            'N_Obs': int(res.nobs),
            'N_Treated': n_treated,
            'N_Control': n_control,
            'R2': res.rsquared,
        }
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return None


def create_coefficient_plot(results_df, outcome, output_path):
    """Create coefficient plot across buffer specifications."""
    import matplotlib.pyplot as plt
    
    df = results_df[
        (results_df['Outcome'] == outcome) & 
        (results_df['Controls'] == 'None')
    ].copy()
    
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by buffer inner distance
    df['inner'] = df['Buffer_Spec'].apply(lambda x: int(x.split('-')[0].replace('m', '').replace('km', '000')))
    df = df.sort_values('inner')
    
    x = range(len(df))
    
    # Plot coefficients with error bars
    colors = ['green' if p < 0.05 else ('orange' if p < 0.1 else 'gray') for p in df['P_value']]
    
    ax.errorbar(x, df['Coeff'], yerr=1.96*df['SE'], fmt='o', 
                capsize=5, capthick=2, markersize=10, 
                color='darkblue', ecolor='gray', elinewidth=2)
    
    # Color points by significance
    for i, (xi, coef, pval) in enumerate(zip(x, df['Coeff'], df['P_value'])):
        color = 'green' if pval < 0.05 else ('orange' if pval < 0.1 else 'gray')
        ax.scatter([xi], [coef], c=color, s=100, zorder=5)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Buffer_Spec'], rotation=45, ha='right')
    ax.set_xlabel('Buffer Specification (Treatment - Control)', fontsize=11)
    ax.set_ylabel(f'DiD Coefficient ({outcome})', fontsize=11)
    ax.set_title(f'Myanmar Earthquake: UNDP Impact on {outcome}\nBuffer-Based DiD (Gaza Approach)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='p < 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='p < 0.10'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='p ≥ 0.10'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print("MYANMAR: BUFFER-BASED DiD (GAZA APPROACH)")
    print("=" * 70)
    print("\nMethodology:")
    print("  - Treatment: Admin units within X meters of UNDP GPS coordinates")
    print("  - Control: Donut buffer (units between X and Y meters)")
    print("  - Multiple buffer specs for robustness")
    print("  - Treatment timing from undp_projects.csv start dates")
    
    if not HAS_LINEARMODELS:
        print("\n⚠ Need linearmodels for regression")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    panel = load_panel_data()
    if panel is None:
        return
    print(f"  Panel: {len(panel):,} obs, {panel['unit_id'].nunique()} units")
    
    gdf, id_col = load_boundaries()
    if gdf is None:
        return
    print(f"  Boundaries: {len(gdf)} units")
    
    gdf_sites = load_undp_sites()
    if gdf_sites is None:
        return
    print(f"  UNDP sites: {len(gdf_sites)} GPS coordinates")
    
    # Load projects for treatment timing
    print("\n--- Treatment Timing ---")
    undp_projects, intervention_date = load_undp_projects()
    print(f"  Using intervention date: {intervention_date.strftime('%Y-%m-%d')}")
    
    # Calculate distances
    print("\n--- Calculating Distances ---")
    distances = calculate_min_distance_to_sites(gdf, gdf_sites, id_col)
    
    dist_values = list(distances.values())
    print(f"  Min distance: {min(dist_values)/1000:.1f} km")
    print(f"  Max distance: {max(dist_values)/1000:.1f} km")
    print(f"  Median distance: {np.median(dist_values)/1000:.1f} km")
    
    # Count units by distance
    for inner, outer in BUFFER_SPECS:
        n_treated = sum(1 for d in dist_values if d <= inner)
        n_control = sum(1 for d in dist_values if inner < d <= outer)
        print(f"  Buffer {inner/1000:.1f}km-{outer/1000:.1f}km: {n_treated} treated, {n_control} control")
    
    # Run regressions for each buffer spec
    print("\n" + "=" * 70)
    print("RUNNING BUFFER-BASED DiD REGRESSIONS")
    print("=" * 70)
    
    all_results = []
    
    for inner_m, outer_m in BUFFER_SPECS:
        spec_name = f"{inner_m/1000:.0f}km-{outer_m/1000:.0f}km"
        print(f"\n--- Buffer: {spec_name} ---")
        
        df_buffer = create_buffer_treatment(panel, distances, inner_m, outer_m, intervention_date)
        
        n_treated = df_buffer[df_buffer['is_treated'] == 1]['unit_id'].nunique()
        n_control = df_buffer[df_buffer['is_control'] == 1]['unit_id'].nunique()
        print(f"  Treated: {n_treated}, Control: {n_control}")
        
        for outcome in OUTCOMES:
            if outcome not in df_buffer.columns:
                continue
            
            # Without controls
            res = run_twfe_regression(df_buffer, outcome, spec_name, controls=[])
            if res:
                all_results.append(res)
            
            # With rainfall control
            res = run_twfe_regression(df_buffer, outcome, spec_name, controls=['rainfall'])
            if res:
                all_results.append(res)
    
    if not all_results:
        print("\n⚠ No results generated")
        return
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'did_buffer_results.csv', index=False)
    print(f"\n✓ Saved: {RESULTS_DIR / 'did_buffer_results.csv'}")
    
    # Create coefficient plots
    print("\n--- Creating Coefficient Plots ---")
    for outcome in ['log_nightlights', 'vh', 'ndvi']:
        if outcome in results_df['Outcome'].values:
            create_coefficient_plot(results_df, outcome, 
                                   FIGURES_DIR / f'coef_plot_{outcome}_buffer.png')
            print(f"  ✓ coef_plot_{outcome}_buffer.png")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Key Results (log_nightlights)")
    print("=" * 70)
    
    key_results = results_df[
        (results_df['Outcome'] == 'log_nightlights') & 
        (results_df['Controls'] == 'None')
    ].sort_values('Coeff', ascending=False)
    
    print(f"\n{'Buffer Spec':<15} {'Coeff':>10} {'SE':>12} {'Stars':>6} {'% Effect':>10} {'N_Treat':>8} {'N_Ctrl':>8}")
    print("-" * 75)
    
    for _, row in key_results.iterrows():
        print(f"{row['Buffer_Spec']:<15} {row['Coeff_Fmt']:>10} {row['SE_Fmt']:>12} {row['Stars']:>6} {row['Pct_Effect']:>10} {row['N_Treated']:>8} {row['N_Control']:>8}")
    
    # SAR results
    print("\n" + "=" * 70)
    print("SUMMARY: SAR VH Results (Radar - Cloud-Free)")
    print("=" * 70)
    
    sar_results = results_df[
        (results_df['Outcome'] == 'vh') & 
        (results_df['Controls'] == 'None')
    ].sort_values('Coeff', ascending=False)
    
    print(f"\n{'Buffer Spec':<15} {'Coeff':>10} {'SE':>12} {'Stars':>6} {'N_Treat':>8} {'N_Ctrl':>8}")
    print("-" * 65)
    
    for _, row in sar_results.iterrows():
        print(f"{row['Buffer_Spec']:<15} {row['Coeff_Fmt']:>10} {row['SE_Fmt']:>12} {row['Stars']:>6} {row['N_Treated']:>8} {row['N_Control']:>8}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
