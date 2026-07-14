#!/usr/bin/env python3
"""
analysis/mmr/02_did_buffer_fixed.py
===================================
Buffer-Based DiD with FIXED treatment date selection.

KEY FIX: Uses Infrastructure & Housing project (Sept 2025)
NOT the Environment project (April 2025).

Usage:
    python analysis/mmr/02_did_buffer_fixed.py
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

EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")

# Buffer specifications (meters)
BUFFER_SPECS = [
    (500, 1500),
    (1000, 3000),
    (2000, 5000),
    (3000, 8000),
    (5000, 15000),
    (8000, 20000),
]

OUTCOMES = ['nightlights', 'log_nightlights', 'ndvi', 'vh', 'vv', 'rvi', 'conflict_events']

# FULL CONTROLS
CONTROLS = ['rainfall', 'conflict_events']


def load_panel_data():
    if not PANEL_FILE.exists():
        return None
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    if 'nightlights' in df.columns and 'log_nightlights' not in df.columns:
        df['log_nightlights'] = np.log(df['nightlights'] + 1)
    return df


def load_boundaries():
    if not BOUNDARIES_FILE.exists():
        return None, None
    gdf = gpd.read_file(BOUNDARIES_FILE)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower() or c == 'shapeID'), None)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['center_lon'] = gdf['centroid'].x
    gdf['center_lat'] = gdf['centroid'].y
    return gdf, id_col


def load_undp_sites():
    if not UNDP_SITES_FILE.exists():
        return None
    df = pd.read_csv(UNDP_SITES_FILE)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if lat_col is None or lon_col is None:
        return None
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df = df.dropna(subset=['lat', 'lon'])
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def get_infrastructure_treatment_date():
    """
    Get treatment date from Infrastructure & Housing project (Sept 2025).
    NOT the Environment project (April 2025).
    """
    if not UNDP_PROJECTS_FILE.exists():
        print("  ⚠ Using default date (no projects file)")
        return pd.Timestamp("2025-09-01"), "Default"
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter to post-earthquake
    df = df[df['start_date'] >= EARTHQUAKE_DATE]
    
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    print("\n  Available post-earthquake UNDP projects:")
    for _, row in df.iterrows():
        cat = row.get('ml_category', 'Unknown')
        date = row['start_date'].strftime('%Y-%m-%d') if pd.notna(row['start_date']) else 'N/A'
        title = str(row.get('title', ''))[:50]
        print(f"    [{date}] {cat}: {title}")
    
    # PRIORITY: Infrastructure & Housing (this is the relevant reconstruction project)
    infra = df[df['ml_category'].str.contains('Infrastructure|Housing', case=False, na=False)]
    if len(infra) > 0:
        project = infra.iloc[0]
        date = project['start_date']
        cat = project.get('ml_category', 'Infrastructure & Housing')
        print(f"\n  → SELECTED: {cat} ({date.strftime('%Y-%m-%d')})")
        print(f"    (This is the relevant reconstruction project, NOT Environment)")
        return date, cat
    
    # Fallback: Debris & Solid Waste
    debris = df[df['ml_category'].str.contains('Debris', case=False, na=False)]
    if len(debris) > 0:
        project = debris.iloc[0]
        return project['start_date'], project.get('ml_category', 'Debris')
    
    # Last resort: Latest project (most likely reconstruction)
    if len(df) > 0:
        project = df.sort_values('start_date').iloc[-1]
        return project['start_date'], project.get('ml_category', 'Unknown')
    
    return pd.Timestamp("2025-09-01"), "Default"


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calculate_min_distance_to_sites(gdf_boundaries, gdf_sites, id_col):
    distances = {}
    site_lats = gdf_sites['lat'].values
    site_lons = gdf_sites['lon'].values
    
    for _, row in gdf_boundaries.iterrows():
        unit_id = row[id_col]
        unit_lat = row['center_lat']
        unit_lon = row['center_lon']
        dists = haversine_distance(unit_lat, unit_lon, site_lats, site_lons)
        distances[unit_id] = np.min(dists)
    
    return distances


def create_buffer_treatment(panel, distances, inner_m, outer_m, intervention_date):
    df = panel.copy()
    df['min_dist'] = df['unit_id'].map(distances)
    df['is_treated'] = (df['min_dist'] <= inner_m).astype(int)
    df['is_control'] = ((df['min_dist'] > inner_m) & (df['min_dist'] <= outer_m)).astype(int)
    df = df[(df['is_treated'] == 1) | (df['is_control'] == 1)].copy()
    df['post'] = (df['period_start'] >= intervention_date).astype(int)
    df['treated_x_post'] = df['is_treated'] * df['post']
    return df


def run_twfe_regression(df, outcome, spec_name, controls=None):
    if outcome not in df.columns:
        return None
    
    df_reg = df.dropna(subset=[outcome, 'treated_x_post']).copy()
    
    if len(df_reg) < 50:
        return None
    
    n_treated = df_reg[df_reg['is_treated'] == 1]['unit_id'].nunique()
    n_control = df_reg[df_reg['is_control'] == 1]['unit_id'].nunique()
    
    if n_treated == 0 or n_control == 0:
        return None
    
    # Handle controls
    if controls:
        valid_controls = [c for c in controls if c in df_reg.columns and c != outcome]
        df_reg = df_reg.dropna(subset=valid_controls)
        control_str = ' + '.join(valid_controls) if valid_controls else ''
    else:
        control_str = ''
    
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
        return None


def create_coefficient_plot(results_df, outcome, output_path):
    """Create coefficient plot - FIXED parsing."""
    import matplotlib.pyplot as plt
    
    df = results_df[
        (results_df['Outcome'] == outcome) & 
        (results_df['Controls'] == 'None')
    ].copy()
    
    if len(df) == 0:
        return
    
    # FIXED: Better parsing for buffer specs like "0.5km-1.5km"
    def parse_inner(spec):
        try:
            inner = spec.split('-')[0]
            inner = inner.replace('km', '')
            return float(inner)
        except:
            return 0
    
    df['inner'] = df['Buffer_Spec'].apply(parse_inner)
    df = df.sort_values('inner')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    
    ax.errorbar(x, df['Coeff'], yerr=1.96*df['SE'], fmt='o', 
                capsize=5, capthick=2, markersize=10, 
                color='darkblue', ecolor='gray', elinewidth=2)
    
    for i, (xi, coef, pval) in enumerate(zip(x, df['Coeff'], df['P_value'])):
        color = 'green' if pval < 0.05 else ('orange' if pval < 0.1 else 'gray')
        ax.scatter([xi], [coef], c=color, s=100, zorder=5)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Buffer_Spec'], rotation=45, ha='right')
    ax.set_xlabel('Buffer Specification (Treatment - Control)', fontsize=11)
    ax.set_ylabel(f'DiD Coefficient ({outcome})', fontsize=11)
    ax.set_title(f'Myanmar: UNDP Impact on {outcome}\nBuffer-Based DiD', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
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
    print("MYANMAR: BUFFER-BASED DiD (FIXED)")
    print("=" * 70)
    print("\n  Using Infrastructure & Housing project (Sept 2025)")
    print("  NOT Environment project (April 2025)")
    
    if not HAS_LINEARMODELS:
        print("\n⚠ Need linearmodels")
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
    
    # Get INFRASTRUCTURE treatment date (Sept 2025)
    print("\n--- Treatment Timing ---")
    intervention_date, project_category = get_infrastructure_treatment_date()
    print(f"\n  TREATMENT DATE: {intervention_date.strftime('%Y-%m-%d')}")
    print(f"  Project: {project_category}")
    
    # Calculate distances
    print("\n--- Calculating Distances ---")
    distances = calculate_min_distance_to_sites(gdf, gdf_sites, id_col)
    
    dist_values = list(distances.values())
    print(f"  Min: {min(dist_values)/1000:.1f} km")
    print(f"  Max: {max(dist_values)/1000:.1f} km")
    print(f"  Median: {np.median(dist_values)/1000:.1f} km")
    
    for inner, outer in BUFFER_SPECS:
        n_treated = sum(1 for d in dist_values if d <= inner)
        n_control = sum(1 for d in dist_values if inner < d <= outer)
        print(f"  Buffer {inner/1000:.1f}km-{outer/1000:.1f}km: {n_treated} treated, {n_control} control")
    
    # Run regressions
    print("\n" + "=" * 70)
    print("RUNNING BUFFER-BASED DiD REGRESSIONS")
    print("=" * 70)
    
    all_results = []
    
    for inner_m, outer_m in BUFFER_SPECS:
        spec_name = f"{inner_m/1000:.1f}km-{outer_m/1000:.1f}km"
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
            
            # With controls
            valid_controls = [c for c in CONTROLS if c != outcome]
            res = run_twfe_regression(df_buffer, outcome, f"{spec_name} + controls", controls=valid_controls)
            if res:
                all_results.append(res)
    
    if not all_results:
        print("\n⚠ No results")
        return
    
    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'did_buffer_results.csv', index=False)
    print(f"\n✓ Saved: {RESULTS_DIR / 'did_buffer_results.csv'}")
    
    # Coefficient plots
    print("\n--- Creating Coefficient Plots ---")
    for outcome in ['log_nightlights', 'vh', 'ndvi']:
        if outcome in results_df['Outcome'].values:
            create_coefficient_plot(results_df, outcome, 
                                   FIGURES_DIR / f'coef_plot_{outcome}_buffer.png')
            print(f"  ✓ coef_plot_{outcome}_buffer.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY RESULTS: log_nightlights")
    print("=" * 70)
    
    key = results_df[(results_df['Outcome'] == 'log_nightlights') & (results_df['Controls'] == 'None')]
    
    print(f"\n{'Buffer':<15} {'Coeff':>10} {'SE':>12} {'Stars':>5} {'% Effect':>10}")
    print("-" * 55)
    for _, row in key.iterrows():
        print(f"{row['Buffer_Spec']:<15} {row['Coeff_Fmt']:>10} {row['SE_Fmt']:>12} {row['Stars']:>5} {row['Pct_Effect']:>10}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
