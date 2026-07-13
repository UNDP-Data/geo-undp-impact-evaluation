#!/usr/bin/env python3
"""
analysis/mmr/02_did_analysis_twfe.py
====================================
Difference-in-Differences Analysis with PROPER TWFE Specification.

Treatment Definition (from old script):
- Treatment = Townships in UNDP project list
- Treatment timing = UNDP_START_DATE (May 1, 2025) OR staggered by project
- Post = period_start >= treatment_date

Two approaches:
1. FIXED DATE: All treated units switch on May 1, 2025 (simpler, old approach)
2. STAGGERED: Each project uses its actual start_date from undp_projects.csv

This script supports BOTH approaches.

Usage:
    python analysis/mmr/02_did_analysis_twfe.py
    python analysis/mmr/02_did_analysis_twfe.py --staggered
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("⚠ linearmodels not installed. Run: pip install linearmodels --break-system-packages")

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
PROJECT_FILE = DATA_DIR / "undp" / "project_data.xlsx"
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
UNDP_START_DATE = pd.Timestamp("2025-05-01")  # From old analysis script
MAX_DATE = pd.Timestamp("2025-09-30")  # Exclude data after September 2025

# Outcomes to analyze
OUTCOMES = [
    'nightlights', 'log_nightlights',
    'ndvi', 'vh', 'vv', 'rvi',
    'conflict_events', 'rainfall'
]

# Controls
CONTROLS = ['rainfall']


def clean_name(name):
    """Clean township name for matching."""
    if pd.isna(name):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def load_panel_data():
    """Load panel data."""
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel not found: {PANEL_FILE}")
        return None
    
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Create log variables if not exist
    if 'nightlights' in df.columns and 'log_nightlights' not in df.columns:
        df['log_nightlights'] = np.log(df['nightlights'] + 1)
    
    if 'conflict_events' in df.columns and 'log_conflict' not in df.columns:
        df['log_conflict'] = np.log(df['conflict_events'] + 1)
    
    return df


def load_boundaries():
    """Load boundaries and get township info."""
    if not BOUNDARIES_FILE.exists():
        return None, None
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    
    # Find ID and name columns
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    name_col = next((c for c in gdf.columns if c.lower() in ['ts', 'adm3_en', 'name', 'township']), None)
    
    if not name_col:
        # Try to find any name-like column
        for col in gdf.columns:
            if 'name' in col.lower() or 'en' in col.lower():
                name_col = col
                break
    
    return gdf, (id_col, name_col)


def load_undp_treatment_from_project_file():
    """
    Load UNDP treatment definition from project_data.xlsx.
    Returns set of treated township clean names.
    """
    if not PROJECT_FILE.exists():
        print(f"  ⚠ Project file not found: {PROJECT_FILE}")
        return None
    
    try:
        df_proj = pd.read_excel(PROJECT_FILE)
    except Exception as e:
        print(f"  ⚠ Error reading project file: {e}")
        return None
    
    # Get township column
    town_col = next((c for c in df_proj.columns if 'Township' in c and 'State' not in c), None)
    if not town_col:
        print("  ⚠ Township column not found in project file")
        return None
    
    df_proj['clean_name'] = df_proj[town_col].apply(clean_name)
    
    treated_townships = set(df_proj['clean_name'].unique())
    treated_townships.discard('')  # Remove empty
    
    return treated_townships


def load_undp_projects_with_dates():
    """
    Load UNDP projects from undp_projects.csv with start dates.
    For staggered treatment timing.
    """
    if not UNDP_PROJECTS_FILE.exists():
        return None
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter out umbrella projects
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    # Only post-earthquake projects
    df = df[df['start_date'] >= EARTHQUAKE_DATE]
    
    return df


def create_treatment_panel_fixed_date(panel, treated_townships, gdf, cols):
    """
    Create treatment panel with FIXED treatment date (May 1, 2025).
    All treated units switch on the same date.
    """
    id_col, name_col = cols
    
    if name_col is None:
        print("  ⚠ Cannot match townships - name column not found")
        return None
    
    # Create township clean names in boundaries
    gdf['clean_name'] = gdf[name_col].apply(clean_name)
    
    # Create treatment indicator
    gdf['treated'] = gdf['clean_name'].isin(treated_townships).astype(int)
    
    # Create mapping: unit_id -> treated
    if 'unit_id' in panel.columns:
        # Match panel unit_id to boundaries id_col
        treatment_map = gdf.set_index(id_col)['treated'].to_dict()
        panel['treated'] = panel['unit_id'].map(treatment_map).fillna(0).astype(int)
    else:
        # Try direct matching
        panel['clean_name'] = panel.get('township_name', panel.get('name', '')).apply(clean_name)
        panel['treated'] = panel['clean_name'].isin(treated_townships).astype(int)
    
    # Create POST indicator (fixed date)
    panel['post'] = (panel['period_start'] >= UNDP_START_DATE).astype(int)
    
    # Create interaction
    panel['treated_x_post'] = panel['treated'] * panel['post']
    
    # Filter date range
    panel = panel[panel['period_start'] <= MAX_DATE]
    
    n_treated = panel[panel['treated'] == 1]['unit_id'].nunique()
    n_control = panel[panel['treated'] == 0]['unit_id'].nunique()
    
    print(f"\n  Treatment (Fixed Date: {UNDP_START_DATE.strftime('%Y-%m-%d')}):")
    print(f"    Treated units: {n_treated}")
    print(f"    Control units: {n_control}")
    print(f"    Pre-treatment periods: {panel[panel['post'] == 0]['period_start'].nunique()}")
    print(f"    Post-treatment periods: {panel[panel['post'] == 1]['period_start'].nunique()}")
    
    return panel


def create_treatment_panel_staggered(panel, undp_projects, gdf, cols):
    """
    Create treatment panel with STAGGERED treatment timing.
    Each project uses its actual start_date.
    """
    id_col, name_col = cols
    
    if undp_projects is None or len(undp_projects) == 0:
        print("  ⚠ No UNDP projects for staggered treatment")
        return None
    
    # Get earliest project start date per... this is tricky without township mapping
    # Since undp_projects.csv doesn't have township codes, we use the earliest date
    
    earliest_date = undp_projects['start_date'].min()
    print(f"  Using earliest project date: {earliest_date}")
    
    # For now, fall back to using the earliest project start date as treatment
    # A more sophisticated approach would require matching projects to townships
    
    panel['post'] = (panel['period_start'] >= earliest_date).astype(int)
    
    # Load treatment townships from project file
    treated_townships = load_undp_treatment_from_project_file()
    
    if treated_townships:
        gdf['clean_name'] = gdf[name_col].apply(clean_name)
        gdf['treated'] = gdf['clean_name'].isin(treated_townships).astype(int)
        
        treatment_map = gdf.set_index(id_col)['treated'].to_dict()
        panel['treated'] = panel['unit_id'].map(treatment_map).fillna(0).astype(int)
    else:
        panel['treated'] = 0
    
    panel['treated_x_post'] = panel['treated'] * panel['post']
    
    panel = panel[panel['period_start'] <= MAX_DATE]
    
    return panel


def get_spatial_subset(df, gdf, id_col, radius_deg, min_radius_deg=0):
    """
    Get spatial subset: treated units + control units within radius.
    Creates donut if min_radius_deg > 0.
    """
    treated_ids = df[df['treated'] == 1]['unit_id'].unique()
    
    if len(treated_ids) == 0:
        return df
    
    # Get treated geometries
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    
    if len(treated_geoms) == 0:
        return df
    
    # Buffer around treated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buffer_outer = treated_geoms.geometry.buffer(radius_deg).unary_union
        
        if min_radius_deg > 0:
            buffer_inner = treated_geoms.geometry.buffer(min_radius_deg).unary_union
            buffer_zone = buffer_outer.difference(buffer_inner)
        else:
            buffer_zone = buffer_outer
    
    # Find control units in buffer
    control_in_buffer = gdf[
        (~gdf[id_col].isin(treated_ids)) & 
        (gdf.geometry.intersects(buffer_zone))
    ][id_col].unique()
    
    # Keep treated + control in buffer
    keep_ids = set(treated_ids) | set(control_in_buffer)
    
    return df[df['unit_id'].isin(keep_ids)]


def run_twfe_regression(df, outcome, sample_name, controls=None):
    """
    Run Two-Way Fixed Effects regression using PanelOLS.
    
    Model: Y_it = α_i + λ_t + β(Treated_i × Post_t) + γX_it + ε_it
    """
    if outcome not in df.columns:
        return None
    
    df_reg = df.dropna(subset=[outcome, 'treated_x_post']).copy()
    
    if len(df_reg) < 100:
        return None
    
    # Build formula
    control_str = ' + '.join(controls) if controls else ''
    
    if HAS_LINEARMODELS:
        # Use linearmodels PanelOLS
        if control_str:
            formula = f"{outcome} ~ treated_x_post + {control_str} + EntityEffects + TimeEffects"
        else:
            formula = f"{outcome} ~ treated_x_post + EntityEffects + TimeEffects"
        
        try:
            # Need to set index for panel
            df_panel = df_reg.set_index(['unit_id', 'period_start'])
            
            mod = PanelOLS.from_formula(formula, data=df_panel, check_rank=False)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            
            return {
                'Outcome': outcome,
                'Sample': sample_name,
                'Controls': control_str if control_str else 'None',
                'Coeff': f"{res.params['treated_x_post']:.4f}",
                'SE': f"({res.std_errors['treated_x_post']:.4f})",
                'P-value': res.pvalues['treated_x_post'],
                'Stars': '***' if res.pvalues['treated_x_post'] < 0.01 else ('**' if res.pvalues['treated_x_post'] < 0.05 else ('*' if res.pvalues['treated_x_post'] < 0.1 else '')),
                'Obs': int(res.nobs),
                'Units': df_reg['unit_id'].nunique(),
                'R2': f"{res.rsquared:.3f}"
            }
        except Exception as e:
            print(f"    ⚠ Regression error: {e}")
            return None
    
    elif HAS_STATSMODELS:
        # Fallback to statsmodels OLS with dummies
        print("    Using statsmodels fallback (slower)")
        
        # Create dummies
        df_reg = pd.get_dummies(df_reg, columns=['unit_id'], drop_first=True, prefix='fe')
        time_dummies = pd.get_dummies(df_reg['period_start'].astype(str), drop_first=True, prefix='t')
        df_reg = pd.concat([df_reg, time_dummies], axis=1)
        
        # Build X
        x_cols = ['treated_x_post'] + (controls if controls else [])
        x_cols += [c for c in df_reg.columns if c.startswith('fe_') or c.startswith('t_')]
        
        X = df_reg[x_cols]
        X = sm.add_constant(X)
        y = df_reg[outcome]
        
        try:
            model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_reg['unit_id']})
            
            return {
                'Outcome': outcome,
                'Sample': sample_name,
                'Controls': control_str if control_str else 'None',
                'Coeff': f"{model.params['treated_x_post']:.4f}",
                'SE': f"({model.bse['treated_x_post']:.4f})",
                'P-value': model.pvalues['treated_x_post'],
                'Stars': '***' if model.pvalues['treated_x_post'] < 0.01 else ('**' if model.pvalues['treated_x_post'] < 0.05 else ('*' if model.pvalues['treated_x_post'] < 0.1 else '')),
                'Obs': int(model.nobs),
                'Units': df_reg['unit_id'].nunique(),
                'R2': f"{model.rsquared:.3f}"
            }
        except Exception as e:
            print(f"    ⚠ Regression error: {e}")
            return None
    
    else:
        print("  ⚠ No regression package available")
        return None


def main():
    parser = argparse.ArgumentParser(description='TWFE DiD Analysis')
    parser.add_argument('--staggered', action='store_true', 
                       help='Use staggered treatment timing from project dates')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MYANMAR EARTHQUAKE: TWFE DIFFERENCE-IN-DIFFERENCES")
    print("=" * 70)
    
    if not HAS_LINEARMODELS and not HAS_STATSMODELS:
        print("\n⚠ Need linearmodels or statsmodels for regression")
        print("  pip install linearmodels --break-system-packages")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    panel = load_panel_data()
    if panel is None:
        return
    print(f"  Panel: {len(panel):,} obs, {panel['unit_id'].nunique()} units")
    
    gdf, cols = load_boundaries()
    if gdf is None:
        print("  ⚠ Boundaries not loaded")
        return
    print(f"  Boundaries: {len(gdf)} admin units")
    
    id_col, name_col = cols
    print(f"  ID column: {id_col}")
    print(f"  Name column: {name_col}")
    
    # Load treatment definition
    print("\n--- Treatment Definition ---")
    treated_townships = load_undp_treatment_from_project_file()
    
    if treated_townships is None or len(treated_townships) == 0:
        print("  ⚠ No treatment townships found")
        return
    
    print(f"  UNDP-treated townships: {len(treated_townships)}")
    
    # Create treatment panel
    if args.staggered:
        print("\n--- Using STAGGERED Treatment Timing ---")
        undp_projects = load_undp_projects_with_dates()
        df = create_treatment_panel_staggered(panel, undp_projects, gdf, cols)
    else:
        print("\n--- Using FIXED Treatment Date ---")
        df = create_treatment_panel_fixed_date(panel, treated_townships, gdf, cols)
    
    if df is None or len(df) == 0:
        print("  ⚠ Failed to create treatment panel")
        return
    
    # Define spatial subsets (from old script)
    subsets = {
        'National (All)': df,
        'Spatial 25km': get_spatial_subset(df, gdf, id_col, 0.25, 0),
        'Spatial 50km': get_spatial_subset(df, gdf, id_col, 0.50, 0),
        'Donut (15-50km)': get_spatial_subset(df, gdf, id_col, 0.50, 0.15),
    }
    
    # Run regressions
    print("\n" + "=" * 70)
    print("MAIN ANALYSIS: TWFE Regressions")
    print("=" * 70)
    
    results = []
    
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        
        print(f"\n  {outcome}:")
        
        for sample_name, df_sub in subsets.items():
            # Without controls
            res = run_twfe_regression(df_sub, outcome, sample_name, controls=[])
            if res:
                results.append(res)
                print(f"    {sample_name}: {res['Coeff']} {res['SE']} {res['Stars']}")
            
            # With controls
            res = run_twfe_regression(df_sub, outcome, f"{sample_name} + controls", controls=CONTROLS)
            if res:
                results.append(res)
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = RESULTS_DIR / 'did_results_twfe.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY: Key Results (log_nightlights, Spatial 25km)")
        print("=" * 70)
        
        key_results = results_df[
            (results_df['Outcome'] == 'log_nightlights') & 
            (results_df['Sample'].str.contains('25km'))
        ]
        
        if len(key_results) > 0:
            for _, row in key_results.iterrows():
                print(f"  {row['Sample']}: β = {row['Coeff']} {row['SE']} {row['Stars']}")
                print(f"    Obs: {row['Obs']}, Units: {row['Units']}, R²: {row['R2']}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
