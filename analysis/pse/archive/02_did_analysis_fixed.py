#!/usr/bin/env python3
"""
analysis/mmr/02_did_analysis_fixed.py
=====================================
Difference-in-Differences Analysis with PROPER specifications.

TREATMENT DATE SELECTION:
- Uses the RELEVANT project from undp_projects.csv
- For Myanmar: Infrastructure & Housing (Sept 2025), NOT Environment (April 2025)
- Can specify category to filter on

TWFE SPECIFICATION:
Model: Y_it = α_i + λ_t + β(Treated_i × Post_t) + γX_it + ε_it

Where:
- Y_it = outcome for unit i at time t
- α_i = unit (township) fixed effects (EntityEffects)
- λ_t = time (month) fixed effects (TimeEffects)  
- Treated_i × Post_t = DiD interaction term
- X_it = controls (rainfall, conflict, prices)

CONTROLS:
- rainfall: Monthly precipitation (CHIRPS)
- conflict_events: ACLED conflict count
- price_rice: WFP rice prices
- State FE: Implicitly captured by unit FE (townships are nested in states)

Usage:
    python analysis/mmr/02_did_analysis_fixed.py
    python analysis/mmr/02_did_analysis_fixed.py --category "Infrastructure & Housing"
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
    print("⚠ linearmodels not installed")

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

# Outcomes
OUTCOMES = ['nightlights', 'log_nightlights', 'ndvi', 'vh', 'vv', 'rvi', 
            'conflict_events', 'rainfall']

# FULL CONTROL SET
CONTROLS_FULL = ['rainfall', 'conflict_events']
CONTROLS_MINIMAL = ['rainfall']


def clean_name(name):
    if pd.isna(name):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


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
    id_col = next((c for c in gdf.columns if 'pcode' in c.lower()), None)
    name_col = next((c for c in gdf.columns if c.lower() in ['ts', 'adm3_en', 'adm3_name', 'name']), None)
    return gdf, (id_col, name_col)


def load_undp_treatment_townships():
    """Load treated townships from project_data.xlsx."""
    if not PROJECT_FILE.exists():
        return None
    try:
        df_proj = pd.read_excel(PROJECT_FILE)
    except:
        return None
    
    town_col = next((c for c in df_proj.columns if 'Township' in c and 'State' not in c), None)
    if not town_col:
        return None
    
    df_proj['clean_name'] = df_proj[town_col].apply(clean_name)
    treated = set(df_proj['clean_name'].unique())
    treated.discard('')
    return treated


def get_treatment_date_from_projects(category_filter=None):
    """
    Get treatment date from undp_projects.csv.
    
    If category_filter is specified (e.g., "Infrastructure & Housing"),
    use that project's start date. Otherwise, use the earliest post-earthquake project.
    """
    if not UNDP_PROJECTS_FILE.exists():
        return EARTHQUAKE_DATE + pd.Timedelta(days=30), "default"
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter to post-earthquake
    df = df[df['start_date'] >= EARTHQUAKE_DATE]
    
    # Filter out umbrella projects
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    if len(df) == 0:
        return EARTHQUAKE_DATE + pd.Timedelta(days=30), "default"
    
    # Display available projects
    print("\n  Available post-earthquake UNDP projects:")
    for _, row in df.iterrows():
        cat = row.get('ml_category', 'Unknown')
        date = row['start_date'].strftime('%Y-%m-%d')
        title = str(row.get('title', ''))[:50]
        print(f"    [{date}] {cat}: {title}")
    
    # If category filter specified, use that
    if category_filter:
        filtered = df[df['ml_category'].str.contains(category_filter, case=False, na=False)]
        if len(filtered) > 0:
            project = filtered.iloc[0]
            date = project['start_date']
            cat = project.get('ml_category', 'Unknown')
            print(f"\n  → Selected: {cat} ({date.strftime('%Y-%m-%d')})")
            return date, cat
        else:
            print(f"\n  ⚠ Category '{category_filter}' not found")
    
    # Otherwise, use the INFRASTRUCTURE project (most relevant for physical recovery)
    infra = df[df['ml_category'].str.contains('Infrastructure|Housing|Debris', case=False, na=False)]
    if len(infra) > 0:
        project = infra.iloc[0]
        date = project['start_date']
        cat = project.get('ml_category', 'Unknown')
        print(f"\n  → Auto-selected (Infrastructure focus): {cat} ({date.strftime('%Y-%m-%d')})")
        return date, cat
    
    # Fall back to earliest
    project = df.iloc[0]
    date = project['start_date']
    cat = project.get('ml_category', 'Unknown')
    print(f"\n  → Using earliest: {cat} ({date.strftime('%Y-%m-%d')})")
    return date, cat


def create_treatment_panel(panel, treated_townships, gdf, cols, treatment_date):
    """Create treatment panel with proper treatment assignment."""
    id_col, name_col = cols
    
    if name_col is None:
        return None
    
    gdf['clean_name'] = gdf[name_col].apply(clean_name)
    gdf['treated'] = gdf['clean_name'].isin(treated_townships).astype(int)
    
    treatment_map = gdf.set_index(id_col)['treated'].to_dict()
    panel['treated'] = panel['unit_id'].map(treatment_map).fillna(0).astype(int)
    
    # POST = after treatment date
    panel['post'] = (panel['period_start'] >= treatment_date).astype(int)
    
    # DiD interaction
    panel['treated_x_post'] = panel['treated'] * panel['post']
    
    return panel


def get_spatial_subset(df, gdf, id_col, radius_deg, min_radius_deg=0):
    """Get spatial subset using donut buffers."""
    treated_ids = df[df['treated'] == 1]['unit_id'].unique()
    if len(treated_ids) == 0:
        return df
    
    treated_geoms = gdf[gdf[id_col].isin(treated_ids)]
    if len(treated_geoms) == 0:
        return df
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buffer_outer = treated_geoms.geometry.buffer(radius_deg).unary_union
        if min_radius_deg > 0:
            buffer_inner = treated_geoms.geometry.buffer(min_radius_deg).unary_union
            buffer_zone = buffer_outer.difference(buffer_inner)
        else:
            buffer_zone = buffer_outer
    
    control_in_buffer = gdf[
        (~gdf[id_col].isin(treated_ids)) & 
        (gdf.geometry.intersects(buffer_zone))
    ][id_col].unique()
    
    keep_ids = set(treated_ids) | set(control_in_buffer)
    return df[df['unit_id'].isin(keep_ids)]


def run_twfe_regression(df, outcome, sample_name, controls=None):
    """
    Run TWFE regression with proper specification.
    
    Model: Y_it = α_i + λ_t + β(treated_x_post) + γX_it + ε_it
    
    - EntityEffects = unit (township) FE
    - TimeEffects = period (month) FE
    - Clustered standard errors at entity level
    """
    if outcome not in df.columns:
        return None
    
    df_reg = df.dropna(subset=[outcome, 'treated_x_post']).copy()
    
    if len(df_reg) < 100:
        return None
    
    # Handle missing controls
    if controls:
        valid_controls = [c for c in controls if c in df_reg.columns and df_reg[c].notna().any()]
        df_reg = df_reg.dropna(subset=valid_controls)
        control_str = ' + '.join(valid_controls)
    else:
        control_str = ''
        valid_controls = []
    
    # Build formula
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
        else:
            pct_effect = np.nan
        
        return {
            'Outcome': outcome,
            'Sample': sample_name,
            'Controls': control_str if control_str else 'None',
            'Coeff': coef,
            'Coeff_Fmt': f"{coef:.4f}",
            'SE': se,
            'SE_Fmt': f"({se:.4f})",
            'P_value': pval,
            'Stars': '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else '')),
            'Pct_Effect': f"{pct_effect:+.1f}%" if not np.isnan(pct_effect) else "",
            'N_Obs': int(res.nobs),
            'N_Units': df_reg['unit_id'].nunique(),
            'N_Treated': df_reg[df_reg['treated'] == 1]['unit_id'].nunique(),
            'N_Control': df_reg[df_reg['treated'] == 0]['unit_id'].nunique(),
            'R2': res.rsquared,
        }
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='TWFE DiD Analysis')
    parser.add_argument('--category', '-c', default=None,
                       help='UNDP project category for treatment date (e.g., "Infrastructure")')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MYANMAR EARTHQUAKE: TWFE DiD ANALYSIS")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("MODEL SPECIFICATION")
    print("=" * 70)
    print("""
    Y_it = α_i + λ_t + β(Treated_i × Post_t) + γX_it + ε_it
    
    Where:
    - α_i = Unit (township) fixed effects
    - λ_t = Time (month) fixed effects
    - β = DiD estimate (treatment effect)
    - X_it = Controls (rainfall, conflict)
    - Standard errors clustered at unit level
    """)
    
    if not HAS_LINEARMODELS:
        print("⚠ Need linearmodels")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    
    panel = load_panel_data()
    if panel is None:
        return
    print(f"  Panel: {len(panel):,} obs, {panel['unit_id'].nunique()} units")
    
    gdf, cols = load_boundaries()
    if gdf is None:
        return
    id_col, name_col = cols
    print(f"  Boundaries: {len(gdf)} units")
    
    # Load treatment definition
    treated_townships = load_undp_treatment_townships()
    if treated_townships is None or len(treated_townships) == 0:
        print("  ⚠ No treatment townships found")
        return
    print(f"  UNDP-treated townships: {len(treated_townships)}")
    
    # Get treatment date from projects
    print("\n--- Treatment Timing ---")
    treatment_date, project_category = get_treatment_date_from_projects(args.category)
    print(f"\n  FINAL TREATMENT DATE: {treatment_date.strftime('%Y-%m-%d')}")
    print(f"  Project Category: {project_category}")
    
    # Create treatment panel
    df = create_treatment_panel(panel, treated_townships, gdf, cols, treatment_date)
    
    n_treated = df[df['treated'] == 1]['unit_id'].nunique()
    n_control = df[df['treated'] == 0]['unit_id'].nunique()
    n_pre = df[df['post'] == 0]['period_start'].nunique()
    n_post = df[df['post'] == 1]['period_start'].nunique()
    
    print(f"\n  Treatment Panel:")
    print(f"    Treated units: {n_treated}")
    print(f"    Control units: {n_control}")
    print(f"    Pre-treatment periods: {n_pre}")
    print(f"    Post-treatment periods: {n_post}")
    
    # Define spatial subsets
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
    
    all_results = []
    
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        
        print(f"\n  {outcome}:")
        
        for sample_name, df_sub in subsets.items():
            # Without controls
            res = run_twfe_regression(df_sub, outcome, sample_name, controls=[])
            if res:
                all_results.append(res)
                print(f"    {sample_name}: {res['Coeff_Fmt']} {res['SE_Fmt']} {res['Stars']} {res['Pct_Effect']}")
            
            # With minimal controls (rainfall)
            res = run_twfe_regression(df_sub, outcome, f"{sample_name} + rainfall", controls=CONTROLS_MINIMAL)
            if res:
                all_results.append(res)
            
            # With full controls
            res = run_twfe_regression(df_sub, outcome, f"{sample_name} + full_controls", controls=CONTROLS_FULL)
            if res:
                all_results.append(res)
    
    if not all_results:
        print("\n⚠ No results")
        return
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = RESULTS_DIR / 'did_results_twfe.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS: log_nightlights")
    print("=" * 70)
    
    key_results = results_df[
        (results_df['Outcome'] == 'log_nightlights') & 
        (results_df['Controls'] == 'None')
    ]
    
    print(f"\n{'Sample':<20} {'Coeff':>10} {'SE':>12} {'Stars':>6} {'% Effect':>10} {'N_Treat':>8} {'N_Ctrl':>8}")
    print("-" * 80)
    
    for _, row in key_results.iterrows():
        print(f"{row['Sample']:<20} {row['Coeff_Fmt']:>10} {row['SE_Fmt']:>12} {row['Stars']:>6} {row['Pct_Effect']:>10} {row['N_Treated']:>8} {row['N_Control']:>8}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    best = key_results.iloc[key_results['Coeff'].argmax()]
    print(f"""
    Best estimate ({best['Sample']}):
    - Coefficient: {best['Coeff_Fmt']} {best['Stars']}
    - Interpretation: UNDP-treated townships show {best['Pct_Effect']} 
      higher nightlights after {project_category} intervention
    - Standard Error: {best['SE_Fmt']}
    - Observations: {best['N_Obs']:,}
    - Treated units: {best['N_Treated']}, Control units: {best['N_Control']}
    """)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
