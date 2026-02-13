#!/usr/bin/env python3
"""
analysis/mmr/02_did_analysis.py
===============================
Difference-in-Differences Analysis for Myanmar Earthquake (March 28, 2025).

Methodology:
- Treatment: Areas near UNDP intervention sites (within treatment buffer)
- Control: Donut-shaped buffer areas (far enough to avoid spillovers)
- Pre-period: Oct 2023 - Mar 2025 (18 months)
- Post-period: Apr 2025 - Jan 2026 (10 months)

Outcomes:
- Nightlights (VIIRS) - economic activity proxy
- NDVI - vegetation/agriculture recovery
- SAR VH/VV - structural changes
- Conflict events - security situation

Usage:
    python analysis/mmr/02_did_analysis.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
INTERVENTION_DATE = pd.Timestamp("2025-04-01")

# Buffer specifications (in meters) - wider than Gaza due to rural context
BUFFER_SPECS = [
    (1000, 3000, "1km-3km", "tight"),
    (2000, 5000, "2km-5km", "moderate"),
    (3000, 8000, "3km-8km", "baseline"),
    (5000, 12000, "5km-12km", "wide"),
    (8000, 20000, "8km-20km", "very_wide"),
]

DEG_PER_M = 1 / 111000

# Outcomes to analyze
OUTCOMES = {
    'nightlights': 'Nightlights (VIIRS)',
    'log_nightlights': 'Log Nightlights',
    'ndvi': 'NDVI',
    'vh': 'SAR VH',
    'vv': 'SAR VV',
    'rvi': 'Radar Vegetation Index',
    'conflict_events': 'Conflict Events',
    'rainfall': 'Rainfall',
}


def load_data():
    """Load panel data and UNDP sites."""
    
    # Panel
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel not found: {PANEL_FILE}")
        return None, None, None
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    # Add log transforms
    if 'nightlights' in panel.columns:
        panel['log_nightlights'] = np.log(panel['nightlights'].clip(lower=0) + 1)
    
    # UNDP sites
    sites = None
    if SITES_FILE.exists():
        sites = pd.read_csv(SITES_FILE)
    
    # Boundaries
    gdf = None
    for path in [BOUNDARIES_FILE,
                 DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson"]:
        if path.exists():
            gdf = gpd.read_file(path).to_crs("EPSG:4326")
            break
    
    return panel, sites, gdf


def assign_treatment_status(panel, sites, gdf, inner_m, outer_m):
    """
    Assign treatment status based on distance to UNDP sites.
    
    Treatment = 1 if admin unit centroid is within inner_m of any UNDP site
    Control = 1 if admin unit centroid is between inner_m and outer_m (donut)
    """
    
    if sites is None or len(sites) == 0:
        return panel
    
    # Get admin unit centroids
    id_col = None
    for col in ['adm3_pcode', 'TS_PCODE', 'ADM3_PCODE']:
        if col in gdf.columns:
            id_col = col
            break
    
    if id_col is None:
        print("  ⚠ Could not find ID column in boundaries")
        return panel
    
    # Compute centroids
    if 'center_lat' in gdf.columns and 'center_lon' in gdf.columns:
        admin_coords = gdf[[id_col, 'center_lon', 'center_lat']].copy()
        admin_coords.columns = ['unit_id', 'lon', 'lat']
    else:
        centroids = gdf.geometry.centroid
        admin_coords = pd.DataFrame({
            'unit_id': gdf[id_col],
            'lon': centroids.x,
            'lat': centroids.y
        })
    
    # UNDP site coordinates
    site_coords = np.array([[row['longitude'], row['latitude']] for _, row in sites.iterrows()])
    tree = cKDTree(site_coords)
    
    # Convert buffers to degrees
    inner_deg = inner_m * DEG_PER_M
    outer_deg = outer_m * DEG_PER_M
    
    # Assign treatment status
    treatment_status = []
    for _, row in admin_coords.iterrows():
        dist, _ = tree.query([row['lon'], row['lat']])
        
        if dist <= inner_deg:
            status = 'treated'
        elif dist <= outer_deg:
            status = 'control'
        else:
            status = 'far'  # Exclude from analysis
        
        treatment_status.append({
            'unit_id': row['unit_id'],
            'min_dist_to_undp': dist / DEG_PER_M,  # Convert back to meters
            'treatment_status': status
        })
    
    treatment_df = pd.DataFrame(treatment_status)
    
    # Merge with panel
    panel_merged = panel.merge(treatment_df, on='unit_id', how='left')
    
    return panel_merged


def run_did_regression(panel, outcome, spec_label):
    """
    Run difference-in-differences regression.
    
    Model: Y = β0 + β1*Post + β2*Treated + β3*(Post×Treated) + ε
    
    With TWFE: Y = β0 + β1*(Post×Treated) + αi + γt + ε
    """
    
    # Filter to treatment and control only
    df = panel[panel['treatment_status'].isin(['treated', 'control'])].copy()
    
    if len(df) == 0:
        return None
    
    # Create indicators
    df['post'] = (df['period_start'] >= INTERVENTION_DATE).astype(int)
    df['treated'] = (df['treatment_status'] == 'treated').astype(int)
    df['post_treated'] = df['post'] * df['treated']
    
    # Create period identifier for FE
    df['period'] = df['period_start'].dt.to_period('M').astype(str)
    
    # Check outcome availability
    if outcome not in df.columns or df[outcome].isna().all():
        return None
    
    # Drop missing values
    df_clean = df[[outcome, 'post', 'treated', 'post_treated', 'unit_id', 'period']].dropna()
    
    if len(df_clean) < 100:
        return None
    
    n_treated = df_clean[df_clean['treated'] == 1]['unit_id'].nunique()
    n_control = df_clean[df_clean['treated'] == 0]['unit_id'].nunique()
    
    results = {
        'outcome': outcome,
        'spec': spec_label,
        'n_treated': n_treated,
        'n_control': n_control,
        'n_obs': len(df_clean),
    }
    
    try:
        # Simple DiD
        model_simple = smf.ols(f'{outcome} ~ post + treated + post_treated', data=df_clean).fit()
        
        results['simple_coef'] = model_simple.params.get('post_treated', np.nan)
        results['simple_se'] = model_simple.bse.get('post_treated', np.nan)
        results['simple_pval'] = model_simple.pvalues.get('post_treated', np.nan)
        
        # TWFE (Two-Way Fixed Effects)
        try:
            model_twfe = smf.ols(f'{outcome} ~ post_treated + C(unit_id) + C(period)', data=df_clean).fit()
            results['twfe_coef'] = model_twfe.params.get('post_treated', np.nan)
            results['twfe_se'] = model_twfe.bse.get('post_treated', np.nan)
            results['twfe_pval'] = model_twfe.pvalues.get('post_treated', np.nan)
        except:
            results['twfe_coef'] = np.nan
            results['twfe_se'] = np.nan
            results['twfe_pval'] = np.nan
        
        # Calculate effect size (% change from control mean)
        control_mean = df_clean[df_clean['treated'] == 0][outcome].mean()
        if control_mean != 0:
            results['pct_effect'] = (results['simple_coef'] / abs(control_mean)) * 100
        else:
            results['pct_effect'] = np.nan
        
    except Exception as e:
        print(f"    ⚠ Regression error: {e}")
        return None
    
    return results


def create_results_table(all_results):
    """Create formatted results table."""
    
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        return df
    
    # Add significance stars
    def add_stars(pval):
        if pd.isna(pval):
            return ''
        elif pval < 0.01:
            return '***'
        elif pval < 0.05:
            return '**'
        elif pval < 0.10:
            return '*'
        return ''
    
    df['stars'] = df['simple_pval'].apply(add_stars)
    
    # Format coefficient with stars
    df['coef_formatted'] = df.apply(
        lambda x: f"{x['simple_coef']:.4f}{x['stars']}" if not pd.isna(x['simple_coef']) else 'N/A',
        axis=1
    )
    
    return df


def plot_coefficients(results_df, output_path):
    """Create coefficient plot across specifications."""
    
    # Group by outcome
    outcomes = results_df['outcome'].unique()
    
    fig, axes = plt.subplots(len(outcomes), 1, figsize=(12, 4*len(outcomes)))
    if len(outcomes) == 1:
        axes = [axes]
    
    for ax, outcome in zip(axes, outcomes):
        subset = results_df[results_df['outcome'] == outcome].copy()
        
        if len(subset) == 0:
            continue
        
        x = range(len(subset))
        coefs = subset['simple_coef'].values
        ses = subset['simple_se'].values
        specs = subset['spec'].values
        
        # Plot
        ax.errorbar(x, coefs, yerr=1.96*ses, fmt='o', capsize=5, capthick=2,
                   color='#3498db', markersize=8, elinewidth=2)
        
        ax.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(specs, rotation=45, ha='right')
        ax.set_ylabel('DiD Coefficient')
        ax.set_title(f'{OUTCOMES.get(outcome, outcome)}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Myanmar Earthquake: DiD Coefficients by Buffer Specification\n(95% CI)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading Data ---")
    panel, sites, gdf = load_data()
    
    if panel is None:
        print("  ⚠ No panel data. Run preprocessing first.")
        return
    
    print(f"  Panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")
    print(f"  Period: {panel['period_start'].min()} to {panel['period_start'].max()}")
    
    if sites is None:
        print("  ⚠ No UNDP sites. Run: python -m pipeline.parse_undp_sites --config config/myanmar.yaml")
        return
    
    print(f"  UNDP sites: {len(sites)}")
    
    if gdf is None:
        print("  ⚠ No boundaries file")
        return
    
    print(f"  Admin units: {len(gdf)}")
    
    # Available outcomes
    available = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"\n  Available outcomes: {', '.join(available)}")
    
    # Run analysis for each buffer specification
    all_results = []
    
    for inner_m, outer_m, spec_label, spec_key in BUFFER_SPECS:
        print(f"\n{'='*70}")
        print(f" BUFFER: {spec_label}")
        print("=" * 70)
        
        # Assign treatment
        panel_treated = assign_treatment_status(panel, sites, gdf, inner_m, outer_m)
        
        n_treated = panel_treated[panel_treated['treatment_status'] == 'treated']['unit_id'].nunique()
        n_control = panel_treated[panel_treated['treatment_status'] == 'control']['unit_id'].nunique()
        
        print(f"  Treated units: {n_treated}")
        print(f"  Control units: {n_control}")
        
        if n_treated == 0 or n_control == 0:
            print(f"  ⚠ Insufficient treatment/control units")
            continue
        
        # Run regressions
        print(f"\n  {'Outcome':<25} {'Coef':>12} {'SE':>10} {'p-val':>10} {'Effect %':>10}")
        print("  " + "-" * 70)
        
        for outcome in available:
            result = run_did_regression(panel_treated, outcome, spec_label)
            
            if result is not None:
                all_results.append(result)
                
                coef = result['simple_coef']
                se = result['simple_se']
                pval = result['simple_pval']
                pct = result.get('pct_effect', np.nan)
                
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                
                print(f"  {OUTCOMES[outcome]:<25} {coef:>12.4f} {se:>10.4f} {pval:>10.4f} {pct:>9.1f}%{stars}")
            else:
                print(f"  {OUTCOMES[outcome]:<25} {'N/A':>12}")
    
    # Create results table
    if all_results:
        results_df = create_results_table(all_results)
        
        # Save results
        results_df.to_csv(RESULTS_DIR / 'did_results.csv', index=False)
        print(f"\n  ✓ Saved: {RESULTS_DIR / 'did_results.csv'}")
        
        # Create coefficient plot
        plot_coefficients(results_df, RESULTS_DIR / 'did_coefficients.png')
        print(f"  ✓ Saved: {RESULTS_DIR / 'did_coefficients.png'}")
        
        # Summary
        print("\n" + "=" * 70)
        print(" SUMMARY: Significant Results (p < 0.10)")
        print("=" * 70)
        
        sig_results = results_df[results_df['simple_pval'] < 0.10]
        
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                direction = "+" if row['simple_coef'] > 0 else "-"
                print(f"  {row['outcome']} ({row['spec']}): {direction}{abs(row['pct_effect']):.1f}% (p={row['simple_pval']:.3f})")
        else:
            print("  No statistically significant results at p < 0.10")
        
        print(f"\n  Output: {RESULTS_DIR}")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
