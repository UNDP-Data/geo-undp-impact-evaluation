#!/usr/bin/env python3
"""
State-Level Difference-in-Differences Analysis - CORRECTED VERSION
==================================================================

FIXES:
1. Properly excludes March 2025 data (contaminated by earthquake on March 28)
2. Uses explicit PRE_PERIOD (Jan-Feb) and POST_PERIOD (Apr-Sep) definitions
3. Adds data validation to ensure no March observations in analysis

Proper econometric specification:
    Y_st = β0 + β1*POST_t + β2*TREATED_s + β3*(POST_t × TREATED_s) + ε_st

Where:
    Y_st = Nightlight radiance in state s at time t
    POST_t = Dummy for post-earthquake period (April-Sept 2025 ONLY)
    TREATED_s = Dummy for states with UNDP presence
    β3 = Difference-in-differences estimator (causal effect)

CRITICAL: March 2025 data is EXCLUDED because:
- Earthquake occurred March 28, 2025
- Monthly VIIRS data aggregates entire month (March 1-31)
- March data contains both pre (March 1-27) and post (March 28-31) earthquake periods
- Including March would contaminate the treatment definition

Usage:
    python state_level_did_regression_corrected.py \\
        --viirs-dir data/viirs_25 \\
        --boundaries-dir data/boundaries/processed \\
        --undp-support undp_interventions.geojson \\
        --output-dir outputs/regression_analysis_corrected
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import rioxarray as rxr
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    DEPS_OK = True
except ImportError:
    DEPS_OK = False
    print("⚠ Missing dependencies. Install with:")
    print("  pip install rioxarray statsmodels")


# Earthquake date
EARTHQUAKE_DATE = datetime(2025, 3, 28)

# CORRECTED: Use explicit period definitions, NOT earthquake date
# March is EXCLUDED because monthly data includes both pre/post earthquake days
PRE_PERIOD_START = datetime(2025, 1, 1)
PRE_PERIOD_END = datetime(2025, 2, 28)    # End of February
POST_PERIOD_START = datetime(2025, 4, 1)  # Start of April (skip March!)
POST_PERIOD_END = datetime(2025, 9, 30)


def find_viirs_files(viirs_dir, start_date=None, end_date=None, exclude_months=None):
    """
    Find VIIRS files in directory.
    
    Parameters:
        viirs_dir: Directory containing VIIRS files
        start_date: Optional start date filter
        end_date: Optional end date filter
        exclude_months: Optional list of (year, month) tuples to exclude
    """
    viirs_dir = Path(viirs_dir)
    tif_files = list(viirs_dir.glob("VIIRS_*.tif")) + list(viirs_dir.glob("SVDNB_*.tif"))
    tif_files = [f for f in tif_files if not f.name.endswith('.tif.gz')]
    
    files_with_dates = []
    for f in tif_files:
        parts = f.name.split('_')
        date = None
        
        if f.name.startswith('VIIRS_') and len(parts) >= 2:
            yyyymm = parts[1]
            if len(yyyymm) == 6 and yyyymm.isdigit():
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        elif f.name.startswith('SVDNB_') and len(parts) >= 3:
            date_range = parts[2].split('-')[0]
            if len(date_range) >= 6 and date_range[:6].isdigit():
                yyyymm = date_range[:6]
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])
                date = datetime(year, month, 1)
        
        if date:
            # Filter by date range
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            
            # CRITICAL: Exclude specific months (e.g., March 2025)
            if exclude_months and (date.year, date.month) in exclude_months:
                print(f"  ⚠ EXCLUDING contaminated month: {date.strftime('%Y-%m')} (contains earthquake date)")
                continue
            
            files_with_dates.append((date, f))
    
    files_with_dates.sort(key=lambda x: x[0])
    return files_with_dates


def load_and_clip_viirs(tif_file, bbox):
    """Load VIIRS file and clip to bounding box."""
    raster = rxr.open_rasterio(tif_file, masked=True).squeeze()
    lon_min, lat_min, lon_max, lat_max = bbox
    raster_clipped = raster.rio.clip_box(
        minx=lon_min, miny=lat_min,
        maxx=lon_max, maxy=lat_max
    )
    return raster_clipped


def calculate_state_mean_nightlight(raster, state_geometry):
    """Calculate mean nightlight for a state (properly masked to geometry)."""
    from rasterio.features import geometry_mask
    
    try:
        mask = geometry_mask(
            [state_geometry],
            out_shape=raster.shape,
            transform=raster.rio.transform(),
            invert=True
        )
        
        values = raster.values[mask]
        values = values[~np.isnan(values)]
        
        if len(values) > 0:
            return float(np.mean(values))
        else:
            return np.nan
    except Exception as e:
        return np.nan


def classify_treatment_status(states_gdf, undp_gdf):
    """
    Classify states as treated (UNDP presence) or control (no UNDP).
    """
    print("\n" + "="*80)
    print("CLASSIFYING TREATMENT STATUS")
    print("="*80)
    
    states_gdf = states_gdf.copy()
    states_gdf['treated'] = 0
    states_gdf['undp_count'] = 0
    
    # Count UNDP interventions per state
    for idx, state in states_gdf.iterrows():
        if undp_gdf is not None:
            intersecting = undp_gdf[undp_gdf.intersects(state.geometry)]
            count = len(intersecting)
            states_gdf.at[idx, 'undp_count'] = count
            if count > 0:
                states_gdf.at[idx, 'treated'] = 1
    
    # Print treatment assignment
    print("\nTreatment assignment:")
    for idx, row in states_gdf.iterrows():
        state_name = row.get('ST', row.get('name', row.get('ST_RG', f'State_{idx}')))
        status = "TREATED" if row['treated'] == 1 else "CONTROL"
        count = int(row['undp_count'])
        print(f"  {state_name:20s}: {status:8s} (n={count:3d} interventions)")
    
    n_treated = states_gdf['treated'].sum()
    n_control = len(states_gdf) - n_treated
    
    print(f"\n{'='*80}")
    print(f"Treatment groups:")
    print(f"  Treated states: {n_treated}")
    print(f"  Control states: {n_control}")
    print(f"{'='*80}")
    
    return states_gdf


def create_panel_data(viirs_files, states_gdf):
    """
    Create state-level panel dataset.
    CORRECTED: Uses explicit period definitions instead of earthquake date.
    """
    print("\n" + "="*80)
    print("CREATING PANEL DATASET")
    print("="*80)
    
    panel_data = []
    
    # Get bounding box
    bounds = states_gdf.total_bounds
    if (bounds[2] - bounds[0]) < 5:
        bbox = (92.0, 9.5, 101.5, 28.5)
    else:
        buffer = 0.5
        bbox = (bounds[0] - buffer, bounds[1] - buffer, 
                bounds[2] + buffer, bounds[3] + buffer)
    
    print(f"\nProcessing {len(viirs_files)} months...")
    print(f"PRE-period:  {PRE_PERIOD_START.strftime('%Y-%m')} to {PRE_PERIOD_END.strftime('%Y-%m')}")
    print(f"POST-period: {POST_PERIOD_START.strftime('%Y-%m')} to {POST_PERIOD_END.strftime('%Y-%m')}")
    print(f"EXCLUDED:    March 2025 (contaminated by earthquake on {EARTHQUAKE_DATE.strftime('%Y-%m-%d')})")
    
    for date, file in viirs_files:
        print(f"\n  {date.strftime('%Y-%m')}:")
        
        try:
            raster = load_and_clip_viirs(file, bbox)
            
            for idx, state in states_gdf.iterrows():
                state_name = state.get('ST', state.get('name', state.get('ST_RG', f'State_{idx}')))
                mean_nl = calculate_state_mean_nightlight(raster, state.geometry)
                
                if not np.isnan(mean_nl):
                    panel_data.append({
                        'state': state_name,
                        'state_id': idx,
                        'date': date,
                        'year_month': date.strftime('%Y-%m'),
                        'nightlight': mean_nl,
                        'treated': state['treated'],
                        'undp_count': state['undp_count']
                    })
                    print(f"    {state_name:20s}: {mean_nl:.6f}")
        
        except Exception as e:
            print(f"    ✗ Error processing {date.strftime('%Y-%m')}: {e}")
            continue
    
    df = pd.DataFrame(panel_data)
    
    if df.empty:
        print("\n✗ No panel data created")
        return None
    
    # CORRECTED: Define post indicator using EXPLICIT period boundaries
    # This ensures March is never included in either period
    df['post'] = ((df['date'] >= POST_PERIOD_START) & (df['date'] <= POST_PERIOD_END)).astype(int)
    df['pre'] = ((df['date'] >= PRE_PERIOD_START) & (df['date'] <= PRE_PERIOD_END)).astype(int)
    
    # DATA VALIDATION: Ensure every observation is either pre or post, never both or neither
    df['period_check'] = df['pre'] + df['post']
    
    print("\n" + "="*80)
    print("DATA VALIDATION CHECK")
    print("="*80)
    
    valid_obs = (df['period_check'] == 1).sum()
    invalid_obs = (df['period_check'] != 1).sum()
    
    print(f"Valid observations (in PRE or POST period): {valid_obs}")
    print(f"Invalid observations (not in either period): {invalid_obs}")
    
    if invalid_obs > 0:
        print("\n⚠ WARNING: Found observations not in PRE or POST period!")
        print("These will be EXCLUDED from the analysis:")
        excluded = df[df['period_check'] != 1][['state', 'year_month', 'date']]
        print(excluded.to_string())
        df = df[df['period_check'] == 1].copy()
    
    # Drop validation columns
    df = df.drop(columns=['pre', 'period_check'])
    
    # Create interaction term
    df['treated_x_post'] = df['treated'] * df['post']
    
    print(f"\n✓ Clean panel dataset created: {len(df)} observations")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Time periods: {df['date'].nunique()}")
    
    # Summary by period
    print(f"\nObservations by period:")
    print(f"  Pre-earthquake (Jan-Feb):  {(df['post'] == 0).sum()}")
    print(f"  Post-earthquake (Apr-Sep): {(df['post'] == 1).sum()}")
    
    return df


def run_did_regression(df, output_dir):
    """
    Run difference-in-differences regression.
    """
    print("\n" + "="*80)
    print("DIFFERENCE-IN-DIFFERENCES REGRESSION")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # DiD regression: Y = β0 + β1*POST + β2*TREATED + β3*(POST × TREATED) + ε
    formula = 'nightlight ~ treated + post + treated_x_post'
    
    model = ols(formula, data=df).fit()
    
    print("\n" + model.summary().as_text())
    
    # Extract key results
    did_coef = model.params['treated_x_post']
    did_se = model.bse['treated_x_post']
    did_pval = model.pvalues['treated_x_post']
    did_ci = model.conf_int().loc['treated_x_post']
    
    print("\n" + "="*80)
    print("DIFFERENCE-IN-DIFFERENCES ESTIMATE")
    print("="*80)
    print(f"\nCausal Effect (β3): {did_coef:+.6f}")
    print(f"Standard Error:     {did_se:.6f}")
    print(f"P-value:            {did_pval:.6f}")
    print(f"95% CI:             [{did_ci[0]:+.6f}, {did_ci[1]:+.6f}]")
    
    if did_pval < 0.01:
        sig = "***"
    elif did_pval < 0.05:
        sig = "**"
    elif did_pval < 0.10:
        sig = "*"
    else:
        sig = ""
    
    print(f"\nSignificance: {sig}")
    
    if did_coef > 0 and did_pval < 0.05:
        pct_effect = (did_coef / df[df['post'] == 0]['nightlight'].mean()) * 100
        print(f"\n✓ UNDP interventions increased nightlight recovery by {did_coef:.6f}")
        print(f"  ({pct_effect:.1f}% relative to pre-earthquake baseline)")
    elif did_coef < 0 and did_pval < 0.05:
        print(f"\n⚠ Negative effect detected: {did_coef:.6f}")
        print(f"  This may indicate UNDP targeted more severely affected areas")
    else:
        print(f"\n→ No statistically significant effect detected")
    
    # Save results
    results_df = pd.DataFrame({
        'coefficient': model.params,
        'std_error': model.bse,
        'p_value': model.pvalues,
        'ci_lower': model.conf_int()[0],
        'ci_upper': model.conf_int()[1]
    })
    
    results_file = output_dir / 'did_regression_results_corrected.csv'
    results_df.to_csv(results_file)
    print(f"\n✓ Saved regression results: {results_file}")
    
    # Save full summary
    summary_file = output_dir / 'regression_summary_corrected.txt'
    with open(summary_file, 'w') as f:
        f.write("CORRECTED DIFFERENCE-IN-DIFFERENCES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("SPECIFICATION NOTES:\n")
        f.write(f"- Pre-period:  {PRE_PERIOD_START.strftime('%Y-%m')} to {PRE_PERIOD_END.strftime('%Y-%m')}\n")
        f.write(f"- Post-period: {POST_PERIOD_START.strftime('%Y-%m')} to {POST_PERIOD_END.strftime('%Y-%m')}\n")
        f.write(f"- Earthquake:  {EARTHQUAKE_DATE.strftime('%Y-%m-%d')}\n")
        f.write(f"- March 2025:  EXCLUDED (contaminated by earthquake mid-month)\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(model.summary().as_text())
    
    print(f"✓ Saved full summary: {summary_file}")
    
    return model, results_df


def plot_did_visualization(df, model, output_dir):
    """Create DiD visualization."""
    output_dir = Path(output_dir)
    
    # Calculate means by group and period
    treated_pre = df[(df['treated'] == 1) & (df['post'] == 0)]['nightlight'].mean()
    treated_post = df[(df['treated'] == 1) & (df['post'] == 1)]['nightlight'].mean()
    control_pre = df[(df['treated'] == 0) & (df['post'] == 0)]['nightlight'].mean()
    control_post = df[(df['treated'] == 0) & (df['post'] == 1)]['nightlight'].mean()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Parallel trends visualization
    ax1 = axes[0]
    periods = ['Pre\n(Jan-Feb)', 'Post\n(Apr-Sep)']
    x_pos = [0, 1]
    
    ax1.plot(x_pos, [treated_pre, treated_post], 'o-', color='#2E86AB', 
             linewidth=2.5, markersize=10, label='Treated (UNDP States)')
    ax1.plot(x_pos, [control_pre, control_post], 's-', color='#A23B72',
             linewidth=2.5, markersize=10, label='Control States')
    
    # Add counterfactual
    counterfactual = treated_pre + (control_post - control_pre)
    ax1.plot(x_pos, [treated_pre, counterfactual], '--', color='#2E86AB',
             linewidth=1.5, alpha=0.5, label='Counterfactual (no treatment)')
    
    # Annotate DiD effect
    did_effect = model.params['treated_x_post']
    mid_x = 1
    mid_y = (treated_post + counterfactual) / 2
    ax1.annotate('', xy=(mid_x+0.05, treated_post), xytext=(mid_x+0.05, counterfactual),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(mid_x+0.15, mid_y, f'DiD Effect\nβ={did_effect:+.4f}',
             fontsize=9, color='red', fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(periods)
    ax1.set_ylabel('Mean Nightlight Radiance', fontsize=11)
    ax1.set_title('Difference-in-Differences Design\n(March 2025 EXCLUDED)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(0.5, ax1.get_ylim()[1]*0.95, 'Earthquake\n(Mar 28)', 
             ha='center', fontsize=8, color='red')
    
    # Panel 2: Time series by treatment group
    ax2 = axes[1]
    treated_ts = df[df['treated'] == 1].groupby('date')['nightlight'].mean()
    control_ts = df[df['treated'] == 0].groupby('date')['nightlight'].mean()
    
    ax2.plot(treated_ts.index, treated_ts.values, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, label='Treated (UNDP States)')
    ax2.plot(control_ts.index, control_ts.values, 's-', color='#A23B72',
             linewidth=2, markersize=6, label='Control States')
    
    ax2.axvline(EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Shade March (excluded period)
    march_start = datetime(2025, 3, 1)
    march_end = datetime(2025, 3, 31)
    ax2.axvspan(march_start, march_end, alpha=0.2, color='gray', label='March (EXCLUDED)')
    
    # Shade post-earthquake period
    ax2.axvspan(POST_PERIOD_START, df['date'].max(), alpha=0.1, color='red')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Mean Nightlight Radiance', fontsize=11)
    ax2.set_title('Time Series by Treatment Group', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Coefficient plot with confidence intervals
    ax3 = axes[2]
    coefs = model.params[['treated', 'post', 'treated_x_post']]
    errors = model.bse[['treated', 'post', 'treated_x_post']]
    ci = model.conf_int().loc[['treated', 'post', 'treated_x_post']]
    
    labels = ['Treated\n(β2)', 'Post\n(β1)', 'DiD Effect\n(β3)']
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    y_pos = np.arange(len(labels))
    
    for i, (coef, err, label, color) in enumerate(zip(coefs, errors, labels, colors)):
        ax3.errorbar(coef, i, xerr=[[coef - ci.iloc[i, 0]], [ci.iloc[i, 1] - coef]], 
                    fmt='o', color=color, markersize=10, capsize=5, capthick=2,
                    linewidth=2, label=label)
    
    ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('Coefficient Estimate', fontsize=11)
    ax3.set_title('Regression Coefficients (95% CI)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    viz_file = output_dir / 'did_analysis_visualization_corrected.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_file}")
    
    plt.close()


def create_state_comparison_table(df, output_dir):
    """Create comparison table of treated vs control states."""
    output_dir = Path(output_dir)
    
    # Calculate pre/post means by state
    state_summary = []
    
    for state in df['state'].unique():
        state_data = df[df['state'] == state]
        treated = state_data['treated'].iloc[0]
        undp_count = state_data['undp_count'].iloc[0]
        
        pre_mean = state_data[state_data['post'] == 0]['nightlight'].mean()
        post_mean = state_data[state_data['post'] == 1]['nightlight'].mean()
        change = post_mean - pre_mean
        pct_change = (change / pre_mean) * 100 if pre_mean > 0 else 0
        
        state_summary.append({
            'State': state,
            'Treatment': 'UNDP' if treated == 1 else 'Control',
            'UNDP_Interventions': int(undp_count),
            'Pre_Mean': pre_mean,
            'Post_Mean': post_mean,
            'Absolute_Change': change,
            'Percent_Change': pct_change
        })
    
    summary_df = pd.DataFrame(state_summary)
    summary_df = summary_df.sort_values('Percent_Change', ascending=False)
    
    summary_file = output_dir / 'state_comparison_table_corrected.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved state comparison: {summary_file}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="CORRECTED State-level difference-in-differences regression analysis"
    )
    
    parser.add_argument("--viirs-dir", type=str, default="./data/viirs_25")
    parser.add_argument("--boundaries-dir", type=str, default="./data/boundaries/processed")
    parser.add_argument("--undp-support", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/regression_analysis_corrected")
    
    args = parser.parse_args()
    
    if not DEPS_OK:
        print("\n✗ Missing dependencies")
        return 1
    
    print("="*80)
    print("  STATE-LEVEL DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("  Myanmar 2025 Earthquake - UNDP Intervention Effect")
    print("  CORRECTED VERSION (March 2025 excluded)")
    print("="*80)
    
    # Find VIIRS files - EXCLUDE MARCH 2025
    start = datetime(2025, 1, 1)
    end = datetime(2025, 9, 30)
    exclude_months = [(2025, 3)]  # Exclude March 2025
    
    viirs_files = find_viirs_files(args.viirs_dir, start, end, exclude_months=exclude_months)
    
    if not viirs_files:
        print("\n✗ No VIIRS files found")
        return 1
    
    print(f"\n✓ Found {len(viirs_files)} VIIRS files ({viirs_files[0][0].strftime('%Y-%m')} to {viirs_files[-1][0].strftime('%Y-%m')})")
    
    # Load boundaries (states/regions)
    boundaries_dir = Path(args.boundaries_dir)
    boundary_file = boundaries_dir / 'mmr_admin3.geojson'
    
    if not boundary_file.exists():
        print(f"\n✗ Boundary file not found: {boundary_file}")
        return 1
    
    states_gdf = gpd.read_file(boundary_file)
    print(f"✓ Loaded {len(states_gdf)} states/regions")
    
    # Load UNDP interventions
    undp_file = Path(args.undp_support)
    if not undp_file.exists():
        print(f"\n✗ UNDP file not found: {undp_file}")
        return 1
    
    undp_gdf = gpd.read_file(undp_file)
    print(f"✓ Loaded {len(undp_gdf)} UNDP intervention locations")
    
    # Ensure same CRS
    if states_gdf.crs != undp_gdf.crs:
        undp_gdf = undp_gdf.to_crs(states_gdf.crs)
    
    # Classify treatment status
    states_gdf = classify_treatment_status(states_gdf, undp_gdf)
    
    # Create panel data
    panel_df = create_panel_data(viirs_files, states_gdf)
    
    if panel_df is None or panel_df.empty:
        print("\n✗ Could not create panel dataset")
        return 1
    
    # Save panel data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    panel_file = output_dir / 'panel_data_corrected.csv'
    panel_df.to_csv(panel_file, index=False)
    print(f"\n✓ Saved panel dataset: {panel_file}")
    
    # Run diff-in-diff regression
    model, results_df = run_did_regression(panel_df, output_dir)
    
    # Create visualizations
    plot_did_visualization(panel_df, model, output_dir)
    
    # State comparison table
    state_table = create_state_comparison_table(panel_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. panel_data_corrected.csv - State-level panel dataset (March excluded)")
    print("  2. did_regression_results_corrected.csv - Regression coefficients with SEs")
    print("  3. regression_summary_corrected.txt - Full regression output")
    print("  4. did_analysis_visualization_corrected.png - Three-panel visualization")
    print("  5. state_comparison_table_corrected.csv - State-by-state comparison")
    
    print("\n" + "="*80)
    print("SPECIFICATION NOTES")
    print("="*80)
    print(f"✓ Pre-period:  {PRE_PERIOD_START.strftime('%B %Y')} - {PRE_PERIOD_END.strftime('%B %Y')}")
    print(f"✓ Post-period: {POST_PERIOD_START.strftime('%B %Y')} - {POST_PERIOD_END.strftime('%B %Y')}")
    print(f"✓ Earthquake:  {EARTHQUAKE_DATE.strftime('%B %d, %Y')}")
    print(f"✓ March 2025:  EXCLUDED (contaminated by mid-month earthquake)")
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
