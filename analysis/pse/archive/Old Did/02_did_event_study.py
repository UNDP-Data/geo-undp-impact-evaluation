#!/usr/bin/env python3
"""
analysis/pse/02_did_event_study.py
==================================
Event Study Analysis with Donut Controls and Multiple Buffer Specifications.

Key Features:
1. Event study plots showing dynamic treatment effects
2. Multiple donut buffer specifications for sensitivity analysis
3. Comparison of treated vs donut control trajectories
4. Focus on outcomes with valid parallel trends (SAR_VH, BSI)

Usage:
    python analysis/pse/02_did_event_study.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import statsmodels.formula.api as smf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from scipy.spatial import cKDTree
    import rasterio
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
FIGURES_DIR = OUTPUT_DIR / "figures" / "event_studies"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")
DEG_PER_M = 1 / 100000

# =============================================================================
# MULTIPLE DONUT BUFFER SPECIFICATIONS FOR SENSITIVITY ANALYSIS
# =============================================================================
# Each specification: (inner_buffer_m, outer_buffer_m, name)
DONUT_SPECIFICATIONS = [
    (300, 1000, "Tight (300m-1km)"),      # Closest controls
    (500, 1500, "Baseline (500m-1.5km)"), # Your current specification
    (500, 2000, "Wide (500m-2km)"),       # Wider ring
    (750, 2000, "Far (750m-2km)"),        # Farther controls
]

CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Social Cohesion': ['Governance & Rule of Law', 'Social Inclusion'],
    'Other': ['Health & Emergency'],
    'Health': ['Health & Emergency']
}

OUTCOME_NAMES = {
    'viirs': 'Nighttime Lights (VIIRS)',
    'log_viirs': 'Log Nighttime Lights',
    'ndvi': 'Vegetation Index (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'sar_vh': 'SAR Backscatter VH',
    'sar_vv': 'SAR Backscatter VV',
}

# Outcomes with valid parallel trends (from your results)
CREDIBLE_OUTCOMES = ['sar_vh', 'bsi']

# =============================================================================
# DONUT CONTROL GENERATION
# =============================================================================

def generate_donut_controls(gdf_boundary, gdf_treated, inner_m, outer_m, n_controls=None):
    """Generate control points in the donut zone."""
    from shapely.geometry import Point
    
    if n_controls is None:
        n_controls = len(gdf_treated)
    
    inner_deg = inner_m * DEG_PER_M
    outer_deg = outer_m * DEG_PER_M
    
    treated_coords = np.array([[p.x, p.y] for p in gdf_treated.geometry])
    tree = cKDTree(treated_coords)
    
    control_points = []
    minx, miny, maxx, maxy = gdf_boundary.total_bounds
    
    attempts = 0
    max_attempts = 300000
    
    while len(control_points) < n_controls and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        
        pnt = Point(x, y)
        if not gdf_boundary.contains(pnt).any():
            continue
        
        dist, _ = tree.query([x, y])
        if dist < inner_deg or dist > outer_deg:
            continue
        
        control_points.append({
            'site_id': f'CTRL_{len(control_points):04d}',
            'intervention_type': 'Control',
            'latitude': y,
            'longitude': x,
            'group': 'Control',
            'project_start': pd.NaT
        })
    
    return pd.DataFrame(control_points)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_treated_sites():
    """Load UNDP intervention sites."""
    if not SITES_CSV.exists():
        return None
    
    sites = pd.read_csv(SITES_CSV)
    
    if PROJECTS_CSV.exists():
        projects = pd.read_csv(PROJECTS_CSV)
        projects['start_date'] = pd.to_datetime(projects['start_date'], errors='coerce')
        projects = projects[projects['start_date'] >= CONFLICT_START]
        
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
    """Extract raster values at all sites."""
    records = []
    
    for indicator, dates in rasters.items():
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
                            
                            # Calculate relative month
                            if pd.notna(site.get('project_start')):
                                diff = (date.year - site['project_start'].year) * 12 + \
                                       (date.month - site['project_start'].month)
                                if diff < -6:
                                    rel_month = -7
                                elif diff > 12:
                                    rel_month = 13
                                else:
                                    rel_month = diff
                            else:
                                rel_month = np.nan
                            
                            records.append({
                                'site_id': site['site_id'],
                                'group': site['group'],
                                'type': site['intervention_type'],
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                                'rel_month': rel_month,
                                'project_start': site.get('project_start', pd.NaT)
                            })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


# =============================================================================
# EVENT STUDY ANALYSIS
# =============================================================================

def run_event_study_regression(df, indicator):
    """Run event study regression and return coefficients."""
    subset = df[(df['indicator'] == indicator) & (df['group'] == 'Treated')].copy()
    subset = subset.dropna(subset=['rel_month'])
    
    if len(subset) < 50:
        return None
    
    formula = "value ~ C(rel_month, Treatment(reference=-1)) + C(type) + C(date)"
    
    try:
        model = smf.ols(formula, data=subset).fit(cov_type='HC1')
        
        coef_data = []
        for t in range(-7, 14):
            if t == -1:
                coef_data.append({'rel_month': t, 'coef': 0, 'se': 0, 'ci_lower': 0, 'ci_upper': 0})
                continue
            
            name = f"C(rel_month, Treatment(reference=-1))[T.{t}]"
            if name in model.params:
                ci = model.conf_int().loc[name]
                coef_data.append({
                    'rel_month': t,
                    'coef': model.params[name],
                    'se': model.bse[name],
                    'ci_lower': ci[0],
                    'ci_upper': ci[1]
                })
        
        return pd.DataFrame(coef_data)
    except:
        return None


def run_did_regression(df, indicator):
    """Run simple DiD regression comparing treated vs control."""
    subset = df[df['indicator'] == indicator].copy()
    
    if len(subset) < 100:
        return None
    
    subset['treated'] = (subset['group'] == 'Treated').astype(int)
    subset['post'] = 0
    for idx, row in subset.iterrows():
        if pd.notna(row.get('project_start')) and row['date'] >= row['project_start']:
            subset.at[idx, 'post'] = 1
    
    # For controls, use median treatment start
    median_start = subset[subset['treated']==1]['project_start'].dropna().median()
    subset.loc[(subset['treated']==0) & (subset['date'] >= median_start), 'post'] = 1
    
    subset['date_str'] = subset['date'].dt.strftime('%Y-%m')
    
    formula = "value ~ treated + post + treated:post + C(date_str)"
    
    try:
        model = smf.ols(formula, data=subset).fit(cov_type='HC1')
        
        if 'treated:post' in model.params:
            return {
                'coef': model.params['treated:post'],
                'se': model.bse['treated:post'],
                'pval': model.pvalues['treated:post'],
                'ci_lower': model.conf_int().loc['treated:post'][0],
                'ci_upper': model.conf_int().loc['treated:post'][1]
            }
    except:
        pass
    
    return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_event_study_with_controls(df, indicator, output_path, donut_spec_name):
    """
    Create event study plot showing:
    1. Dynamic treatment effects over time
    2. Pre-treatment coefficients (for parallel trends validation)
    3. Post-treatment coefficients (treatment effect)
    """
    coef_df = run_event_study_regression(df, indicator)
    
    if coef_df is None or len(coef_df) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    outcome_name = OUTCOME_NAMES.get(indicator, indicator.upper())
    is_credible = indicator in CREDIBLE_OUTCOMES
    
    # Shaded regions
    ax.axvspan(-7.5, -0.5, alpha=0.15, color='#3498db', label='Pre-intervention period')
    ax.axvspan(-0.5, 13.5, alpha=0.15, color='#27ae60', label='Post-intervention period')
    
    # Reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax.axvline(-0.5, color='#e74c3c', linestyle='--', linewidth=2.5, label='Intervention Start')
    
    # Confidence interval fill
    ax.fill_between(coef_df['rel_month'], coef_df['ci_lower'], coef_df['ci_upper'],
                    alpha=0.3, color='#2c3e50')
    
    # Point estimates with line
    ax.plot(coef_df['rel_month'], coef_df['coef'], 'o-', 
            color='#2c3e50', linewidth=2.5, markersize=10, 
            label='Point Estimate (95% CI)')
    
    # Highlight significant coefficients
    for _, row in coef_df.iterrows():
        if row['ci_lower'] > 0 or row['ci_upper'] < 0:
            ax.plot(row['rel_month'], row['coef'], 'o', 
                   color='#e74c3c', markersize=14, zorder=5, markeredgecolor='white', markeredgewidth=2)
    
    # Pre-trends assessment
    pre_coefs = coef_df[coef_df['rel_month'] < -1]
    if len(pre_coefs) > 0:
        pre_sig = sum((pre_coefs['ci_lower'] > 0) | (pre_coefs['ci_upper'] < 0))
        if pre_sig == 0:
            pretrend_status = "✓ Pre-trends flat (parallel trends likely hold)"
            pretrend_color = '#27ae60'
        else:
            pretrend_status = f"⚠ {pre_sig} significant pre-trend coefs"
            pretrend_color = '#e74c3c'
        
        ax.annotate(pretrend_status, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=11, ha='left', va='top', fontweight='bold', color=pretrend_color,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=pretrend_color))
    
    # Credibility badge
    if is_credible:
        ax.annotate("★ CREDIBLE OUTCOME", xy=(0.98, 0.98), xycoords='axes fraction',
                   fontsize=11, ha='right', va='top', fontweight='bold', color='#27ae60',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', alpha=0.9, edgecolor='#27ae60'))
    
    ax.set_xlabel('Months Relative to Intervention Start', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Effect on {outcome_name}', fontsize=13, fontweight='bold')
    ax.set_title(f'Event Study: {outcome_name}\n'
                 f'Donut Control: {donut_spec_name}',
                 fontsize=15, fontweight='bold')
    
    # X-axis
    xticks = list(range(-7, 14))
    xlabels = ['≤-6' if x == -7 else '≥12' if x == 13 else str(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_xlim(-7.5, 13.5)
    
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_treated_vs_control_trajectory(df, indicator, output_path, donut_spec_name):
    """
    Create plot showing raw trajectories of treated vs control groups.
    """
    treated = df[(df['indicator'] == indicator) & (df['group'] == 'Treated')]
    control = df[(df['indicator'] == indicator) & (df['group'] == 'Control')]
    
    if treated.empty or control.empty:
        return
    
    # Aggregate by date
    treat_agg = treated.groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    treat_agg['se'] = treat_agg['std'] / np.sqrt(treat_agg['count'])
    treat_agg['ci_lower'] = treat_agg['mean'] - 1.96 * treat_agg['se']
    treat_agg['ci_upper'] = treat_agg['mean'] + 1.96 * treat_agg['se']
    
    ctrl_agg = control.groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    ctrl_agg['se'] = ctrl_agg['std'] / np.sqrt(ctrl_agg['count'])
    ctrl_agg['ci_lower'] = ctrl_agg['mean'] - 1.96 * ctrl_agg['se']
    ctrl_agg['ci_upper'] = ctrl_agg['mean'] + 1.96 * ctrl_agg['se']
    
    # Median intervention start
    starts = treated['project_start'].dropna().unique()
    median_start = pd.Timestamp(np.median([s.value for s in starts])) if len(starts) > 0 else None
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    outcome_name = OUTCOME_NAMES.get(indicator, indicator.upper())
    conflict_marker = datetime(2023, 10, 7)
    
    # Shaded periods
    ax.axvspan(treat_agg['date'].min(), conflict_marker, alpha=0.05, color='gray')
    if median_start:
        ax.axvspan(conflict_marker, median_start, alpha=0.1, color='#e74c3c')
        ax.axvspan(median_start, treat_agg['date'].max(), alpha=0.1, color='#27ae60')
    
    # Control with CI
    ax.fill_between(ctrl_agg['date'], ctrl_agg['ci_lower'], ctrl_agg['ci_upper'],
                    alpha=0.2, color='#3498db')
    ax.plot(ctrl_agg['date'], ctrl_agg['mean'],
            color='#3498db', linewidth=2.5, linestyle='--', marker='s', markersize=5,
            label=f'Donut Control')
    
    # Treated with CI
    ax.fill_between(treat_agg['date'], treat_agg['ci_lower'], treat_agg['ci_upper'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treat_agg['date'], treat_agg['mean'],
            color='#e74c3c', linewidth=2.5, marker='o', markersize=6,
            label='Treated (UNDP Sites)')
    
    # Markers
    ax.axvline(conflict_marker, color='black', linestyle=':', linewidth=2, label='Conflict Start')
    if median_start:
        ax.axvline(median_start, color='#27ae60', linestyle='-', linewidth=2.5, 
                   alpha=0.8, label='Median Intervention Start')
    
    ax.set_title(f'Trajectory: {outcome_name}\n'
                 f'Donut Control: {donut_spec_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(outcome_name, fontsize=12, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    n_treat = len(treated['site_id'].unique())
    n_ctrl = len(control['site_id'].unique())
    ax.annotate(f'N: {n_treat} treated, {n_ctrl} control',
               xy=(0.02, 0.02), xycoords='axes fraction',
               fontsize=10, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_sensitivity_analysis(all_results, output_path):
    """
    Create sensitivity analysis plot showing DiD effects across donut specifications.
    """
    if not all_results:
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Focus on credible outcomes
    credible_results = results_df[results_df['indicator'].isin(CREDIBLE_OUTCOMES)]
    
    if credible_results.empty:
        credible_results = results_df
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: SAR_VH sensitivity
    ax1 = axes[0]
    sar_results = results_df[results_df['indicator'] == 'sar_vh']
    
    if not sar_results.empty:
        specs = sar_results['donut_spec'].tolist()
        coefs = sar_results['coef'].tolist()
        errors = [(sar_results['coef'] - sar_results['ci_lower']).tolist(),
                  (sar_results['ci_upper'] - sar_results['coef']).tolist()]
        
        colors = ['#27ae60' if p < 0.05 else '#f39c12' if p < 0.1 else '#95a5a6' 
                  for p in sar_results['pval']]
        
        bars = ax1.barh(range(len(specs)), coefs, xerr=errors, color=colors, 
                       edgecolor='black', capsize=5, height=0.6)
        ax1.set_yticks(range(len(specs)))
        ax1.set_yticklabels(specs, fontsize=11)
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('DiD Coefficient (Treatment Effect)', fontsize=12)
        ax1.set_title('SAR VH: Sensitivity to Donut Specification\n★ Parallel trends hold for all specs',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add significance stars
        for i, (coef, pval) in enumerate(zip(coefs, sar_results['pval'])):
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            ax1.annotate(stars, xy=(coef, i), xytext=(5, 0), textcoords='offset points',
                        fontsize=12, fontweight='bold', va='center')
    
    # Plot 2: BSI sensitivity
    ax2 = axes[1]
    bsi_results = results_df[results_df['indicator'] == 'bsi']
    
    if not bsi_results.empty:
        specs = bsi_results['donut_spec'].tolist()
        coefs = bsi_results['coef'].tolist()
        errors = [(bsi_results['coef'] - bsi_results['ci_lower']).tolist(),
                  (bsi_results['ci_upper'] - bsi_results['coef']).tolist()]
        
        colors = ['#27ae60' if p < 0.05 else '#f39c12' if p < 0.1 else '#95a5a6' 
                  for p in bsi_results['pval']]
        
        bars = ax2.barh(range(len(specs)), coefs, xerr=errors, color=colors,
                       edgecolor='black', capsize=5, height=0.6)
        ax2.set_yticks(range(len(specs)))
        ax2.set_yticklabels(specs, fontsize=11)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('DiD Coefficient (Treatment Effect)', fontsize=12)
        ax2.set_title('BSI (Bare Soil): Sensitivity to Donut Specification\n★ Parallel trends hold (4/5 intervention types)',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (coef, pval) in enumerate(zip(coefs, bsi_results['pval'])):
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            ax2.annotate(stars, xy=(coef, i), xytext=(5, 0), textcoords='offset points',
                        fontsize=12, fontweight='bold', va='center')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='p < 0.05'),
        Patch(facecolor='#f39c12', edgecolor='black', label='p < 0.10'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Not significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" EVENT STUDY ANALYSIS WITH MULTIPLE DONUT SPECIFICATIONS")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    treated_sites = load_treated_sites()
    
    if treated_sites is None:
        print("❌ Failed to load sites")
        return
    
    print(f"  Treated sites: {len(treated_sites)}")
    
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    treated_gdf = gpd.GeoDataFrame(
        treated_sites,
        geometry=gpd.points_from_xy(treated_sites.longitude, treated_sites.latitude),
        crs="EPSG:4326"
    )
    
    rasters = find_all_rasters()
    print(f"  Rasters: {list(rasters.keys())}")
    
    # Run analysis for each donut specification
    all_sensitivity_results = []
    
    for inner_m, outer_m, spec_name in DONUT_SPECIFICATIONS:
        print(f"\n{'='*70}")
        print(f" DONUT SPECIFICATION: {spec_name}")
        print("=" * 70)
        
        # Generate controls
        print(f"  Generating donut controls ({inner_m}m - {outer_m}m)...")
        controls = generate_donut_controls(boundaries, treated_gdf, inner_m, outer_m, 
                                           n_controls=len(treated_sites))
        
        if len(controls) < len(treated_sites) * 0.5:
            print(f"  ⚠ Insufficient controls ({len(controls)}), skipping")
            continue
        
        print(f"  Controls generated: {len(controls)}")
        
        # Combine and extract time series
        all_sites = pd.concat([treated_sites, controls], ignore_index=True)
        df = extract_time_series(all_sites, rasters)
        
        if df.empty:
            print("  ⚠ No data extracted")
            continue
        
        print(f"  Observations: {len(df):,}")
        
        # Generate plots for each indicator
        spec_clean = spec_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        for indicator in df['indicator'].unique():
            # Event study plot
            out_path = FIGURES_DIR / f"event_{indicator}_{spec_clean}.png"
            plot_event_study_with_controls(df, indicator, out_path, spec_name)
            
            # Trajectory plot
            out_path = FIGURES_DIR / f"trajectory_{indicator}_{spec_clean}.png"
            plot_treated_vs_control_trajectory(df, indicator, out_path, spec_name)
            
            print(f"    ✓ {indicator}")
        
        # Run DiD for sensitivity analysis
        for indicator in df['indicator'].unique():
            did_result = run_did_regression(df, indicator)
            if did_result:
                all_sensitivity_results.append({
                    'donut_spec': spec_name,
                    'inner_m': inner_m,
                    'outer_m': outer_m,
                    'indicator': indicator,
                    **did_result
                })
    
    # Sensitivity analysis plot
    if all_sensitivity_results:
        print("\n--- Creating Sensitivity Analysis Plot ---")
        plot_sensitivity_analysis(all_sensitivity_results, FIGURES_DIR / "sensitivity_donut_specs.png")
        
        # Save results table
        sens_df = pd.DataFrame(all_sensitivity_results)
        sens_df['stars'] = sens_df['pval'].apply(
            lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '')
        sens_df.to_csv(TABLES_DIR / "sensitivity_donut_specs.csv", index=False)
        print(f"  ✓ Saved: sensitivity_donut_specs.csv")
        
        # Summary
        print("\n" + "=" * 70)
        print(" SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 70)
        
        for indicator in CREDIBLE_OUTCOMES:
            ind_results = sens_df[sens_df['indicator'] == indicator]
            if not ind_results.empty:
                print(f"\n  {OUTCOME_NAMES.get(indicator, indicator)}:")
                for _, row in ind_results.iterrows():
                    print(f"    {row['donut_spec']:<25}: {row['coef']:>8.4f} {row['stars']}")
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
