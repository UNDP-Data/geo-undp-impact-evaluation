#!/usr/bin/env python3
"""
analysis/pse/02_did_pretrends_sensitivity.py
============================================
Pre-Trends Analysis Across Multiple Donut Buffer Specifications.

Creates parallel trends plots and statistical tests for each buffer spec:
- 200m-600m (very tight)
- 300m-800m (tight)  
- 400m-1km (moderate)
- 400m-1.2km (goldilocks)
- 500m-1.5km (baseline)
- 600m-1.5km (conservative)
- 750m-2km (wide)

Outputs:
1. Pre-trends plots for each specification × outcome
2. Summary heatmap showing which specs pass parallel trends
3. CSV with all test results

Usage:
    python analysis/pse/02_did_pretrends_sensitivity.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
FIGURES_DIR = OUTPUT_DIR / "figures" / "pretrends_sensitivity"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2023-10-07")
DEG_PER_M = 1 / 111000  # More accurate conversion

# =============================================================================
# DONUT SPECIFICATIONS TO TEST
# =============================================================================
DONUT_SPECS = [
    (200, 600, "200m-600m", "very_tight"),
    (300, 800, "300m-800m", "tight"),
    (400, 1000, "400m-1km", "moderate"),
    (400, 1200, "400m-1.2km", "goldilocks"),
    (500, 1500, "500m-1.5km", "baseline"),
    (600, 1500, "600m-1.5km", "conservative"),
    (750, 2000, "750m-2km", "wide"),
]

CATEGORY_MAP = {
    'WASH': ['Water & Sanitation (WASH)'],
    'Debris': ['Debris & Solid Waste', 'Environment'],
    'Shelter': ['Infrastructure & Housing', 'Social Inclusion'],
    'Employment': ['Livelihoods & Economy'],
    'Social Cohesion': ['Governance & Rule of Law', 'Social Inclusion'],
    'Other': ['Health & Emergency'],
}

OUTCOME_LABELS = {
    'sar_vh': 'SAR VH (Structures)',
    'bsi': 'BSI (Bare Soil)',
    'viirs': 'VIIRS (Nightlights)',
    'ndbi': 'NDBI (Built-up)',
    'ndvi': 'NDVI (Vegetation)',
}

# =============================================================================
# DATA FUNCTIONS
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
    max_attempts = 500000
    
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
        })
    
    return pd.DataFrame(control_points)


def load_treated_sites():
    """Load UNDP intervention sites."""
    sites = pd.read_csv(SITES_CSV)
    sites['group'] = 'Treated'
    return sites


def find_rasters():
    """Find all raster files."""
    rasters = {}
    for f in RASTER_DIR.glob("*.tif"):
        parts = f.stem.split('_')
        if 'sar' in f.name.lower():
            ind = 'sar_vh' if 'vh' in f.name.lower() else 'sar_vv' if 'vv' in f.name.lower() else None
        else:
            ind = parts[0].lower()
        
        if ind:
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
    """Extract raster values for all sites over time."""
    records = []
    
    for indicator, dates in rasters.items():
        for date, path in sorted(dates.items()):
            try:
                with rasterio.open(path) as src:
                    band = src.read(1)
                    for _, site in sites.iterrows():
                        try:
                            r, c = src.index(site['longitude'], site['latitude'])
                            val = band[r, c]
                            if np.isnan(val) or val == src.nodata:
                                continue
                            
                            records.append({
                                'site_id': site['site_id'],
                                'group': site['group'],
                                'type': site.get('intervention_type', 'Unknown'),
                                'date': date,
                                'indicator': indicator,
                                'value': float(val),
                            })
                        except:
                            continue
            except:
                continue
    
    return pd.DataFrame(records)


# =============================================================================
# PRE-TRENDS ANALYSIS
# =============================================================================

def test_parallel_trends(df, indicator):
    """
    Test for parallel pre-trends using interaction term.
    
    Model: value ~ time + treated + time × treated
    
    H0: coefficient on time × treated = 0 (parallel trends)
    """
    pre_data = df[(df['date'] < CONFLICT_START) & (df['indicator'] == indicator)].copy()
    
    if len(pre_data) < 50:
        return None, None, None
    
    pre_data['month_idx'] = pre_data['date'].dt.year * 12 + pre_data['date'].dt.month
    pre_data['is_treated'] = (pre_data['group'] == 'Treated').astype(int)
    
    try:
        model = smf.ols('value ~ month_idx * is_treated', data=pre_data).fit()
        
        if 'month_idx:is_treated' in model.pvalues:
            pval = model.pvalues['month_idx:is_treated']
            coef = model.params['month_idx:is_treated']
            passed = pval > 0.10
            return passed, pval, coef
    except:
        pass
    
    return None, None, None


def compute_group_means(df, indicator):
    """Compute mean values by group and date for plotting."""
    subset = df[df['indicator'] == indicator].copy()
    
    if subset.empty:
        return None, None
    
    treated = subset[subset['group'] == 'Treated'].groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    treated['se'] = treated['std'] / np.sqrt(treated['count'])
    treated['group'] = 'Treated'
    
    control = subset[subset['group'] == 'Control'].groupby('date')['value'].agg(['mean', 'std', 'count']).reset_index()
    control['se'] = control['std'] / np.sqrt(control['count'])
    control['group'] = 'Control'
    
    return treated, control


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pretrends_single(df, indicator, spec_label, passed, pval, output_path):
    """Create a single pre-trends plot for one specification × outcome."""
    treated, control = compute_group_means(df, indicator)
    
    if treated is None or control is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    outcome_label = OUTCOME_LABELS.get(indicator, indicator.upper())
    
    # Background shading
    ax.axvspan(treated['date'].min(), CONFLICT_START, alpha=0.1, color='#3498db', label='Pre-conflict')
    ax.axvspan(CONFLICT_START, treated['date'].max(), alpha=0.1, color='#e74c3c', label='Post-conflict')
    
    # Conflict line
    ax.axvline(CONFLICT_START, color='#c0392b', linewidth=2.5, linestyle='--', label='Conflict Start')
    
    # Treated trajectory
    ax.fill_between(treated['date'], 
                    treated['mean'] - 1.96*treated['se'],
                    treated['mean'] + 1.96*treated['se'],
                    alpha=0.2, color='#e74c3c')
    ax.plot(treated['date'], treated['mean'], 'o-', color='#e74c3c', 
            linewidth=2, markersize=5, label='Treated (UNDP sites)')
    
    # Control trajectory
    ax.fill_between(control['date'],
                    control['mean'] - 1.96*control['se'],
                    control['mean'] + 1.96*control['se'],
                    alpha=0.2, color='#3498db')
    ax.plot(control['date'], control['mean'], 's--', color='#3498db',
            linewidth=2, markersize=5, label=f'Control ({spec_label})')
    
    # Pre-trends test result
    if passed is not None:
        if passed:
            status_text = f"✓ Parallel trends HOLD (p={pval:.3f})"
            status_color = '#27ae60'
        else:
            status_text = f"⚠ Parallel trends VIOLATED (p={pval:.3f})"
            status_color = '#e74c3c'
        
        ax.annotate(status_text, xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=12, ha='left', va='top', fontweight='bold', color=status_color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=status_color, alpha=0.95))
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel(outcome_label, fontsize=11, fontweight='bold')
    ax.set_title(f'Pre-Trends Test: {outcome_label}\nControl Group: {spec_label}',
                fontsize=13, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pretrends_comparison(all_results, indicator, output_path):
    """
    Create a multi-panel comparison showing pre-trends across all specifications.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    outcome_label = OUTCOME_LABELS.get(indicator, indicator.upper())
    
    for i, (inner, outer, spec_label, spec_key) in enumerate(DONUT_SPECS):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        key = (spec_key, indicator)
        if key not in all_results:
            ax.set_visible(False)
            continue
        
        result = all_results[key]
        treated = result['treated']
        control = result['control']
        passed = result['passed']
        pval = result['pval']
        
        if treated is None or control is None:
            ax.set_visible(False)
            continue
        
        # Background
        ax.axvspan(treated['date'].min(), CONFLICT_START, alpha=0.1, color='#3498db')
        ax.axvspan(CONFLICT_START, treated['date'].max(), alpha=0.1, color='#e74c3c')
        ax.axvline(CONFLICT_START, color='#c0392b', linewidth=2, linestyle='--')
        
        # Lines
        ax.plot(treated['date'], treated['mean'], 'o-', color='#e74c3c', 
                linewidth=1.5, markersize=3, label='Treated')
        ax.plot(control['date'], control['mean'], 's--', color='#3498db',
                linewidth=1.5, markersize=3, label='Control')
        
        # Status badge
        if passed is not None:
            badge_color = '#27ae60' if passed else '#e74c3c'
            badge_text = f"p={pval:.2f}" if pval else "N/A"
            badge_symbol = "✓" if passed else "⚠"
            ax.annotate(f'{badge_symbol} {badge_text}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=10, ha='left', va='top', fontweight='bold', color=badge_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=badge_color, alpha=0.9))
        
        ax.set_title(spec_label, fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        if i == 0:
            ax.legend(fontsize=8, loc='lower right')
        
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplot
    if len(DONUT_SPECS) < len(axes):
        axes[-1].set_visible(False)
    
    fig.suptitle(f'Pre-Trends Comparison: {outcome_label}\nAcross Donut Control Specifications',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_pretrends_heatmap(all_test_results, output_path):
    """
    Create a heatmap showing which specification × outcome combinations pass parallel trends.
    """
    # Build matrix
    specs = [s[3] for s in DONUT_SPECS]  # spec_key
    outcomes = list(OUTCOME_LABELS.keys())
    
    matrix = np.full((len(outcomes), len(specs)), np.nan)
    pval_matrix = np.empty((len(outcomes), len(specs)), dtype=object)
    
    for i, outcome in enumerate(outcomes):
        for j, spec_key in enumerate(specs):
            key = (spec_key, outcome)
            if key in all_test_results:
                result = all_test_results[key]
                if result['pval'] is not None:
                    matrix[i, j] = 1 if result['passed'] else 0
                    pval_matrix[i, j] = f"{result['pval']:.2f}"
                else:
                    pval_matrix[i, j] = "N/A"
            else:
                pval_matrix[i, j] = "N/A"
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Custom colormap: red (fail) -> yellow (marginal) -> green (pass)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(matrix, ax=ax, annot=pval_matrix, fmt='', cmap=cmap,
                xticklabels=[s[2] for s in DONUT_SPECS],  # spec_label
                yticklabels=[OUTCOME_LABELS[o] for o in outcomes],
                cbar_kws={'label': 'Parallel Trends (1=Pass, 0=Fail)'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, center=0.5)
    
    ax.set_title('Pre-Trends Test Results: Specification × Outcome\n'
                 '(Green = Parallel trends hold [p>0.10], Red = Violated [p<0.10])',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Donut Control Specification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_sar_bsi_focus(all_results, output_path):
    """
    Create a focused comparison for SAR_VH and BSI (the credible outcomes).
    """
    fig, axes = plt.subplots(2, len(DONUT_SPECS), figsize=(24, 10))
    
    outcomes = ['sar_vh', 'bsi']
    
    for row, outcome in enumerate(outcomes):
        outcome_label = OUTCOME_LABELS.get(outcome, outcome)
        
        for col, (inner, outer, spec_label, spec_key) in enumerate(DONUT_SPECS):
            ax = axes[row, col]
            
            key = (spec_key, outcome)
            if key not in all_results:
                ax.set_visible(False)
                continue
            
            result = all_results[key]
            treated = result['treated']
            control = result['control']
            passed = result['passed']
            pval = result['pval']
            
            if treated is None or control is None:
                ax.set_visible(False)
                continue
            
            # Pre-conflict data only for cleaner view
            pre_treat = treated[treated['date'] < CONFLICT_START]
            pre_ctrl = control[control['date'] < CONFLICT_START]
            
            if pre_treat.empty or pre_ctrl.empty:
                ax.set_visible(False)
                continue
            
            # Plot
            ax.fill_between(pre_treat['date'], 
                           pre_treat['mean'] - 1.96*pre_treat['se'],
                           pre_treat['mean'] + 1.96*pre_treat['se'],
                           alpha=0.25, color='#e74c3c')
            ax.plot(pre_treat['date'], pre_treat['mean'], 'o-', color='#e74c3c',
                   linewidth=2, markersize=4, label='Treated')
            
            ax.fill_between(pre_ctrl['date'],
                           pre_ctrl['mean'] - 1.96*pre_ctrl['se'],
                           pre_ctrl['mean'] + 1.96*pre_ctrl['se'],
                           alpha=0.25, color='#3498db')
            ax.plot(pre_ctrl['date'], pre_ctrl['mean'], 's--', color='#3498db',
                   linewidth=2, markersize=4, label='Control')
            
            # Status
            if passed is not None:
                color = '#27ae60' if passed else '#e74c3c'
                symbol = '✓' if passed else '⚠'
                ax.annotate(f'{symbol} p={pval:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, ha='left', va='top', fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor=color, alpha=0.9, linewidth=2))
            
            # Title and formatting
            if row == 0:
                ax.set_title(spec_label, fontsize=11, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(outcome_label, fontsize=11, fontweight='bold')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='lower right')
    
    fig.suptitle('Pre-Trends Comparison: SAR_VH and BSI (Credible Outcomes)\n'
                 'Pre-Conflict Period Only | Shaded areas = 95% CI',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" PRE-TRENDS ANALYSIS ACROSS DONUT SPECIFICATIONS")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("\n⚠ Missing dependencies. Exiting.")
        return
    
    # Load data
    print("\n--- Loading Data ---")
    treated_sites = load_treated_sites()
    print(f"  Treated sites: {len(treated_sites)}")
    
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    treated_gdf = gpd.GeoDataFrame(
        treated_sites,
        geometry=gpd.points_from_xy(treated_sites.longitude, treated_sites.latitude),
        crs="EPSG:4326"
    )
    
    rasters = find_rasters()
    print(f"  Rasters: {list(rasters.keys())}")
    
    # Store all results
    all_results = {}
    all_test_results = {}
    summary_rows = []
    
    # Analyze each specification
    for inner, outer, spec_label, spec_key in DONUT_SPECS:
        print(f"\n{'='*70}")
        print(f" SPECIFICATION: {spec_label}")
        print("=" * 70)
        
        # Generate controls
        controls = generate_donut_controls(boundaries, treated_gdf, inner, outer,
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
        
        # Test pre-trends for each indicator
        print(f"\n  Pre-trends tests:")
        
        for indicator in df['indicator'].unique():
            passed, pval, coef = test_parallel_trends(df, indicator)
            treated_agg, control_agg = compute_group_means(df, indicator)
            
            # Store results
            key = (spec_key, indicator)
            all_results[key] = {
                'treated': treated_agg,
                'control': control_agg,
                'passed': passed,
                'pval': pval,
                'coef': coef,
            }
            all_test_results[key] = {'passed': passed, 'pval': pval, 'coef': coef}
            
            # Print result
            if passed is not None:
                status = "✓ PASS" if passed else "⚠ FAIL"
                print(f"    {indicator:<10}: {status} (p={pval:.3f})")
            else:
                print(f"    {indicator:<10}: ? (insufficient data)")
            
            # Save to summary
            summary_rows.append({
                'specification': spec_label,
                'spec_key': spec_key,
                'inner_m': inner,
                'outer_m': outer,
                'indicator': indicator,
                'passed': passed,
                'pval': pval,
                'coef': coef,
            })
            
            # Create individual plot
            output_path = FIGURES_DIR / f"pretrends_{indicator}_{spec_key}.png"
            plot_pretrends_single(df, indicator, spec_label, passed, pval, output_path)
    
    # Create summary visualizations
    print(f"\n{'='*70}")
    print(" CREATING SUMMARY VISUALIZATIONS")
    print("=" * 70)
    
    # Heatmap
    print("\n  Creating heatmap...")
    plot_pretrends_heatmap(all_test_results, FIGURES_DIR / "pretrends_heatmap.png")
    print("  ✓ Saved: pretrends_heatmap.png")
    
    # Comparison plots for each outcome
    for indicator in OUTCOME_LABELS.keys():
        print(f"  Creating comparison for {indicator}...")
        plot_pretrends_comparison(all_results, indicator, 
                                 FIGURES_DIR / f"pretrends_comparison_{indicator}.png")
        print(f"  ✓ Saved: pretrends_comparison_{indicator}.png")
    
    # Focused SAR/BSI plot
    print("  Creating SAR/BSI focus plot...")
    plot_sar_bsi_focus(all_results, FIGURES_DIR / "pretrends_sar_bsi_focus.png")
    print("  ✓ Saved: pretrends_sar_bsi_focus.png")
    
    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "pretrends_sensitivity_results.csv", index=False)
    print(f"\n✓ Saved: pretrends_sensitivity_results.csv")
    
    # Final summary
    print(f"\n{'='*70}")
    print(" SUMMARY: PARALLEL TRENDS BY SPECIFICATION")
    print("=" * 70)
    
    print("\n  Pass rates by specification:")
    for spec_label in [s[2] for s in DONUT_SPECS]:
        spec_results = summary_df[summary_df['specification'] == spec_label]
        if not spec_results.empty:
            passed = spec_results['passed'].sum()
            total = spec_results['passed'].notna().sum()
            pct = passed / total * 100 if total > 0 else 0
            print(f"    {spec_label:<20}: {passed}/{total} pass ({pct:.0f}%)")
    
    print("\n  Pass rates by outcome:")
    for indicator, label in OUTCOME_LABELS.items():
        ind_results = summary_df[summary_df['indicator'] == indicator]
        if not ind_results.empty:
            passed = ind_results['passed'].sum()
            total = ind_results['passed'].notna().sum()
            pct = passed / total * 100 if total > 0 else 0
            status = "★ CREDIBLE" if pct >= 70 else ""
            print(f"    {label:<25}: {passed}/{total} pass ({pct:.0f}%) {status}")
    
    print(f"\n{'='*70}")
    print(" ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
  Output files:
    - {FIGURES_DIR}/pretrends_heatmap.png
    - {FIGURES_DIR}/pretrends_comparison_*.png (one per outcome)
    - {FIGURES_DIR}/pretrends_sar_bsi_focus.png
    - {FIGURES_DIR}/pretrends_*_*.png (individual plots)
    - {TABLES_DIR}/pretrends_sensitivity_results.csv
    """)


if __name__ == "__main__":
    main()
