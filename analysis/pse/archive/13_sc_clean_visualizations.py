#!/usr/bin/env python3
"""
analysis/pse/13_sc_clean_visualizations.py
==========================================
Clean, publication-ready visualizations:

1. SYNTHETIC CONTROL (for outcomes with donor data):
   - nightlights, ndvi, vh, vv
   - Shows: Actual vs Synthetic trajectory with UNDP dates

2. INTERRUPTED TIME SERIES (for outcomes WITHOUT donor data):
   - Prices, Google Trends, RVI
   - Shows: Gaza trajectory with trend lines and UNDP dates

Usage:
    python analysis/pse/13_sc_clean_visualizations.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "synthetic_control"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# Outcomes that CAN use SC (have donor data)
SC_OUTCOMES = ['nightlights', 'ndvi', 'vh', 'vv']

# Outcomes for ITS only (no donor data)
ITS_OUTCOMES = [
    'price_bread', 'price_flour', 'price_rice', 'price_milk', 
    'price_oil', 'price_sugar', 'price_chicken',
    'gt_food', 'gt_water', 'gt_hospital', 'gt_help', 'gt_electricity',
    'gt_shelter', 'gt_medicine', 'gt_bread', 'gt_flour',
    'rvi', 'conflict_events', 'conflict_fatalities'
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'vh': 'SAR Backscatter (VH dB)',
    'vv': 'SAR Backscatter (VV dB)',
    'rvi': 'Radar Vegetation Index',
    'price_bread': 'Bread Price (ILS/kg)',
    'price_flour': 'Flour Price (ILS/kg)',
    'price_rice': 'Rice Price (ILS/kg)',
    'price_milk': 'Milk Price (ILS/L)',
    'price_oil': 'Cooking Oil Price (ILS/L)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_chicken': 'Chicken Price (ILS/kg)',
    'price_wheat': 'Wheat Price (ILS/kg)',
    'gt_food': 'Google Trends: Food',
    'gt_water': 'Google Trends: Water',
    'gt_hospital': 'Google Trends: Hospital',
    'gt_help': 'Google Trends: Help',
    'gt_electricity': 'Google Trends: Electricity',
    'gt_shelter': 'Google Trends: Shelter',
    'gt_medicine': 'Google Trends: Medicine',
    'gt_bread': 'Google Trends: Bread',
    'gt_flour': 'Google Trends: Flour',
    'conflict_events': 'Conflict Events (ACLED)',
    'conflict_fatalities': 'Conflict Fatalities (ACLED)',
}

# Key interventions to show (simplified - not all)
KEY_INTERVENTIONS = [
    'Any UNDP',
    'Debris & Solid Waste', 
    'Infrastructure & Housing',
    'Food Security',
    'Health & Emergency'
]


def fit_synthetic_control(Y1_pre, Y0_pre, Y1_post, Y0_post):
    """Fit SC."""
    J = Y0_pre.shape[1]
    def objective(W):
        return np.mean((Y1_pre - Y0_pre @ W) ** 2)
    constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
    bounds = [(0, 1) for _ in range(J)]
    W0 = np.ones(J) / J
    result = minimize(objective, W0, method='SLSQP', bounds=bounds,
                     constraints=constraints, options={'maxiter': 1000})
    weights = result.x if result.success else W0
    Y_synth_pre = Y0_pre @ weights
    Y_synth_post = Y0_post @ weights
    
    ss_tot = np.sum((Y1_pre - Y1_pre.mean()) ** 2)
    ss_res = np.sum((Y1_pre - Y_synth_pre) ** 2)
    pre_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'Y_synth_pre': Y_synth_pre,
        'Y_synth_post': Y_synth_post,
        'att': np.mean(Y1_post - Y_synth_post),
        'pre_r2': pre_r2
    }


def prepare_sc_data(region_df, donor_df, outcome, pre_times, post_times):
    """Prepare SC data."""
    agg = region_df.groupby('period_start')[outcome].mean().reset_index()
    agg = agg.sort_values('period_start')
    
    Y1_pre = agg[agg['period_start'].isin(pre_times)][outcome].values
    Y1_post = agg[agg['period_start'].isin(post_times)][outcome].values
    
    if len(Y1_pre) == 0 or np.all(np.isnan(Y1_pre)):
        return None
    
    valid_donors, Y0_pre_list, Y0_post_list = [], [], []
    for unit in donor_df['admin_id'].unique():
        udf = donor_df[donor_df['admin_id'] == unit]
        upre = [udf[udf['period_start'] == t][outcome].values[0] 
                if len(udf[udf['period_start'] == t]) > 0 else np.nan for t in pre_times]
        upost = [udf[udf['period_start'] == t][outcome].values[0] 
                 if len(udf[udf['period_start'] == t]) > 0 else np.nan for t in post_times]
        upre, upost = np.array(upre), np.array(upost)
        
        if np.sum(np.isnan(upre)) > len(pre_times) * 0.3:
            continue
        upre[np.isnan(upre)] = np.nanmean(upre)
        upost[np.isnan(upost)] = np.nanmean(upost) if not np.all(np.isnan(upost)) else 0
        
        valid_donors.append(unit)
        Y0_pre_list.append(upre)
        Y0_post_list.append(upost)
    
    if len(valid_donors) < 3:
        return None
    
    return {
        'Y1_pre': Y1_pre, 'Y1_post': Y1_post,
        'Y0_pre': np.array(Y0_pre_list).T, 'Y0_post': np.array(Y0_post_list).T
    }


def create_clean_sc_plot(outcome, times_dt, Y_actual, Y_synthetic, 
                         intervention_dates, pre_r2, att, output_path):
    """
    Create CLEAN SC plot showing:
    - Top: Actual vs Synthetic trajectories
    - Bottom: Gap (Actual - Synthetic) with UNDP dates
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Align lengths
    min_len = min(len(times_dt), len(Y_actual), len(Y_synthetic))
    times_plot = times_dt[:min_len]
    Y_actual = Y_actual[:min_len]
    Y_synthetic = Y_synthetic[:min_len]
    gap = Y_actual - Y_synthetic
    
    # =========================================================================
    # TOP: Actual vs Synthetic Trajectories
    # =========================================================================
    ax1 = axes[0]
    
    # Shade pre/post conflict
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.1, color='green', label='Pre-conflict')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.1, color='red', label='Post-conflict')
    
    # Plot trajectories
    ax1.plot(times_plot, Y_actual, 'b-', linewidth=2.5, marker='o', markersize=5, 
             label='Gaza (Actual)', zorder=5)
    ax1.plot(times_plot, Y_synthetic, 'r--', linewidth=2.5, marker='s', markersize=4, 
             label='Synthetic Gaza (Counterfactual)', zorder=4)
    
    # Conflict line
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2, label='Oct 7, 2023')
    
    # Key UNDP interventions (only show a few, not all)
    colors = {'Any UNDP': 'purple', 'Debris & Solid Waste': 'brown', 
              'Infrastructure & Housing': 'orange', 'Food Security': 'green',
              'Health & Emergency': 'red'}
    
    for cat in KEY_INTERVENTIONS:
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if date >= times_plot[0] and date <= times_plot[-1]:
                ax1.axvline(x=date, color=colors.get(cat, 'gray'), linestyle=':', 
                           linewidth=2, alpha=0.8, label=f'{cat}')
    
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=12)
    ax1.set_title(f'Synthetic Control: {OUTCOME_LABELS.get(outcome, outcome)}\n'
                  f'Pre-treatment fit: R² = {pre_r2:.3f}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # =========================================================================
    # BOTTOM: Gap (Actual - Synthetic)
    # =========================================================================
    ax2 = axes[1]
    
    # Color bars by pre/post conflict
    n_pre = sum(1 for t in times_plot if t <= CONFLICT_DATE)
    colors_bar = ['gray'] * n_pre + ['coral'] * (len(gap) - n_pre)
    
    ax2.bar(times_plot, gap, width=20, color=colors_bar, alpha=0.7, edgecolor='darkgray')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.axhline(y=att, color='darkred', linestyle='--', linewidth=2, label=f'ATT = {att:.3f}')
    
    # Conflict and UNDP lines
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    for cat in KEY_INTERVENTIONS:
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if date >= times_plot[0] and date <= times_plot[-1]:
                ax2.axvline(x=date, color=colors.get(cat, 'gray'), linestyle=':', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Gap (Actual - Synthetic)', fontsize=12)
    ax2.set_title('Treatment Effect: Gap shows deviation from counterfactual', fontsize=11, style='italic')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Add interpretation
    if att < 0:
        interp = f"Gaza is {abs(att):.3f} units BELOW counterfactual on average"
    else:
        interp = f"Gaza is {att:.3f} units ABOVE counterfactual on average"
    
    ax2.text(0.02, 0.95, interp, transform=ax2.transAxes, fontsize=10, 
             fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_its_plot(outcome, times_dt, Y_values, intervention_dates, output_path):
    """
    Create ITS plot for outcomes WITHOUT donor data.
    Shows Gaza trajectory with trend lines and UNDP dates.
    """
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Align
    min_len = min(len(times_dt), len(Y_values))
    times_plot = times_dt[:min_len]
    Y_plot = Y_values[:min_len]
    
    # Remove NaN
    valid_mask = ~np.isnan(Y_plot)
    if valid_mask.sum() < 5:
        plt.close()
        return False
    
    times_valid = times_plot[valid_mask]
    Y_valid = Y_plot[valid_mask]
    
    # Shade pre/post conflict
    ax.axvspan(times_valid[0], CONFLICT_DATE, alpha=0.1, color='green', label='Pre-conflict')
    ax.axvspan(CONFLICT_DATE, times_valid[-1], alpha=0.1, color='red', label='Post-conflict')
    
    # Plot trajectory
    ax.plot(times_valid, Y_valid, 'b-', linewidth=2.5, marker='o', markersize=6, 
            label='Gaza', zorder=5)
    
    # Fit trend lines
    # Pre-conflict trend
    pre_mask = times_valid <= CONFLICT_DATE
    if pre_mask.sum() > 2:
        pre_times = times_valid[pre_mask]
        pre_Y = Y_valid[pre_mask]
        pre_x = np.arange(len(pre_times))
        
        try:
            pre_coef = np.polyfit(pre_x, pre_Y, 1)
            pre_trend = np.poly1d(pre_coef)(pre_x)
            ax.plot(pre_times, pre_trend, 'g--', linewidth=2, label=f'Pre-trend: {pre_coef[0]:+.4f}/month')
        except:
            pass
    
    # Post-conflict trend
    post_mask = times_valid > CONFLICT_DATE
    if post_mask.sum() > 2:
        post_times = times_valid[post_mask]
        post_Y = Y_valid[post_mask]
        post_x = np.arange(len(post_times))
        
        try:
            post_coef = np.polyfit(post_x, post_Y, 1)
            post_trend = np.poly1d(post_coef)(post_x)
            ax.plot(post_times, post_trend, 'r--', linewidth=2, label=f'Post-trend: {post_coef[0]:+.4f}/month')
        except:
            pass
    
    # Conflict line
    ax.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2.5, label='Oct 7, 2023')
    
    # UNDP interventions
    colors = {'Any UNDP': 'purple', 'Debris & Solid Waste': 'brown', 
              'Infrastructure & Housing': 'orange', 'Food Security': 'green',
              'Health & Emergency': 'red'}
    
    for cat in KEY_INTERVENTIONS:
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if date >= times_valid[0] and date <= times_valid[-1]:
                ax.axvline(x=date, color=colors.get(cat, 'gray'), linestyle=':', 
                          linewidth=2, alpha=0.8, label=f'{cat}')
    
    # Calculate pre/post means
    pre_mean = Y_valid[pre_mask].mean() if pre_mask.sum() > 0 else np.nan
    post_mean = Y_valid[post_mask].mean() if post_mask.sum() > 0 else np.nan
    change = post_mean - pre_mean if not np.isnan(pre_mean) and not np.isnan(post_mean) else np.nan
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=12)
    ax.set_title(f'Interrupted Time Series: {OUTCOME_LABELS.get(outcome, outcome)}\n'
                 f'Pre-mean: {pre_mean:.2f} | Post-mean: {post_mean:.2f} | Change: {change:+.2f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("\n" + "=" * 70)
    print("CLEAN VISUALIZATIONS: SC + ITS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    
    # Donor pool
    donor_df = None
    if DONOR_POOL.exists():
        donor_df = pd.read_csv(DONOR_POOL)
        donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
        print(f"  Donors: {len(donor_df)} obs")
    
    # UNDP dates
    intervention_dates = {}
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE]
        
        if 'ml_category' in undp.columns:
            for cat in undp['ml_category'].dropna().unique():
                date = undp[undp['ml_category'] == cat]['start_date'].min()
                if pd.notna(date):
                    intervention_dates[cat] = date
        intervention_dates['Any UNDP'] = undp['start_date'].min()
    except:
        intervention_dates = {'Any UNDP': pd.Timestamp('2024-01-01')}
    
    print(f"  UNDP interventions: {len(intervention_dates)}")
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    # =========================================================================
    # PART 1: SYNTHETIC CONTROL PLOTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: SYNTHETIC CONTROL (with donor data)")
    print("=" * 70)
    
    sc_results = []
    
    for outcome in SC_OUTCOMES:
        if outcome not in gaza_df.columns or outcome not in donor_df.columns:
            print(f"  {outcome}: ✗ Missing data")
            continue
        
        print(f"\n  {outcome}...")
        
        # Prepare data
        data = prepare_sc_data(gaza_df, donor_df, outcome, pre_times, post_times)
        if data is None:
            print(f"    ⚠ Could not prepare data")
            continue
        
        # Fit SC
        sc = fit_synthetic_control(data['Y1_pre'], data['Y0_pre'], 
                                   data['Y1_post'], data['Y0_post'])
        
        print(f"    R² = {sc['pre_r2']:.3f}, ATT = {sc['att']:.3f}")
        
        # Combine actual values
        Y_actual = np.concatenate([data['Y1_pre'], data['Y1_post']])
        Y_synthetic = np.concatenate([sc['Y_synth_pre'], sc['Y_synth_post']])
        
        # Create plot
        create_clean_sc_plot(
            outcome, times_dt, Y_actual, Y_synthetic,
            intervention_dates, sc['pre_r2'], sc['att'],
            FIGURES_DIR / f'sc_clean_{outcome}.png'
        )
        print(f"    ✓ sc_clean_{outcome}.png")
        
        sc_results.append({
            'Outcome': outcome,
            'Method': 'Synthetic Control',
            'Pre_R2': sc['pre_r2'],
            'ATT': sc['att']
        })
    
    # =========================================================================
    # PART 2: ITS PLOTS (no donor data)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: INTERRUPTED TIME SERIES (no donor data)")
    print("=" * 70)
    
    its_results = []
    
    for outcome in ITS_OUTCOMES:
        if outcome not in gaza_df.columns:
            continue
        
        if gaza_df[outcome].notna().sum() < 10:
            continue
        
        print(f"\n  {outcome}...")
        
        # Aggregate Gaza
        agg = gaza_df.groupby('period_start')[outcome].mean().reset_index()
        agg = agg.sort_values('period_start')
        
        # Get values aligned to times_dt
        Y_values = []
        for t in times_dt:
            val = agg[agg['period_start'] == t][outcome].values
            Y_values.append(val[0] if len(val) > 0 else np.nan)
        Y_values = np.array(Y_values)
        
        # Create plot
        success = create_its_plot(
            outcome, times_dt, Y_values,
            intervention_dates,
            FIGURES_DIR / f'its_{outcome}.png'
        )
        
        if success:
            print(f"    ✓ its_{outcome}.png")
            
            # Calculate stats
            pre_mask = times_dt <= CONFLICT_DATE
            post_mask = times_dt > CONFLICT_DATE
            pre_mean = np.nanmean(Y_values[pre_mask])
            post_mean = np.nanmean(Y_values[post_mask])
            
            its_results.append({
                'Outcome': outcome,
                'Method': 'ITS',
                'Pre_Mean': pre_mean,
                'Post_Mean': post_mean,
                'Change': post_mean - pre_mean
            })
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if sc_results:
        pd.DataFrame(sc_results).to_csv(TABLES_DIR / 'sc_clean_results.csv', index=False)
        print(f"  ✓ sc_clean_results.csv")
    
    if its_results:
        pd.DataFrame(its_results).to_csv(TABLES_DIR / 'its_results.csv', index=False)
        print(f"  ✓ its_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nSYNTHETIC CONTROL RESULTS:")
    print("-" * 50)
    for r in sc_results:
        print(f"  {r['Outcome']:<15} R²={r['Pre_R2']:.3f}  ATT={r['ATT']:+.3f}")
    
    print("\nITS RESULTS (top changes):")
    print("-" * 50)
    its_df = pd.DataFrame(its_results)
    if len(its_df) > 0:
        its_df['Abs_Change'] = its_df['Change'].abs()
        its_df = its_df.sort_values('Abs_Change', ascending=False)
        for _, r in its_df.head(10).iterrows():
            print(f"  {r['Outcome']:<25} Pre={r['Pre_Mean']:>8.2f}  Post={r['Post_Mean']:>8.2f}  Δ={r['Change']:+.2f}")
    
    print("\n" + "=" * 70)
    print("NOTE: Missing Outcomes")
    print("=" * 70)
    print("""
    The following outcomes CANNOT be analyzed because they're missing:
    
    ❌ NDBI (Built-up Index) - Not in panel data
       → Need to add to GEE download pipeline
    
    ❌ BSI (Bare Soil Index) - Not in panel data  
       → Need to add to GEE download pipeline
    
    ❌ SAR (if you mean different from VH/VV) - VH and VV ARE included
       → VH and VV are the SAR polarizations
    
    ❌ Wheat Price - 0% coverage in panel
       → WFP may not report wheat prices for Gaza
    
    To add NDBI/BSI:
    1. Update your GEE pipeline to include Landsat spectral indices
    2. Re-run: python -m pipeline.gee_raster_download --indices ndbi bsi
    3. Re-build panel: python -m pipeline.build_panel
    """)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"  SC plots: {FIGURES_DIR}/sc_clean_*.png")
    print(f"  ITS plots: {FIGURES_DIR}/its_*.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
