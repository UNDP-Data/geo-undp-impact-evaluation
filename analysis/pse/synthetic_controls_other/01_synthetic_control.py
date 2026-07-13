#!/usr/bin/env python3
"""
analysis/pse/00_synthetic_control.py
====================================
MAIN Synthetic Control Analysis with West Bank Placebo Tests

Creates 4-panel SC plots for ALL available outcomes:
- Panel 1: Gaza Actual vs Synthetic
- Panel 2: West Bank Actual vs Synthetic (PLACEBO)
- Panel 3: Gaza Gap (Actual - Synthetic)
- Panel 4: West Bank Gap (should be ~0 if valid)

For outcomes WITHOUT donor data: Creates ITS plots instead.

Usage:
    python analysis/pse/00_synthetic_control.py
    
    Or via Makefile:
    make sc COUNTRY=gaza
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

# Try multiple possible donor pool locations (in order of preference)
DONOR_POOL_PATHS = [
    # Full donor pool with prices/GT/conflict
    DATA_DIR / "synthetic_control" / "donor_pool_full.csv",
    # Satellite-only donor pool (fallback)
    DATA_DIR / "synthetic_control" / "donor_pool_combined.csv",
    DATA_DIR / "pse" / "synthetic_control" / "donor_pool_combined.csv",
    OUTPUT_DIR / "donor_pool_combined.csv",
]

UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# ALL outcomes to try for SC (script will check which have data in both panel AND donors)
ALL_SC_OUTCOMES = [
    # Satellite indices
    'nightlights', 'ndvi', 'vh', 'vv', 'bsi', 'ndbi',
    # Prices (WFP)
    'price_bread', 'price_flour', 'price_rice', 'price_milk', 
    'price_oil', 'price_sugar', 'price_chicken',
    # Google Trends
    'gt_food', 'gt_water', 'gt_hospital', 'gt_help', 'gt_electricity',
    'gt_shelter', 'gt_medicine', 'gt_bread', 'gt_flour',
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'vh': 'SAR Backscatter (VH dB)',
    'vv': 'SAR Backscatter (VV dB)',
    'bsi': 'Bare Soil Index (BSI)',
    'ndbi': 'Built-up Index (NDBI)',
    'rvi': 'Radar Vegetation Index',
    'price_bread': 'Bread Price (ILS/kg)',
    'price_flour': 'Flour Price (ILS/kg)',
    'price_rice': 'Rice Price (ILS/kg)',
    'price_milk': 'Milk Price (ILS/L)',
    'price_oil': 'Cooking Oil Price (ILS/L)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_chicken': 'Chicken Price (ILS/kg)',
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

# Key UNDP interventions
KEY_INTERVENTIONS = [
    'Any UNDP',
    'Debris & Solid Waste', 
    'Infrastructure & Housing',
    'Health & Emergency'
]


def fit_synthetic_control(Y1_pre, Y0_pre, Y1_post, Y0_post):
    """Fit synthetic control weights."""
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
        'weights': weights,
        'Y_synth_pre': Y_synth_pre,
        'Y_synth_post': Y_synth_post,
        'att': np.mean(Y1_post - Y_synth_post),
        'pre_r2': max(0, pre_r2)
    }


def prepare_sc_data(region_df, donor_df, outcome, pre_times, post_times):
    """Prepare data for SC."""
    agg = region_df.groupby('period_start')[outcome].mean().reset_index()
    agg = agg.sort_values('period_start')
    
    Y1_pre = np.array([agg[agg['period_start'] == t][outcome].values[0] 
                       if len(agg[agg['period_start'] == t]) > 0 else np.nan 
                       for t in pre_times])
    Y1_post = np.array([agg[agg['period_start'] == t][outcome].values[0] 
                        if len(agg[agg['period_start'] == t]) > 0 else np.nan 
                        for t in post_times])
    
    if np.all(np.isnan(Y1_pre)) or len(Y1_pre) == 0:
        return None
    
    # Fill NaN with mean
    Y1_pre = np.where(np.isnan(Y1_pre), np.nanmean(Y1_pre), Y1_pre)
    Y1_post = np.where(np.isnan(Y1_post), np.nanmean(Y1_post) if not np.all(np.isnan(Y1_post)) else 0, Y1_post)
    
    # Donor pool
    valid_donors, Y0_pre_list, Y0_post_list = [], [], []
    
    for unit in donor_df['admin_id'].unique():
        udf = donor_df[donor_df['admin_id'] == unit]
        
        upre = np.array([udf[udf['period_start'] == t][outcome].values[0] 
                        if len(udf[udf['period_start'] == t]) > 0 else np.nan 
                        for t in pre_times])
        upost = np.array([udf[udf['period_start'] == t][outcome].values[0] 
                         if len(udf[udf['period_start'] == t]) > 0 else np.nan 
                         for t in post_times])
        
        if np.sum(np.isnan(upre)) > len(pre_times) * 0.3:
            continue
        
        upre = np.where(np.isnan(upre), np.nanmean(upre), upre)
        upost = np.where(np.isnan(upost), np.nanmean(upost) if not np.all(np.isnan(upost)) else 0, upost)
        
        valid_donors.append(unit)
        Y0_pre_list.append(upre)
        Y0_post_list.append(upost)
    
    if len(valid_donors) < 3:
        return None
    
    return {
        'Y1_pre': Y1_pre,
        'Y1_post': Y1_post,
        'Y0_pre': np.array(Y0_pre_list).T,
        'Y0_post': np.array(Y0_post_list).T
    }


def create_4panel_sc_plot(outcome, times_dt, 
                          gaza_actual, gaza_synth, gaza_r2, gaza_att,
                          wb_actual, wb_synth, wb_r2, wb_att,
                          intervention_dates, output_path):
    """
    Create 4-panel SC plot with West Bank placebo.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Align lengths
    min_len = min(len(times_dt), len(gaza_actual), len(gaza_synth))
    times_plot = times_dt[:min_len]
    gaza_actual = gaza_actual[:min_len]
    gaza_synth = gaza_synth[:min_len]
    gaza_gap = gaza_actual - gaza_synth
    
    wb_actual = wb_actual[:min_len]
    wb_synth = wb_synth[:min_len]
    wb_gap = wb_actual - wb_synth
    
    label = OUTCOME_LABELS.get(outcome, outcome)
    
    # Color scheme
    colors = {
        'actual': '#1f77b4',
        'synth': '#d62728',
        'wb_actual': '#2ca02c',
        'wb_synth': '#ff7f0e'
    }
    
    # =========================================================================
    # Panel 1: Gaza Actual vs Synthetic
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='red')
    
    ax1.plot(times_plot, gaza_actual, 'o-', color=colors['actual'], 
             linewidth=2.5, markersize=5, label='Gaza (Actual)')
    ax1.plot(times_plot, gaza_synth, 's--', color=colors['synth'], 
             linewidth=2.5, markersize=4, label='Synthetic Gaza')
    
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    
    # UNDP dates
    undp_colors = ['purple', 'brown', 'orange', 'red']
    for i, cat in enumerate(KEY_INTERVENTIONS):
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if times_plot[0] <= date <= times_plot[-1]:
                ax1.axvline(x=date, color=undp_colors[i % len(undp_colors)], 
                           linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax1.set_ylabel(label, fontsize=11)
    ax1.set_title(f'Gaza: Actual vs Synthetic\nPre-R² = {gaza_r2:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 2: West Bank Actual vs Synthetic (PLACEBO)
    # =========================================================================
    ax2 = axes[0, 1]
    ax2.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax2.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='lightblue')
    
    ax2.plot(times_plot, wb_actual, 'o-', color=colors['wb_actual'], 
             linewidth=2.5, markersize=5, label='West Bank (Actual)')
    ax2.plot(times_plot, wb_synth, 's--', color=colors['wb_synth'], 
             linewidth=2.5, markersize=4, label='Synthetic WB')
    
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    
    ax2.set_ylabel(label, fontsize=11)
    ax2.set_title(f'West Bank PLACEBO: Actual vs Synthetic\nPre-R² = {wb_r2:.3f}', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 3: Gaza Gap
    # =========================================================================
    ax3 = axes[1, 0]
    
    n_pre = sum(1 for t in times_plot if t <= CONFLICT_DATE)
    bar_colors = ['gray'] * n_pre + ['coral'] * (len(gaza_gap) - n_pre)
    
    ax3.bar(times_plot, gaza_gap, width=20, color=bar_colors, alpha=0.7, edgecolor='darkgray')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.axhline(y=gaza_att, color='darkred', linestyle='--', linewidth=2, 
                label=f'ATT = {gaza_att:.3f}')
    ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    
    # UNDP dates
    for i, cat in enumerate(KEY_INTERVENTIONS):
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if times_plot[0] <= date <= times_plot[-1]:
                ax3.axvline(x=date, color=undp_colors[i % len(undp_colors)], 
                           linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax3.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Gaza Treatment Effect', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Panel 4: West Bank Gap (PLACEBO)
    # =========================================================================
    ax4 = axes[1, 1]
    
    bar_colors_wb = ['gray'] * n_pre + ['lightblue'] * (len(wb_gap) - n_pre)
    
    ax4.bar(times_plot, wb_gap, width=20, color=bar_colors_wb, alpha=0.7, edgecolor='darkgray')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax4.axhline(y=wb_att, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'ATT = {wb_att:.3f}')
    ax4.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    
    ax4.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title('West Bank PLACEBO (should be ≈ 0)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Overall title with validity assessment
    # =========================================================================
    ratio = abs(wb_att / gaza_att) if abs(gaza_att) > 0.001 else 999
    validity = "✓ VALID" if ratio < 0.30 else "⚠ CAUTION"
    
    fig.suptitle(f'Synthetic Control: {label}\n'
                 f'Gaza ATT = {gaza_att:.3f} | WB Placebo ATT = {wb_att:.3f} | '
                 f'Ratio = {ratio:.2f} → {validity}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_clean_sc_plot(outcome, times_dt, gaza_actual, gaza_synth, gaza_r2, gaza_att,
                         intervention_dates, output_path):
    """
    Create CLEAN 2-panel SC plot (without WB placebo).
    - Top panel: Gaza Actual vs Synthetic counterfactual
    - Bottom panel: Gap (Actual - Synthetic)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    min_len = min(len(times_dt), len(gaza_actual), len(gaza_synth))
    times_plot = times_dt[:min_len]
    gaza_actual = gaza_actual[:min_len]
    gaza_synth = gaza_synth[:min_len]
    gaza_gap = gaza_actual - gaza_synth
    
    label = OUTCOME_LABELS.get(outcome, outcome)
    
    # =========================================================================
    # Top Panel: Gaza Actual vs Synthetic
    # =========================================================================
    ax1 = axes[0]
    
    # Background shading
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green', label='Pre-conflict')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='red', label='Post-conflict')
    
    # Lines
    ax1.plot(times_plot, gaza_actual, 'o-', color='#1f77b4', 
             linewidth=2.5, markersize=6, label='Gaza (Actual)')
    ax1.plot(times_plot, gaza_synth, 's--', color='#d62728', 
             linewidth=2.5, markersize=5, label='Synthetic Gaza (Counterfactual)')
    
    # Oct 7 line
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2.5, label='Oct 7, 2023')
    
    # UNDP intervention dates
    undp_colors = {'Any UNDP': 'purple', 'Debris & Solid Waste': 'brown', 
                   'Infrastructure & Housing': 'orange', 'Health & Emergency': 'red'}
    for cat in KEY_INTERVENTIONS:
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if times_plot[0] <= date <= times_plot[-1]:
                ax1.axvline(x=date, color=undp_colors.get(cat, 'gray'), 
                           linestyle=':', linewidth=2, alpha=0.8, label=cat)
    
    ax1.set_ylabel(label, fontsize=12)
    ax1.set_title(f'Synthetic Control: {label}\nPre-treatment fit: R² = {gaza_r2:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # =========================================================================
    # Bottom Panel: Gap (Treatment Effect)
    # =========================================================================
    ax2 = axes[1]
    
    n_pre = sum(1 for t in times_plot if t <= CONFLICT_DATE)
    bar_colors = ['gray'] * n_pre + ['coral'] * (len(gaza_gap) - n_pre)
    
    ax2.bar(times_plot, gaza_gap, width=20, color=bar_colors, alpha=0.7, edgecolor='darkgray')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.axhline(y=gaza_att, color='darkred', linestyle='--', linewidth=2.5, 
                label=f'ATT = {gaza_att:.3f}')
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='-', linewidth=2)
    
    # UNDP dates on gap panel
    for cat in KEY_INTERVENTIONS:
        if cat in intervention_dates:
            date = intervention_dates[cat]
            if times_plot[0] <= date <= times_plot[-1]:
                ax2.axvline(x=date, color=undp_colors.get(cat, 'gray'), 
                           linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Annotation box
    direction = "BELOW" if gaza_att < 0 else "ABOVE"
    ax2.annotate(f'Gaza is {abs(gaza_att):.3f} units {direction} counterfactual on average',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Treatment Effect: Gap shows deviation from counterfactual', fontsize=11)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL WITH WEST BANK PLACEBO")
    print("=" * 70)
    
    # Load panel data
    print("\nLoading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    wb_df = panel[~panel['unit_id'].str.startswith('PS02')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    print(f"  West Bank: {len(wb_df)} obs")
    
    # Load donor pool
    donor_df = None
    donor_path_used = None
    for path in DONOR_POOL_PATHS:
        if path.exists():
            donor_df = pd.read_csv(path)
            donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
            donor_path_used = path
            print(f"  Donors: {len(donor_df)} obs from {path.name}")
            break
    
    if donor_df is None:
        print("  ⚠ No donor pool found - cannot run SC analysis")
        return
    
    # Dynamically determine which outcomes have data in BOTH panel AND donors
    sc_outcomes = []
    donor_cols = set(donor_df.columns)
    
    for outcome in ALL_SC_OUTCOMES:
        if outcome in gaza_df.columns and outcome in donor_cols:
            # Check for sufficient data
            gaza_has = gaza_df[outcome].notna().sum() >= 5
            donor_has = donor_df[outcome].notna().sum() >= 10
            if gaza_has and donor_has:
                sc_outcomes.append(outcome)
    
    print(f"\n  SC outcomes available: {len(sc_outcomes)}")
    for o in sc_outcomes:
        print(f"    ✓ {o}")
    
    # Load UNDP dates
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
    except Exception as e:
        print(f"  ⚠ Could not load UNDP dates: {e}")
        intervention_dates = {'Any UNDP': pd.Timestamp('2024-01-01')}
    
    print(f"  UNDP interventions: {len(intervention_dates)}")
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    # =========================================================================
    # =========================================================================
    # SYNTHETIC CONTROL (4-panel with WB placebo)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL WITH WB PLACEBO")
    print("=" * 70)
    
    sc_results = []
    
    for outcome in sc_outcomes:
        if outcome not in gaza_df.columns:
            print(f"\n  {outcome}: ✗ Not in panel")
            continue
        if outcome not in donor_df.columns:
            print(f"\n  {outcome}: ✗ Not in donors")
            continue
        
        print(f"\n  {outcome}...")
        
        # Gaza SC
        gaza_data = prepare_sc_data(gaza_df, donor_df, outcome, pre_times, post_times)
        if gaza_data is None:
            print(f"    ⚠ Could not prepare Gaza data")
            continue
        
        gaza_sc = fit_synthetic_control(
            gaza_data['Y1_pre'], gaza_data['Y0_pre'],
            gaza_data['Y1_post'], gaza_data['Y0_post']
        )
        
        # West Bank SC (PLACEBO)
        wb_data = prepare_sc_data(wb_df, donor_df, outcome, pre_times, post_times)
        if wb_data is None:
            print(f"    ⚠ Could not prepare WB data")
            continue
        
        wb_sc = fit_synthetic_control(
            wb_data['Y1_pre'], wb_data['Y0_pre'],
            wb_data['Y1_post'], wb_data['Y0_post']
        )
        
        print(f"    Gaza:  R² = {gaza_sc['pre_r2']:.3f}, ATT = {gaza_sc['att']:.3f}")
        print(f"    WB:    R² = {wb_sc['pre_r2']:.3f}, ATT = {wb_sc['att']:.3f}")
        
        # Combine time series
        gaza_actual = np.concatenate([gaza_data['Y1_pre'], gaza_data['Y1_post']])
        gaza_synth = np.concatenate([gaza_sc['Y_synth_pre'], gaza_sc['Y_synth_post']])
        wb_actual = np.concatenate([wb_data['Y1_pre'], wb_data['Y1_post']])
        wb_synth = np.concatenate([wb_sc['Y_synth_pre'], wb_sc['Y_synth_post']])
        
        # Create CLEAN 2-panel plot (main result)
        create_clean_sc_plot(
            outcome, times_dt,
            gaza_actual, gaza_synth, gaza_sc['pre_r2'], gaza_sc['att'],
            intervention_dates,
            FIGURES_DIR / f'sc_clean_{outcome}.png'
        )
        print(f"    ✓ sc_clean_{outcome}.png")
        
        # Create 4-panel plot (with WB placebo)
        create_4panel_sc_plot(
            outcome, times_dt,
            gaza_actual, gaza_synth, gaza_sc['pre_r2'], gaza_sc['att'],
            wb_actual, wb_synth, wb_sc['pre_r2'], wb_sc['att'],
            intervention_dates,
            FIGURES_DIR / f'sc_4panel_{outcome}.png'
        )
        print(f"    ✓ sc_4panel_{outcome}.png")
        
        # Validity assessment
        ratio = abs(wb_sc['att'] / gaza_sc['att']) if abs(gaza_sc['att']) > 0.001 else 999
        valid = "Yes" if ratio < 0.30 else "No"
        
        sc_results.append({
            'Outcome': outcome,
            'Gaza_Pre_R2': gaza_sc['pre_r2'],
            'Gaza_ATT': gaza_sc['att'],
            'WB_Pre_R2': wb_sc['pre_r2'],
            'WB_ATT': wb_sc['att'],
            'WB_Gaza_Ratio': ratio,
            'Valid': valid
        })
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if sc_results:
        sc_df = pd.DataFrame(sc_results)
        sc_df.to_csv(TABLES_DIR / 'sc_wb_placebo_results.csv', index=False)
        print(f"  ✓ sc_wb_placebo_results.csv")
        
        print("\nSYNTHETIC CONTROL RESULTS:")
        print("-" * 70)
        print(sc_df.to_string(index=False))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDITY SUMMARY")
    print("=" * 70)
    
    if sc_results:
        valid_count = sum(1 for r in sc_results if r['Valid'] == 'Yes')
        print(f"\n  SC Placebo Test: {valid_count}/{len(sc_results)} outcomes pass")
        
        for r in sc_results:
            status = "✓" if r['Valid'] == 'Yes' else "⚠"
            print(f"    {status} {r['Outcome']}: Gaza ATT = {r['Gaza_ATT']:.3f}, "
                  f"WB ATT = {r['WB_ATT']:.3f}, Ratio = {r['WB_Gaza_Ratio']:.2f}")
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"  4-panel SC plots: {FIGURES_DIR}/sc_4panel_*.png")
    print(f"  Tables: {TABLES_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
