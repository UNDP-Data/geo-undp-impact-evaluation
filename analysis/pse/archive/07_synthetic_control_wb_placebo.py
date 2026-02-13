#!/usr/bin/env python3
"""
analysis/pse/07_synthetic_control_wb_placebo.py
===============================================
Synthetic Control with West Bank Placebo Test.

LOGIC:
1. Run SC for Gaza → Estimates conflict impact
2. Run SAME SC for West Bank → Should show ~0 gap if model is valid

If the model can accurately predict West Bank (which wasn't directly bombed),
then we have more confidence that it can construct valid counterfactuals.

If the model CANNOT predict West Bank, then the donor pool is inadequate
and Gaza results should be interpreted with caution.

Outcomes analyzed:
- nightlights, ndvi, ndbi, bsi, vh (SAR)
- price_bread, price_rice, price_flour, price_wheat

Usage:
    python analysis/pse/07_synthetic_control_wb_placebo.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
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

# Files
PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# All outcomes to analyze
OUTCOMES = [
    'nightlights', 'ndvi', 'ndbi', 'bsi', 'vh',
    'price_bread', 'price_rice', 'price_flour', 'price_wheat'
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights',
    'ndvi': 'Vegetation (NDVI)',
    'ndbi': 'Built-up (NDBI)',
    'bsi': 'Bare Soil (BSI)',
    'vh': 'SAR Backscatter (VH)',
    'price_bread': 'Bread Price',
    'price_rice': 'Rice Price',
    'price_flour': 'Flour Price',
    'price_wheat': 'Wheat Price',
}


def fit_synthetic_control(Y1_pre, Y0_pre, Y1_post, Y0_post):
    """
    Fit synthetic control weights and compute results.
    
    Args:
        Y1_pre: Treated unit pre-treatment outcomes (T_pre,)
        Y0_pre: Donor units pre-treatment outcomes (T_pre, J)
        Y1_post: Treated unit post-treatment outcomes (T_post,)
        Y0_post: Donor units post-treatment outcomes (T_post, J)
    
    Returns:
        dict with weights, synthetic values, gaps, ATT, fit metrics
    """
    
    J = Y0_pre.shape[1]
    
    # Objective: minimize pre-treatment MSE
    def objective(W):
        Y_synth = Y0_pre @ W
        return np.mean((Y1_pre - Y_synth) ** 2)
    
    # Constraints: weights sum to 1, non-negative
    constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
    bounds = [(0, 1) for _ in range(J)]
    W0 = np.ones(J) / J
    
    result = minimize(objective, W0, method='SLSQP', bounds=bounds,
                     constraints=constraints, options={'maxiter': 1000})
    
    weights = result.x if result.success else W0
    
    # Compute synthetic values
    Y_synth_pre = Y0_pre @ weights
    Y_synth_post = Y0_post @ weights
    
    # Fit metrics
    pre_rmse = np.sqrt(np.mean((Y1_pre - Y_synth_pre) ** 2))
    ss_tot = np.sum((Y1_pre - Y1_pre.mean()) ** 2)
    ss_res = np.sum((Y1_pre - Y_synth_pre) ** 2)
    pre_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Gaps and ATT
    gaps_pre = Y1_pre - Y_synth_pre
    gaps_post = Y1_post - Y_synth_post
    att = np.mean(gaps_post)
    
    return {
        'weights': weights,
        'Y_synth_pre': Y_synth_pre,
        'Y_synth_post': Y_synth_post,
        'gaps_pre': gaps_pre,
        'gaps_post': gaps_post,
        'att': att,
        'pre_rmse': pre_rmse,
        'pre_r2': pre_r2
    }


def prepare_region_data(region_df, donor_df, outcome, time_col, 
                        donor_unit_col, pre_times, post_times):
    """
    Prepare data for a single region (Gaza or West Bank).
    
    Returns:
        Y1_pre, Y0_pre, Y1_post, Y0_post, donor_ids
    """
    
    # Aggregate region to single time series
    region_agg = region_df.groupby(time_col)[outcome].mean().reset_index()
    region_agg = region_agg.sort_values(time_col)
    
    # Extract Y1
    Y1_pre = region_agg[region_agg[time_col].isin(pre_times)].sort_values(time_col)[outcome].values
    Y1_post = region_agg[region_agg[time_col].isin(post_times)].sort_values(time_col)[outcome].values
    
    if len(Y1_pre) == 0 or np.all(np.isnan(Y1_pre)):
        return None, None, None, None, None
    
    # Build donor matrix
    donor_units = donor_df[donor_unit_col].unique()
    
    valid_donors = []
    Y0_pre_list = []
    Y0_post_list = []
    
    for unit in donor_units:
        unit_df = donor_df[donor_df[donor_unit_col] == unit].sort_values(time_col)
        
        # Get values
        unit_pre = []
        unit_post = []
        
        for t in pre_times:
            val = unit_df[unit_df[time_col] == t][outcome].values
            unit_pre.append(val[0] if len(val) > 0 and not np.isnan(val[0]) else np.nan)
        
        for t in post_times:
            val = unit_df[unit_df[time_col] == t][outcome].values
            unit_post.append(val[0] if len(val) > 0 and not np.isnan(val[0]) else np.nan)
        
        unit_pre = np.array(unit_pre)
        unit_post = np.array(unit_post)
        
        # Skip if too much missing
        if np.sum(np.isnan(unit_pre)) > len(pre_times) * 0.3:
            continue
        
        # Impute missing
        if np.any(np.isnan(unit_pre)):
            unit_pre[np.isnan(unit_pre)] = np.nanmean(unit_pre)
        if np.any(np.isnan(unit_post)):
            mean_val = np.nanmean(unit_post) if not np.all(np.isnan(unit_post)) else 0
            unit_post[np.isnan(unit_post)] = mean_val
        
        valid_donors.append(unit)
        Y0_pre_list.append(unit_pre)
        Y0_post_list.append(unit_post)
    
    if len(valid_donors) < 3:
        return None, None, None, None, None
    
    Y0_pre = np.array(Y0_pre_list).T
    Y0_post = np.array(Y0_post_list).T
    
    return Y1_pre, Y0_pre, Y1_post, Y0_post, valid_donors


def create_comparison_figure(outcome, times_dt, 
                             gaza_results, wb_results,
                             Y1_gaza, Y1_wb,
                             output_path):
    """
    Create side-by-side comparison: Gaza (treatment) vs West Bank (placebo).
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.15)
    
    n_pre = len(gaza_results['Y_synth_pre'])
    
    # Combine values
    Y_gaza_actual = np.concatenate([Y1_gaza[:n_pre], Y1_gaza[n_pre:] if len(Y1_gaza) > n_pre else gaza_results['gaps_post'] + gaza_results['Y_synth_post']])
    Y_gaza_synth = np.concatenate([gaza_results['Y_synth_pre'], gaza_results['Y_synth_post']])
    
    Y_wb_actual = np.concatenate([Y1_wb[:n_pre], Y1_wb[n_pre:] if len(Y1_wb) > n_pre else wb_results['gaps_post'] + wb_results['Y_synth_post']])
    Y_wb_synth = np.concatenate([wb_results['Y_synth_pre'], wb_results['Y_synth_post']])
    
    gaza_gaps = np.concatenate([gaza_results['gaps_pre'], gaza_results['gaps_post']])
    wb_gaps = np.concatenate([wb_results['gaps_pre'], wb_results['gaps_post']])
    
    # Ensure times match data length
    min_len = min(len(times_dt), len(Y_gaza_actual), len(Y_gaza_synth))
    times_dt = times_dt[:min_len]
    Y_gaza_actual = Y_gaza_actual[:min_len]
    Y_gaza_synth = Y_gaza_synth[:min_len]
    Y_wb_actual = Y_wb_actual[:min_len]
    Y_wb_synth = Y_wb_synth[:min_len]
    gaza_gaps = gaza_gaps[:min_len]
    wb_gaps = wb_gaps[:min_len]
    
    # =========================================================================
    # TOP LEFT: Gaza (Treatment)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.axvspan(times_dt[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax1.axvspan(CONFLICT_DATE, times_dt[-1], alpha=0.15, color='red')
    
    ax1.plot(times_dt, Y_gaza_actual, 'b-', linewidth=2.5, marker='o', markersize=4, label='Gaza (Actual)')
    ax1.plot(times_dt, Y_gaza_synth, 'r--', linewidth=2, marker='s', markersize=3, label='Synthetic Gaza')
    
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=11)
    ax1.set_title(f'GAZA (Treatment)\nPre-R² = {gaza_results["pre_r2"]:.3f}', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # TOP RIGHT: West Bank (Placebo)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.axvspan(times_dt[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax2.axvspan(CONFLICT_DATE, times_dt[-1], alpha=0.15, color='lightblue')
    
    ax2.plot(times_dt, Y_wb_actual, 'b-', linewidth=2.5, marker='o', markersize=4, label='West Bank (Actual)')
    ax2.plot(times_dt, Y_wb_synth, 'orange', linewidth=2, linestyle='--', marker='s', markersize=3, label='Synthetic West Bank')
    
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    
    ax2.set_title(f'WEST BANK (Placebo - No Direct Bombing)\nPre-R² = {wb_results["pre_r2"]:.3f}', 
                 fontsize=12, fontweight='bold', color='darkblue')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # BOTTOM LEFT: Gaza Gap
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    colors_gaza = ['gray'] * n_pre + ['red'] * (len(gaza_gaps) - n_pre)
    colors_gaza = colors_gaza[:len(gaza_gaps)]
    ax3.bar(times_dt, gaza_gaps, width=15, color=colors_gaza, alpha=0.7)
    
    ax3.axhline(y=gaza_results['att'], color='darkred', linestyle=':', linewidth=2.5,
               label=f'ATT = {gaza_results["att"]:.3f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    
    ax3.set_ylabel('Gap (Actual - Synthetic)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('Gaza Gap: EXPECTED to be large negative', fontsize=10, style='italic')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # BOTTOM RIGHT: West Bank Gap
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    colors_wb = ['gray'] * n_pre + ['steelblue'] * (len(wb_gaps) - n_pre)
    colors_wb = colors_wb[:len(wb_gaps)]
    ax4.bar(times_dt, wb_gaps, width=15, color=colors_wb, alpha=0.7)
    
    ax4.axhline(y=wb_results['att'], color='darkblue', linestyle=':', linewidth=2.5,
               label=f'ATT = {wb_results["att"]:.3f}')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    
    ax4.set_ylabel('Gap (Actual - Synthetic)', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_title('West Bank Gap: EXPECTED to be ~0 (no treatment)', fontsize=10, style='italic')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # Interpretation box
    # =========================================================================
    # Determine validity
    wb_gap_magnitude = np.abs(wb_results['att'])
    gaza_gap_magnitude = np.abs(gaza_results['att'])
    
    if gaza_gap_magnitude > 0:
        ratio = wb_gap_magnitude / gaza_gap_magnitude
    else:
        ratio = np.inf
    
    if wb_results['pre_r2'] > 0.5 and ratio < 0.3:
        validity = "✓ VALID: WB gap small, model predicts well"
        color = 'green'
    elif wb_results['pre_r2'] > 0.3 and ratio < 0.5:
        validity = "⚠ CAUTION: WB shows some gap"
        color = 'orange'
    else:
        validity = "✗ CONCERN: WB gap large or poor fit"
        color = 'red'
    
    fig.text(0.5, 0.02, 
             f'PLACEBO VALIDITY: {validity}\n'
             f'Gaza ATT: {gaza_results["att"]:.3f} | West Bank ATT: {wb_results["att"]:.3f} | Ratio: {ratio:.2f}',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=color, linewidth=2))
    
    plt.suptitle(f'Synthetic Control: {OUTCOME_LABELS.get(outcome, outcome)}\nGaza (Treatment) vs West Bank (Placebo)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    """Run SC analysis with West Bank placebo for all outcomes."""
    
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL WITH WEST BANK PLACEBO TEST")
    print("=" * 70)
    print("""
    APPROACH:
    1. Run SC for Gaza → Estimate conflict impact
    2. Run SAME SC for West Bank → Validate model
    
    INTERPRETATION:
    - If WB gap ≈ 0: Model is valid, donors can predict Palestinian trajectories
    - If WB gap ≠ 0: Model may be invalid, or WB has spillover effects
    """)
    
    # Load data
    print("\nLoading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    wb_df = panel[panel['unit_id'].str.startswith('PS01')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs ({gaza_df['unit_id'].nunique()} districts)")
    print(f"  West Bank: {len(wb_df)} obs ({wb_df['unit_id'].nunique()} districts)")
    
    # Load donors
    if DONOR_POOL.exists():
        donor_df = pd.read_csv(DONOR_POOL)
        donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
        donor_unit_col = 'admin_id'
        print(f"  Donors: {len(donor_df)} obs from {donor_df['country'].nunique()} countries")
    else:
        print(f"\n⚠ Donor pool not found: {DONOR_POOL}")
        print("  Cannot run proper synthetic control without external donors.")
        print("  Run: python -m pipeline.gee_donor_pool --countries jordan lebanon egypt")
        return
    
    # Get time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    print(f"\n  Pre-treatment: {len(pre_times)} periods")
    print(f"  Post-treatment: {len(post_times)} periods")
    
    # Check available outcomes
    print("\n" + "=" * 70)
    print("CHECKING OUTCOME AVAILABILITY")
    print("=" * 70)
    
    available = []
    for outcome in OUTCOMES:
        in_gaza = outcome in gaza_df.columns and gaza_df[outcome].notna().sum() > 20
        in_wb = outcome in wb_df.columns and wb_df[outcome].notna().sum() > 20
        in_donor = outcome in donor_df.columns and donor_df[outcome].notna().sum() > 50
        
        if in_gaza and in_wb and in_donor:
            available.append(outcome)
            print(f"  {outcome}: ✓ Available in all datasets")
        else:
            reasons = []
            if not in_gaza: reasons.append("missing in Gaza")
            if not in_wb: reasons.append("missing in WB")
            if not in_donor: reasons.append("missing in donors")
            print(f"  {outcome}: ✗ {', '.join(reasons)}")
    
    if not available:
        print("\n❌ No outcomes available in all three datasets")
        return
    
    # Run analysis
    results = []
    
    for outcome in available:
        print(f"\n{'='*60}")
        print(f"OUTCOME: {outcome.upper()}")
        print(f"{'='*60}")
        
        # Prepare Gaza data
        gaza_data = prepare_region_data(
            gaza_df, donor_df, outcome, 'period_start', 
            donor_unit_col, pre_times, post_times
        )
        
        if gaza_data[0] is None:
            print(f"  ⚠ Could not prepare Gaza data")
            continue
        
        Y1_gaza_pre, Y0_pre, Y1_gaza_post, Y0_post, donor_ids = gaza_data
        
        # Prepare West Bank data (SAME donors)
        wb_data = prepare_region_data(
            wb_df, donor_df, outcome, 'period_start',
            donor_unit_col, pre_times, post_times
        )
        
        if wb_data[0] is None:
            print(f"  ⚠ Could not prepare West Bank data")
            continue
        
        Y1_wb_pre, _, Y1_wb_post, _, _ = wb_data
        
        print(f"\n  Data prepared:")
        print(f"    Pre-treatment periods: {len(Y1_gaza_pre)}")
        print(f"    Post-treatment periods: {len(Y1_gaza_post)}")
        print(f"    Donors: {len(donor_ids)}")
        
        # Fit SC for Gaza
        print(f"\n  Fitting Gaza SC...")
        gaza_results = fit_synthetic_control(Y1_gaza_pre, Y0_pre, Y1_gaza_post, Y0_post)
        print(f"    Pre-R²: {gaza_results['pre_r2']:.4f}")
        print(f"    ATT: {gaza_results['att']:.4f}")
        
        # Fit SC for West Bank (SAME specification)
        print(f"\n  Fitting West Bank SC (placebo)...")
        wb_results = fit_synthetic_control(Y1_wb_pre, Y0_pre, Y1_wb_post, Y0_post)
        print(f"    Pre-R²: {wb_results['pre_r2']:.4f}")
        print(f"    ATT: {wb_results['att']:.4f}")
        
        # Interpretation
        ratio = np.abs(wb_results['att']) / np.abs(gaza_results['att']) if gaza_results['att'] != 0 else np.inf
        
        if wb_results['pre_r2'] > 0.5 and ratio < 0.3:
            validity = "VALID"
            print(f"\n  ✓ VALID: WB gap is small ({ratio:.1%} of Gaza gap)")
        elif ratio < 0.5:
            validity = "CAUTION"
            print(f"\n  ⚠ CAUTION: WB gap is {ratio:.1%} of Gaza gap")
        else:
            validity = "CONCERN"
            print(f"\n  ✗ CONCERN: WB gap is {ratio:.1%} of Gaza gap")
        
        # Store results
        results.append({
            'Outcome': outcome,
            'Gaza_Pre_R2': gaza_results['pre_r2'],
            'Gaza_ATT': gaza_results['att'],
            'WB_Pre_R2': wb_results['pre_r2'],
            'WB_ATT': wb_results['att'],
            'WB_Gaza_Ratio': ratio,
            'Validity': validity
        })
        
        # Create visualization
        print(f"\n  Creating visualization...")
        Y1_gaza = np.concatenate([Y1_gaza_pre, Y1_gaza_post])
        Y1_wb = np.concatenate([Y1_wb_pre, Y1_wb_post])
        
        create_comparison_figure(
            outcome, times_dt,
            gaza_results, wb_results,
            Y1_gaza, Y1_wb,
            FIGURES_DIR / f'sc_wb_placebo_{outcome}.png'
        )
        print(f"  ✓ sc_wb_placebo_{outcome}.png")
    
    # Save summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / 'sc_wb_placebo_summary.csv', index=False)
        print(f"\n✓ Saved: sc_wb_placebo_summary.csv")
        
        print("\n" + results_df.to_string(index=False))
        
        # Interpretation guide
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")
        print("""
    WB_Gaza_Ratio = |WB ATT| / |Gaza ATT|
    
    VALIDITY CRITERIA:
    - VALID:   WB_Gaza_Ratio < 0.30 AND WB Pre-R² > 0.5
               → Model predicts WB well, WB gap is small
               → Gaza results are credible
    
    - CAUTION: WB_Gaza_Ratio 0.30-0.50 OR WB Pre-R² 0.3-0.5
               → Some issues with model or spillover effects
               → Gaza results should be interpreted carefully
    
    - CONCERN: WB_Gaza_Ratio > 0.50 OR WB Pre-R² < 0.3
               → Model cannot predict WB trajectory
               → Gaza results may not be valid
    
    WHAT IT MEANS:
    - If WB gap ≈ 0: Donors are good counterfactuals for Palestinian territories
    - If WB gap ≠ 0: Either model is wrong OR WB has spillover effects from Gaza conflict
    
    FOR REPORT:
    Use SC results only where Validity = "VALID"
    For other outcomes, fall back to ITS or DiD methods
        """)
    
    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
