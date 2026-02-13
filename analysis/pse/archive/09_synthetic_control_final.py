#!/usr/bin/env python3
"""
analysis/pse/09_synthetic_control_final.py
==========================================
Final Synthetic Control Analysis with:
1. Gaza vs Synthetic Gaza (treatment effect)
2. West Bank Placebo (model validation)
3. UNDP Effect Test on Gap (intervention timing)

Generates clean 5-panel figures for ALL outcomes:
- Satellite: nightlights, ndvi, ndbi, bsi, vh
- Prices: bread, rice, flour, sugar, oil, milk, chicken

Usage:
    python analysis/pse/09_synthetic_control_final.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.formula.api as smf
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

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# ALL outcomes to analyze
OUTCOMES = [
    # Satellite indices
    'nightlights', 'ndvi', 'ndbi', 'bsi', 'vh',
    # Food prices
    'price_bread', 'price_rice', 'price_flour', 'price_sugar',
    'price_oil', 'price_milk', 'price_chicken', 'price_wheat'
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'vh': 'SAR Backscatter (VH)',
    'price_bread': 'Bread Price (ILS/kg)',
    'price_rice': 'Rice Price (ILS/kg)',
    'price_flour': 'Flour Price (ILS/kg)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_oil': 'Cooking Oil Price (ILS/L)',
    'price_milk': 'Milk Price (ILS/L)',
    'price_chicken': 'Chicken Price (ILS/kg)',
    'price_wheat': 'Wheat Price (ILS/kg)',
}

# Expected direction of conflict effect
EXPECTED_DIRECTION = {
    'nightlights': 'negative',
    'ndvi': 'negative',
    'ndbi': 'negative',
    'bsi': 'positive',
    'vh': 'negative',
    'price_bread': 'positive',
    'price_rice': 'positive',
    'price_flour': 'positive',
    'price_sugar': 'positive',
    'price_oil': 'positive',
    'price_milk': 'positive',
    'price_chicken': 'positive',
    'price_wheat': 'positive',
}


def fit_synthetic_control(Y1_pre, Y0_pre, Y1_post, Y0_post):
    """Fit SC and return results."""
    
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
    
    pre_rmse = np.sqrt(np.mean((Y1_pre - Y_synth_pre) ** 2))
    ss_tot = np.sum((Y1_pre - Y1_pre.mean()) ** 2)
    ss_res = np.sum((Y1_pre - Y_synth_pre) ** 2)
    pre_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
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
    """Prepare SC data for a region."""
    
    region_agg = region_df.groupby(time_col)[outcome].mean().reset_index()
    region_agg = region_agg.sort_values(time_col)
    
    Y1_pre = region_agg[region_agg[time_col].isin(pre_times)].sort_values(time_col)[outcome].values
    Y1_post = region_agg[region_agg[time_col].isin(post_times)].sort_values(time_col)[outcome].values
    
    if len(Y1_pre) == 0 or np.all(np.isnan(Y1_pre)):
        return None
    
    donor_units = donor_df[donor_unit_col].unique()
    
    valid_donors = []
    Y0_pre_list = []
    Y0_post_list = []
    
    for unit in donor_units:
        unit_df = donor_df[donor_df[donor_unit_col] == unit].sort_values(time_col)
        
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
        
        if np.sum(np.isnan(unit_pre)) > len(pre_times) * 0.3:
            continue
        
        if np.any(np.isnan(unit_pre)):
            unit_pre[np.isnan(unit_pre)] = np.nanmean(unit_pre)
        if np.any(np.isnan(unit_post)):
            mean_val = np.nanmean(unit_post) if not np.all(np.isnan(unit_post)) else 0
            unit_post[np.isnan(unit_post)] = mean_val
        
        valid_donors.append(unit)
        Y0_pre_list.append(unit_pre)
        Y0_post_list.append(unit_post)
    
    if len(valid_donors) < 3:
        return None
    
    return {
        'Y1_pre': Y1_pre,
        'Y1_post': Y1_post,
        'Y0_pre': np.array(Y0_pre_list).T,
        'Y0_post': np.array(Y0_post_list).T,
        'donor_ids': valid_donors
    }


def test_undp_effect_on_gap(gaps, times, undp_date, conflict_date):
    """Test if UNDP intervention changed the gap trajectory."""
    
    df = pd.DataFrame({
        'gap': gaps,
        'date': pd.to_datetime(times)
    })
    
    df['time_idx'] = np.arange(len(df))
    df['post_conflict'] = (df['date'] > conflict_date).astype(int)
    df['post_undp'] = (df['date'] >= undp_date).astype(int)
    
    undp_time = df[df['date'] >= undp_date]['time_idx'].min() if df['post_undp'].sum() > 0 else len(df)
    df['time_since_undp'] = (df['time_idx'] - undp_time).clip(lower=0)
    
    df_post = df[df['post_conflict'] == 1].copy()
    
    if len(df_post) < 6:
        return None
    
    df_post['time_post'] = np.arange(len(df_post))
    undp_time_post = df_post[df_post['post_undp'] == 1]['time_post'].min() if df_post['post_undp'].sum() > 0 else len(df_post)
    df_post['time_since_undp'] = (df_post['time_post'] - undp_time_post).clip(lower=0)
    
    try:
        model = smf.ols('gap ~ time_post + post_undp + time_since_undp', data=df_post).fit(cov_type='HC3')
        
        return {
            'level_shift': model.params.get('post_undp', np.nan),
            'level_pval': model.pvalues.get('post_undp', np.nan),
            'trend_change': model.params.get('time_since_undp', np.nan),
            'trend_pval': model.pvalues.get('time_since_undp', np.nan),
            'baseline_trend': model.params.get('time_post', np.nan),
            'intercept': model.params.get('Intercept', np.nan),
            'r2': model.rsquared,
            'n_obs': int(model.nobs),
            'model': model,
            'data': df_post
        }
    except:
        return None


def create_figure(outcome, times_dt, gaza_results, wb_results,
                  Y1_gaza, Y1_wb, undp_results, undp_date, output_path):
    """
    Create 5-panel figure:
    Row 1: Gaza SC | WB Placebo
    Row 2: Gaza Gap | WB Gap  
    Row 3: UNDP Effect Test (full width)
    """
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, height_ratios=[1.2, 1, 1.2], hspace=0.28, wspace=0.15)
    
    n_pre = len(gaza_results['Y_synth_pre'])
    
    # Combine values
    Y_gaza_actual = np.concatenate([Y1_gaza[:n_pre], Y1_gaza[n_pre:] if len(Y1_gaza) > n_pre else []])
    Y_gaza_synth = np.concatenate([gaza_results['Y_synth_pre'], gaza_results['Y_synth_post']])
    Y_wb_actual = np.concatenate([Y1_wb[:n_pre], Y1_wb[n_pre:] if len(Y1_wb) > n_pre else []])
    Y_wb_synth = np.concatenate([wb_results['Y_synth_pre'], wb_results['Y_synth_post']])
    
    gaza_gaps = np.concatenate([gaza_results['gaps_pre'], gaza_results['gaps_post']])
    wb_gaps = np.concatenate([wb_results['gaps_pre'], wb_results['gaps_post']])
    
    # Align lengths
    min_len = min(len(times_dt), len(Y_gaza_actual), len(Y_gaza_synth), 
                  len(Y_wb_actual), len(Y_wb_synth), len(gaza_gaps), len(wb_gaps))
    times_plot = times_dt[:min_len]
    Y_gaza_actual = Y_gaza_actual[:min_len]
    Y_gaza_synth = Y_gaza_synth[:min_len]
    Y_wb_actual = Y_wb_actual[:min_len]
    Y_wb_synth = Y_wb_synth[:min_len]
    gaza_gaps = gaza_gaps[:min_len]
    wb_gaps = wb_gaps[:min_len]
    
    # =========================================================================
    # ROW 1: Time Series
    # =========================================================================
    
    # Gaza (left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='red')
    ax1.plot(times_plot, Y_gaza_actual, 'b-', linewidth=2.5, marker='o', markersize=4, label='Gaza (Actual)')
    ax1.plot(times_plot, Y_gaza_synth, 'r--', linewidth=2, marker='s', markersize=3, label='Synthetic Gaza')
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=11)
    ax1.set_title(f'GAZA (Treatment)\nPre-R² = {gaza_results["pre_r2"]:.3f}', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # West Bank (right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax2.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='lightblue')
    ax2.plot(times_plot, Y_wb_actual, 'b-', linewidth=2.5, marker='o', markersize=4, label='West Bank (Actual)')
    ax2.plot(times_plot, Y_wb_synth, 'orange', linewidth=2, linestyle='--', marker='s', markersize=3, label='Synthetic West Bank')
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax2.set_title(f'WEST BANK (Placebo - No Direct Bombing)\nPre-R² = {wb_results["pre_r2"]:.3f}', 
                 fontsize=12, fontweight='bold', color='darkblue')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # ROW 2: Gap plots
    # =========================================================================
    
    # Gaza gap
    ax3 = fig.add_subplot(gs[1, 0])
    colors_gaza = ['gray'] * n_pre + ['red'] * (len(gaza_gaps) - n_pre)
    colors_gaza = colors_gaza[:len(gaza_gaps)]
    ax3.bar(times_plot, gaza_gaps, width=15, color=colors_gaza, alpha=0.7)
    ax3.axhline(y=gaza_results['att'], color='darkred', linestyle=':', linewidth=2.5,
               label=f'ATT = {gaza_results["att"]:.3f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax3.set_ylabel('Gap (Actual - Synthetic)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('Gaza Gap: EXPECTED to be large', fontsize=10, style='italic')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # West Bank gap
    ax4 = fig.add_subplot(gs[1, 1])
    colors_wb = ['gray'] * n_pre + ['steelblue'] * (len(wb_gaps) - n_pre)
    colors_wb = colors_wb[:len(wb_gaps)]
    ax4.bar(times_plot, wb_gaps, width=15, color=colors_wb, alpha=0.7)
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
    # ROW 3: UNDP Effect Test (full width)
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, :])
    
    # Post-conflict data only
    post_mask = np.array([t > CONFLICT_DATE for t in times_plot])
    times_post = times_plot[post_mask]
    gaps_post_plot = gaza_gaps[post_mask]
    
    if len(times_post) > 0 and len(gaps_post_plot) > 0:
        # Shade pre/post UNDP
        ax5.axvspan(times_post[0], undp_date, alpha=0.15, color='salmon', label='Pre-UNDP')
        ax5.axvspan(undp_date, times_post[-1], alpha=0.15, color='lightgreen', label='Post-UNDP')
        
        # Plot gaps
        ax5.plot(times_post, gaps_post_plot, 'o-', color='darkred', linewidth=2, markersize=6, label='Gaza Gap')
        
        # UNDP line
        ax5.axvline(x=undp_date, color='purple', linestyle='--', linewidth=3, label='UNDP Start')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add fitted ITS lines
        if undp_results is not None and 'model' in undp_results:
            try:
                df_fit = undp_results['data']
                model = undp_results['model']
                intercept = undp_results['intercept']
                baseline = undp_results['baseline_trend']
                level = undp_results['level_shift']
                trend = undp_results['trend_change']
                
                # Pre-UNDP fitted
                pre_undp_mask = df_fit['post_undp'] == 0
                n_pre_undp = pre_undp_mask.sum()
                if n_pre_undp > 1:
                    pre_x = np.arange(n_pre_undp)
                    pre_fitted = intercept + baseline * pre_x
                    ax5.plot(times_post[:n_pre_undp], pre_fitted, 'r--', linewidth=2.5, alpha=0.8, label='Pre-UNDP trend')
                
                # Post-UNDP fitted
                post_undp_mask = df_fit['post_undp'] == 1
                n_post_undp = post_undp_mask.sum()
                if n_post_undp > 1:
                    post_x = np.arange(n_pre_undp, n_pre_undp + n_post_undp)
                    post_tsu = np.arange(n_post_undp)
                    post_fitted = intercept + baseline * post_x + level + trend * post_tsu
                    ax5.plot(times_post[-n_post_undp:], post_fitted, 'g--', linewidth=2.5, alpha=0.8, label='Post-UNDP trend')
            except:
                pass
        
        ax5.set_xlabel('Date', fontsize=11)
        ax5.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
        ax5.set_title('UNDP EFFECT TEST: Does the Gap Improve After UNDP Intervention?', 
                     fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=9, ncol=3)
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # UNDP results box
        if undp_results is not None:
            level_sig = '***' if undp_results['level_pval'] < 0.01 else ('**' if undp_results['level_pval'] < 0.05 else ('*' if undp_results['level_pval'] < 0.1 else ''))
            trend_sig = '***' if undp_results['trend_pval'] < 0.01 else ('**' if undp_results['trend_pval'] < 0.05 else ('*' if undp_results['trend_pval'] < 0.1 else ''))
            
            # Determine if improvement
            expected = EXPECTED_DIRECTION.get(outcome, 'negative')
            if expected == 'negative':
                improving = undp_results['level_shift'] > 0 or undp_results['trend_change'] > 0
            else:
                improving = undp_results['level_shift'] < 0 or undp_results['trend_change'] < 0
            
            significant = undp_results['level_pval'] < 0.1 or undp_results['trend_pval'] < 0.1
            
            if improving and significant:
                interp = "✓ Evidence of recovery at UNDP timing"
                box_color = 'lightgreen'
            elif improving:
                interp = "Gap improving but not significant"
                box_color = 'lightyellow'
            else:
                interp = "No improvement at UNDP timing"
                box_color = 'lightcoral'
            
            results_text = (
                f"UNDP EFFECT ON GAP:\n"
                f"  Level shift: {undp_results['level_shift']:+.4f}{level_sig} (p={undp_results['level_pval']:.3f})\n"
                f"  Trend change: {undp_results['trend_change']:+.5f}{trend_sig} (p={undp_results['trend_pval']:.3f})\n"
                f"  {interp}"
            )
            
            ax5.text(0.02, 0.98, results_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
    
    # =========================================================================
    # Validity box
    # =========================================================================
    ratio = np.abs(wb_results['att']) / np.abs(gaza_results['att']) if gaza_results['att'] != 0 else np.inf
    
    if wb_results['pre_r2'] > 0.5 and ratio < 0.3:
        validity = "✓ VALID: WB gap small, model predicts well"
        v_color = 'green'
    elif wb_results['pre_r2'] > 0.3 and ratio < 0.5:
        validity = "⚠ CAUTION: Some WB gap or moderate fit"
        v_color = 'orange'
    else:
        validity = "✗ CONCERN: Large WB gap or poor fit"
        v_color = 'red'
    
    fig.text(0.5, 0.01,
             f'PLACEBO VALIDITY: {validity}\n'
             f'Gaza ATT: {gaza_results["att"]:.3f} | West Bank ATT: {wb_results["att"]:.3f} | Ratio: {ratio:.2f}',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=v_color, linewidth=2))
    
    plt.suptitle(f'Synthetic Control: {OUTCOME_LABELS.get(outcome, outcome)}\nGaza (Treatment) vs West Bank (Placebo)',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    """Run SC analysis for all outcomes."""
    
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL ANALYSIS - ALL OUTCOMES")
    print("With West Bank Placebo & UNDP Effect Test")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    wb_df = panel[panel['unit_id'].str.startswith('PS01')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    print(f"  West Bank: {len(wb_df)} obs")
    
    # Load donors
    if not DONOR_POOL.exists():
        print(f"\n❌ Donor pool not found: {DONOR_POOL}")
        return
    
    donor_df = pd.read_csv(DONOR_POOL)
    donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
    donor_unit_col = 'admin_id'
    print(f"  Donors: {len(donor_df)} obs")
    
    # UNDP date
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE]
        undp_date = undp['start_date'].min()
    except:
        undp_date = pd.Timestamp('2024-01-01')
    print(f"  UNDP start: {undp_date.strftime('%Y-%m-%d')}")
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    # Check outcomes
    print("\n" + "=" * 70)
    print("CHECKING OUTCOMES")
    print("=" * 70)
    
    available = []
    for outcome in OUTCOMES:
        in_gaza = outcome in gaza_df.columns and gaza_df[outcome].notna().sum() > 20
        in_wb = outcome in wb_df.columns and wb_df[outcome].notna().sum() > 20
        in_donor = outcome in donor_df.columns and donor_df[outcome].notna().sum() > 50
        
        if in_gaza and in_wb and in_donor:
            available.append(outcome)
            print(f"  {outcome}: ✓")
        else:
            print(f"  {outcome}: ✗ Missing data")
    
    # Run analysis
    results = []
    
    for outcome in available:
        print(f"\n{'='*60}")
        print(f"{outcome.upper()}")
        print(f"{'='*60}")
        
        gaza_data = prepare_region_data(gaza_df, donor_df, outcome, 'period_start',
                                        donor_unit_col, pre_times, post_times)
        wb_data = prepare_region_data(wb_df, donor_df, outcome, 'period_start',
                                      donor_unit_col, pre_times, post_times)
        
        if gaza_data is None or wb_data is None:
            print(f"  ⚠ Could not prepare data")
            continue
        
        # Fit SC
        gaza_results = fit_synthetic_control(
            gaza_data['Y1_pre'], gaza_data['Y0_pre'],
            gaza_data['Y1_post'], gaza_data['Y0_post']
        )
        wb_results = fit_synthetic_control(
            wb_data['Y1_pre'], wb_data['Y0_pre'],
            wb_data['Y1_post'], wb_data['Y0_post']
        )
        
        print(f"  Gaza: R²={gaza_results['pre_r2']:.3f}, ATT={gaza_results['att']:.3f}")
        print(f"  WB:   R²={wb_results['pre_r2']:.3f}, ATT={wb_results['att']:.3f}")
        
        # UNDP test
        all_gaps = np.concatenate([gaza_results['gaps_pre'], gaza_results['gaps_post']])
        all_times_list = pre_times + post_times
        undp_results = test_undp_effect_on_gap(all_gaps, all_times_list, undp_date, CONFLICT_DATE)
        
        if undp_results:
            print(f"  UNDP Level: {undp_results['level_shift']:+.4f} (p={undp_results['level_pval']:.3f})")
            print(f"  UNDP Trend: {undp_results['trend_change']:+.5f} (p={undp_results['trend_pval']:.3f})")
        
        # Store
        ratio = np.abs(wb_results['att']) / np.abs(gaza_results['att']) if gaza_results['att'] != 0 else np.inf
        
        results.append({
            'Outcome': outcome,
            'Gaza_Pre_R2': gaza_results['pre_r2'],
            'Gaza_ATT': gaza_results['att'],
            'WB_Pre_R2': wb_results['pre_r2'],
            'WB_ATT': wb_results['att'],
            'WB_Gaza_Ratio': ratio,
            'UNDP_Level': undp_results['level_shift'] if undp_results else np.nan,
            'UNDP_Level_p': undp_results['level_pval'] if undp_results else np.nan,
            'UNDP_Trend': undp_results['trend_change'] if undp_results else np.nan,
            'UNDP_Trend_p': undp_results['trend_pval'] if undp_results else np.nan,
        })
        
        # Create figure
        Y1_gaza = np.concatenate([gaza_data['Y1_pre'], gaza_data['Y1_post']])
        Y1_wb = np.concatenate([wb_data['Y1_pre'], wb_data['Y1_post']])
        
        create_figure(
            outcome, times_dt,
            gaza_results, wb_results,
            Y1_gaza, Y1_wb,
            undp_results, undp_date,
            FIGURES_DIR / f'sc_final_{outcome}.png'
        )
        print(f"  ✓ sc_final_{outcome}.png")
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / 'sc_final_summary.csv', index=False)
        print(f"\n✓ {TABLES_DIR / 'sc_final_summary.csv'}\n")
        
        print(results_df.round(4).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
