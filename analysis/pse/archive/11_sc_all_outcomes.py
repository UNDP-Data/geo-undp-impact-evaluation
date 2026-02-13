#!/usr/bin/env python3
"""
analysis/pse/11_sc_all_outcomes.py
==================================
Synthetic Control Analysis for ALL available outcomes with:
1. Clean 4-panel SC plots (Gaza vs WB Placebo)
2. Simplified intervention timeline (only show relevant interventions)
3. Summary tables

Checks for ALL possible outcomes:
- Satellite: nightlights, ndvi, ndbi, bsi, vh, vv
- Prices: bread, wheat, rice, milk, flour, sugar, oil, chicken
- Google Trends: all available gt_* columns
- Population: worldpop (if available)

Usage:
    python analysis/pse/11_sc_all_outcomes.py
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

# Outcome labels
OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'vh': 'SAR Backscatter (VH)',
    'vv': 'SAR Backscatter (VV)',
    'rvi': 'Radar Vegetation Index',
    'price_bread': 'Bread Price',
    'price_wheat': 'Wheat Price',
    'price_rice': 'Rice Price',
    'price_milk': 'Milk Price',
    'price_flour': 'Flour Price',
    'price_sugar': 'Sugar Price',
    'price_oil': 'Cooking Oil Price',
    'price_chicken': 'Chicken Price',
    'worldpop': 'Population (WorldPop)',
    'population': 'Population',
}

# Which interventions are relevant for each outcome
RELEVANT_INTERVENTIONS = {
    'nightlights': ['Any UNDP', 'Livelihoods & Economy', 'Energy & Power', 'Infrastructure & Housing'],
    'ndvi': ['Any UNDP', 'Water & Sanitation (WASH)', 'Food Security'],
    'ndbi': ['Any UNDP', 'Debris & Solid Waste', 'Infrastructure & Housing'],
    'bsi': ['Any UNDP', 'Debris & Solid Waste', 'Infrastructure & Housing'],
    'vh': ['Any UNDP', 'Debris & Solid Waste', 'Infrastructure & Housing'],
    'vv': ['Any UNDP', 'Debris & Solid Waste', 'Infrastructure & Housing'],
    'price_bread': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_wheat': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_rice': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_milk': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_flour': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_sugar': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_oil': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
    'price_chicken': ['Any UNDP', 'Food Security', 'Livelihoods & Economy'],
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
    
    return {
        'weights': weights,
        'Y_synth_pre': Y_synth_pre,
        'Y_synth_post': Y_synth_post,
        'gaps_pre': Y1_pre - Y_synth_pre,
        'gaps_post': Y1_post - Y_synth_post,
        'att': np.mean(Y1_post - Y_synth_post),
        'pre_rmse': pre_rmse,
        'pre_r2': pre_r2
    }


def prepare_region_data(region_df, donor_df, outcome, time_col, 
                        donor_unit_col, pre_times, post_times):
    """Prepare SC data."""
    region_agg = region_df.groupby(time_col)[outcome].mean().reset_index()
    region_agg = region_agg.sort_values(time_col)
    
    Y1_pre = region_agg[region_agg[time_col].isin(pre_times)].sort_values(time_col)[outcome].values
    Y1_post = region_agg[region_agg[time_col].isin(post_times)].sort_values(time_col)[outcome].values
    
    if len(Y1_pre) == 0 or np.all(np.isnan(Y1_pre)):
        return None
    
    donor_units = donor_df[donor_unit_col].unique()
    valid_donors, Y0_pre_list, Y0_post_list = [], [], []
    
    for unit in donor_units:
        unit_df = donor_df[donor_df[donor_unit_col] == unit].sort_values(time_col)
        
        unit_pre = []
        for t in pre_times:
            vals = unit_df[unit_df[time_col] == t][outcome].values
            unit_pre.append(vals[0] if len(vals) > 0 and not np.isnan(vals[0]) else np.nan)
        
        unit_post = []
        for t in post_times:
            vals = unit_df[unit_df[time_col] == t][outcome].values
            unit_post.append(vals[0] if len(vals) > 0 and not np.isnan(vals[0]) else np.nan)
        
        unit_pre = np.array(unit_pre)
        unit_post = np.array(unit_post)
        
        if np.sum(np.isnan(unit_pre)) > len(pre_times) * 0.3:
            continue
        
        if np.any(np.isnan(unit_pre)):
            unit_pre[np.isnan(unit_pre)] = np.nanmean(unit_pre)
        if np.any(np.isnan(unit_post)):
            unit_post[np.isnan(unit_post)] = np.nanmean(unit_post) if not np.all(np.isnan(unit_post)) else 0
        
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


def test_intervention_effect(gaps, times, intervention_date, conflict_date):
    """Test effect of intervention on gap."""
    df = pd.DataFrame({'gap': gaps, 'date': pd.to_datetime(times)})
    df['post_conflict'] = (df['date'] > conflict_date).astype(int)
    df['post_int'] = (df['date'] >= intervention_date).astype(int)
    
    df_post = df[df['post_conflict'] == 1].copy()
    if len(df_post) < 6 or df_post['post_int'].sum() < 2:
        return None
    
    df_post['time_post'] = np.arange(len(df_post))
    int_time = df_post[df_post['post_int'] == 1]['time_post'].min()
    df_post['time_since_int'] = (df_post['time_post'] - int_time).clip(lower=0)
    
    try:
        model = smf.ols('gap ~ time_post + post_int + time_since_int', data=df_post).fit(cov_type='HC3')
        return {
            'level_shift': model.params.get('post_int', np.nan),
            'level_pval': model.pvalues.get('post_int', np.nan),
            'trend_change': model.params.get('time_since_int', np.nan),
            'trend_pval': model.pvalues.get('time_since_int', np.nan),
        }
    except:
        return None


def create_4panel_plot(outcome, times_dt, gaza_sc, wb_sc, Y1_gaza, Y1_wb, output_path):
    """Create clean 4-panel SC plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    n_pre = len(gaza_sc['Y_synth_pre'])
    
    Y_gaza_actual = np.concatenate([Y1_gaza[:n_pre], Y1_gaza[n_pre:]])
    Y_gaza_synth = np.concatenate([gaza_sc['Y_synth_pre'], gaza_sc['Y_synth_post']])
    Y_wb_actual = np.concatenate([Y1_wb[:n_pre], Y1_wb[n_pre:]])
    Y_wb_synth = np.concatenate([wb_sc['Y_synth_pre'], wb_sc['Y_synth_post']])
    gaza_gaps = np.concatenate([gaza_sc['gaps_pre'], gaza_sc['gaps_post']])
    wb_gaps = np.concatenate([wb_sc['gaps_pre'], wb_sc['gaps_post']])
    
    min_len = min(len(times_dt), len(Y_gaza_actual), len(gaza_gaps))
    times_plot = times_dt[:min_len]
    Y_gaza_actual = Y_gaza_actual[:min_len]
    Y_gaza_synth = Y_gaza_synth[:min_len]
    Y_wb_actual = Y_wb_actual[:min_len]
    Y_wb_synth = Y_wb_synth[:min_len]
    gaza_gaps = gaza_gaps[:min_len]
    wb_gaps = wb_gaps[:min_len]
    
    # Top-left: Gaza
    ax1 = axes[0, 0]
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='red')
    ax1.plot(times_plot, Y_gaza_actual, 'b-', linewidth=2, marker='o', markersize=4, label='Gaza (Actual)')
    ax1.plot(times_plot, Y_gaza_synth, 'r--', linewidth=2, marker='s', markersize=3, label='Synthetic Gaza')
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=11)
    ax1.set_title(f'GAZA (Treatment)\nPre-R² = {gaza_sc["pre_r2"]:.3f}', fontsize=12, fontweight='bold', color='darkred')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # Top-right: West Bank
    ax2 = axes[0, 1]
    ax2.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax2.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='lightblue')
    ax2.plot(times_plot, Y_wb_actual, 'b-', linewidth=2, marker='o', markersize=4, label='West Bank (Actual)')
    ax2.plot(times_plot, Y_wb_synth, 'orange', linewidth=2, linestyle='--', marker='s', markersize=3, label='Synthetic West Bank')
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax2.set_title(f'WEST BANK (Placebo - No Direct Bombing)\nPre-R² = {wb_sc["pre_r2"]:.3f}', fontsize=12, fontweight='bold', color='darkblue')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # Bottom-left: Gaza Gap
    ax3 = axes[1, 0]
    colors_gaza = ['gray'] * n_pre + ['red'] * (len(gaza_gaps) - n_pre)
    ax3.bar(times_plot, gaza_gaps, width=15, color=colors_gaza[:len(gaza_gaps)], alpha=0.7)
    ax3.axhline(y=gaza_sc['att'], color='darkred', linestyle=':', linewidth=2.5, label=f'ATT = {gaza_sc["att"]:.3f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax3.set_ylabel('Gap (Actual - Synthetic)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('Gaza Gap: EXPECTED to be large', fontsize=10, style='italic')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # Bottom-right: WB Gap
    ax4 = axes[1, 1]
    colors_wb = ['gray'] * n_pre + ['steelblue'] * (len(wb_gaps) - n_pre)
    ax4.bar(times_plot, wb_gaps, width=15, color=colors_wb[:len(wb_gaps)], alpha=0.7)
    ax4.axhline(y=wb_sc['att'], color='darkblue', linestyle=':', linewidth=2.5, label=f'ATT = {wb_sc["att"]:.3f}')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax4.set_ylabel('Gap (Actual - Synthetic)', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_title('West Bank Gap: EXPECTED to be ~0 (no treatment)', fontsize=10, style='italic')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # Validity
    ratio = np.abs(wb_sc['att']) / np.abs(gaza_sc['att']) if gaza_sc['att'] != 0 else np.inf
    if wb_sc['pre_r2'] > 0.5 and ratio < 0.3:
        validity, v_color = "✓ VALID: WB gap small, model predicts well", 'green'
    elif wb_sc['pre_r2'] > 0.3 and ratio < 0.5:
        validity, v_color = "⚠ CAUTION: Some WB gap or moderate fit", 'orange'
    else:
        validity, v_color = "✗ CONCERN: Large WB gap or poor fit", 'red'
    
    fig.text(0.5, 0.01, f'PLACEBO VALIDITY: {validity}\nGaza ATT: {gaza_sc["att"]:.3f} | West Bank ATT: {wb_sc["att"]:.3f} | Ratio: {ratio:.2f}',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=v_color, linewidth=2))
    
    plt.suptitle(f'Synthetic Control: {OUTCOME_LABELS.get(outcome, outcome)}\nGaza (Treatment) vs West Bank (Placebo)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_intervention_plot(outcome, times_dt, gaza_gaps, intervention_dates, 
                             int_results, output_path):
    """Create CLEANER intervention timeline showing only relevant interventions."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Post-conflict only
    post_mask = np.array([t > CONFLICT_DATE for t in times_dt])
    times_post = times_dt[post_mask]
    gaps_post = gaza_gaps[post_mask]
    
    if len(times_post) == 0:
        plt.close()
        return
    
    # Plot gap
    ax.plot(times_post, gaps_post, 'o-', color='darkred', linewidth=2.5, markersize=8, label='Gaza Gap')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Only show RELEVANT interventions for this outcome
    relevant = RELEVANT_INTERVENTIONS.get(outcome, ['Any UNDP'])
    colors = {'Any UNDP': 'gray', 'Debris & Solid Waste': 'brown', 'Infrastructure & Housing': 'orange',
              'Livelihoods & Economy': 'green', 'Energy & Power': 'gold', 'Water & Sanitation (WASH)': 'blue',
              'Health & Emergency': 'red', 'Food Security': 'purple'}
    
    for cat, date in intervention_dates.items():
        if cat in relevant and date >= times_post[0] and date <= times_post[-1]:
            # Get effect results if available
            effect_text = ""
            if cat in int_results:
                res = int_results[cat]
                if res and res['level_pval'] < 0.1:
                    sig = '***' if res['level_pval'] < 0.01 else ('**' if res['level_pval'] < 0.05 else '*')
                    effect_text = f" ({res['level_shift']:+.3f}{sig})"
            
            ax.axvline(x=date, color=colors.get(cat, 'gray'), linestyle='--', linewidth=2.5,
                      label=f'{cat}: {date.strftime("%b %Y")}{effect_text}')
    
    # Shade and annotate
    att = np.mean(gaps_post)
    ax.axhline(y=att, color='darkred', linestyle=':', linewidth=2, alpha=0.5, label=f'Mean Gap = {att:.3f}')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gap (Actual - Synthetic)', fontsize=12)
    ax.set_title(f'{OUTCOME_LABELS.get(outcome, outcome)}: Gap with Relevant UNDP Interventions\n'
                 f'(Gap < 0 = Below counterfactual | Improvement = Gap moving toward 0)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add interpretation box
    interp = ("Reading this plot:\n"
              "• Gap = Actual Gaza - Synthetic Gaza\n"
              "• Negative gap = Gaza below counterfactual (damage)\n"
              "• Gap moving UP = Recovery toward counterfactual\n"
              "• Check if recovery occurs after intervention lines")
    ax.text(0.02, 0.98, interp, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL - ALL OUTCOMES")
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
    
    # List ALL columns in panel
    print(f"\n  All columns in panel ({len(panel.columns)}):")
    for col in sorted(panel.columns):
        if col not in ['period_start', 'unit_id', 'year_month', 'year_month_str']:
            coverage = panel[col].notna().sum() / len(panel) * 100
            print(f"    {col}: {coverage:.0f}% coverage")
    
    # Donors
    if not DONOR_POOL.exists():
        print(f"\n❌ Donor pool not found: {DONOR_POOL}")
        return
    
    donor_df = pd.read_csv(DONOR_POOL)
    donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
    donor_unit_col = 'admin_id'
    print(f"\n  Donor pool: {len(donor_df)} obs")
    print(f"  Donor columns: {[c for c in donor_df.columns if c not in ['period_start', 'admin_id', 'country']]}")
    
    # UNDP
    intervention_dates = {}
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE]
        
        if 'ml_category' in undp.columns:
            for cat in undp['ml_category'].dropna().unique():
                cat_date = undp[undp['ml_category'] == cat]['start_date'].min()
                if pd.notna(cat_date):
                    intervention_dates[cat] = cat_date
        intervention_dates['Any UNDP'] = undp['start_date'].min()
    except:
        intervention_dates = {'Any UNDP': pd.Timestamp('2024-01-01')}
    
    print(f"\n  UNDP interventions: {len(intervention_dates)}")
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    # Find ALL available outcomes
    print("\n" + "=" * 70)
    print("CHECKING ALL OUTCOMES")
    print("=" * 70)
    
    # Check satellite
    satellite_outcomes = ['nightlights', 'ndvi', 'ndbi', 'bsi', 'vh', 'vv', 'rvi']
    
    # Check prices
    price_outcomes = [c for c in panel.columns if c.startswith('price_')]
    
    # Check Google Trends
    gt_outcomes = [c for c in panel.columns if c.startswith('gt_')]
    
    # Check population
    pop_outcomes = [c for c in panel.columns if 'pop' in c.lower() or 'worldpop' in c.lower()]
    
    all_outcomes = satellite_outcomes + price_outcomes + gt_outcomes + pop_outcomes
    
    available = []
    not_available = {'not_in_gaza': [], 'not_in_wb': [], 'not_in_donors': []}
    
    for outcome in all_outcomes:
        in_gaza = outcome in gaza_df.columns and gaza_df[outcome].notna().sum() > 20
        in_wb = outcome in wb_df.columns and wb_df[outcome].notna().sum() > 20
        in_donor = outcome in donor_df.columns and donor_df[outcome].notna().sum() > 50
        
        if in_gaza and in_wb and in_donor:
            available.append(outcome)
            print(f"  ✓ {outcome}")
        else:
            reasons = []
            if not in_gaza: 
                reasons.append("Gaza")
                not_available['not_in_gaza'].append(outcome)
            if not in_wb: 
                reasons.append("WB")
                not_available['not_in_wb'].append(outcome)
            if not in_donor: 
                reasons.append("donors")
                not_available['not_in_donors'].append(outcome)
            print(f"  ✗ {outcome} (missing: {', '.join(reasons)})")
    
    print(f"\n  Available: {len(available)}")
    print(f"  Not in donors: {not_available['not_in_donors']}")
    
    # Run analysis
    results = []
    intervention_effects = []
    
    for outcome in available:
        print(f"\n{'='*60}")
        print(f"{outcome.upper()}")
        print(f"{'='*60}")
        
        # Prepare data
        gaza_data = prepare_region_data(gaza_df, donor_df, outcome, 'period_start',
                                        donor_unit_col, pre_times, post_times)
        wb_data = prepare_region_data(wb_df, donor_df, outcome, 'period_start',
                                      donor_unit_col, pre_times, post_times)
        
        if gaza_data is None or wb_data is None:
            print(f"  ⚠ Could not prepare data")
            continue
        
        # Fit SC
        gaza_sc = fit_synthetic_control(gaza_data['Y1_pre'], gaza_data['Y0_pre'],
                                        gaza_data['Y1_post'], gaza_data['Y0_post'])
        wb_sc = fit_synthetic_control(wb_data['Y1_pre'], wb_data['Y0_pre'],
                                      wb_data['Y1_post'], wb_data['Y0_post'])
        
        print(f"  Gaza: R²={gaza_sc['pre_r2']:.3f}, ATT={gaza_sc['att']:.3f}")
        print(f"  WB:   R²={wb_sc['pre_r2']:.3f}, ATT={wb_sc['att']:.3f}")
        
        ratio = np.abs(wb_sc['att']) / np.abs(gaza_sc['att']) if gaza_sc['att'] != 0 else np.inf
        
        results.append({
            'Outcome': outcome,
            'Gaza_Pre_R2': gaza_sc['pre_r2'],
            'Gaza_ATT': gaza_sc['att'],
            'WB_Pre_R2': wb_sc['pre_r2'],
            'WB_ATT': wb_sc['att'],
            'Ratio': ratio,
            'Valid': 'Yes' if wb_sc['pre_r2'] > 0.5 and ratio < 0.3 else ('Caution' if ratio < 0.5 else 'No')
        })
        
        # Create 4-panel plot
        Y1_gaza = np.concatenate([gaza_data['Y1_pre'], gaza_data['Y1_post']])
        Y1_wb = np.concatenate([wb_data['Y1_pre'], wb_data['Y1_post']])
        
        create_4panel_plot(outcome, times_dt, gaza_sc, wb_sc, Y1_gaza, Y1_wb,
                          FIGURES_DIR / f'sc_{outcome}.png')
        print(f"  ✓ sc_{outcome}.png")
        
        # Test interventions
        all_gaps = np.concatenate([gaza_sc['gaps_pre'], gaza_sc['gaps_post']])
        all_times_list = list(pre_times) + list(post_times)
        
        int_results = {}
        for cat, date in intervention_dates.items():
            res = test_intervention_effect(all_gaps, all_times_list, date, CONFLICT_DATE)
            if res:
                int_results[cat] = res
                intervention_effects.append({
                    'Outcome': outcome,
                    'Intervention': cat,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Level_Shift': res['level_shift'],
                    'Level_pval': res['level_pval'],
                    'Trend_Change': res['trend_change'],
                    'Trend_pval': res['trend_pval']
                })
        
        # Create intervention plot
        create_intervention_plot(outcome, times_dt, all_gaps, intervention_dates,
                                int_results, FIGURES_DIR / f'sc_int_{outcome}.png')
        print(f"  ✓ sc_int_{outcome}.png")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / 'sc_all_outcomes.csv', index=False)
        print(f"  ✓ sc_all_outcomes.csv")
        print(f"\n{results_df.to_string(index=False)}")
    
    if intervention_effects:
        int_df = pd.DataFrame(intervention_effects)
        int_df.to_csv(TABLES_DIR / 'intervention_effects_all.csv', index=False)
        print(f"\n  ✓ intervention_effects_all.csv")
        
        # Significant effects
        sig = int_df[(int_df['Level_pval'] < 0.1) | (int_df['Trend_pval'] < 0.1)]
        if len(sig) > 0:
            print(f"\n  Significant effects (p < 0.10):")
            for _, row in sig.iterrows():
                if row['Level_pval'] < 0.1:
                    s = '***' if row['Level_pval'] < 0.01 else ('**' if row['Level_pval'] < 0.05 else '*')
                    print(f"    {row['Outcome']} + {row['Intervention']}: Level {row['Level_Shift']:+.4f}{s}")
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
