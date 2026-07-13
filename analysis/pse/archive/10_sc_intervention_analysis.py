#!/usr/bin/env python3
"""
analysis/pse/10_sc_intervention_analysis.py
===========================================
Comprehensive Synthetic Control Analysis with:

1. CLEAN 4-PANEL PLOTS (like the VH example) for all outcomes
2. INTERVENTION-SPECIFIC TESTS - match UNDP categories to relevant outcomes
3. CONFLICT-ADJUSTED ANALYSIS - control for bombing intensity from ACLED
4. PELT CHANGEPOINT DETECTION - data-driven break detection

Outcomes analyzed:
- Satellite: nightlights, ndvi, ndbi, bsi, vh
- Prices: bread, wheat, rice, milk
- Google Trends: gt_water, gt_food, gt_hospital, gt_help, etc.

Usage:
    python analysis/pse/10_sc_intervention_analysis.py
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

# Try to import ruptures for PELT
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("Note: Install 'ruptures' for PELT changepoint detection: pip install ruptures")

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
ACLED_FILE = DATA_DIR / "pse" / "external" / "acled_events.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# All outcomes to analyze
OUTCOMES = [
    # Satellite
    'nightlights', 'ndvi', 'ndbi', 'bsi', 'vh',
    # Prices
    'price_bread', 'price_wheat', 'price_rice', 'price_milk',
    # Google Trends (will check which exist)
    'gt_water', 'gt_food', 'gt_hospital', 'gt_help', 'gt_electricity',
    'gt_aid', 'gt_shelter', 'gt_medicine', 'gt_bread', 'gt_flour'
]

OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'vh': 'SAR Backscatter (VH)',
    'price_bread': 'Bread Price',
    'price_wheat': 'Wheat Price',
    'price_rice': 'Rice Price',
    'price_milk': 'Milk Price',
    'gt_water': 'Google Trends: Water',
    'gt_food': 'Google Trends: Food',
    'gt_hospital': 'Google Trends: Hospital',
    'gt_help': 'Google Trends: Help',
    'gt_electricity': 'Google Trends: Electricity',
    'gt_aid': 'Google Trends: Aid',
    'gt_shelter': 'Google Trends: Shelter',
    'gt_medicine': 'Google Trends: Medicine',
    'gt_bread': 'Google Trends: Bread',
    'gt_flour': 'Google Trends: Flour',
}

# UNDP intervention categories and their expected outcome effects
INTERVENTION_OUTCOME_MAP = {
    'Debris & Solid Waste': ['ndbi', 'bsi', 'vh', 'nightlights'],
    'Infrastructure & Housing': ['ndbi', 'vh', 'nightlights'],
    'Livelihoods & Economy': ['nightlights', 'price_bread', 'price_wheat'],
    'Energy & Power': ['nightlights', 'gt_electricity'],
    'Water & Sanitation (WASH)': ['gt_water', 'ndvi'],
    'Health & Emergency': ['gt_hospital', 'gt_medicine', 'gt_help'],
    'Food Security': ['price_bread', 'price_wheat', 'price_rice', 'gt_food', 'gt_bread'],
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
        
        unit_pre = [unit_df[unit_df[time_col] == t][outcome].values[0] 
                   if len(unit_df[unit_df[time_col] == t]) > 0 and 
                   not np.isnan(unit_df[unit_df[time_col] == t][outcome].values[0]) 
                   else np.nan for t in pre_times]
        unit_post = [unit_df[unit_df[time_col] == t][outcome].values[0] 
                    if len(unit_df[unit_df[time_col] == t]) > 0 and 
                    not np.isnan(unit_df[unit_df[time_col] == t][outcome].values[0]) 
                    else np.nan for t in post_times]
        
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
    """Test effect of specific intervention on gap."""
    
    df = pd.DataFrame({'gap': gaps, 'date': pd.to_datetime(times)})
    df['post_conflict'] = (df['date'] > conflict_date).astype(int)
    df['post_intervention'] = (df['date'] >= intervention_date).astype(int)
    
    df_post = df[df['post_conflict'] == 1].copy()
    if len(df_post) < 6:
        return None
    
    df_post['time_post'] = np.arange(len(df_post))
    int_time = df_post[df_post['post_intervention'] == 1]['time_post'].min() if df_post['post_intervention'].sum() > 0 else len(df_post)
    df_post['time_since_int'] = (df_post['time_post'] - int_time).clip(lower=0)
    
    try:
        model = smf.ols('gap ~ time_post + post_intervention + time_since_int', data=df_post).fit(cov_type='HC3')
        return {
            'level_shift': model.params.get('post_intervention', np.nan),
            'level_pval': model.pvalues.get('post_intervention', np.nan),
            'trend_change': model.params.get('time_since_int', np.nan),
            'trend_pval': model.pvalues.get('time_since_int', np.nan),
            'n_obs': int(model.nobs)
        }
    except:
        return None


def detect_changepoints_pelt(gaps, times, n_bkps=3):
    """Detect changepoints using PELT algorithm."""
    
    if not HAS_RUPTURES:
        return []
    
    signal = np.array(gaps).reshape(-1, 1)
    
    # Remove NaN
    valid_mask = ~np.isnan(signal.flatten())
    if valid_mask.sum() < 10:
        return []
    
    signal_clean = signal[valid_mask]
    times_clean = np.array(times)[valid_mask]
    
    try:
        # PELT with RBF kernel
        algo = rpt.Pelt(model="rbf", min_size=3).fit(signal_clean)
        bkps = algo.predict(pen=1)
        
        # Convert to dates
        changepoint_dates = []
        for bkp in bkps[:-1]:  # Exclude last (end of signal)
            if bkp < len(times_clean):
                changepoint_dates.append(pd.Timestamp(times_clean[bkp]))
        
        return changepoint_dates
    except:
        return []


def test_conflict_adjusted_effect(gaps, times, acled_df, intervention_date, conflict_date):
    """
    Test UNDP effect while controlling for conflict intensity.
    
    Model: Gap_t = α + β₁·Conflict_Intensity_t + β₂·Post_UNDP + β₃·time_since_UNDP + ε
    
    This asks: "Given the bombing intensity, did UNDP improve outcomes?"
    """
    
    df = pd.DataFrame({'gap': gaps, 'date': pd.to_datetime(times)})
    
    # Merge ACLED conflict intensity
    if acled_df is not None and len(acled_df) > 0:
        acled_monthly = acled_df.groupby(pd.Grouper(key='event_date', freq='MS')).agg({
            'fatalities': 'sum',
            'event_id_cnty': 'count'
        }).reset_index()
        acled_monthly.columns = ['date', 'fatalities', 'n_events']
        acled_monthly['date'] = pd.to_datetime(acled_monthly['date'])
        
        df = df.merge(acled_monthly, on='date', how='left')
        df['fatalities'] = df['fatalities'].fillna(0)
        df['n_events'] = df['n_events'].fillna(0)
        df['conflict_intensity'] = df['fatalities'] + df['n_events']  # Combined measure
    else:
        df['conflict_intensity'] = 0
    
    df['post_conflict'] = (df['date'] > conflict_date).astype(int)
    df['post_undp'] = (df['date'] >= intervention_date).astype(int)
    
    df_post = df[df['post_conflict'] == 1].copy()
    if len(df_post) < 6:
        return None
    
    df_post['time_post'] = np.arange(len(df_post))
    undp_time = df_post[df_post['post_undp'] == 1]['time_post'].min() if df_post['post_undp'].sum() > 0 else len(df_post)
    df_post['time_since_undp'] = (df_post['time_post'] - undp_time).clip(lower=0)
    
    try:
        # Model with conflict intensity control
        model = smf.ols('gap ~ conflict_intensity + time_post + post_undp + time_since_undp', 
                       data=df_post).fit(cov_type='HC3')
        
        return {
            'conflict_effect': model.params.get('conflict_intensity', np.nan),
            'conflict_pval': model.pvalues.get('conflict_intensity', np.nan),
            'undp_level': model.params.get('post_undp', np.nan),
            'undp_level_pval': model.pvalues.get('post_undp', np.nan),
            'undp_trend': model.params.get('time_since_undp', np.nan),
            'undp_trend_pval': model.pvalues.get('time_since_undp', np.nan),
            'r2': model.rsquared,
            'n_obs': int(model.nobs)
        }
    except:
        return None


def create_clean_4panel_plot(outcome, times_dt, gaza_results, wb_results,
                              Y1_gaza, Y1_wb, output_path):
    """Create clean 4-panel plot like the VH example."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    n_pre = len(gaza_results['Y_synth_pre'])
    
    # Combine values
    Y_gaza_actual = np.concatenate([Y1_gaza[:n_pre], Y1_gaza[n_pre:]])
    Y_gaza_synth = np.concatenate([gaza_results['Y_synth_pre'], gaza_results['Y_synth_post']])
    Y_wb_actual = np.concatenate([Y1_wb[:n_pre], Y1_wb[n_pre:]])
    Y_wb_synth = np.concatenate([wb_results['Y_synth_pre'], wb_results['Y_synth_post']])
    
    gaza_gaps = np.concatenate([gaza_results['gaps_pre'], gaza_results['gaps_post']])
    wb_gaps = np.concatenate([wb_results['gaps_pre'], wb_results['gaps_post']])
    
    # Align
    min_len = min(len(times_dt), len(Y_gaza_actual), len(Y_gaza_synth), len(gaza_gaps))
    times_plot = times_dt[:min_len]
    Y_gaza_actual = Y_gaza_actual[:min_len]
    Y_gaza_synth = Y_gaza_synth[:min_len]
    Y_wb_actual = Y_wb_actual[:min_len]
    Y_wb_synth = Y_wb_synth[:min_len]
    gaza_gaps = gaza_gaps[:min_len]
    wb_gaps = wb_gaps[:min_len]
    
    # =========================================================================
    # TOP LEFT: Gaza
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax1.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='red')
    ax1.plot(times_plot, Y_gaza_actual, 'b-', linewidth=2, marker='o', markersize=4, label='Gaza (Actual)')
    ax1.plot(times_plot, Y_gaza_synth, 'r--', linewidth=2, marker='s', markersize=3, label='Synthetic Gaza')
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=11)
    ax1.set_title(f'GAZA (Treatment)\nPre-R² = {gaza_results["pre_r2"]:.3f}', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    
    # =========================================================================
    # TOP RIGHT: West Bank
    # =========================================================================
    ax2 = axes[0, 1]
    ax2.axvspan(times_plot[0], CONFLICT_DATE, alpha=0.15, color='green')
    ax2.axvspan(CONFLICT_DATE, times_plot[-1], alpha=0.15, color='lightblue')
    ax2.plot(times_plot, Y_wb_actual, 'b-', linewidth=2, marker='o', markersize=4, label='West Bank (Actual)')
    ax2.plot(times_plot, Y_wb_synth, 'orange', linewidth=2, linestyle='--', marker='s', markersize=3, label='Synthetic West Bank')
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
    ax3 = axes[1, 0]
    colors_gaza = ['gray'] * n_pre + ['red'] * (len(gaza_gaps) - n_pre)
    ax3.bar(times_plot, gaza_gaps, width=15, color=colors_gaza[:len(gaza_gaps)], alpha=0.7)
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
    ax4 = axes[1, 1]
    colors_wb = ['gray'] * n_pre + ['steelblue'] * (len(wb_gaps) - n_pre)
    ax4.bar(times_plot, wb_gaps, width=15, color=colors_wb[:len(wb_gaps)], alpha=0.7)
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
    
    # Validity box
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
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_intervention_effect_plot(outcome, times_dt, gaza_gaps, intervention_dates,
                                    changepoints, output_path):
    """Create plot showing gap with intervention dates and changepoints."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Post-conflict only
    post_mask = np.array([t > CONFLICT_DATE for t in times_dt])
    times_post = times_dt[post_mask]
    gaps_post = gaza_gaps[post_mask]
    
    if len(times_post) == 0:
        plt.close()
        return False
    
    # Plot gap
    ax.plot(times_post, gaps_post, 'o-', color='darkred', linewidth=2, markersize=6, label='Gaza Gap')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add intervention dates
    colors = plt.cm.Set2(np.linspace(0, 1, len(intervention_dates)))
    for i, (cat, date) in enumerate(intervention_dates.items()):
        if date >= times_post[0] and date <= times_post[-1]:
            ax.axvline(x=date, color=colors[i], linestyle='--', linewidth=2, 
                      label=f'{cat}: {date.strftime("%b %Y")}')
    
    # Add PELT changepoints
    for cp in changepoints:
        if cp >= times_post[0] and cp <= times_post[-1]:
            ax.axvline(x=cp, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
    ax.set_title(f'{OUTCOME_LABELS.get(outcome, outcome)}: Gap with UNDP Interventions',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add changepoint note
    if changepoints:
        ax.text(0.02, 0.02, f'PELT changepoints (purple dotted): {len(changepoints)}',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    """Run comprehensive SC analysis with intervention-specific tests."""
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SYNTHETIC CONTROL ANALYSIS")
    print("With Intervention-Specific Tests & PELT Changepoint Detection")
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
    
    # Donors
    if not DONOR_POOL.exists():
        print(f"❌ Donor pool not found: {DONOR_POOL}")
        return
    
    donor_df = pd.read_csv(DONOR_POOL)
    donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
    donor_unit_col = 'admin_id'
    print(f"  Donors: {len(donor_df)} obs")
    
    # ACLED (for conflict-adjusted analysis)
    acled_df = None
    if ACLED_FILE.exists():
        try:
            acled_df = pd.read_csv(ACLED_FILE)
            acled_df['event_date'] = pd.to_datetime(acled_df['event_date'], errors='coerce')
            print(f"  ACLED: {len(acled_df)} events")
        except:
            pass
    
    # UNDP intervention dates by category
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
        
        # Also get overall first date
        intervention_dates['Any UNDP'] = undp['start_date'].min()
        
        print(f"\n  UNDP intervention dates:")
        for cat, date in sorted(intervention_dates.items(), key=lambda x: x[1]):
            print(f"    {cat}: {date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  Could not load UNDP: {e}")
        intervention_dates = {'Any UNDP': pd.Timestamp('2024-01-01')}
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    times_dt = pd.to_datetime(pre_times + post_times)
    
    # Check available outcomes
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
            print(f"  {outcome}: ✗")
    
    print(f"\n  Available outcomes: {len(available)}")
    
    # Storage for results
    sc_results = []
    intervention_results = []
    conflict_adjusted_results = []
    changepoint_results = []
    
    # ==========================================================================
    # PART 1: Clean 4-panel SC plots for all outcomes
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 1: SYNTHETIC CONTROL ANALYSIS")
    print("=" * 70)
    
    for outcome in available:
        print(f"\n--- {outcome} ---")
        
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
        
        sc_results.append({
            'Outcome': outcome,
            'Gaza_Pre_R2': gaza_sc['pre_r2'],
            'Gaza_ATT': gaza_sc['att'],
            'WB_Pre_R2': wb_sc['pre_r2'],
            'WB_ATT': wb_sc['att'],
            'WB_Gaza_Ratio': ratio
        })
        
        # Create 4-panel plot
        Y1_gaza = np.concatenate([gaza_data['Y1_pre'], gaza_data['Y1_post']])
        Y1_wb = np.concatenate([wb_data['Y1_pre'], wb_data['Y1_post']])
        
        create_clean_4panel_plot(
            outcome, times_dt, gaza_sc, wb_sc, Y1_gaza, Y1_wb,
            FIGURES_DIR / f'sc_{outcome}.png'
        )
        print(f"  ✓ sc_{outcome}.png")
        
        # Store gap for later analyses
        all_gaps = np.concatenate([gaza_sc['gaps_pre'], gaza_sc['gaps_post']])
        all_times_list = list(pre_times) + list(post_times)
        
        # ==================================================================
        # PART 2: Intervention-specific tests
        # ==================================================================
        for cat, int_date in intervention_dates.items():
            # Check if this outcome is expected to be affected by this intervention
            expected_outcomes = INTERVENTION_OUTCOME_MAP.get(cat, [])
            is_expected = outcome in expected_outcomes or cat == 'Any UNDP'
            
            result = test_intervention_effect(all_gaps, all_times_list, int_date, CONFLICT_DATE)
            
            if result:
                intervention_results.append({
                    'Outcome': outcome,
                    'Intervention': cat,
                    'Int_Date': int_date.strftime('%Y-%m-%d'),
                    'Expected_Effect': 'Yes' if is_expected else 'No',
                    'Level_Shift': result['level_shift'],
                    'Level_pval': result['level_pval'],
                    'Trend_Change': result['trend_change'],
                    'Trend_pval': result['trend_pval'],
                    'N_obs': result['n_obs']
                })
        
        # ==================================================================
        # PART 3: Conflict-adjusted analysis
        # ==================================================================
        if acled_df is not None:
            for cat, int_date in intervention_dates.items():
                result = test_conflict_adjusted_effect(all_gaps, all_times_list, acled_df, 
                                                      int_date, CONFLICT_DATE)
                if result:
                    conflict_adjusted_results.append({
                        'Outcome': outcome,
                        'Intervention': cat,
                        'Conflict_Effect': result['conflict_effect'],
                        'Conflict_pval': result['conflict_pval'],
                        'UNDP_Level': result['undp_level'],
                        'UNDP_Level_pval': result['undp_level_pval'],
                        'UNDP_Trend': result['undp_trend'],
                        'UNDP_Trend_pval': result['undp_trend_pval'],
                        'R2': result['r2']
                    })
        
        # ==================================================================
        # PART 4: PELT changepoint detection
        # ==================================================================
        changepoints = detect_changepoints_pelt(all_gaps, all_times_list)
        
        for cp in changepoints:
            # Find closest intervention
            closest_int = None
            min_diff = float('inf')
            for cat, int_date in intervention_dates.items():
                diff = abs((cp - int_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_int = cat
            
            changepoint_results.append({
                'Outcome': outcome,
                'Changepoint': cp.strftime('%Y-%m-%d'),
                'Closest_Intervention': closest_int,
                'Days_Difference': min_diff
            })
        
        # Create intervention effect plot
        create_intervention_effect_plot(
            outcome, times_dt, all_gaps, intervention_dates, changepoints,
            FIGURES_DIR / f'sc_interventions_{outcome}.png'
        )
        print(f"  ✓ sc_interventions_{outcome}.png")
    
    # ==========================================================================
    # SAVE ALL RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # SC summary
    if sc_results:
        pd.DataFrame(sc_results).to_csv(TABLES_DIR / 'sc_summary.csv', index=False)
        print(f"  ✓ sc_summary.csv")
    
    # Intervention effects
    if intervention_results:
        int_df = pd.DataFrame(intervention_results)
        int_df.to_csv(TABLES_DIR / 'intervention_effects.csv', index=False)
        print(f"  ✓ intervention_effects.csv")
        
        # Highlight significant results
        sig_df = int_df[(int_df['Level_pval'] < 0.1) | (int_df['Trend_pval'] < 0.1)]
        if len(sig_df) > 0:
            print(f"\n  Significant intervention effects (p < 0.10):")
            for _, row in sig_df.iterrows():
                l_sig = '***' if row['Level_pval'] < 0.01 else ('**' if row['Level_pval'] < 0.05 else '*')
                t_sig = '***' if row['Trend_pval'] < 0.01 else ('**' if row['Trend_pval'] < 0.05 else '*')
                expected_mark = '✓' if row['Expected_Effect'] == 'Yes' else ''
                if row['Level_pval'] < 0.1:
                    print(f"    {row['Outcome']} + {row['Intervention']}: Level {row['Level_Shift']:+.4f}{l_sig} {expected_mark}")
                if row['Trend_pval'] < 0.1:
                    print(f"    {row['Outcome']} + {row['Intervention']}: Trend {row['Trend_Change']:+.5f}{t_sig} {expected_mark}")
    
    # Conflict-adjusted
    if conflict_adjusted_results:
        pd.DataFrame(conflict_adjusted_results).to_csv(TABLES_DIR / 'conflict_adjusted_effects.csv', index=False)
        print(f"  ✓ conflict_adjusted_effects.csv")
    
    # Changepoints
    if changepoint_results:
        cp_df = pd.DataFrame(changepoint_results)
        cp_df.to_csv(TABLES_DIR / 'pelt_changepoints.csv', index=False)
        print(f"  ✓ pelt_changepoints.csv")
        
        # Check if changepoints are near intervention dates (within 60 days)
        near_intervention = cp_df[cp_df['Days_Difference'] <= 60]
        if len(near_intervention) > 0:
            print(f"\n  Changepoints within 60 days of interventions:")
            for _, row in near_intervention.iterrows():
                print(f"    {row['Outcome']}: {row['Changepoint']} near {row['Closest_Intervention']} ({row['Days_Difference']} days)")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    if sc_results:
        sc_df = pd.DataFrame(sc_results)
        print(f"\nSynthetic Control Results:")
        print(sc_df.round(4).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
    INTERVENTION-SPECIFIC TESTS:
    - Tests each UNDP category against each outcome
    - "Expected_Effect = Yes" means this outcome SHOULD respond to this intervention
    - Look for significant effects (p < 0.10) on expected outcome-intervention pairs
    
    CONFLICT-ADJUSTED ANALYSIS:
    - Controls for bombing intensity from ACLED
    - Asks: "Given the conflict, did UNDP improve outcomes?"
    - Conflict_Effect shows impact of bombing on gap
    - UNDP_Level/Trend show UNDP effect net of conflict
    
    PELT CHANGEPOINTS:
    - Data-driven detection of structural breaks in gap series
    - If changepoints occur near UNDP dates → supportive evidence
    - Less causal than ITS, but useful as corroboration
    
    VALIDITY:
    - Results valid where WB placebo passes (ratio < 0.30)
    - For outcomes with poor SC fit, interpret with caution
    """)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
