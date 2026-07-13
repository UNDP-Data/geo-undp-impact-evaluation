#!/usr/bin/env python3
"""
analysis/pse/05_did_robustness.py
=================================
Rigorous Difference-in-Differences with:
1. Proper interaction terms (Gaza × Post_UNDP)
2. West Bank placebo test (falsification)
3. Pre-trends test (parallel trends assumption)
4. Event study with leads and lags

This addresses: "Are we actually measuring anything related to UNDP timing?"

Usage:
    python analysis/pse/05_did_robustness.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/pse")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
RASTER_PANEL_FILE = OUTPUT_DIR / "raster_panel.csv"
UNDP_FILE = DATA_DIR / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")


def load_data():
    """Load and prepare panel data."""
    
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load main panel
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    # Try to merge raster data for NDBI/BSI
    if RASTER_PANEL_FILE.exists():
        raster = pd.read_csv(RASTER_PANEL_FILE)
        print(f"  Raster panel: {len(raster)} rows")
        
        # Find the date column
        date_col = None
        for col in ['date', 'year_month', 'period', 'month', 'time']:
            if col in raster.columns:
                date_col = col
                break
        
        # Find the unit column
        unit_col = None
        for col in ['adm2_pcode', 'unit_id', 'admin_id', 'ADM2_PCODE']:
            if col in raster.columns:
                unit_col = col
                break
        
        if date_col and unit_col:
            raster[date_col] = pd.to_datetime(raster[date_col], errors='coerce')
            
            for outcome_col in ['ndbi', 'bsi', 'viirs', 'ndvi']:
                if outcome_col in raster.columns and (outcome_col not in panel.columns or panel[outcome_col].isna().all()):
                    raster_monthly = raster.groupby(
                        [unit_col, pd.Grouper(key=date_col, freq='MS')]
                    )[outcome_col].mean().reset_index()
                    raster_monthly.columns = ['unit_id', 'period_start', outcome_col]
                    
                    if outcome_col in panel.columns:
                        panel = panel.drop(columns=[outcome_col])
                    panel = panel.merge(raster_monthly, on=['unit_id', 'period_start'], how='left')
                    print(f"  Merged {outcome_col} from raster panel")
    
    # Region indicators
    panel['is_gaza'] = panel['unit_id'].str.startswith('PS02')
    panel['gaza'] = panel['is_gaza'].astype(int)
    panel['region'] = panel['is_gaza'].map({True: 'Gaza', False: 'West Bank'})
    
    # Time variables
    panel['year_month_str'] = panel['period_start'].dt.strftime('%Y_%m')
    panel['time_idx'] = (panel['period_start'].dt.year - 2023) * 12 + panel['period_start'].dt.month
    
    # Conflict indicator
    panel['post_conflict'] = (panel['period_start'] >= CONFLICT_DATE).astype(int)
    
    print(f"\nPanel: {len(panel)} observations")
    print(f"  Gaza: {panel['is_gaza'].sum()} obs ({panel[panel['is_gaza']]['unit_id'].nunique()} districts)")
    print(f"  West Bank: {(~panel['is_gaza']).sum()} obs ({panel[~panel['is_gaza']]['unit_id'].nunique()} districts)")
    
    # Load UNDP dates
    treatment_dates = {}
    
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE]
        
        treatment_dates['any_undp'] = undp['start_date'].min()
        
        if 'ml_category' in undp.columns:
            recovery_cats = ['Debris & Solid Waste', 'Infrastructure & Housing', 
                           'Livelihoods & Economy', 'Energy & Power']
            recovery = undp[undp['ml_category'].isin(recovery_cats)]
            if len(recovery) > 0:
                treatment_dates['recovery'] = recovery['start_date'].min()
            
            humanitarian_cats = ['Health & Emergency', 'Water & Sanitation (WASH)', 'Education']
            humanitarian = undp[undp['ml_category'].isin(humanitarian_cats)]
            if len(humanitarian) > 0:
                treatment_dates['humanitarian'] = humanitarian['start_date'].min()
        
        print(f"\nUNDP treatment dates:")
        for name, date in sorted(treatment_dates.items(), key=lambda x: x[1]):
            print(f"  {name}: {date.strftime('%Y-%m-%d')}")
            
    except Exception as e:
        print(f"  Could not load UNDP data: {e}")
        treatment_dates = {
            'any_undp': pd.Timestamp('2024-01-01'),
            'recovery': pd.Timestamp('2024-10-01'),
            'humanitarian': pd.Timestamp('2024-07-01')
        }
    
    # Create post-treatment indicators (for ALL units, not just Gaza)
    for name, date in treatment_dates.items():
        panel[f'post_{name}'] = (panel['period_start'] >= date).astype(int)
        # Interaction term: Gaza × Post_Treatment
        panel[f'gaza_x_post_{name}'] = panel['gaza'] * panel[f'post_{name}']
    
    return panel, treatment_dates


def get_outcomes(panel):
    """Get all outcomes with sufficient data."""
    
    candidates = ['nightlights', 'ndvi', 'ndbi', 'bsi', 'vh', 'vv', 'rvi',
                  'conflict_events', 'conflict_fatalities'] + \
                 [c for c in panel.columns if c.startswith('price_')] + \
                 [c for c in panel.columns if c.startswith('gt_')]
    
    outcomes = []
    for var in candidates:
        if var in panel.columns:
            coverage = panel[var].notna().sum() / len(panel)
            if coverage >= 0.3:
                outcomes.append(var)
    
    print(f"\nOutcomes to analyze ({len(outcomes)}): {outcomes}")
    return outcomes


def run_did_with_interactions(panel, outcome, treatment_name, controls=None):
    """
    Proper DiD with interaction terms.
    
    Model: Y_it = α + β₁·Gaza_i + β₂·Post_t + β₃·(Gaza×Post)_it + γX_it + μ_i + λ_t + ε_it
    
    β₃ is the DiD estimator: differential change in Gaza after treatment
    """
    
    post_var = f'post_{treatment_name}'
    interaction_var = f'gaza_x_post_{treatment_name}'
    
    # Full model with unit and time FE
    formula = f'{outcome} ~ gaza + {post_var} + {interaction_var}'
    formula += ' + C(unit_id) + C(year_month_str)'
    
    if controls:
        formula += ' + ' + ' + '.join(controls)
    
    try:
        data = panel.dropna(subset=[outcome])
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['unit_id']}
        )
        return model
    except Exception as e:
        print(f"    DiD failed: {e}")
        return None


def run_westbank_placebo(panel, outcome, treatment_name, controls=None):
    """
    Placebo test: Run DiD as if West Bank were "treated" at the same time.
    
    If we find "effects" in West Bank, our identification is questionable.
    
    Model: Same as DiD but using West Bank as treated, Gaza as control
    """
    
    # Create "fake" treatment for West Bank
    panel = panel.copy()
    panel['is_westbank'] = (~panel['is_gaza']).astype(int)
    
    post_var = f'post_{treatment_name}'
    panel['wb_x_post'] = panel['is_westbank'] * panel[post_var]
    
    formula = f'{outcome} ~ is_westbank + {post_var} + wb_x_post'
    formula += ' + C(unit_id) + C(year_month_str)'
    
    if controls:
        formula += ' + ' + ' + '.join(controls)
    
    try:
        data = panel.dropna(subset=[outcome])
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['unit_id']}
        )
        return model
    except:
        return None


def test_parallel_trends(panel, outcome, treatment_date, controls=None):
    """
    Test parallel trends assumption in pre-treatment period.
    
    Model: Y_it = α + β·(Gaza × Time)_it + μ_i + λ_t + ε_it
    
    For pre-treatment period only. If β ≠ 0, parallel trends violated.
    """
    
    # Pre-treatment data only
    pre_data = panel[panel['period_start'] < treatment_date].copy()
    
    if len(pre_data) < 50:
        return None, None
    
    # Create Gaza × Time interaction
    pre_data['gaza_x_time'] = pre_data['gaza'] * pre_data['time_idx']
    
    formula = f'{outcome} ~ gaza + time_idx + gaza_x_time + C(unit_id)'
    
    if controls:
        valid_controls = [c for c in controls if c in pre_data.columns]
        if valid_controls:
            formula += ' + ' + ' + '.join(valid_controls)
    
    try:
        model = smf.ols(formula, data=pre_data.dropna(subset=[outcome])).fit(
            cov_type='cluster', cov_kwds={'groups': pre_data.dropna(subset=[outcome])['unit_id']}
        )
        
        # Test if gaza_x_time coefficient is significantly different from 0
        coef = model.params.get('gaza_x_time', np.nan)
        pval = model.pvalues.get('gaza_x_time', np.nan)
        
        return coef, pval
    except:
        return None, None


def run_event_study_interactions(panel, outcome, treatment_date, controls=None):
    """
    Event study with Gaza × Time interactions.
    
    Model: Y_it = α + Σ_k β_k·(Gaza × 1(t=k))_it + μ_i + λ_t + ε_it
    
    Shows dynamics of differential Gaza effect over time.
    """
    
    panel = panel.copy()
    
    # Calculate relative time
    panel['rel_time'] = (
        (panel['period_start'].dt.year - treatment_date.year) * 12 +
        (panel['period_start'].dt.month - treatment_date.month)
    ).clip(-6, 12)
    
    # Create interaction dummies (Gaza × relative time)
    # Reference period: t = -1
    interaction_vars = []
    for t in range(-6, 13):
        if t == -1:
            continue
        
        col = f'gaza_t{"m" if t < 0 else "p"}{abs(t)}'
        panel[col] = (panel['gaza'] * (panel['rel_time'] == t)).astype(int)
        interaction_vars.append(col)
    
    formula = f'{outcome} ~ gaza + ' + ' + '.join(interaction_vars)
    formula += ' + C(unit_id) + C(year_month_str)'
    
    if controls:
        formula += ' + ' + ' + '.join(controls)
    
    try:
        data = panel.dropna(subset=[outcome])
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['unit_id']}
        )
        
        # Extract coefficients
        results = []
        for t in range(-6, 13):
            if t == -1:
                results.append({'rel_time': t, 'coef': 0, 'se': 0, 'pval': np.nan})
            else:
                col = f'gaza_t{"m" if t < 0 else "p"}{abs(t)}'
                if col in model.params.index:
                    results.append({
                        'rel_time': t,
                        'coef': model.params[col],
                        'se': model.bse[col],
                        'pval': model.pvalues[col]
                    })
        
        return pd.DataFrame(results), model
    except Exception as e:
        print(f"    Event study failed: {e}")
        return None, None


def create_event_study_plot(results_df, outcome, treatment_name, output_path):
    """Create event study plot showing Gaza differential effect over time."""
    
    if results_df is None or len(results_df) == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_df['ci_low'] = results_df['coef'] - 1.96 * results_df['se']
    results_df['ci_high'] = results_df['coef'] + 1.96 * results_df['se']
    
    # Plot
    ax.plot(results_df['rel_time'], results_df['coef'], 'o-', 
            color='#1f77b4', linewidth=2, markersize=8, label='Gaza × Time')
    ax.fill_between(results_df['rel_time'], results_df['ci_low'], results_df['ci_high'],
                    alpha=0.3, color='#1f77b4')
    
    # Reference lines
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label=f'{treatment_name} start')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Shade pre/post
    ax.axvspan(-6, 0, alpha=0.1, color='green', label='Pre-treatment')
    ax.axvspan(0, 12, alpha=0.1, color='red', label='Post-treatment')
    
    ax.set_xlabel('Months Relative to Treatment', fontsize=12)
    ax.set_ylabel(f'Differential Effect (Gaza vs West Bank)', fontsize=12)
    ax.set_title(f'Event Study: {outcome}\nGaza × Time Interactions (DiD Coefficients)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation note
    ax.text(0.02, 0.98, 'Pre-treatment coefficients ≈ 0 → Parallel trends hold\nPost-treatment ≠ 0 → Differential Gaza effect',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_comparison_plot(gaza_model, wb_model, outcome, treatment_name, output_path):
    """Create side-by-side comparison of Gaza effect vs West Bank placebo."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract coefficients
    labels = []
    coefs = []
    errors = []
    colors = []
    
    # Gaza effect
    if gaza_model is not None:
        interaction_var = f'gaza_x_post_{treatment_name}'
        if interaction_var in gaza_model.params.index:
            labels.append(f'Gaza × Post\n(Actual Treatment)')
            coefs.append(gaza_model.params[interaction_var])
            errors.append(1.96 * gaza_model.bse[interaction_var])
            pval = gaza_model.pvalues[interaction_var]
            colors.append('#2ca02c' if pval < 0.05 else ('#ff7f0e' if pval < 0.1 else '#1f77b4'))
    
    # West Bank placebo
    if wb_model is not None:
        if 'wb_x_post' in wb_model.params.index:
            labels.append(f'West Bank × Post\n(Placebo)')
            coefs.append(wb_model.params['wb_x_post'])
            errors.append(1.96 * wb_model.bse['wb_x_post'])
            pval = wb_model.pvalues['wb_x_post']
            colors.append('#d62728' if pval < 0.05 else ('#ff7f0e' if pval < 0.1 else '#7f7f7f'))
    
    if not coefs:
        return False
    
    y_pos = np.arange(len(coefs))
    ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.7, capsize=8, height=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('DiD Coefficient (Treatment Effect)', fontsize=12)
    ax.set_title(f'{outcome}: Gaza Effect vs West Bank Placebo\n({treatment_name})', 
                 fontsize=13, fontweight='bold')
    
    # Add interpretation
    ax.text(0.98, 0.02, 
            'Green = p<0.05, Orange = p<0.10, Blue/Gray = n.s.\nRed bar = Placebo should be ~0 if model is valid',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    """Run DiD analysis with robustness checks."""
    
    print("\n" + "=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES WITH ROBUSTNESS CHECKS")
    print("=" * 70)
    
    # Load data
    panel, treatment_dates = load_data()
    
    # Get outcomes
    outcomes = get_outcomes(panel)
    
    # Controls
    controls = []
    if 'rainfall' in panel.columns and panel['rainfall'].notna().sum() > 100:
        controls.append('rainfall')
    
    # Store results
    all_results = []
    
    # ==========================================================================
    # ANALYSIS 1: DiD WITH INTERACTION TERMS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: DiD WITH INTERACTION TERMS")
    print("=" * 70)
    print("Model: Y = α + β₁·Gaza + β₂·Post + β₃·(Gaza×Post) + FE + ε")
    print("β₃ = Differential change in Gaza after UNDP intervention\n")
    
    for outcome in outcomes:
        print(f"\n--- {outcome} ---")
        
        for treat_name in treatment_dates.keys():
            model = run_did_with_interactions(panel, outcome, treat_name, controls)
            
            if model:
                interaction_var = f'gaza_x_post_{treat_name}'
                if interaction_var in model.params.index:
                    coef = model.params[interaction_var]
                    se = model.bse[interaction_var]
                    pval = model.pvalues[interaction_var]
                    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                    
                    print(f"  {treat_name}: β₃ = {coef:.4f} (SE: {se:.4f}) {sig}")
                    
                    all_results.append({
                        'Analysis': 'DiD_Interaction',
                        'Outcome': outcome,
                        'Treatment': treat_name,
                        'Coefficient': coef,
                        'Std_Error': se,
                        'p_value': pval,
                        'N': int(model.nobs),
                        'R2': model.rsquared
                    })
    
    # ==========================================================================
    # ANALYSIS 2: WEST BANK PLACEBO TEST
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: WEST BANK PLACEBO TEST")
    print("=" * 70)
    print("Question: If we pretend West Bank was 'treated', do we find effects?")
    print("Expected: NO significant effects (WB wasn't treated by UNDP Gaza projects)\n")
    
    placebo_results = []
    
    for outcome in outcomes:
        print(f"\n--- {outcome} ---")
        
        for treat_name in treatment_dates.keys():
            # Gaza (actual)
            gaza_model = run_did_with_interactions(panel, outcome, treat_name, controls)
            
            # West Bank (placebo)
            wb_model = run_westbank_placebo(panel, outcome, treat_name, controls)
            
            if gaza_model and wb_model:
                interaction_var = f'gaza_x_post_{treat_name}'
                
                gaza_coef = gaza_model.params.get(interaction_var, np.nan)
                gaza_pval = gaza_model.pvalues.get(interaction_var, np.nan)
                
                wb_coef = wb_model.params.get('wb_x_post', np.nan)
                wb_pval = wb_model.pvalues.get('wb_x_post', np.nan)
                
                gaza_sig = '***' if gaza_pval < 0.01 else ('**' if gaza_pval < 0.05 else ('*' if gaza_pval < 0.1 else ''))
                wb_sig = '***' if wb_pval < 0.01 else ('**' if wb_pval < 0.05 else ('*' if wb_pval < 0.1 else ''))
                
                print(f"  {treat_name}:")
                print(f"    Gaza (actual):      β = {gaza_coef:.4f} {gaza_sig}")
                print(f"    West Bank (placebo): β = {wb_coef:.4f} {wb_sig}")
                
                # Interpretation
                if gaza_pval < 0.1 and wb_pval >= 0.1:
                    print(f"    → ✓ VALID: Effect in Gaza, no false positive in WB")
                elif gaza_pval < 0.1 and wb_pval < 0.1:
                    print(f"    → ⚠ CAUTION: Effects in both - possible regional trend or spillover")
                elif gaza_pval >= 0.1 and wb_pval < 0.1:
                    print(f"    → ❌ PROBLEM: No Gaza effect but WB 'effect' - model misspecified?")
                else:
                    print(f"    → No significant effects detected")
                
                placebo_results.append({
                    'Outcome': outcome,
                    'Treatment': treat_name,
                    'Gaza_Coef': gaza_coef,
                    'Gaza_pval': gaza_pval,
                    'WB_Placebo_Coef': wb_coef,
                    'WB_Placebo_pval': wb_pval,
                    'Valid': 'Yes' if (gaza_pval < 0.1 and wb_pval >= 0.1) else 'No'
                })
            
            # Create comparison plot for key outcomes
            if outcome in ['nightlights', 'ndbi', 'ndvi', 'vh'] and treat_name == 'any_undp':
                create_comparison_plot(
                    gaza_model, wb_model, outcome, treat_name,
                    FIGURES_DIR / f'placebo_comparison_{outcome}.png'
                )
                print(f"    ✓ placebo_comparison_{outcome}.png")
    
    # ==========================================================================
    # ANALYSIS 3: PARALLEL TRENDS TEST
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PARALLEL TRENDS TEST")
    print("=" * 70)
    print("Question: Were Gaza and WB on similar trajectories BEFORE treatment?")
    print("Test: Gaza × Time interaction in pre-treatment period")
    print("Expected: Coefficient ≈ 0 (not significantly different)\n")
    
    trends_results = []
    
    for outcome in outcomes:
        for treat_name, treat_date in treatment_dates.items():
            coef, pval = test_parallel_trends(panel, outcome, treat_date, controls)
            
            if coef is not None:
                sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                
                if pval < 0.1:
                    status = "⚠ VIOLATED"
                else:
                    status = "✓ Holds"
                
                print(f"  {outcome} ({treat_name}): Gaza×Time = {coef:.4f} {sig} - {status}")
                
                trends_results.append({
                    'Outcome': outcome,
                    'Treatment': treat_name,
                    'PreTrend_Coef': coef,
                    'PreTrend_pval': pval,
                    'Parallel_Trends': 'Holds' if pval >= 0.1 else 'Violated'
                })
    
    # ==========================================================================
    # ANALYSIS 4: EVENT STUDY WITH INTERACTIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: EVENT STUDY (Gaza × Time Interactions)")
    print("=" * 70)
    print("Shows differential Gaza effect at each time point\n")
    
    key_outcomes = ['nightlights', 'ndbi', 'ndvi', 'vh', 'bsi']
    key_outcomes = [o for o in key_outcomes if o in outcomes]
    
    for outcome in key_outcomes:
        print(f"\n--- {outcome} ---")
        
        for treat_name, treat_date in treatment_dates.items():
            if treat_name != 'any_undp':  # Focus on main treatment
                continue
            
            es_df, es_model = run_event_study_interactions(panel, outcome, treat_date, controls)
            
            if es_df is not None:
                # Check pre-trend coefficients
                pre_coefs = es_df[es_df['rel_time'] < 0]['coef'].values
                post_coefs = es_df[es_df['rel_time'] > 0]['coef'].values
                
                pre_mean = np.mean(np.abs(pre_coefs)) if len(pre_coefs) > 0 else 0
                post_mean = np.mean(np.abs(post_coefs)) if len(post_coefs) > 0 else 0
                
                print(f"  {treat_name}:")
                print(f"    Pre-treatment avg |coef|: {pre_mean:.4f}")
                print(f"    Post-treatment avg |coef|: {post_mean:.4f}")
                
                if post_mean > pre_mean * 2:
                    print(f"    → Effect emerges post-treatment ✓")
                
                # Create plot
                create_event_study_plot(
                    es_df, outcome, treat_name,
                    FIGURES_DIR / f'event_study_did_{outcome}.png'
                )
                print(f"    ✓ event_study_did_{outcome}.png")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Main DiD results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(TABLES_DIR / 'did_interaction_results.csv', index=False)
        print(f"  ✓ did_interaction_results.csv")
    
    # Placebo results
    if placebo_results:
        placebo_df = pd.DataFrame(placebo_results)
        placebo_df.to_csv(TABLES_DIR / 'placebo_test_results.csv', index=False)
        print(f"  ✓ placebo_test_results.csv")
    
    # Parallel trends
    if trends_results:
        trends_df = pd.DataFrame(trends_results)
        trends_df.to_csv(TABLES_DIR / 'parallel_trends_test.csv', index=False)
        print(f"  ✓ parallel_trends_test.csv")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        sig_results = results_df[results_df['p_value'] < 0.1]
        
        print(f"\nSignificant DiD effects (p < 0.10): {len(sig_results)}/{len(results_df)}")
        
        if len(sig_results) > 0:
            print("\nKey findings:")
            for _, row in sig_results.head(10).iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '*')
                direction = "↑" if row['Coefficient'] > 0 else "↓"
                print(f"  {row['Outcome']} + {row['Treatment']}: {direction} {row['Coefficient']:.3f}{sig}")
    
    if placebo_results:
        placebo_df = pd.DataFrame(placebo_results)
        valid_count = (placebo_df['Valid'] == 'Yes').sum()
        print(f"\nPlacebo test validity: {valid_count}/{len(placebo_df)} specifications pass")
        
        # Flag problematic outcomes
        problematic = placebo_df[placebo_df['Valid'] == 'No']['Outcome'].unique()
        if len(problematic) > 0:
            print(f"  ⚠ Outcomes with WB placebo effects: {list(problematic)}")
    
    if trends_results:
        trends_df = pd.DataFrame(trends_results)
        holds_count = (trends_df['Parallel_Trends'] == 'Holds').sum()
        print(f"\nParallel trends: {holds_count}/{len(trends_df)} specifications hold")
        
        violated = trends_df[trends_df['Parallel_Trends'] == 'Violated']['Outcome'].unique()
        if len(violated) > 0:
            print(f"  ⚠ Outcomes with violated pre-trends: {list(violated)}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
    WHAT MAKES A CREDIBLE DiD ESTIMATE:
    
    1. Significant Gaza × Post effect (β₃ ≠ 0)
    2. NO West Bank placebo effect (model isn't picking up noise)
    3. Parallel pre-trends (Gaza and WB were similar before)
    4. Effect emerges AT treatment time (not before)
    
    IF ALL FOUR HOLD:
    → "Outcomes changed differentially in Gaza around UNDP timing"
    → Still cannot prove UNDP CAUSED it (other agencies, etc.)
    
    IF PLACEBO TEST FAILS:
    → Regional trend affecting both Gaza and WB
    → Model may be misspecified
    → Results should be interpreted with caution
    
    IF PRE-TRENDS VIOLATED:
    → Gaza and WB were already diverging before treatment
    → DiD assumption violated
    → Consider synthetic control instead
    """)
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
