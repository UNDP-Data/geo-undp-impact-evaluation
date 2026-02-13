#!/usr/bin/env python3
"""
analysis/pse/01_did_analysis.py
===============================
Difference-in-Differences Analysis for Gaza Impact Evaluation

Design:
- Treatment: Gaza Strip (PS02* districts)
- Control: West Bank (PS01* districts)  
- Event 1: Conflict start (October 7, 2023)
- Event 2: UNDP intervention (June 2025)

Methodology:
- Two-Way Fixed Effects (TWFE) with district and time FE
- Multiple specifications: levels, logs, with/without controls
- Clustered standard errors at district level
- Interaction terms for heterogeneous effects

Outputs:
- Regression results CSV (all specifications)
- Publication-ready tables (like Myanmar format)
- Balance table (descriptive statistics)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels.panel import PanelOLS
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/pse")
OUTPUT_DIR = Path("outputs/pse")
TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
UNDP_DATE = pd.Timestamp("2025-06-01")

# Outcomes to analyze (matching Myanmar format)
OUTCOMES_CONFIG = {
    'nightlights': {'label': 'Nightlights (Radiance)', 'log': True},
    'ndvi': {'label': 'Vegetation (NDVI)', 'log': False},
    'ndbi': {'label': 'Built-up Index (NDBI)', 'log': False},
    'rvi': {'label': 'Radar Vegetation (RVI)', 'log': False},
    'conflict_events': {'label': 'Conflict Events', 'log': True},
    'rainfall': {'label': 'Rainfall (mm)', 'log': False},
    'price_bread': {'label': 'Bread Price (ILS)', 'log': True},
    'vh': {'label': 'Structure Proxy (VH dB)', 'log': False},
    'vv': {'label': 'Flood Proxy (VV dB)', 'log': False},
}

# Control variables
CONTROLS_RAIN = ['rainfall']
CONTROLS_FULL = ['rainfall', 'conflict_events']

# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel_data():
    """Load and prepare panel data for analysis."""
    
    print("=" * 70)
    print("GAZA DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)
    
    # Load panel
    panel_path = OUTPUT_DIR / "panel_data.csv"
    if not panel_path.exists():
        print(f"✗ Panel data not found: {panel_path}")
        print("  Run: make preprocess COUNTRY=gaza")
        return None
    
    df = pd.read_csv(panel_path)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    print(f"\nLoaded panel: {len(df)} observations")
    print(f"  Districts: {df['unit_id'].nunique()}")
    print(f"  Period: {df['period_start'].min().date()} to {df['period_start'].max().date()}")
    
    # Define treatment (Gaza = 1, West Bank = 0)
    df['treated'] = df['unit_id'].str.startswith('PS02').astype(int)
    df['group'] = df['treated'].map({1: 'Gaza', 0: 'West Bank'})
    
    # Define post periods
    df['post_conflict'] = (df['period_start'] >= CONFLICT_DATE).astype(int)
    df['post_undp'] = (df['period_start'] >= UNDP_DATE).astype(int)
    
    # Interaction terms
    df['treated_x_post'] = df['treated'] * df['post_conflict']
    df['treated_x_undp'] = df['treated'] * df['post_undp']
    
    # Triple difference: UNDP effect within conflict period
    # This captures: did UNDP arrival change trajectory for Gaza relative to WB?
    df['treated_x_post_x_undp'] = df['treated'] * df['post_conflict'] * df['post_undp']
    
    # Log transformations
    for outcome, config in OUTCOMES_CONFIG.items():
        if outcome in df.columns and config.get('log', False):
            df[f'log_{outcome}'] = np.log1p(df[outcome].clip(lower=0))
    
    # Summary
    n_gaza = df[df['treated'] == 1]['unit_id'].nunique()
    n_wb = df[df['treated'] == 0]['unit_id'].nunique()
    
    print(f"\n  Gaza (treatment): {n_gaza} districts")
    print(f"  West Bank (control): {n_wb} districts")
    print(f"\n  Pre-conflict obs: {(df['post_conflict'] == 0).sum()}")
    print(f"  Post-conflict (pre-UNDP): {((df['post_conflict'] == 1) & (df['post_undp'] == 0)).sum()}")
    print(f"  Post-UNDP obs: {(df['post_undp'] == 1).sum()}")
    
    return df


# =============================================================================
# BALANCE TABLE (DESCRIPTIVE STATISTICS)
# =============================================================================

def create_balance_table(df):
    """Create balance table comparing Gaza vs West Bank, pre/post conflict."""
    
    print("\n" + "=" * 70)
    print("CREATING BALANCE TABLE")
    print("=" * 70)
    
    rows = []
    
    # Header row
    rows.append({
        'Variable': 'PANEL A: GAZA VS WEST BANK',
        'Period': '',
        'Gaza Mean': '', 'Gaza SD': '',
        'West Bank Mean': '', 'West Bank SD': '',
        'Diff': '',
        'Obs Gaza': '', 'Obs WB': ''
    })
    
    for outcome, config in OUTCOMES_CONFIG.items():
        if outcome not in df.columns:
            continue
            
        label = config['label']
        
        # Pre-conflict
        pre = df[df['post_conflict'] == 0]
        gaza_pre = pre[pre['treated'] == 1][outcome].dropna()
        wb_pre = pre[pre['treated'] == 0][outcome].dropna()
        
        if len(gaza_pre) > 0 and len(wb_pre) > 0:
            rows.append({
                'Variable': label,
                'Period': 'Pre-Conflict',
                'Gaza Mean': f"{gaza_pre.mean():.2f}",
                'Gaza SD': f"({gaza_pre.std():.3f})",
                'West Bank Mean': f"{wb_pre.mean():.2f}",
                'West Bank SD': f"({wb_pre.std():.3f})",
                'Diff': f"{gaza_pre.mean() - wb_pre.mean():.2f}",
                'Obs Gaza': str(len(gaza_pre)),
                'Obs WB': str(len(wb_pre))
            })
        
        # Post-conflict
        post = df[df['post_conflict'] == 1]
        gaza_post = post[post['treated'] == 1][outcome].dropna()
        wb_post = post[post['treated'] == 0][outcome].dropna()
        
        if len(gaza_post) > 0 and len(wb_post) > 0:
            rows.append({
                'Variable': '',
                'Period': 'Post-Conflict',
                'Gaza Mean': f"{gaza_post.mean():.2f}",
                'Gaza SD': f"({gaza_post.std():.3f})",
                'West Bank Mean': f"{wb_post.mean():.2f}",
                'West Bank SD': f"({wb_post.std():.3f})",
                'Diff': f"{gaza_post.mean() - wb_post.mean():.2f}",
                'Obs Gaza': str(len(gaza_post)),
                'Obs WB': str(len(wb_post))
            })
    
    balance_df = pd.DataFrame(rows)
    return balance_df


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_twfe_regression(df, outcome, treatment_var='treated_x_post', controls=None, 
                        use_log=False, sample_name="Full Sample"):
    """
    Run TWFE regression with entity and time fixed effects.
    
    Returns dict with results or None if failed.
    """
    
    # Determine outcome variable
    outcome_var = f'log_{outcome}' if use_log and f'log_{outcome}' in df.columns else outcome
    
    if outcome_var not in df.columns:
        return None
    
    # Prepare data
    cols_needed = [outcome_var, treatment_var]
    if controls:
        cols_needed.extend([c for c in controls if c in df.columns and c != outcome])
    
    model_data = df[['unit_id', 'period_start'] + cols_needed].dropna()
    
    if len(model_data) < 50:
        return None
    
    if model_data[outcome_var].std() < 1e-10:
        return None
    
    # Set panel index
    model_data = model_data.set_index(['unit_id', 'period_start'])
    
    # Build formula
    valid_controls = []
    if controls:
        for c in controls:
            if c in model_data.columns and c != outcome and model_data[c].std() > 1e-10:
                valid_controls.append(c)
    
    formula = f"{outcome_var} ~ {treatment_var} + EntityEffects + TimeEffects"
    if valid_controls:
        formula += " + " + " + ".join(valid_controls)
    
    try:
        mod = PanelOLS.from_formula(formula, data=model_data, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coef = res.params[treatment_var]
        se = res.std_errors[treatment_var]
        pval = res.pvalues[treatment_var]
        tstat = res.tstats[treatment_var]
        
        # Significance stars
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        
        result = {
            'Outcome': outcome,
            'Outcome_Label': OUTCOMES_CONFIG.get(outcome, {}).get('label', outcome),
            'Specification': 'Log' if use_log else 'Levels',
            'Sample': sample_name,
            'Treatment_Var': treatment_var,
            'Controls': ', '.join(valid_controls) if valid_controls else 'None',
            'Coefficient': coef,
            'Coeff_Display': f"{coef:.4f}{stars}",
            'SE': se,
            'SE_Display': f"({se:.4f})",
            'T-stat': tstat,
            'P-value': pval,
            'N': int(res.nobs),
            'R2': res.rsquared,
            'N_Units': model_data.index.get_level_values(0).nunique(),
            'N_Periods': model_data.index.get_level_values(1).nunique()
        }
        
        return result
        
    except Exception as e:
        return None


def run_all_regressions(df):
    """Run full suite of TWFE regressions matching Myanmar format."""
    
    results = []
    
    print("\n" + "=" * 70)
    print("RUNNING TWFE REGRESSIONS")
    print("=" * 70)
    
    # ==========================================================================
    # PART 1: CONFLICT IMPACT (Oct 2023)
    # ==========================================================================
    print("\n--- Part 1: Conflict Impact (Treated × Post) ---")
    
    for outcome in OUTCOMES_CONFIG.keys():
        if outcome not in df.columns:
            continue
            
        config = OUTCOMES_CONFIG[outcome]
        use_log = config.get('log', False)
        
        print(f"\n  {outcome}:")
        
        # Specification 1: Levels, no controls
        res = run_twfe_regression(df, outcome, 'treated_x_post', 
                                  controls=None, use_log=False, sample_name='No Controls')
        if res:
            results.append(res)
            print(f"    Levels (raw): {res['Coeff_Display']}")
        
        # Specification 2: Levels, rainfall control
        if outcome != 'rainfall':
            res = run_twfe_regression(df, outcome, 'treated_x_post',
                                      controls=CONTROLS_RAIN, use_log=False, sample_name='+ Rainfall')
            if res:
                results.append(res)
                print(f"    Levels (+rain): {res['Coeff_Display']}")
        
        # Specification 3: Log, no controls (if applicable)
        if use_log:
            res = run_twfe_regression(df, outcome, 'treated_x_post',
                                      controls=None, use_log=True, sample_name='No Controls')
            if res:
                results.append(res)
                print(f"    Log (raw): {res['Coeff_Display']}")
            
            # Specification 4: Log, rainfall control
            if outcome != 'rainfall':
                res = run_twfe_regression(df, outcome, 'treated_x_post',
                                          controls=CONTROLS_RAIN, use_log=True, sample_name='+ Rainfall')
                if res:
                    results.append(res)
                    print(f"    Log (+rain): {res['Coeff_Display']}")
    
    # ==========================================================================
    # PART 2: UNDP HETEROGENEOUS EFFECT
    # ==========================================================================
    print("\n--- Part 2: UNDP Intervention Effect (Interaction) ---")
    print("  Testing if UNDP arrival changed trajectory for Gaza")
    
    for outcome in ['nightlights', 'ndvi', 'ndbi', 'price_bread']:
        if outcome not in df.columns:
            continue
            
        config = OUTCOMES_CONFIG[outcome]
        use_log = config.get('log', False)
        
        print(f"\n  {outcome}:")
        
        # UNDP effect: treated_x_undp
        res = run_twfe_regression(df, outcome, 'treated_x_undp',
                                  controls=CONTROLS_RAIN, use_log=use_log, 
                                  sample_name='UNDP Period')
        if res:
            res['Event'] = 'UNDP'
            results.append(res)
            print(f"    UNDP effect: {res['Coeff_Display']}")
        
        # Triple interaction: treated_x_post_x_undp
        # This shows: additional effect of UNDP beyond conflict effect
        res = run_twfe_regression(df, outcome, 'treated_x_post_x_undp',
                                  controls=CONTROLS_RAIN, use_log=use_log,
                                  sample_name='Triple Diff')
        if res:
            res['Event'] = 'UNDP×Conflict'
            results.append(res)
            print(f"    Triple diff: {res['Coeff_Display']}")
    
    return pd.DataFrame(results)


# =============================================================================
# PUBLICATION TABLES
# =============================================================================

def create_twfe_table(results_df, outcome, title):
    """Create publication-ready TWFE table for one outcome (like Myanmar Table 2)."""
    
    sub = results_df[
        (results_df['Outcome'] == outcome) & 
        (results_df['Treatment_Var'] == 'treated_x_post')
    ].copy()
    
    if sub.empty:
        return None
    
    # Build table structure
    rows = []
    
    # Get specifications
    specs = []
    for _, r in sub.iterrows():
        spec_label = f"{r['Specification']}\n({r['Sample']})"
        if spec_label not in [s[0] for s in specs]:
            specs.append((spec_label, r))
    
    # Header
    header = [''] + [s[0] for s in specs]
    rows.append(header)
    
    # Column numbers
    col_nums = [''] + [f"({i+1})" for i in range(len(specs))]
    rows.append(col_nums)
    
    # Coefficient row
    coef_row = ['Treated × Post'] + [s[1]['Coeff_Display'] for s in specs]
    rows.append(coef_row)
    
    # SE row
    se_row = [''] + [s[1]['SE_Display'] for s in specs]
    rows.append(se_row)
    
    # Controls
    rain_row = ['Rainfall Control'] + ['Yes' if 'rainfall' in str(s[1]['Controls']).lower() else 'No' for s in specs]
    rows.append(rain_row)
    
    # Fixed effects
    rows.append(['District FE'] + ['Yes'] * len(specs))
    rows.append(['Time FE'] + ['Yes'] * len(specs))
    
    # Observations
    obs_row = ['Observations'] + [f"{int(s[1]['N']):,}" for s in specs]
    rows.append(obs_row)
    
    # R-squared
    r2_row = ['R²'] + [f"{s[1]['R2']:.2f}" for s in specs]
    rows.append(r2_row)
    
    table_df = pd.DataFrame(rows)
    return table_df


# =============================================================================
# BREAD PRICE PREDICTION (SYNTHETIC CONTROL IDEA)
# =============================================================================

def bread_price_counterfactual(df):
    """
    Estimate counterfactual bread prices using nightlights-price relationship.
    
    Idea: Use pre-conflict correlation between nightlights and bread prices
    to predict what prices would have been without conflict.
    """
    
    print("\n" + "=" * 70)
    print("BREAD PRICE COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    
    if 'price_bread' not in df.columns or 'nightlights' not in df.columns:
        print("  ✗ Missing price_bread or nightlights columns")
        return None, None
    
    # Focus on Gaza only
    gaza = df[df['treated'] == 1].copy()
    
    # Pre-conflict period for training
    pre = gaza[gaza['post_conflict'] == 0][['nightlights', 'price_bread', 'period_start']].dropna()
    
    if len(pre) < 10:
        print("  ✗ Insufficient pre-conflict data")
        return None, None
    
    # Fit relationship: price = a + b * nightlights
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        pre['nightlights'], pre['price_bread']
    )
    
    print(f"\n  Pre-conflict relationship (Gaza):")
    print(f"    Price = {intercept:.2f} + {slope:.4f} × Nightlights")
    print(f"    R² = {r_value**2:.3f}")
    print(f"    P-value = {p_value:.4f}")
    
    if r_value**2 < 0.1:
        print("  ⚠ Weak relationship - counterfactual may not be reliable")
    
    # Predict counterfactual for post-conflict period
    post = gaza[gaza['post_conflict'] == 1].copy()
    
    # What nightlights WOULD have been (use pre-conflict mean as proxy)
    pre_nl_mean = pre['nightlights'].mean()
    
    # Counterfactual price (if nightlights had stayed at pre-conflict level)
    post['price_counterfactual'] = intercept + slope * pre_nl_mean
    
    # Actual observed price
    post_prices = post.groupby('period_start').agg({
        'price_bread': 'mean',
        'nightlights': 'mean',
        'price_counterfactual': 'mean'
    }).reset_index()
    
    # Price increase due to conflict
    post_prices['price_increase'] = post_prices['price_bread'] - post_prices['price_counterfactual']
    post_prices['price_increase_pct'] = (post_prices['price_increase'] / post_prices['price_counterfactual']) * 100
    
    print(f"\n  Post-conflict comparison:")
    print(f"    Avg observed price: {post_prices['price_bread'].mean():.2f} ILS")
    print(f"    Avg counterfactual: {post_prices['price_counterfactual'].mean():.2f} ILS")
    print(f"    Avg price increase: {post_prices['price_increase'].mean():.2f} ILS ({post_prices['price_increase_pct'].mean():.1f}%)")
    
    # Save results
    results = {
        'pre_conflict_intercept': intercept,
        'pre_conflict_slope': slope,
        'pre_conflict_r2': r_value**2,
        'pre_conflict_nl_mean': pre_nl_mean,
        'post_conflict_price_mean': post_prices['price_bread'].mean(),
        'counterfactual_price_mean': post_prices['price_counterfactual'].mean(),
        'price_increase_mean': post_prices['price_increase'].mean(),
        'price_increase_pct': post_prices['price_increase_pct'].mean()
    }
    
    # Time series of counterfactual
    counterfactual_ts = post_prices[['period_start', 'price_bread', 'price_counterfactual', 'price_increase']]
    
    return results, counterfactual_ts


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    df = load_panel_data()
    if df is None:
        return
    
    # ==========================================================================
    # 1. BALANCE TABLE
    # ==========================================================================
    balance_table = create_balance_table(df)
    balance_path = TABLES_DIR / "table_balance.csv"
    balance_table.to_csv(balance_path, index=False)
    print(f"\n✓ Balance table saved: {balance_path}")
    
    # ==========================================================================
    # 2. TWFE REGRESSIONS
    # ==========================================================================
    results = run_all_regressions(df)
    
    if results.empty:
        print("\n✗ No regression results generated")
        return
    
    # Save full results
    results_path = OUTPUT_DIR / "regression_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\n✓ Full results saved: {results_path}")
    
    # ==========================================================================
    # 3. PUBLICATION TABLES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREATING PUBLICATION TABLES")
    print("=" * 70)
    
    for outcome in ['nightlights', 'ndvi', 'conflict_events', 'price_bread']:
        table = create_twfe_table(results, outcome, f"Impact on {outcome}")
        if table is not None:
            table_path = TABLES_DIR / f"table_twfe_{outcome}.csv"
            table.to_csv(table_path, index=False, header=False)
            print(f"  ✓ {table_path.name}")
    
    # Combined conflict impact table
    conflict_results = results[results['Treatment_Var'] == 'treated_x_post'].copy()
    conflict_path = TABLES_DIR / "table_conflict_impact.csv"
    conflict_results.to_csv(conflict_path, index=False)
    print(f"  ✓ table_conflict_impact.csv")
    
    # UNDP impact table
    undp_results = results[results['Treatment_Var'].str.contains('undp')].copy()
    if not undp_results.empty:
        undp_path = TABLES_DIR / "table_undp_impact.csv"
        undp_results.to_csv(undp_path, index=False)
        print(f"  ✓ table_undp_impact.csv")
    
    # ==========================================================================
    # 4. BREAD PRICE COUNTERFACTUAL
    # ==========================================================================
    counterfactual_results, counterfactual_ts = bread_price_counterfactual(df)
    
    if counterfactual_ts is not None:
        cf_path = TABLES_DIR / "bread_price_counterfactual.csv"
        counterfactual_ts.to_csv(cf_path, index=False)
        print(f"\n✓ Bread price counterfactual: {cf_path}")
    
    # ==========================================================================
    # 5. SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    
    # Conflict impact (levels, no controls)
    conflict_raw = results[
        (results['Treatment_Var'] == 'treated_x_post') &
        (results['Specification'] == 'Levels') &
        (results['Sample'] == 'No Controls')
    ]
    
    print("\n--- Conflict Impact (Oct 2023) ---")
    print("    Outcome                  Coeff        SE       p-value")
    print("    " + "-" * 55)
    for _, row in conflict_raw.iterrows():
        sig = "***" if row['P-value'] < 0.01 else "**" if row['P-value'] < 0.05 else "*" if row['P-value'] < 0.1 else ""
        print(f"    {row['Outcome']:20s}  {row['Coefficient']:+10.3f}  {row['SE']:8.3f}  {row['P-value']:8.4f} {sig}")
    
    # UNDP impact
    undp_raw = results[results['Treatment_Var'] == 'treated_x_undp']
    if not undp_raw.empty:
        print("\n--- UNDP Intervention Effect (June 2025) ---")
        print("    Outcome                  Coeff        SE       p-value")
        print("    " + "-" * 55)
        for _, row in undp_raw.iterrows():
            sig = "***" if row['P-value'] < 0.01 else "**" if row['P-value'] < 0.05 else "*" if row['P-value'] < 0.1 else ""
            print(f"    {row['Outcome']:20s}  {row['Coefficient']:+10.3f}  {row['SE']:8.3f}  {row['P-value']:8.4f} {sig}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"  Tables saved to: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
