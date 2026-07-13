#!/usr/bin/env python3
"""
analysis/pse/02_impact_analysis.py
==================================
Comprehensive econometric analysis of UNDP interventions in Gaza.

FIXED VERSION:
- Variable names sanitized for statsmodels formula parser
- Period converted to string for categorical FE
- Event study uses 'tm6' instead of 'rel_-6' to avoid parser issues

Usage:
    python analysis/pse/02_impact_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import re
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
UNDP_FILE = DATA_DIR / "external" / "undp_projects.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")

# =============================================================================
# INTERVENTION CATEGORIES
# =============================================================================
RECOVERY_CATEGORIES = [
    'Debris & Solid Waste',
    'Infrastructure & Housing', 
    'Livelihoods & Economy',
    'Energy & Power',
]

HUMANITARIAN_CATEGORIES = [
    'Health & Emergency',
    'Water & Sanitation (WASH)',
    'Education',
]

GOVERNANCE_CATEGORIES = [
    'Governance & Rule of Law',
    'Social Inclusion',
]

# Keywords for granular classification
DEBRIS_KEYWORDS = ['debris', 'rubble', 'solid waste', 'clearance', 'removal', 'demolition']
CASH_KEYWORDS = ['cash', 'transfer', 'grant', 'livelihood', 'employment', 'job', 'income']
RECONSTRUCTION_KEYWORDS = ['reconstruct', 'rebuild', 'shelter', 'housing', 'infrastructure']


def sanitize_column_name(name):
    """
    Sanitize column names for statsmodels formula compatibility.
    Replaces special characters with underscores.
    """
    # Replace problematic characters
    name = re.sub(r'[/\(\)&\s\-]+', '_', str(name))
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Replace multiple underscores with single
    name = re.sub(r'_+', '_', name)
    # Make lowercase
    name = name.lower()
    return name


def load_and_prepare_data():
    """Load panel data and UNDP projects, create treatment variables."""
    
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load panel
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    # Identify Gaza vs West Bank
    panel['is_gaza'] = panel['unit_id'].str.startswith('PS02')
    panel['region'] = panel['is_gaza'].map({True: 'Gaza', False: 'West Bank'})
    
    # Time variables - FIXED: Convert to string for formula compatibility
    panel['year'] = panel['period_start'].dt.year
    panel['month'] = panel['period_start'].dt.month
    panel['year_month_str'] = panel['period_start'].dt.strftime('%Y_%m')  # String format
    panel['time_idx'] = (panel['period_start'].dt.year - 2023) * 12 + panel['period_start'].dt.month
    
    # Post-conflict indicator
    panel['post_conflict'] = (panel['period_start'] >= CONFLICT_DATE).astype(int)
    
    # Months since conflict
    panel['months_since_conflict'] = (
        (panel['period_start'].dt.year - CONFLICT_DATE.year) * 12 + 
        (panel['period_start'].dt.month - CONFLICT_DATE.month)
    )
    
    print(f"\nPanel: {len(panel)} observations")
    print(f"  Units: {panel['unit_id'].nunique()} ({panel[panel['is_gaza']]['unit_id'].nunique()} Gaza, "
          f"{panel[~panel['is_gaza']]['unit_id'].nunique()} West Bank)")
    print(f"  Period: {panel['period_start'].min().strftime('%Y-%m')} to {panel['period_start'].max().strftime('%Y-%m')}")
    
    # Load UNDP projects
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'end_date' in undp.columns:
            undp['end_date'] = pd.to_datetime(undp['end_date'], errors='coerce')
        
        # Filter to post-conflict projects
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE].copy()
        
        print(f"\nUNDP Projects (post-conflict): {len(undp)}")
        if 'ml_category' in undp.columns:
            print(f"  Categories: {undp['ml_category'].value_counts().to_dict()}")
        
        # Create category flags
        if 'ml_category' in undp.columns:
            undp['is_recovery'] = undp['ml_category'].isin(RECOVERY_CATEGORIES)
            undp['is_humanitarian'] = undp['ml_category'].isin(HUMANITARIAN_CATEGORIES)
            undp['is_governance'] = undp['ml_category'].isin(GOVERNANCE_CATEGORIES)
        
        # Create keyword-based flags
        desc_col = 'full_description_text' if 'full_description_text' in undp.columns else 'title'
        undp['text'] = (undp['title'].fillna('') + ' ' + undp.get(desc_col, undp['title']).fillna('')).str.lower()
        
        undp['is_debris'] = undp['text'].str.contains('|'.join(DEBRIS_KEYWORDS), regex=True, na=False)
        undp['is_cash'] = undp['text'].str.contains('|'.join(CASH_KEYWORDS), regex=True, na=False)
        undp['is_reconstruction'] = undp['text'].str.contains('|'.join(RECONSTRUCTION_KEYWORDS), regex=True, na=False)
        
        print(f"\n  Recovery projects: {undp['is_recovery'].sum() if 'is_recovery' in undp.columns else 'N/A'}")
        print(f"  Humanitarian projects: {undp['is_humanitarian'].sum() if 'is_humanitarian' in undp.columns else 'N/A'}")
        print(f"  Debris-related: {undp['is_debris'].sum()}")
        print(f"  Cash-related: {undp['is_cash'].sum()}")
        print(f"  Reconstruction: {undp['is_reconstruction'].sum()}")
        
        # Create treatment timing variables
        treatment_dates = {}
        
        if 'ml_category' in undp.columns:
            for category in undp['ml_category'].unique():
                cat_projects = undp[undp['ml_category'] == category]
                if len(cat_projects) > 0:
                    safe_name = sanitize_column_name(category)
                    treatment_dates[safe_name] = cat_projects['start_date'].min()
        
        # Aggregate treatment dates
        if 'is_recovery' in undp.columns and undp['is_recovery'].any():
            treatment_dates['recovery'] = undp[undp['is_recovery']]['start_date'].min()
        if 'is_humanitarian' in undp.columns and undp['is_humanitarian'].any():
            treatment_dates['humanitarian'] = undp[undp['is_humanitarian']]['start_date'].min()
        if undp['is_debris'].any():
            treatment_dates['debris'] = undp[undp['is_debris']]['start_date'].min()
        if undp['is_cash'].any():
            treatment_dates['cash'] = undp[undp['is_cash']]['start_date'].min()
        
        treatment_dates['any_undp'] = undp['start_date'].min()
        
        print(f"\nFirst treatment dates:")
        for name, date in sorted(treatment_dates.items(), key=lambda x: x[1]):
            print(f"  {name}: {date.strftime('%Y-%m-%d')}")
        
        # Create treatment indicators in panel (FIXED: sanitized names)
        for name, treat_date in treatment_dates.items():
            col_name = f'post_{name}'
            panel[col_name] = ((panel['period_start'] >= treat_date) & panel['is_gaza']).astype(int)
        
        # Create cumulative treatment intensity
        panel['undp_projects_cumulative'] = 0
        panel['undp_budget_cumulative'] = 0.0
        
        for idx, row in panel[panel['is_gaza']].iterrows():
            period_end = row['period_start'] + pd.DateOffset(months=1)
            projects_started = undp[undp['start_date'] < period_end]
            panel.loc[idx, 'undp_projects_cumulative'] = len(projects_started)
            if 'total_budget' in undp.columns:
                panel.loc[idx, 'undp_budget_cumulative'] = projects_started['total_budget'].sum()
        
    except FileNotFoundError:
        print(f"\n⚠ UNDP file not found: {UNDP_FILE}")
        undp = pd.DataFrame()
        treatment_dates = {'any_undp': pd.Timestamp('2024-01-01')}
        panel['post_any_undp'] = ((panel['period_start'] >= treatment_dates['any_undp']) & panel['is_gaza']).astype(int)
    
    return panel, undp, treatment_dates


def identify_outcomes_and_controls(panel):
    """Identify which variables are suitable as outcomes vs controls."""
    
    available = panel.columns.tolist()
    
    outcomes = {
        'primary': [],
        'secondary': [],
    }
    
    controls = []
    
    # Primary outcomes
    primary_candidates = ['nightlights', 'ndbi', 'vh']
    for var in primary_candidates:
        if var in available and panel[var].notna().sum() > 50:
            outcomes['primary'].append(var)
    
    # Secondary outcomes
    secondary_candidates = ['ndvi', 'bsi', 'vv', 'rvi']
    for var in secondary_candidates:
        if var in available and panel[var].notna().sum() > 50:
            outcomes['secondary'].append(var)
    
    # Price outcomes
    price_vars = [c for c in available if c.startswith('price_')]
    for var in price_vars:
        if panel[var].notna().sum() > 50:
            outcomes['secondary'].append(var)
    
    # Controls
    if 'rainfall' in available and panel['rainfall'].notna().sum() > 50:
        controls.append('rainfall')
    
    print("\n" + "=" * 70)
    print("VARIABLE CLASSIFICATION")
    print("=" * 70)
    print(f"\nPrimary outcomes: {outcomes['primary']}")
    print(f"Secondary outcomes: {outcomes['secondary']}")
    print(f"Controls: {controls}")
    
    return outcomes, controls


def run_event_study(panel, outcome, treatment_date, treatment_name, controls=None):
    """
    Run event study design.
    FIXED: Use 'tm6' instead of 'rel_-6' to avoid formula parser issues.
    """
    
    panel = panel.copy()
    panel['rel_time'] = (
        (panel['period_start'].dt.year - treatment_date.year) * 12 + 
        (panel['period_start'].dt.month - treatment_date.month)
    )
    
    # Bin extreme values
    panel['rel_time_binned'] = panel['rel_time'].clip(-6, 12)
    
    # Create dummy variables with SAFE names (tm = t minus, tp = t plus)
    for t in range(-6, 13):
        if t == -1:  # Reference period
            continue
        if t < 0:
            col_name = f'tm{abs(t)}'  # tm6 for t=-6
        else:
            col_name = f'tp{t}'  # tp0 for t=0
        panel[col_name] = ((panel['rel_time_binned'] == t) & panel['is_gaza']).astype(int)
    
    # Build formula with safe variable names
    rel_vars = []
    for t in range(-6, 13):
        if t == -1:
            continue
        if t < 0:
            rel_vars.append(f'tm{abs(t)}')
        else:
            rel_vars.append(f'tp{t}')
    
    formula = f'{outcome} ~ ' + ' + '.join(rel_vars)
    formula += ' + C(unit_id) + C(year_month_str)'
    
    if controls:
        formula += ' + ' + ' + '.join(controls)
    
    # Run regression
    try:
        data = panel.dropna(subset=[outcome])
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['unit_id']}
        )
        
        # Extract coefficients and map back to relative time
        results = []
        for t in range(-6, 13):
            if t == -1:
                results.append({'rel_time': t, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
            else:
                if t < 0:
                    var_name = f'tm{abs(t)}'
                else:
                    var_name = f'tp{t}'
                
                if var_name in model.params.index:
                    coef = model.params[var_name]
                    se = model.bse[var_name]
                    results.append({
                        'rel_time': t,
                        'coef': coef,
                        'se': se,
                        'ci_low': coef - 1.96 * se,
                        'ci_high': coef + 1.96 * se
                    })
        
        return pd.DataFrame(results), model
        
    except Exception as e:
        print(f"    Event study failed: {e}")
        return None, None


def run_twfe_did(panel, outcome, treatment_var, controls=None):
    """
    Two-Way Fixed Effects DiD.
    FIXED: Using year_month_str instead of year_month Period.
    """
    
    formula = f'{outcome} ~ {treatment_var} + C(unit_id) + C(year_month_str)'
    
    if controls:
        formula += ' + ' + ' + '.join(controls)
    
    try:
        data = panel.dropna(subset=[outcome])
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['unit_id']}
        )
        return model
        
    except Exception as e:
        print(f"    TWFE failed: {e}")
        return None


def run_interrupted_time_series(panel, outcome, treatment_date, controls=None):
    """Interrupted Time Series for Gaza only."""
    
    gaza = panel[panel['is_gaza']].copy()
    
    # Aggregate to regional level
    agg_dict = {outcome: 'mean', 'time_idx': 'first'}
    if controls:
        for c in controls:
            if c in gaza.columns:
                agg_dict[c] = 'mean'
    
    gaza_agg = gaza.groupby('period_start').agg(agg_dict).reset_index()
    
    # Create ITS variables
    treat_time = (treatment_date.year - 2023) * 12 + treatment_date.month
    gaza_agg['post'] = (gaza_agg['period_start'] >= treatment_date).astype(int)
    gaza_agg['time_since_treatment'] = (gaza_agg['time_idx'] - treat_time).clip(lower=0)
    
    # Build formula
    formula = f'{outcome} ~ time_idx + post + time_since_treatment'
    if controls:
        valid_controls = [c for c in controls if c in gaza_agg.columns]
        if valid_controls:
            formula += ' + ' + ' + '.join(valid_controls)
    
    try:
        model = smf.ols(formula, data=gaza_agg.dropna(subset=[outcome])).fit(cov_type='HC3')
        return model, gaza_agg
        
    except Exception as e:
        print(f"    ITS failed: {e}")
        return None, None


def run_staggered_did(panel, outcome, treatment_vars, controls=None):
    """
    Staggered DiD with multiple treatment timing.
    FIXED: Filter to only safe column names.
    """
    
    # Filter treatment vars to only those that exist and have valid names
    valid_treatment_vars = []
    for var in treatment_vars:
        if var in panel.columns and panel[var].sum() > 0:
            # Check for safe name (no special characters)
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                valid_treatment_vars.append(var)
    
    if not valid_treatment_vars:
        print("    No valid treatment variables found")
        return None
    
    formula = f'{outcome} ~ ' + ' + '.join(valid_treatment_vars)
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
        print(f"    Staggered DiD failed: {e}")
        return None


def create_event_study_plot(results_dict, outcome, output_path):
    """Create event study plot for multiple treatments."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, (name, df) in enumerate(results_dict.items()):
        if df is None or len(df) == 0:
            continue
        
        color = colors[i % len(colors)]
        
        # Plot coefficients
        ax.plot(df['rel_time'], df['coef'], 'o-', color=color, label=name, linewidth=2, markersize=6)
        
        # Confidence intervals
        ax.fill_between(df['rel_time'], df['ci_low'], df['ci_high'], alpha=0.2, color=color)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Treatment')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Months Relative to Treatment', fontsize=12)
    ax.set_ylabel(f'Effect on {outcome}', fontsize=12)
    ax.set_title(f'Event Study: Effect of UNDP Interventions on {outcome}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ {output_path.name}")
    return True


def main():
    """Run full impact analysis."""
    
    print("\n" + "=" * 70)
    print("UNDP GAZA IMPACT EVALUATION")
    print("Econometric Analysis (FIXED)")
    print("=" * 70)
    
    # Load data
    panel, undp, treatment_dates = load_and_prepare_data()
    
    # Identify outcomes and controls
    outcomes, controls = identify_outcomes_and_controls(panel)
    
    all_results = []
    
    # ==========================================================================
    # ANALYSIS 1: TWFE DiD
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: TWO-WAY FIXED EFFECTS DiD")
    print("=" * 70)
    
    twfe_results = {}
    
    for outcome in outcomes['primary']:
        print(f"\n--- Outcome: {outcome} ---")
        
        # Get treatment variables that exist
        treatment_vars = ['post_any_undp', 'post_recovery', 'post_humanitarian', 
                          'post_debris', 'post_cash']
        treatment_vars = [v for v in treatment_vars if v in panel.columns]
        
        for treat_var in treatment_vars:
            model = run_twfe_did(panel, outcome, treat_var, controls)
            
            if model:
                key = f"{outcome}_{treat_var}"
                twfe_results[key] = model
                
                coef = model.params[treat_var]
                se = model.bse[treat_var]
                pval = model.pvalues[treat_var]
                sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                
                print(f"  {treat_var}: β = {coef:.4f} (SE: {se:.4f}) {sig}")
                
                all_results.append({
                    'Analysis': 'TWFE',
                    'Outcome': outcome,
                    'Treatment': treat_var,
                    'Coefficient': coef,
                    'Std_Error': se,
                    'p_value': pval,
                    'N': int(model.nobs),
                    'R2': model.rsquared
                })
    
    # ==========================================================================
    # ANALYSIS 2: EVENT STUDY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: EVENT STUDY BY INTERVENTION TYPE")
    print("=" * 70)
    
    event_study_results = {}
    
    for outcome in outcomes['primary']:
        print(f"\n--- Outcome: {outcome} ---")
        
        outcome_results = {}
        
        for treat_name in ['any_undp', 'recovery', 'humanitarian', 'debris', 'cash']:
            if treat_name in treatment_dates:
                es_df, es_model = run_event_study(
                    panel, outcome, treatment_dates[treat_name], treat_name, controls
                )
                
                if es_df is not None:
                    outcome_results[treat_name] = es_df
                    print(f"  {treat_name}: Event study computed ✓")
        
        if outcome_results:
            event_study_results[outcome] = outcome_results
            
            # Create plot
            create_event_study_plot(
                outcome_results, outcome,
                FIGURES_DIR / f'event_study_{outcome}.png'
            )
    
    # ==========================================================================
    # ANALYSIS 3: INTERRUPTED TIME SERIES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: INTERRUPTED TIME SERIES (Gaza only)")
    print("=" * 70)
    
    its_results = {}
    
    for outcome in outcomes['primary']:
        print(f"\n--- Outcome: {outcome} ---")
        
        for treat_name in ['any_undp', 'recovery']:
            if treat_name in treatment_dates:
                treat_date = treatment_dates[treat_name]
                
                model, data = run_interrupted_time_series(panel, outcome, treat_date, controls)
                
                if model:
                    its_results[f"{outcome}_{treat_name}"] = model
                    
                    level_shift = model.params.get('post', np.nan)
                    trend_change = model.params.get('time_since_treatment', np.nan)
                    
                    pval_level = model.pvalues.get('post', np.nan)
                    pval_trend = model.pvalues.get('time_since_treatment', np.nan)
                    
                    sig_level = '***' if pval_level < 0.01 else ('**' if pval_level < 0.05 else ('*' if pval_level < 0.1 else ''))
                    sig_trend = '***' if pval_trend < 0.01 else ('**' if pval_trend < 0.05 else ('*' if pval_trend < 0.1 else ''))
                    
                    print(f"  {treat_name}:")
                    print(f"    Level shift: {level_shift:.4f} {sig_level}")
                    print(f"    Trend change: {trend_change:.4f} {sig_trend}")
                    
                    all_results.append({
                        'Analysis': 'ITS_level',
                        'Outcome': outcome,
                        'Treatment': treat_name,
                        'Coefficient': level_shift,
                        'Std_Error': model.bse.get('post', np.nan),
                        'p_value': pval_level,
                        'N': int(model.nobs),
                        'R2': model.rsquared
                    })
                    
                    all_results.append({
                        'Analysis': 'ITS_trend',
                        'Outcome': outcome,
                        'Treatment': treat_name,
                        'Coefficient': trend_change,
                        'Std_Error': model.bse.get('time_since_treatment', np.nan),
                        'p_value': pval_trend,
                        'N': int(model.nobs),
                        'R2': model.rsquared
                    })
    
    # ==========================================================================
    # ANALYSIS 4: STAGGERED DiD
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: STAGGERED DiD (Multiple Treatments)")
    print("=" * 70)
    
    # Get safe treatment variables
    safe_treatment_vars = [c for c in panel.columns 
                          if c.startswith('post_') and c != 'post_conflict'
                          and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', c)
                          and panel[c].sum() > 0]
    
    print(f"\nSafe treatment variables: {safe_treatment_vars}")
    
    for outcome in outcomes['primary']:
        print(f"\n--- Outcome: {outcome} ---")
        
        model = run_staggered_did(panel, outcome, safe_treatment_vars, controls)
        
        if model:
            print(f"\n  Joint F-test: F={model.fvalue:.2f}, p={model.f_pvalue:.4f}")
            
            for var in safe_treatment_vars:
                if var in model.params.index:
                    coef = model.params[var]
                    pval = model.pvalues[var]
                    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                    print(f"  {var}: β = {coef:.4f} {sig}")
    
    # ==========================================================================
    # ANALYSIS 5: DOSE-RESPONSE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: DOSE-RESPONSE")
    print("=" * 70)
    
    for outcome in outcomes['primary']:
        print(f"\n--- Outcome: {outcome} ---")
        
        formula = f'{outcome} ~ undp_projects_cumulative + C(unit_id) + C(year_month_str)'
        if controls:
            formula += ' + ' + ' + '.join(controls)
        
        try:
            data = panel.dropna(subset=[outcome])
            model = smf.ols(formula, data=data).fit(
                cov_type='cluster', cov_kwds={'groups': data['unit_id']}
            )
            
            coef = model.params['undp_projects_cumulative']
            pval = model.pvalues['undp_projects_cumulative']
            sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
            
            print(f"  Effect per additional project: {coef:.4f} {sig}")
            
            all_results.append({
                'Analysis': 'Dose_Response',
                'Outcome': outcome,
                'Treatment': 'cumulative_projects',
                'Coefficient': coef,
                'Std_Error': model.bse['undp_projects_cumulative'],
                'p_value': pval,
                'N': int(model.nobs),
                'R2': model.rsquared
            })
            
        except Exception as e:
            print(f"  Dose-response failed: {e}")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(TABLES_DIR / 'impact_analysis_summary.csv', index=False)
        print(f"  ✓ {TABLES_DIR / 'impact_analysis_summary.csv'}")
        
        # Print summary
        print("\n--- SUMMARY OF SIGNIFICANT RESULTS ---")
        sig_results = results_df[results_df['p_value'] < 0.1].copy()
        if len(sig_results) > 0:
            print(sig_results[['Analysis', 'Outcome', 'Treatment', 'Coefficient', 'p_value']].to_string(index=False))
        else:
            print("  No statistically significant results at p < 0.10")
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
