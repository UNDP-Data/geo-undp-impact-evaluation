#!/usr/bin/env python3
"""
analysis/pse/04_comprehensive_analysis.py
==========================================
Comprehensive econometric analysis for UNDP Gaza evaluation.

IMPORTANT: WHAT THIS ANALYSIS CAN AND CANNOT MEASURE
=====================================================

WHAT WE CAN MEASURE:
1. Conflict impact: How much did Gaza deviate from synthetic counterfactual?
2. Recovery trajectory: Are outcomes improving post-conflict?
3. Timing correlation: Do improvements coincide with UNDP project dates?
4. West Bank spillover: Did WB also experience effects?

WHAT WE CANNOT MEASURE:
1. UNDP-specific causal effect (no within-Gaza variation)
2. Separation of UNDP from other agencies (UNRWA, WFP, etc.)
3. Attribution of specific % to specific projects

HONEST FRAMING:
- Results show "association" not "causation"
- UNDP projects are PART of a broader response
- We document correlation with recovery, not proof of causation

Usage:
    python analysis/pse/04_comprehensive_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
RASTER_PANEL_FILE = OUTPUT_DIR / "raster_panel.csv"
UNDP_FILE = DATA_DIR / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")


def load_data():
    """Load all data sources and merge if needed."""
    
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load main panel
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    print(f"\nMain panel: {len(panel)} rows, {len(panel.columns)} columns")
    
    # Check for raster panel (has NDBI, BSI at higher temporal resolution)
    if RASTER_PANEL_FILE.exists():
        raster = pd.read_csv(RASTER_PANEL_FILE)
        print(f"Raster panel: {len(raster)} rows, columns: {raster.columns.tolist()[:10]}...")
        
        # Find the date column (could be 'date', 'year_month', 'period', etc.)
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
            # Parse date
            raster[date_col] = pd.to_datetime(raster[date_col], errors='coerce')
            
            # Merge raster outcomes into main panel if not already there
            for outcome_col in ['ndbi', 'bsi', 'viirs', 'ndvi']:
                if outcome_col in raster.columns and (outcome_col not in panel.columns or panel[outcome_col].isna().all()):
                    # Aggregate raster to monthly by admin unit
                    raster_monthly = raster.groupby([unit_col, pd.Grouper(key=date_col, freq='MS')])[outcome_col].mean().reset_index()
                    raster_monthly.columns = ['unit_id', 'period_start', outcome_col]
                    
                    if outcome_col in panel.columns:
                        panel = panel.drop(columns=[outcome_col])
                    panel = panel.merge(raster_monthly, on=['unit_id', 'period_start'], how='left')
                    print(f"  Added {outcome_col} from raster panel")
        else:
            print(f"  Could not identify date/unit columns in raster panel")
    
    # Identify regions
    panel['is_gaza'] = panel['unit_id'].str.startswith('PS02')
    panel['region'] = panel['is_gaza'].map({True: 'Gaza', False: 'West Bank'})
    
    # Time variables
    panel['year_month_str'] = panel['period_start'].dt.strftime('%Y_%m')
    panel['time_idx'] = (panel['period_start'].dt.year - 2023) * 12 + panel['period_start'].dt.month
    panel['post_conflict'] = (panel['period_start'] >= CONFLICT_DATE).astype(int)
    panel['months_since_conflict'] = (
        (panel['period_start'].dt.year - CONFLICT_DATE.year) * 12 + 
        (panel['period_start'].dt.month - CONFLICT_DATE.month)
    )
    
    # Load UNDP projects
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_DATE].copy()
        print(f"UNDP projects (post-conflict): {len(undp)}")
    except:
        undp = pd.DataFrame()
        print("UNDP projects: Not found")
    
    # Create treatment dates
    treatment_dates = {}
    
    if len(undp) > 0 and 'ml_category' in undp.columns:
        # Recovery categories
        recovery_cats = ['Debris & Solid Waste', 'Infrastructure & Housing', 
                        'Livelihoods & Economy', 'Energy & Power']
        recovery_projects = undp[undp['ml_category'].isin(recovery_cats)]
        if len(recovery_projects) > 0:
            treatment_dates['recovery'] = recovery_projects['start_date'].min()
        
        # Humanitarian categories
        humanitarian_cats = ['Health & Emergency', 'Water & Sanitation (WASH)', 'Education']
        humanitarian_projects = undp[undp['ml_category'].isin(humanitarian_cats)]
        if len(humanitarian_projects) > 0:
            treatment_dates['humanitarian'] = humanitarian_projects['start_date'].min()
        
        # Any UNDP
        treatment_dates['any_undp'] = undp['start_date'].min()
    else:
        # Defaults based on typical timeline
        treatment_dates['any_undp'] = pd.Timestamp('2024-01-01')
        treatment_dates['recovery'] = pd.Timestamp('2024-10-01')
        treatment_dates['humanitarian'] = pd.Timestamp('2024-07-01')
    
    # Create treatment indicators
    for name, date in treatment_dates.items():
        col = f'post_{name}'
        panel[col] = ((panel['period_start'] >= date) & panel['is_gaza']).astype(int)
    
    print(f"\nTreatment dates:")
    for name, date in sorted(treatment_dates.items(), key=lambda x: x[1]):
        print(f"  {name}: {date.strftime('%Y-%m-%d')}")
    
    return panel, undp, treatment_dates


def get_all_outcomes(panel):
    """Get all available outcomes with sufficient data."""
    
    # Potential outcomes
    candidates = {
        'primary': ['nightlights', 'ndbi', 'vh', 'ndvi', 'bsi'],
        'secondary': ['vv', 'rvi'],
        'prices': [c for c in panel.columns if c.startswith('price_')],
        'conflict': ['conflict_events', 'conflict_fatalities'],
        'google_trends': [c for c in panel.columns if c.startswith('gt_')],
    }
    
    available = {}
    
    print("\n" + "=" * 70)
    print("AVAILABLE OUTCOMES")
    print("=" * 70)
    
    for category, vars in candidates.items():
        available[category] = []
        for var in vars:
            if var in panel.columns:
                n_valid = panel[var].notna().sum()
                pct = n_valid / len(panel) * 100
                if pct >= 30:  # At least 30% coverage
                    available[category].append(var)
                    print(f"  {var}: {n_valid}/{len(panel)} ({pct:.1f}%) ✓")
                else:
                    print(f"  {var}: {n_valid}/{len(panel)} ({pct:.1f}%) - insufficient")
            else:
                print(f"  {var}: NOT IN DATA")
    
    return available


def run_its_analysis(panel, outcome, treatment_date, treatment_name, controls=None):
    """
    Interrupted Time Series for Gaza.
    
    This measures: trajectory change after treatment_date
    NOT: causal effect of specific intervention
    """
    
    gaza = panel[panel['is_gaza']].copy()
    
    # Aggregate
    agg_dict = {outcome: 'mean', 'time_idx': 'first'}
    if controls:
        for c in controls:
            if c in gaza.columns:
                agg_dict[c] = 'mean'
    
    gaza_agg = gaza.groupby('period_start').agg(agg_dict).reset_index()
    
    # ITS variables
    treat_time = (treatment_date.year - 2023) * 12 + treatment_date.month
    gaza_agg['post'] = (gaza_agg['period_start'] >= treatment_date).astype(int)
    gaza_agg['time_since'] = (gaza_agg['time_idx'] - treat_time).clip(lower=0)
    
    formula = f'{outcome} ~ time_idx + post + time_since'
    if controls:
        valid_controls = [c for c in controls if c in gaza_agg.columns]
        if valid_controls:
            formula += ' + ' + ' + '.join(valid_controls)
    
    try:
        model = smf.ols(formula, data=gaza_agg.dropna(subset=[outcome])).fit(cov_type='HC3')
        return model
    except Exception as e:
        return None


def run_did_analysis(panel, outcome, treatment_var, controls=None):
    """
    Difference-in-Differences with West Bank as control.
    
    CAVEAT: West Bank is imperfect control (different conflict exposure)
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
    except:
        return None


def run_event_study(panel, outcome, treatment_date, controls=None):
    """Event study around treatment date."""
    
    panel = panel.copy()
    panel['rel_time'] = (
        (panel['period_start'].dt.year - treatment_date.year) * 12 + 
        (panel['period_start'].dt.month - treatment_date.month)
    ).clip(-6, 12)
    
    # Create dummies (tm = t minus, tp = t plus)
    for t in range(-6, 13):
        if t == -1:
            continue
        col = f'tm{abs(t)}' if t < 0 else f'tp{t}'
        panel[col] = ((panel['rel_time'] == t) & panel['is_gaza']).astype(int)
    
    rel_vars = [f'tm{abs(t)}' if t < 0 else f'tp{t}' for t in range(-6, 13) if t != -1]
    
    formula = f'{outcome} ~ ' + ' + '.join(rel_vars) + ' + C(unit_id) + C(year_month_str)'
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
                results.append({'rel_time': t, 'coef': 0, 'se': 0})
            else:
                var = f'tm{abs(t)}' if t < 0 else f'tp{t}'
                if var in model.params.index:
                    results.append({
                        'rel_time': t,
                        'coef': model.params[var],
                        'se': model.bse[var]
                    })
        
        return pd.DataFrame(results), model
    except:
        return None, None


def create_event_study_plot(results_dict, outcome, treatment_name, output_path):
    """Create event study visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    
    for i, (name, df) in enumerate(results_dict.items()):
        if df is None or len(df) == 0:
            continue
        
        color = colors[i % len(colors)]
        df['ci_low'] = df['coef'] - 1.96 * df['se']
        df['ci_high'] = df['coef'] + 1.96 * df['se']
        
        ax.plot(df['rel_time'], df['coef'], 'o-', color=color, label=name, linewidth=2)
        ax.fill_between(df['rel_time'], df['ci_low'], df['ci_high'], alpha=0.2, color=color)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Months Relative to Treatment', fontsize=12)
    ax.set_ylabel(f'Effect on {outcome}', fontsize=12)
    ax.set_title(f'Event Study: {outcome}\n(Shows timing correlation, not causation)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    """Run comprehensive analysis on ALL outcomes."""
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE GAZA IMPACT ANALYSIS")
    print("=" * 70)
    print("""
    IMPORTANT CAVEATS:
    - This analysis shows CORRELATION, not CAUSATION
    - We cannot isolate UNDP's effect from other agencies
    - Results show "recovery trajectory" not "UNDP impact"
    - West Bank is an imperfect control group
    """)
    
    # Load data
    panel, undp, treatment_dates = load_data()
    
    # Get all outcomes
    available = get_all_outcomes(panel)
    
    # Controls
    controls = ['rainfall'] if 'rainfall' in panel.columns and panel['rainfall'].notna().sum() > 100 else []
    
    # All outcomes to analyze
    all_outcomes = []
    for cat, vars in available.items():
        all_outcomes.extend(vars)
    
    print(f"\n\nAnalyzing {len(all_outcomes)} outcomes: {all_outcomes}")
    
    # Store all results
    all_results = []
    
    # ==========================================================================
    # ANALYSIS 1: ITS (Gaza only - most honest approach)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: INTERRUPTED TIME SERIES (Gaza only)")
    print("=" * 70)
    print("Interpretation: Tests if trajectory changed after intervention dates")
    print("Caveat: Does NOT prove UNDP caused the change\n")
    
    for outcome in all_outcomes:
        print(f"\n--- {outcome} ---")
        
        for treat_name, treat_date in treatment_dates.items():
            model = run_its_analysis(panel, outcome, treat_date, treat_name, controls)
            
            if model:
                level = model.params.get('post', np.nan)
                trend = model.params.get('time_since', np.nan)
                pval_level = model.pvalues.get('post', np.nan)
                pval_trend = model.pvalues.get('time_since', np.nan)
                
                sig_l = '***' if pval_level < 0.01 else ('**' if pval_level < 0.05 else ('*' if pval_level < 0.1 else ''))
                sig_t = '***' if pval_trend < 0.01 else ('**' if pval_trend < 0.05 else ('*' if pval_trend < 0.1 else ''))
                
                print(f"  {treat_name}: level={level:.4f}{sig_l}, trend={trend:.4f}{sig_t}")
                
                all_results.append({
                    'Analysis': 'ITS',
                    'Outcome': outcome,
                    'Treatment': treat_name,
                    'Coefficient_Level': level,
                    'Coefficient_Trend': trend,
                    'p_value_Level': pval_level,
                    'p_value_Trend': pval_trend,
                    'N': int(model.nobs),
                    'R2': model.rsquared
                })
    
    # ==========================================================================
    # ANALYSIS 2: DiD (with West Bank - caveat: imperfect control)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: DIFFERENCE-IN-DIFFERENCES")
    print("=" * 70)
    print("Interpretation: Gaza vs West Bank change")
    print("Caveat: West Bank had different conflict exposure\n")
    
    for outcome in all_outcomes:
        print(f"\n--- {outcome} ---")
        
        for treat_name in ['post_any_undp', 'post_recovery', 'post_humanitarian']:
            if treat_name in panel.columns:
                model = run_did_analysis(panel, outcome, treat_name, controls)
                
                if model and treat_name in model.params.index:
                    coef = model.params[treat_name]
                    pval = model.pvalues[treat_name]
                    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                    
                    print(f"  {treat_name}: β={coef:.4f}{sig}")
                    
                    all_results.append({
                        'Analysis': 'DiD',
                        'Outcome': outcome,
                        'Treatment': treat_name,
                        'Coefficient': coef,
                        'Std_Error': model.bse[treat_name],
                        'p_value': pval,
                        'N': int(model.nobs),
                        'R2': model.rsquared
                    })
    
    # ==========================================================================
    # ANALYSIS 3: EVENT STUDY (for key outcomes)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: EVENT STUDY")
    print("=" * 70)
    print("Interpretation: Shows timing of changes relative to interventions\n")
    
    key_outcomes = ['nightlights', 'ndbi', 'ndvi', 'vh', 'bsi']
    key_outcomes = [o for o in key_outcomes if o in all_outcomes]
    
    for outcome in key_outcomes:
        print(f"\n--- {outcome} ---")
        
        es_results = {}
        for treat_name, treat_date in treatment_dates.items():
            es_df, es_model = run_event_study(panel, outcome, treat_date, controls)
            if es_df is not None:
                es_results[treat_name] = es_df
                print(f"  {treat_name}: Event study computed ✓")
        
        if es_results:
            create_event_study_plot(
                es_results, outcome, 'UNDP',
                FIGURES_DIR / f'event_study_{outcome}.png'
            )
            print(f"  ✓ event_study_{outcome}.png")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / 'comprehensive_analysis.csv', index=False)
    print(f"  ✓ {TABLES_DIR / 'comprehensive_analysis.csv'}")
    
    # Summary of significant results
    print("\n--- SIGNIFICANT RESULTS (p < 0.10) ---")
    
    # ITS results
    its_results = results_df[results_df['Analysis'] == 'ITS'].copy()
    if len(its_results) > 0:
        sig_its = its_results[
            (its_results['p_value_Level'] < 0.1) | 
            (its_results['p_value_Trend'] < 0.1)
        ]
        if len(sig_its) > 0:
            print("\nITS (Gaza trajectory changes):")
            for _, row in sig_its.iterrows():
                level_sig = '***' if row['p_value_Level'] < 0.01 else ('**' if row['p_value_Level'] < 0.05 else ('*' if row['p_value_Level'] < 0.1 else ''))
                trend_sig = '***' if row['p_value_Trend'] < 0.01 else ('**' if row['p_value_Trend'] < 0.05 else ('*' if row['p_value_Trend'] < 0.1 else ''))
                print(f"  {row['Outcome']} + {row['Treatment']}: level={row['Coefficient_Level']:.3f}{level_sig}, trend={row['Coefficient_Trend']:.3f}{trend_sig}")
    
    # DiD results
    did_results = results_df[results_df['Analysis'] == 'DiD'].copy()
    if len(did_results) > 0 and 'p_value' in did_results.columns:
        sig_did = did_results[did_results['p_value'] < 0.1]
        if len(sig_did) > 0:
            print("\nDiD (Gaza vs West Bank):")
            for _, row in sig_did.iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '*')
                print(f"  {row['Outcome']} + {row['Treatment']}: β={row['Coefficient']:.3f}{sig}")
    
    # ==========================================================================
    # INTERPRETATION GUIDE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
    WHAT THESE RESULTS MEAN:
    
    1. ITS Level Shift (+): Outcomes INCREASED after treatment date
       - For nightlights: More economic activity
       - For NDBI: More built-up area (reconstruction)
       - For BSI: More bare soil (could be debris OR cleared land)
    
    2. ITS Trend Change (+): Recovery is ACCELERATING
       - Positive trend = things getting better faster
       - Negative trend = improvement slowing or reversal
    
    3. DiD Coefficient: Gaza changed MORE than West Bank
       - Positive = Gaza improved relative to WB
       - Negative = Gaza worsened relative to WB
    
    WHAT THESE RESULTS DO NOT MEAN:
    
    ❌ "UNDP caused X% of recovery"
       → We cannot isolate UNDP from other agencies
    
    ❌ "Debris removal led to Y improvement"  
       → Multiple interventions happen simultaneously
    
    ❌ "Without UNDP, outcome would be Z"
       → We don't have that counterfactual
    
    APPROPRIATE CLAIMS:
    
    ✓ "Recovery indicators began improving around [date], 
        coinciding with the start of UNDP [category] projects"
    
    ✓ "The trajectory of [outcome] shows acceleration 
        in the period following UNDP intervention initiation"
    
    ✓ "These patterns are CONSISTENT WITH positive 
        intervention effects, though causation cannot be established"
    """)
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Results: {TABLES_DIR / 'comprehensive_analysis.csv'}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
