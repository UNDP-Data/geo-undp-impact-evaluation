#!/usr/bin/env python3
"""
analysis/pse/05_conflict_counterfactual.py
==========================================
Conflict-Driven Counterfactual Analysis for UNDP Impact Measurement

METHODOLOGY:
============
Instead of using donor countries (SC) or assuming parallel trends (DiD),
this approach uses CONFLICT INTENSITY as the driver of outcome deterioration.

Key insight: We can predict what outcomes WOULD have been based solely on 
the relationship between conflict and outcomes, then compare to actual values
after UNDP interventions begin.

STEPS:
1. Estimate "destruction coefficient": How does conflict affect outcomes?
   - Use period from Oct 7, 2023 to first UNDP intervention (Oct 2024)
   - Regression: Outcome_t = β₀ + β₁ × Conflict_t + ε
   
2. Predict counterfactual trajectory after UNDP intervention:
   - Using the destruction coefficient and ACTUAL conflict intensity
   - Predicted_t = β₀ + β₁ × Actual_Conflict_t
   
3. Calculate UNDP effect:
   - UNDP_Effect = Actual_Outcome - Predicted_Outcome
   - If positive for nightlights/NDVI → UNDP facilitating recovery beyond
     what conflict intensity alone would predict

This is powerful because:
- No donor country data needed
- Uses actual conflict variation (which we have from ACLED)
- Causally intuitive: conflict destroys, UNDP helps recover
- Can test intervention-specific effects (debris removal, health, etc.)

Usage:
    python analysis/pse/05_conflict_counterfactual.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "conflict_counterfactual"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

# Key dates
CONFLICT_START = pd.Timestamp("2023-10-07")
FIRST_UNDP_INTERVENTION = pd.Timestamp("2024-10-01")  # Debris removal

# Outcomes to analyze
OUTCOMES = [
    'nightlights', 'ndvi', 'vh', 'vv', 'rvi',
    'price_bread', 'price_flour', 'price_rice', 'price_milk',
    'price_oil', 'price_sugar', 'price_chicken',
]

# Conflict measures to try
CONFLICT_MEASURES = ['conflict_events', 'conflict_fatalities']

# Expected direction of UNDP effect (positive = improvement)
OUTCOME_DIRECTION = {
    'nightlights': 1,   # Higher is better (recovery)
    'ndvi': 1,          # Higher is better (vegetation recovery)
    'vh': -1,           # More negative is normal (recovery)
    'vv': -1,           # More negative is normal
    'rvi': 1,           # Higher is better
    'price_bread': -1,  # Lower is better (prices stabilizing)
    'price_flour': -1,
    'price_rice': -1,
    'price_milk': -1,
    'price_oil': -1,
    'price_sugar': -1,
    'price_chicken': -1,
}

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
    'price_oil': 'Oil Price (ILS/L)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_chicken': 'Chicken Price (ILS/kg)',
}


def load_and_prepare_data():
    """Load panel data and aggregate to monthly Gaza means."""
    print("Loading data...")
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    # Gaza only
    gaza = panel[panel['unit_id'].str.startswith('PS02')].copy()
    
    # Get only numeric columns for aggregation
    numeric_cols = gaza.select_dtypes(include=[np.number]).columns.tolist()
    
    # Aggregate to monthly means (numeric columns only)
    monthly = gaza.groupby('period_start')[numeric_cols].mean().reset_index()
    
    monthly = monthly.sort_values('period_start')
    
    print(f"  Gaza monthly data: {len(monthly)} observations")
    print(f"  Date range: {monthly['period_start'].min()} to {monthly['period_start'].max()}")
    print(f"  Numeric columns: {len(numeric_cols)}")
    
    return monthly


def load_undp_interventions():
    """Load UNDP intervention dates by category."""
    interventions = {}
    
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        
        # Filter to post-conflict, non-umbrella projects
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_START]
        
        # Only include Gaza-specific physical interventions (not administrative/umbrella)
        physical_categories = [
            'Debris & Solid Waste', 'Infrastructure & Housing', 
            'Health & Emergency', 'Water & Sanitation (WASH)',
            'Energy & Power', 'Education'
        ]
        
        # Get first intervention date by category
        if 'ml_category' in undp.columns:
            for cat in undp['ml_category'].dropna().unique():
                cat_df = undp[undp['ml_category'] == cat]
                if len(cat_df) > 0:
                    interventions[cat] = cat_df['start_date'].min()
        
        # First PHYSICAL intervention (not umbrella/admin)
        physical_undp = undp[undp['ml_category'].isin(physical_categories)] if 'ml_category' in undp.columns else undp
        if len(physical_undp) > 0:
            interventions['First Physical UNDP'] = physical_undp['start_date'].min()
        else:
            interventions['First Physical UNDP'] = pd.Timestamp('2024-10-01')
        
        print(f"  UNDP interventions loaded: {len(interventions)}")
        for name, date in sorted(interventions.items(), key=lambda x: x[1]):
            print(f"    {name}: {date.strftime('%Y-%m-%d')}")
            
    except Exception as e:
        print(f"  ⚠ Could not load UNDP data: {e}")
        interventions = {
            'Debris & Solid Waste': pd.Timestamp('2024-10-01'),
            'First Physical UNDP': pd.Timestamp('2024-10-01'),
        }
    
    return interventions


def estimate_destruction_coefficient(df, outcome, conflict_var, 
                                     estimation_start, estimation_end):
    """
    Estimate the relationship between conflict and outcome.
    
    Uses the period between conflict start and first UNDP intervention
    to estimate: Outcome = β₀ + β₁ × Conflict + ε
    
    Returns:
    --------
    dict with 'beta0', 'beta1', 'r2', 'pvalue', 'model'
    """
    # Filter to estimation period
    mask = (df['period_start'] >= estimation_start) & (df['period_start'] <= estimation_end)
    est_df = df[mask].copy()
    
    if len(est_df) < 4:
        return None
    
    # Check data availability
    y = est_df[outcome].values
    X = est_df[conflict_var].values
    
    valid = ~(np.isnan(y) | np.isnan(X))
    if valid.sum() < 4:
        return None
    
    y = y[valid]
    X = X[valid]
    
    # Add constant
    X_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X_const).fit()
        
        return {
            'beta0': model.params[0],
            'beta1': model.params[1],
            'se_beta1': model.bse[1],
            'r2': model.rsquared,
            'pvalue': model.pvalues[1],
            'n_obs': len(y),
            'model': model,
        }
    except Exception as e:
        print(f"    Estimation error: {e}")
        return None


def predict_counterfactual(df, conflict_var, beta0, beta1):
    """
    Predict counterfactual outcome based on destruction coefficient
    and actual conflict intensity.
    """
    predicted = beta0 + beta1 * df[conflict_var].values
    return predicted


def calculate_cumulative_conflict(df, conflict_var, start_date):
    """
    Calculate cumulative conflict exposure since start_date.
    This captures the "stock" of destruction rather than the flow.
    """
    df = df.copy()
    df['cum_conflict'] = 0.0
    
    mask = df['period_start'] >= start_date
    cum_values = df.loc[mask, conflict_var].cumsum()
    df.loc[mask, 'cum_conflict'] = cum_values.values
    
    return df['cum_conflict'].values


def run_conflict_counterfactual_analysis(df, outcome, conflict_var,
                                         interventions, use_cumulative=False):
    """
    Main analysis function.
    
    Steps:
    1. Estimate destruction coefficient (conflict start → first intervention)
    2. Predict counterfactual for entire post-conflict period
    3. Calculate UNDP effect (actual - predicted)
    4. Test statistical significance
    """
    results = {}
    
    # Estimation period: conflict start to first PHYSICAL UNDP intervention
    estimation_start = CONFLICT_START
    first_physical = interventions.get('First Physical UNDP', FIRST_UNDP_INTERVENTION)
    estimation_end = first_physical - timedelta(days=1)
    
    # Prepare conflict variable
    if use_cumulative:
        df = df.copy()
        df['conflict_measure'] = calculate_cumulative_conflict(df, conflict_var, CONFLICT_START)
    else:
        df['conflict_measure'] = df[conflict_var]
    
    # Step 1: Estimate destruction coefficient
    est_result = estimate_destruction_coefficient(
        df, outcome, 'conflict_measure', estimation_start, estimation_end
    )
    
    if est_result is None:
        return None
    
    results['estimation'] = est_result
    
    # Step 2: Predict counterfactual for all post-conflict periods
    post_conflict = df[df['period_start'] >= CONFLICT_START].copy()
    post_conflict['predicted'] = predict_counterfactual(
        post_conflict, 'conflict_measure', 
        est_result['beta0'], est_result['beta1']
    )
    post_conflict['actual'] = post_conflict[outcome]
    
    # Step 3: Calculate UNDP effect for post-intervention periods
    first_intervention = interventions.get('First Physical UNDP', FIRST_UNDP_INTERVENTION)
    
    post_intervention = post_conflict[post_conflict['period_start'] >= first_intervention].copy()
    pre_intervention = post_conflict[post_conflict['period_start'] < first_intervention].copy()
    
    if len(post_intervention) < 2:
        return None
    
    # Calculate effects
    post_intervention['effect'] = post_intervention['actual'] - post_intervention['predicted']
    pre_intervention['effect'] = pre_intervention['actual'] - pre_intervention['predicted']
    
    # Average treatment effect on the treated (ATT)
    att = post_intervention['effect'].mean()
    att_se = post_intervention['effect'].std() / np.sqrt(len(post_intervention))
    
    # Comparison: pre-intervention "effect" should be ~0 (model fit)
    pre_effect = pre_intervention['effect'].mean() if len(pre_intervention) > 0 else 0
    
    # T-test: is post-intervention effect different from pre?
    if len(pre_intervention) > 2 and len(post_intervention) > 2:
        t_stat, p_value = stats.ttest_ind(
            post_intervention['effect'].dropna(),
            pre_intervention['effect'].dropna()
        )
    else:
        t_stat, p_value = np.nan, np.nan
    
    # Direction-adjusted interpretation
    direction = OUTCOME_DIRECTION.get(outcome, 1)
    adjusted_att = att * direction  # Positive = UNDP helping
    
    results['counterfactual'] = {
        'att': att,
        'att_se': att_se,
        'adjusted_att': adjusted_att,  # Positive = good
        'pre_effect': pre_effect,
        't_stat': t_stat,
        'p_value': p_value,
        'n_post': len(post_intervention),
        'n_pre': len(pre_intervention),
        'post_data': post_intervention,
        'pre_data': pre_intervention,
        'full_data': post_conflict,
    }
    
    return results


def create_counterfactual_plot(outcome, results, interventions, output_path):
    """
    Create a 4-panel plot showing the conflict counterfactual analysis.
    
    Panel 1: Actual vs Predicted trajectory
    Panel 2: Conflict intensity over time
    Panel 3: UNDP Effect (Actual - Predicted)
    Panel 4: Summary statistics table
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    est = results['estimation']
    cf = results['counterfactual']
    data = cf['full_data']
    
    label = OUTCOME_LABELS.get(outcome, outcome)
    direction = OUTCOME_DIRECTION.get(outcome, 1)
    
    times = data['period_start']
    
    # Get first intervention date
    first_int = interventions.get('First Physical UNDP', FIRST_UNDP_INTERVENTION)
    
    # Intervention colors
    int_colors = {
        'Debris & Solid Waste': 'brown',
        'Health & Emergency': 'red',
        'Water & Sanitation (WASH)': 'blue',
        'Infrastructure & Housing': 'orange',
        'Social Inclusion': 'green',
    }
    
    # =========================================================================
    # Panel 1: Actual vs Predicted
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Background shading
    ax1.axvspan(CONFLICT_START, first_int, alpha=0.15, color='orange', 
                label='Estimation Period')
    ax1.axvspan(first_int, times.max(), alpha=0.15, color='lightgreen',
                label='Post-UNDP Period')
    
    # Plot actual and predicted
    ax1.plot(times, data['actual'], 'b-', linewidth=2.5, marker='o', 
             markersize=6, label='Actual', zorder=5)
    ax1.plot(times, data['predicted'], 'r--', linewidth=2, marker='s',
             markersize=5, label='Predicted (conflict-only)', zorder=4)
    
    # Shade the gap
    ax1.fill_between(times, data['actual'], data['predicted'], 
                     alpha=0.3, color='green' if cf['adjusted_att'] > 0 else 'red',
                     label='UNDP Effect' if cf['adjusted_att'] > 0 else 'Shortfall')
    
    # Mark intervention dates
    for name, date in interventions.items():
        if pd.notna(date) and times.min() <= date <= times.max():
            color = int_colors.get(name, 'purple')
            ax1.axvline(x=date, color=color, linestyle=':', linewidth=2, alpha=0.8)
    
    ax1.axvline(x=CONFLICT_START, color='black', linestyle='-', linewidth=2.5, 
                label='Oct 7, 2023')
    
    ax1.set_ylabel(label, fontsize=11)
    ax1.set_title('Actual vs Conflict-Predicted Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 2: Conflict intensity
    # =========================================================================
    ax2 = axes[0, 1]
    
    conflict_var = 'conflict_events' if 'conflict_events' in data.columns else 'conflict_measure'
    if conflict_var in data.columns:
        ax2.fill_between(times, 0, data[conflict_var], alpha=0.4, color='red')
        ax2.plot(times, data[conflict_var], 'r-', linewidth=2)
    
    ax2.axvline(x=CONFLICT_START, color='black', linestyle='-', linewidth=2.5)
    ax2.axvline(x=first_int, color='green', linestyle='--', linewidth=2, 
                label='First UNDP')
    
    ax2.set_ylabel('Conflict Events', fontsize=11)
    ax2.set_title('Conflict Intensity Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 3: Effect over time (Actual - Predicted)
    # =========================================================================
    ax3 = axes[1, 0]
    
    effect = data['actual'] - data['predicted']
    colors = ['green' if e * direction > 0 else 'red' for e in effect]
    
    ax3.bar(times, effect, width=25, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax3.axvline(x=CONFLICT_START, color='black', linestyle='-', linewidth=2.5)
    ax3.axvline(x=first_int, color='green', linestyle='--', linewidth=2)
    
    # Add ATT annotation
    att = cf['att']
    pval = cf['p_value']
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    
    ax3.axhline(y=att, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax3.annotate(f'ATT = {att:.3f}{sig}', xy=(times.iloc[-1], att),
                fontsize=11, fontweight='bold', color='blue',
                ha='right', va='bottom' if att > 0 else 'top')
    
    ax3.set_ylabel(f'Effect (Actual - Predicted)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('UNDP Effect Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 4: Summary statistics
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Build summary table
    beta1_sig = '***' if est['pvalue'] < 0.01 else '**' if est['pvalue'] < 0.05 else '*' if est['pvalue'] < 0.1 else ''
    att_sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    
    interpretation = "✓ UNDP HELPING" if cf['adjusted_att'] > 0 else "⚠ BELOW EXPECTED"
    interp_color = '#90EE90' if cf['adjusted_att'] > 0 else '#FFB6C1'
    
    table_data = [
        ['DESTRUCTION COEFFICIENT', ''],
        ['β₁ (conflict → outcome)', f"{est['beta1']:.4f}{beta1_sig}"],
        ['Standard Error', f"{est['se_beta1']:.4f}"],
        ['R²', f"{est['r2']:.3f}"],
        ['Estimation Period', f"Oct 2023 - Sep 2024"],
        ['N (estimation)', f"{est['n_obs']}"],
        ['', ''],
        ['UNDP EFFECT', ''],
        ['ATT (post-intervention)', f"{att:.4f}{att_sig}"],
        ['SE', f"{cf['att_se']:.4f}"],
        ['Pre-intervention fit', f"{cf['pre_effect']:.4f}"],
        ['N (post-intervention)', f"{cf['n_post']}"],
        ['T-statistic', f"{cf['t_stat']:.2f}" if not np.isnan(cf['t_stat']) else 'N/A'],
        ['', ''],
        ['INTERPRETATION', interpretation],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.5, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color interpretation row
    for i, row in enumerate(table_data):
        if 'INTERPRETATION' in row[0]:
            table[(i, 0)].set_facecolor(interp_color)
            table[(i, 1)].set_facecolor(interp_color)
        elif row[0] in ['DESTRUCTION COEFFICIENT', 'UNDP EFFECT']:
            table[(i, 0)].set_facecolor('#E0E0E0')
            table[(i, 1)].set_facecolor('#E0E0E0')
    
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # =========================================================================
    # Overall title
    # =========================================================================
    fig.suptitle(
        f'Conflict-Driven Counterfactual Analysis: {label}\n'
        f'Destruction Coefficient: β₁={est["beta1"]:.4f} (R²={est["r2"]:.3f}) | '
        f'UNDP Effect: ATT={att:+.4f} ({interpretation})',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_plot(all_results, interventions, output_path):
    """Create summary plot comparing UNDP effects across all outcomes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data
    outcomes = []
    atts = []
    att_ses = []
    pvals = []
    r2s = []
    
    for outcome, res in all_results.items():
        if res is None:
            continue
        
        direction = OUTCOME_DIRECTION.get(outcome, 1)
        adjusted_att = res['counterfactual']['att'] * direction
        
        outcomes.append(OUTCOME_LABELS.get(outcome, outcome))
        atts.append(adjusted_att)
        att_ses.append(res['counterfactual']['att_se'])
        pvals.append(res['counterfactual']['p_value'])
        r2s.append(res['estimation']['r2'])
    
    if not outcomes:
        print("  No results to plot")
        return
    
    # Panel 1: ATT by outcome (forest plot style)
    ax1 = axes[0]
    
    y_pos = np.arange(len(outcomes))
    colors = ['green' if a > 0 else 'red' for a in atts]
    
    ax1.barh(y_pos, atts, xerr=[s*1.96 for s in att_ses], 
             color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add significance stars
    for i, (att, pval) in enumerate(zip(atts, pvals)):
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        ax1.annotate(sig, xy=(att, i), fontsize=12, fontweight='bold',
                    ha='left' if att > 0 else 'right', va='center')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(outcomes, fontsize=10)
    ax1.set_xlabel('UNDP Effect (Direction-Adjusted ATT)', fontsize=11)
    ax1.set_title('UNDP Impact by Outcome\n(Positive = UNDP Helping Recovery)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Model fit (R²)
    ax2 = axes[1]
    
    ax2.barh(y_pos, r2s, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='R²=0.5')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(outcomes, fontsize=10)
    ax2.set_xlabel('R² (Destruction Coefficient Model Fit)', fontsize=11)
    ax2.set_title('How Well Does Conflict Predict Outcomes?\n(Higher = Better Model)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Conflict-Driven Counterfactual: Summary of UNDP Effects',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("CONFLICT-DRIVEN COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    print("""
    METHODOLOGY:
    1. Estimate destruction coefficient (how conflict affects outcomes)
    2. Predict counterfactual trajectory (what would happen without UNDP)
    3. Calculate UNDP effect (actual - predicted)
    """)
    
    # Load data
    df = load_and_prepare_data()
    interventions = load_undp_interventions()
    
    # Check for conflict data
    conflict_var = None
    for cv in CONFLICT_MEASURES:
        if cv in df.columns and df[cv].notna().sum() > 10:
            conflict_var = cv
            break
    
    if conflict_var is None:
        print("\n⚠ No conflict data available. Cannot run analysis.")
        return
    
    print(f"\n  Using conflict measure: {conflict_var}")
    
    # ==========================================================================
    # ANALYZE EACH OUTCOME
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ESTIMATING DESTRUCTION COEFFICIENTS & UNDP EFFECTS")
    print("=" * 70)
    
    all_results = {}
    summary_rows = []
    
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        if df[outcome].notna().sum() < 10:
            continue
        
        print(f"\n  {outcome}...")
        
        # Run analysis
        results = run_conflict_counterfactual_analysis(
            df, outcome, conflict_var, interventions, use_cumulative=False
        )
        
        if results is None:
            print(f"    ⚠ Insufficient data")
            continue
        
        all_results[outcome] = results
        
        # Print results
        est = results['estimation']
        cf = results['counterfactual']
        
        direction = OUTCOME_DIRECTION.get(outcome, 1)
        interpretation = "HELPING" if cf['adjusted_att'] > 0 else "BELOW EXPECTED"
        
        beta_sig = '***' if est['pvalue'] < 0.01 else '**' if est['pvalue'] < 0.05 else '*' if est['pvalue'] < 0.1 else ''
        att_sig = '***' if cf['p_value'] < 0.01 else '**' if cf['p_value'] < 0.05 else '*' if cf['p_value'] < 0.1 else ''
        
        print(f"    β₁ (destruction): {est['beta1']:.4f}{beta_sig} (R²={est['r2']:.3f})")
        print(f"    ATT (UNDP effect): {cf['att']:+.4f}{att_sig} → {interpretation}")
        
        # Create plot
        create_counterfactual_plot(
            outcome, results, interventions,
            FIGURES_DIR / f'cf_{outcome}.png'
        )
        print(f"    ✓ cf_{outcome}.png")
        
        # Store summary
        summary_rows.append({
            'Outcome': outcome,
            'Beta1_Destruction': est['beta1'],
            'Beta1_SE': est['se_beta1'],
            'Beta1_pvalue': est['pvalue'],
            'R2_Model': est['r2'],
            'N_Estimation': est['n_obs'],
            'ATT_UNDP': cf['att'],
            'ATT_SE': cf['att_se'],
            'ATT_pvalue': cf['p_value'],
            'ATT_Adjusted': cf['adjusted_att'],
            'N_PostIntervention': cf['n_post'],
            'Interpretation': 'UNDP Helping' if cf['adjusted_att'] > 0 else 'Below Expected',
        })
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / 'conflict_counterfactual_results.csv', index=False)
    print(f"  ✓ conflict_counterfactual_results.csv")
    
    # Summary plot
    create_summary_plot(all_results, interventions, FIGURES_DIR / 'cf_summary.png')
    print(f"  ✓ cf_summary.png")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(summary_df) > 0:
        helping = summary_df[summary_df['ATT_Adjusted'] > 0]
        sig_helping = helping[helping['ATT_pvalue'] < 0.1]
        
        print(f"\n  Total outcomes analyzed: {len(summary_df)}")
        print(f"  UNDP appears to be helping: {len(helping)}")
        print(f"  Statistically significant (p<0.1): {len(sig_helping)}")
        
        if len(sig_helping) > 0:
            print("\n  SIGNIFICANT POSITIVE UNDP EFFECTS:")
            print("-" * 60)
            for _, row in sig_helping.iterrows():
                sig = '***' if row['ATT_pvalue'] < 0.01 else '**' if row['ATT_pvalue'] < 0.05 else '*'
                print(f"    {row['Outcome']:20} ATT={row['ATT_Adjusted']:+.4f}{sig}")
        
        print(f"""
    INTERPRETATION:
    ---------------
    This analysis estimates what outcomes WOULD have been based solely on 
    the conflict-outcome relationship, then compares to actual values.
    
    Positive ATT (direction-adjusted) means:
    → Actual outcomes are BETTER than predicted by conflict intensity alone
    → UNDP interventions may be facilitating recovery beyond baseline
    
    Model R² indicates how well conflict predicts outcome changes:
    → Higher R² = conflict is strong driver (more reliable counterfactual)
    → Lower R² = other factors matter (counterfactual less reliable)
    
    CAVEATS:
    - Assumes linear conflict-outcome relationship
    - Other factors may contribute to recovery (not just UNDP)
    - Estimation period relatively short (12 months)
        """)
    
    print(f"\n{'='*70}")
    print("✓ CONFLICT COUNTERFACTUAL ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}/cf_*.png")
    print(f"  Results: {TABLES_DIR}/conflict_counterfactual_results.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
