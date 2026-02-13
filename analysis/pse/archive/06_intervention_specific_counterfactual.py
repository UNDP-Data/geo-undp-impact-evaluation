#!/usr/bin/env python3
"""
analysis/pse/06_intervention_specific_counterfactual.py
=======================================================
Intervention-Specific Conflict Counterfactual Analysis

METHODOLOGY:
============
For each UNDP intervention, we:
1. Map the intervention to its EXPECTED affected outcomes
   (e.g., Debris Removal → BSI/NDBI, not nightlights)
2. Estimate destruction coefficient: Outcome = β₀ + β₁ × Conflict
   Using data from conflict start → intervention start
3. Predict counterfactual post-intervention using actual conflict intensity
4. Calculate UNDP effect = Actual - Predicted

This is cleaner than a general "UNDP effect" because:
- UNDP has been operating since before the conflict
- Each intervention type affects specific outcomes
- Estimation period is intervention-specific

Usage:
    python analysis/pse/06_intervention_specific_counterfactual.py
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
FIGURES_DIR = OUTPUT_DIR / "figures" / "intervention_counterfactual"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

CONFLICT_START = pd.Timestamp("2023-10-07")

# =============================================================================
# OUTCOMES AND INTERVENTIONS
# =============================================================================
# ALL satellite and price outcomes to evaluate
# Each intervention will be evaluated on ALL outcomes

ALL_OUTCOMES = [
    # Satellite indices
    'nightlights',  # VIIRS nighttime lights
    'ndvi',         # Vegetation index
    'ndbi',         # Built-up index (from rasters)
    'bsi',          # Bare soil index (from rasters)
    'vh',           # SAR VH backscatter
    'vv',           # SAR VV backscatter
    # Prices
    'price_bread',
    'price_flour', 
    'price_rice',
]

# Expected direction of IMPROVEMENT for each outcome
# Positive = higher is better (recovery), Negative = lower is better (price reduction)
OUTCOME_EXPECTED_DIRECTION = {
    'nightlights': 1,   # More lights = recovery
    'ndvi': 1,          # More vegetation = recovery
    'ndbi': 1,          # More built-up = reconstruction
    'bsi': -1,          # Less bare soil = reconstruction (debris cleared & rebuilt)
    'vh': 1,            # Higher backscatter = surface changes (construction)
    'vv': 1,            # Higher backscatter = surface changes
    'price_bread': -1,  # Lower prices = stability
    'price_flour': -1,
    'price_rice': -1,
}

# Intervention categories and their colors for plotting
INTERVENTION_COLORS = {
    'Debris & Solid Waste': 'brown',
    'Infrastructure & Housing': 'orange',
    'Energy & Power': 'gold',
    'Water & Sanitation (WASH)': 'blue',
    'Health & Emergency': 'red',
    'Livelihoods & Economy': 'green',
    'Education': 'purple',
    'Social Inclusion': 'pink',
    'Governance & Rule of Law': 'gray',
    'Environment': 'darkgreen',
}

OUTCOME_LABELS = {
    'nightlights': 'Nightlights (VIIRS)',
    'ndvi': 'Vegetation (NDVI)',
    'ndbi': 'Built-up (NDBI)',
    'bsi': 'Bare Soil (BSI)',
    'vh': 'SAR VH (dB)',
    'vv': 'SAR VV (dB)',
    'rvi': 'Radar Vegetation Index',
    'price_bread': 'Bread Price (ILS/kg)',
    'price_flour': 'Flour Price (ILS/kg)',
    'price_rice': 'Rice Price (ILS/kg)',
    'price_milk': 'Milk Price (ILS/L)',
    'price_oil': 'Oil Price (ILS/L)',
    'price_sugar': 'Sugar Price (ILS/kg)',
    'price_chicken': 'Chicken Price (ILS/kg)',
}


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    # Gaza only, aggregate to monthly
    gaza = panel[panel['unit_id'].str.startswith('PS02')].copy()
    numeric_cols = gaza.select_dtypes(include=[np.number]).columns.tolist()
    monthly = gaza.groupby('period_start')[numeric_cols].mean().reset_index()
    monthly = monthly.sort_values('period_start')
    
    print(f"  Gaza monthly: {len(monthly)} obs ({monthly['period_start'].min().strftime('%Y-%m')} to {monthly['period_start'].max().strftime('%Y-%m')})")
    
    return monthly


def load_interventions():
    """Load UNDP intervention dates for ALL categories."""
    interventions = {}
    
    try:
        undp = pd.read_csv(UNDP_FILE)
        undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
        
        # Filter to post-conflict, physical interventions
        if 'is_likely_umbrella' in undp.columns:
            undp = undp[undp['is_likely_umbrella'] != True]
        undp = undp[undp['start_date'] >= CONFLICT_START]
        
        # Get first date for each category (ALL categories, not just mapped ones)
        if 'ml_category' in undp.columns:
            for cat in undp['ml_category'].dropna().unique():
                cat_df = undp[undp['ml_category'] == cat]
                if len(cat_df) > 0:
                    interventions[cat] = cat_df['start_date'].min()
        
        print(f"  Interventions found: {len(interventions)}")
        for name, date in sorted(interventions.items(), key=lambda x: x[1]):
            print(f"    {name}: {date.strftime('%Y-%m-%d')}")
            
    except Exception as e:
        print(f"  ⚠ Error loading UNDP data: {e}")
        # Fallback
        interventions = {
            'Debris & Solid Waste': pd.Timestamp('2024-10-01'),
            'Infrastructure & Housing': pd.Timestamp('2024-12-01'),
            'Energy & Power': pd.Timestamp('2024-12-01'),
        }
    
    return interventions


def estimate_destruction_coefficient(df, outcome, conflict_var, start_date, end_date):
    """
    Estimate: Outcome = β₀ + β₁ × Conflict + ε
    
    Returns dict with beta0, beta1, r2, pvalue, se, n_obs
    """
    # Filter to estimation period
    mask = (df['period_start'] >= start_date) & (df['period_start'] < end_date)
    est_df = df[mask].copy()
    
    if len(est_df) < 3:
        return None
    
    y = est_df[outcome].values
    X = est_df[conflict_var].values
    
    valid = ~(np.isnan(y) | np.isnan(X))
    if valid.sum() < 3:
        return None
    
    y, X = y[valid], X[valid]
    X_const = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X_const).fit()
        return {
            'beta0': model.params[0],
            'beta1': model.params[1],
            'se': model.bse[1],
            'r2': model.rsquared,
            'pvalue': model.pvalues[1],
            'n_obs': len(y),
        }
    except:
        return None


def run_intervention_analysis(df, intervention_name, intervention_date, 
                               outcome, conflict_var, expected_direction):
    """
    Run counterfactual analysis for one intervention-outcome pair.
    
    Returns dict with estimation results, counterfactual, effect size
    """
    # Estimation period: conflict start → intervention start
    est_start = CONFLICT_START
    est_end = intervention_date
    
    # Check we have enough pre-intervention data
    pre_data = df[(df['period_start'] >= est_start) & (df['period_start'] < est_end)]
    if len(pre_data) < 3:
        return {'error': f'Insufficient pre-intervention data ({len(pre_data)} obs)'}
    
    # Step 1: Estimate destruction coefficient
    est = estimate_destruction_coefficient(df, outcome, conflict_var, est_start, est_end)
    if est is None:
        return {'error': 'Could not estimate destruction coefficient'}
    
    # Step 2: Predict counterfactual for post-intervention period
    post_mask = df['period_start'] >= intervention_date
    post_df = df[post_mask].copy()
    
    if len(post_df) < 2:
        return {'error': f'Insufficient post-intervention data ({len(post_df)} obs)'}
    
    # Predicted = β₀ + β₁ × Actual_Conflict
    post_df['predicted'] = est['beta0'] + est['beta1'] * post_df[conflict_var]
    post_df['actual'] = post_df[outcome]
    post_df['effect'] = post_df['actual'] - post_df['predicted']
    
    # Step 3: Calculate treatment effect
    att = post_df['effect'].mean()
    att_se = post_df['effect'].std() / np.sqrt(len(post_df))
    
    # Direction-adjusted effect (positive = intervention helping)
    adjusted_att = att * expected_direction
    
    # Statistical test
    t_stat = att / att_se if att_se > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(post_df)-1)) if not np.isnan(t_stat) else np.nan
    
    # Percent improvement over counterfactual
    counterfactual_mean = post_df['predicted'].mean()
    pct_improvement = (att / abs(counterfactual_mean) * 100) if counterfactual_mean != 0 else np.nan
    
    # Get full post-conflict data for plotting
    full_post = df[df['period_start'] >= CONFLICT_START].copy()
    full_post['predicted'] = est['beta0'] + est['beta1'] * full_post[conflict_var]
    full_post['actual'] = full_post[outcome]
    
    return {
        'estimation': est,
        'att': att,
        'att_se': att_se,
        'adjusted_att': adjusted_att,
        't_stat': t_stat,
        'p_value': p_value,
        'pct_improvement': pct_improvement,
        'n_pre': len(pre_data),
        'n_post': len(post_df),
        'post_data': post_df,
        'full_data': full_post,
        'interpretation': 'HELPING' if adjusted_att > 0 else 'BELOW EXPECTED',
    }


def create_intervention_plot(intervention_name, outcome, result, 
                              intervention_date, output_path):
    """Create 2x2 plot for one intervention-outcome analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    est = result['estimation']
    full_data = result['full_data']
    post_data = result['post_data']
    
    times = full_data['period_start']
    label = OUTCOME_LABELS.get(outcome, outcome)
    color = INTERVENTION_COLORS.get(intervention_name, 'purple')
    
    # =========================================================================
    # Panel 1: Actual vs Predicted Trajectory
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Shading
    ax1.axvspan(CONFLICT_START, intervention_date, alpha=0.15, color='orange',
                label='Estimation Period')
    ax1.axvspan(intervention_date, times.max(), alpha=0.15, color='lightgreen',
                label='Post-Intervention')
    
    # Plot
    ax1.plot(times, full_data['actual'], 'b-', linewidth=2.5, marker='o',
             markersize=5, label='Actual', zorder=5)
    ax1.plot(times, full_data['predicted'], 'r--', linewidth=2, marker='s',
             markersize=4, label='Predicted (conflict-only)', zorder=4)
    
    # Fill effect
    fill_color = 'green' if result['adjusted_att'] > 0 else 'red'
    ax1.fill_between(times, full_data['actual'], full_data['predicted'],
                     where=(times >= intervention_date),
                     alpha=0.3, color=fill_color)
    
    ax1.axvline(x=CONFLICT_START, color='black', linewidth=2.5, label='Oct 7, 2023')
    ax1.axvline(x=intervention_date, color=color, linewidth=2.5, linestyle='--',
                label=intervention_name)
    
    ax1.set_ylabel(label, fontsize=11)
    ax1.set_title('Actual vs Conflict-Predicted Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 2: Conflict Intensity
    # =========================================================================
    ax2 = axes[0, 1]
    
    if 'conflict_events' in full_data.columns:
        ax2.fill_between(times, 0, full_data['conflict_events'], alpha=0.4, color='red')
        ax2.plot(times, full_data['conflict_events'], 'r-', linewidth=2)
        ax2.set_ylabel('Conflict Events', fontsize=11)
    
    ax2.axvline(x=CONFLICT_START, color='black', linewidth=2.5)
    ax2.axvline(x=intervention_date, color=color, linewidth=2.5, linestyle='--')
    
    ax2.set_title('Conflict Intensity (Predictor Variable)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 3: Monthly Effect (Actual - Predicted)
    # =========================================================================
    ax3 = axes[1, 0]
    
    post_times = post_data['period_start']
    effects = post_data['effect']
    expected_dir = OUTCOME_EXPECTED_DIRECTION.get(outcome, 1)
    
    colors = ['green' if e * expected_dir > 0 else 'red' for e in effects]
    ax3.bar(post_times, effects, width=25, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axhline(y=result['att'], color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    sig = '***' if result['p_value'] < 0.01 else '**' if result['p_value'] < 0.05 else '*' if result['p_value'] < 0.1 else ''
    ax3.annotate(f"ATT = {result['att']:.4f}{sig}", 
                 xy=(post_times.iloc[-1], result['att']),
                 fontsize=11, fontweight='bold', color='blue',
                 ha='right', va='bottom' if result['att'] > 0 else 'top')
    
    ax3.set_ylabel('Effect (Actual - Predicted)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title(f'Intervention Effect Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # =========================================================================
    # Panel 4: Summary Statistics
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    beta_sig = '***' if est['pvalue'] < 0.01 else '**' if est['pvalue'] < 0.05 else '*' if est['pvalue'] < 0.1 else ''
    att_sig = '***' if result['p_value'] < 0.01 else '**' if result['p_value'] < 0.05 else '*' if result['p_value'] < 0.1 else ''
    
    interpretation = result['interpretation']
    interp_color = '#90EE90' if interpretation == 'HELPING' else '#FFB6C1'
    
    mechanism = f"Testing {intervention_name} effect on {outcome}"
    
    table_data = [
        ['INTERVENTION', intervention_name],
        ['Start Date', intervention_date.strftime('%Y-%m-%d')],
        ['Target Outcome', label],
        ['Mechanism', mechanism[:40] + '...' if len(mechanism) > 40 else mechanism],
        ['', ''],
        ['DESTRUCTION COEFFICIENT', ''],
        [f'β₁ (conflict → {outcome})', f"{est['beta1']:.5f}{beta_sig}"],
        ['SE', f"{est['se']:.5f}"],
        ['R²', f"{est['r2']:.3f}"],
        ['N (pre-intervention)', f"{result['n_pre']}"],
        ['', ''],
        ['INTERVENTION EFFECT', ''],
        ['ATT', f"{result['att']:.5f}{att_sig}"],
        ['SE', f"{result['att_se']:.5f}"],
        ['% vs Counterfactual', f"{result['pct_improvement']:+.1f}%" if not np.isnan(result['pct_improvement']) else 'N/A'],
        ['N (post-intervention)', f"{result['n_post']}"],
        ['', ''],
        ['INTERPRETATION', f"✓ {interpretation}" if interpretation == 'HELPING' else f"⚠ {interpretation}"],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.55, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    # Color rows
    for i, row in enumerate(table_data):
        if 'INTERPRETATION' in row[0]:
            table[(i, 0)].set_facecolor(interp_color)
            table[(i, 1)].set_facecolor(interp_color)
        elif row[0] in ['DESTRUCTION COEFFICIENT', 'INTERVENTION EFFECT', 'INTERVENTION']:
            table[(i, 0)].set_facecolor('#E0E0E0')
            table[(i, 1)].set_facecolor('#E0E0E0')
    
    # =========================================================================
    # Title
    # =========================================================================
    fig.suptitle(
        f'{intervention_name} → {label}\n'
        f'Destruction: β₁={est["beta1"]:.5f} (R²={est["r2"]:.2f}) | '
        f'Effect: ATT={result["att"]:+.4f} ({interpretation})',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_plot(all_results, output_path):
    """Create summary forest plot of all intervention effects."""
    
    # Flatten results
    plot_data = []
    for int_name, outcomes in all_results.items():
        for outcome, result in outcomes.items():
            if 'error' not in result:
                plot_data.append({
                    'intervention': int_name,
                    'outcome': outcome,
                    'label': f"{int_name[:15]}... → {outcome}",
                    'att': result['adjusted_att'],
                    'se': result['att_se'],
                    'pvalue': result['p_value'],
                    'r2': result['estimation']['r2'],
                })
    
    if not plot_data:
        print("  No results to plot")
        return
    
    df = pd.DataFrame(plot_data)
    df = df.sort_values('att', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    
    y_pos = np.arange(len(df))
    colors = ['green' if a > 0 else 'red' for a in df['att']]
    
    ax.barh(y_pos, df['att'], xerr=df['se']*1.96, color=colors, 
            alpha=0.7, edgecolor='black', capsize=4)
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Add significance stars
    for i, (att, pval) in enumerate(zip(df['att'], df['pvalue'])):
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        ax.annotate(sig, xy=(att, i), fontsize=10, fontweight='bold',
                   ha='left' if att > 0 else 'right', va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Intervention Effect (Direction-Adjusted ATT)\nPositive = Intervention Helping', fontsize=11)
    ax.set_title('Summary: UNDP Intervention Effects by Outcome\n(95% CI shown)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("INTERVENTION-SPECIFIC CONFLICT COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    print("""
    For each UNDP intervention:
    1. Map intervention to EXPECTED outcomes (e.g., Debris → surface indices)
    2. Estimate destruction coefficient from pre-intervention data
    3. Predict counterfactual using actual conflict intensity
    4. Calculate intervention effect = Actual - Predicted
    """)
    
    # Load data
    df = load_data()
    interventions = load_interventions()
    
    # Check conflict data
    conflict_var = None
    for cv in ['conflict_events', 'conflict_fatalities']:
        if cv in df.columns and df[cv].notna().sum() > 5:
            conflict_var = cv
            break
    
    if conflict_var is None:
        print("\n⚠ No conflict data available")
        return
    
    print(f"\n  Conflict measure: {conflict_var}")
    
    # ==========================================================================
    # RUN ANALYSIS FOR EACH INTERVENTION-OUTCOME PAIR
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYZING INTERVENTION-OUTCOME PAIRS")
    print(f"  Testing ALL {len(ALL_OUTCOMES)} outcomes for each intervention")
    print("=" * 70)
    
    all_results = {}
    summary_rows = []
    
    for int_name, int_date in interventions.items():
        print(f"\n{'='*60}")
        print(f"{int_name} (started {int_date.strftime('%Y-%m-%d')})")
        print('='*60)
        
        all_results[int_name] = {}
        
        # Evaluate ALL outcomes for this intervention
        for outcome in ALL_OUTCOMES:
            if outcome not in df.columns:
                print(f"  {outcome}: ⚠ Not in data")
                continue
            if df[outcome].notna().sum() < 5:
                print(f"  {outcome}: ⚠ Insufficient data")
                continue
            
            # Get expected direction for this outcome
            expected_dir = OUTCOME_EXPECTED_DIRECTION.get(outcome, 1)
            
            result = run_intervention_analysis(
                df, int_name, int_date, outcome, conflict_var, expected_dir
            )
            
            all_results[int_name][outcome] = result
            
            if 'error' in result:
                print(f"  {outcome}: ⚠ {result['error']}")
                continue
            
            est = result['estimation']
            sig_beta = '***' if est['pvalue'] < 0.01 else '**' if est['pvalue'] < 0.05 else '*' if est['pvalue'] < 0.1 else ''
            sig_att = '***' if result['p_value'] < 0.01 else '**' if result['p_value'] < 0.05 else '*' if result['p_value'] < 0.1 else ''
            
            print(f"  {outcome}:")
            print(f"    β₁={est['beta1']:.5f}{sig_beta} (R²={est['r2']:.2f}, n={result['n_pre']})")
            print(f"    ATT={result['att']:+.5f}{sig_att} → {result['interpretation']}")
            
            # Create plot
            filename = f"int_{int_name.replace(' ', '_').replace('/', '_').replace('&', 'and')}_{outcome}.png"
            create_intervention_plot(
                int_name, outcome, result, int_date,
                FIGURES_DIR / filename
            )
            print(f"    ✓ {filename}")
            
            # Store for summary
            summary_rows.append({
                'Intervention': int_name,
                'Intervention_Date': int_date,
                'Outcome': outcome,
                'Beta1': est['beta1'],
                'Beta1_SE': est['se'],
                'Beta1_pvalue': est['pvalue'],
                'R2': est['r2'],
                'N_Pre': result['n_pre'],
                'ATT': result['att'],
                'ATT_Adjusted': result['adjusted_att'],
                'ATT_SE': result['att_se'],
                'ATT_pvalue': result['p_value'],
                'Pct_Improvement': result['pct_improvement'],
                'N_Post': result['n_post'],
                'Interpretation': result['interpretation'],
            })
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(TABLES_DIR / 'intervention_counterfactual_results.csv', index=False)
        print(f"  ✓ intervention_counterfactual_results.csv ({len(summary_df)} pairs)")
        
        create_summary_plot(all_results, FIGURES_DIR / 'intervention_summary.png')
        print(f"  ✓ intervention_summary.png")
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        helping = summary_df[summary_df['ATT_Adjusted'] > 0]
        sig_helping = helping[helping['ATT_pvalue'] < 0.1]
        
        print(f"\n  Total pairs analyzed: {len(summary_df)}")
        print(f"  Positive effects (intervention helping): {len(helping)}")
        print(f"  Statistically significant (p<0.1): {len(sig_helping)}")
        
        if len(sig_helping) > 0:
            print("\n  SIGNIFICANT POSITIVE EFFECTS:")
            print("-" * 60)
            for _, row in sig_helping.iterrows():
                sig = '***' if row['ATT_pvalue'] < 0.01 else '**' if row['ATT_pvalue'] < 0.05 else '*'
                print(f"    {row['Intervention'][:20]:20} → {row['Outcome']:12} ATT={row['ATT_Adjusted']:+.4f}{sig}")
    
    print(f"""
    INTERPRETATION GUIDE:
    ─────────────────────
    Each intervention is mapped to outcomes it SHOULD affect:
    • Debris Removal → Surface indices (VH, VV, NDVI)
    • Infrastructure → Nightlights, built-up indices
    • Energy/Power → Nightlights
    • Health/Emergency → Food prices
    • WASH → Vegetation (water access)
    
    ATT (Average Treatment Effect on Treated):
    • Positive adjusted ATT = Intervention is helping recovery
    • We compare ACTUAL outcome to what we PREDICT based on conflict alone
    
    Limitations:
    • Small sample sizes (few months post-intervention)
    • Model assumes linear conflict-outcome relationship
    • Other factors may contribute to changes
    """)
    
    print(f"\n{'='*70}")
    print("✓ INTERVENTION-SPECIFIC ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}/int_*.png")
    print(f"  Summary: {FIGURES_DIR}/intervention_summary.png")
    print(f"  Results: {TABLES_DIR}/intervention_counterfactual_results.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
