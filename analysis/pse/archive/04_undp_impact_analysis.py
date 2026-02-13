#!/usr/bin/env python3
"""
analysis/pse/04_undp_impact_analysis.py
=======================================
Honest Assessment of What We Can Measure About UNDP Impact

This script is explicit about identification challenges and focuses on
what we CAN credibly estimate:

1. CONFLICT IMPACT (Synthetic Control): Gaza vs counterfactual
2. RECOVERY DYNAMICS (ITS): Did outcomes improve after UNDP started?
3. INTERVENTION HETEROGENEITY: Do specific UNDP project types matter?
4. DOSE-RESPONSE: Do more UNDP projects → faster recovery?
5. TIMING TESTS: Do outcomes change precisely when UNDP intervenes?

What we CANNOT identify:
- UNDP effects isolated from other agencies (UNRWA, WFP, WHO, etc.)
- Causal UNDP impact without within-Gaza variation

Usage:
    python analysis/pse/04_undp_impact_analysis.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
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


def load_all_data():
    """Load and merge all data sources."""
    
    print("=" * 70)
    print("LOADING ALL DATA SOURCES")
    print("=" * 70)
    
    # Load main panel
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    print(f"\nMain panel: {len(panel)} obs, {panel.shape[1]} columns")
    
    # Load raster panel (has NDBI, BSI from raster extraction)
    raster_panel = None
    if RASTER_PANEL_FILE.exists():
        raster_panel = pd.read_csv(RASTER_PANEL_FILE)
        raster_panel['date'] = pd.to_datetime(raster_panel['date'])
        print(f"Raster panel: {len(raster_panel)} obs")
        
        # Check what variables are in raster panel
        raster_vars = [c for c in raster_panel.columns if c not in ['date', 'admin_id', 'admin_name']]
        print(f"  Raster variables: {raster_vars}")
    
    # Load UNDP projects
    undp = pd.read_csv(UNDP_FILE)
    undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
    
    # Filter to post-conflict, non-umbrella
    if 'is_likely_umbrella' in undp.columns:
        undp = undp[undp['is_likely_umbrella'] != True]
    undp = undp[undp['start_date'] >= CONFLICT_DATE].copy()
    print(f"\nUNDP projects (post-conflict): {len(undp)}")
    
    # Identify Gaza
    panel['is_gaza'] = panel['unit_id'].str.startswith('PS02')
    panel['year_month_str'] = panel['period_start'].dt.strftime('%Y_%m')
    panel['time_idx'] = (panel['period_start'].dt.year - 2023) * 12 + panel['period_start'].dt.month
    
    return panel, raster_panel, undp


def analyze_undp_variation(undp):
    """Analyze what variation exists in UNDP interventions."""
    
    print("\n" + "=" * 70)
    print("ANALYZING UNDP INTERVENTION VARIATION")
    print("=" * 70)
    
    print("\n--- Temporal Variation (When did different interventions start?) ---")
    
    if 'ml_category' in undp.columns:
        timing = undp.groupby('ml_category')['start_date'].min().sort_values()
        print("\nFirst project by category:")
        for cat, date in timing.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {cat}")
    
    print("\n--- Budget Variation ---")
    if 'total_budget' in undp.columns:
        print(f"  Total budget: ${undp['total_budget'].sum():,.0f}")
        print(f"  Mean per project: ${undp['total_budget'].mean():,.0f}")
        print(f"  Range: ${undp['total_budget'].min():,.0f} - ${undp['total_budget'].max():,.0f}")
    
    print("\n--- Key Intervention Dates ---")
    
    # Define intervention types with keywords
    interventions = {
        'any_undp': undp['start_date'].min(),
        'debris_clearance': undp[undp['title'].str.lower().str.contains('debris|rubble|waste', na=False)]['start_date'].min() if len(undp[undp['title'].str.lower().str.contains('debris|rubble|waste', na=False)]) > 0 else None,
        'cash_transfer': undp[undp['title'].str.lower().str.contains('cash|livelihood|employment', na=False)]['start_date'].min() if len(undp[undp['title'].str.lower().str.contains('cash|livelihood|employment', na=False)]) > 0 else None,
        'infrastructure': undp[undp['title'].str.lower().str.contains('infrastructure|housing|shelter|reconstruct', na=False)]['start_date'].min() if len(undp[undp['title'].str.lower().str.contains('infrastructure|housing|shelter|reconstruct', na=False)]) > 0 else None,
        'health': undp[undp['title'].str.lower().str.contains('health|medical|hospital', na=False)]['start_date'].min() if len(undp[undp['title'].str.lower().str.contains('health|medical|hospital', na=False)]) > 0 else None,
    }
    
    interventions = {k: v for k, v in interventions.items() if v is not None and pd.notna(v)}
    
    for name, date in sorted(interventions.items(), key=lambda x: x[1]):
        print(f"  {name}: {date.strftime('%Y-%m-%d')}")
    
    return interventions


def get_all_outcomes(panel, raster_panel):
    """Get all available outcome variables."""
    
    outcomes = {
        'economic': [],
        'physical': [],
        'vegetation': [],
        'prices': [],
        'other': []
    }
    
    # From main panel
    if 'nightlights' in panel.columns and panel['nightlights'].notna().sum() > 50:
        outcomes['economic'].append('nightlights')
    
    for var in ['ndvi', 'ndbi', 'bsi']:
        if var in panel.columns and panel[var].notna().sum() > 50:
            if var == 'ndvi':
                outcomes['vegetation'].append(var)
            else:
                outcomes['physical'].append(var)
    
    for var in ['vv', 'vh', 'rvi']:
        if var in panel.columns and panel[var].notna().sum() > 50:
            outcomes['physical'].append(var)
    
    price_vars = [c for c in panel.columns if c.startswith('price_') and panel[c].notna().sum() > 50]
    outcomes['prices'] = price_vars
    
    # From raster panel (aggregated to Gaza level)
    if raster_panel is not None:
        raster_outcomes = []
        for var in ['viirs_mean', 'ndvi_mean', 'ndbi_mean', 'bsi_mean', 'sar_vh_mean']:
            if var in raster_panel.columns:
                raster_outcomes.append(var)
        outcomes['raster'] = raster_outcomes
    
    print("\n--- Available Outcomes ---")
    for category, vars in outcomes.items():
        if vars:
            print(f"  {category}: {vars}")
    
    return outcomes


def run_its_analysis(panel, outcome, intervention_dates, controls=None):
    """
    Interrupted Time Series for Gaza.
    Tests whether outcome trajectory changes at intervention dates.
    """
    
    gaza = panel[panel['is_gaza']].copy()
    
    # Aggregate to Gaza-level time series
    agg_dict = {outcome: 'mean', 'time_idx': 'first'}
    if controls:
        for c in controls:
            if c in gaza.columns:
                agg_dict[c] = 'mean'
    
    ts = gaza.groupby('period_start').agg(agg_dict).reset_index()
    ts = ts.dropna(subset=[outcome])
    
    if len(ts) < 12:
        return None
    
    results = {}
    
    for intervention_name, intervention_date in intervention_dates.items():
        if pd.isna(intervention_date):
            continue
            
        treat_time = (intervention_date.year - 2023) * 12 + intervention_date.month
        
        ts_copy = ts.copy()
        ts_copy['post'] = (ts_copy['period_start'] >= intervention_date).astype(int)
        ts_copy['time_since'] = (ts_copy['time_idx'] - treat_time).clip(lower=0)
        
        formula = f'{outcome} ~ time_idx + post + time_since'
        if controls:
            valid_controls = [c for c in controls if c in ts_copy.columns]
            if valid_controls:
                formula += ' + ' + ' + '.join(valid_controls)
        
        try:
            model = smf.ols(formula, data=ts_copy).fit(cov_type='HC3')
            
            results[intervention_name] = {
                'level_shift': model.params.get('post', np.nan),
                'level_se': model.bse.get('post', np.nan),
                'level_pval': model.pvalues.get('post', np.nan),
                'trend_change': model.params.get('time_since', np.nan),
                'trend_se': model.bse.get('time_since', np.nan),
                'trend_pval': model.pvalues.get('time_since', np.nan),
                'r2': model.rsquared,
                'n': int(model.nobs)
            }
        except:
            continue
    
    return results


def run_dose_response(panel, outcome, undp, controls=None):
    """
    Test if more UNDP projects/budget → better outcomes.
    This is dose-response analysis at the Gaza level.
    """
    
    gaza = panel[panel['is_gaza']].copy()
    
    # Create cumulative UNDP variables
    gaza['undp_count'] = 0
    gaza['undp_budget'] = 0.0
    
    for idx, row in gaza.iterrows():
        period = row['period_start']
        projects = undp[undp['start_date'] <= period]
        gaza.loc[idx, 'undp_count'] = len(projects)
        if 'total_budget' in undp.columns:
            gaza.loc[idx, 'undp_budget'] = projects['total_budget'].sum() / 1e6  # Millions
    
    # Aggregate to time series
    ts = gaza.groupby('period_start').agg({
        outcome: 'mean',
        'undp_count': 'first',
        'undp_budget': 'first',
        'time_idx': 'first',
        **{c: 'mean' for c in (controls or []) if c in gaza.columns}
    }).reset_index()
    
    ts = ts.dropna(subset=[outcome])
    
    results = {}
    
    # Model 1: Count
    formula = f'{outcome} ~ undp_count + time_idx'
    if controls:
        valid = [c for c in controls if c in ts.columns]
        if valid:
            formula += ' + ' + ' + '.join(valid)
    
    try:
        model = smf.ols(formula, data=ts).fit(cov_type='HC3')
        results['count'] = {
            'coef': model.params['undp_count'],
            'se': model.bse['undp_count'],
            'pval': model.pvalues['undp_count'],
            'r2': model.rsquared
        }
    except:
        pass
    
    # Model 2: Budget (if available)
    if ts['undp_budget'].sum() > 0:
        formula = f'{outcome} ~ undp_budget + time_idx'
        if controls:
            valid = [c for c in controls if c in ts.columns]
            if valid:
                formula += ' + ' + ' + '.join(valid)
        
        try:
            model = smf.ols(formula, data=ts).fit(cov_type='HC3')
            results['budget'] = {
                'coef': model.params['undp_budget'],
                'se': model.bse['undp_budget'],
                'pval': model.pvalues['undp_budget'],
                'r2': model.rsquared
            }
        except:
            pass
    
    return results


def create_comprehensive_timeline_plot(panel, undp, outcomes, output_path):
    """
    Create a comprehensive plot showing:
    - Key outcomes over time
    - UNDP intervention dates marked
    - Clear visual of what we're measuring
    """
    
    gaza = panel[panel['is_gaza']].copy()
    
    # Select key outcomes
    plot_outcomes = []
    for var in ['nightlights', 'ndvi', 'vh']:
        if var in gaza.columns and gaza[var].notna().sum() > 10:
            plot_outcomes.append(var)
    
    if not plot_outcomes:
        return
    
    n_plots = len(plot_outcomes)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Get UNDP intervention dates
    intervention_dates = {
        'First UNDP': undp['start_date'].min(),
    }
    
    if 'ml_category' in undp.columns:
        for cat in undp['ml_category'].unique():
            cat_date = undp[undp['ml_category'] == cat]['start_date'].min()
            if pd.notna(cat_date):
                intervention_dates[cat] = cat_date
    
    # Aggregate to time series
    ts = gaza.groupby('period_start')[plot_outcomes].mean().reset_index()
    
    colors = {'nightlights': '#1f77b4', 'ndvi': '#2ca02c', 'vh': '#d62728', 
              'ndbi': '#ff7f0e', 'bsi': '#9467bd'}
    
    for ax, outcome in zip(axes, plot_outcomes):
        # Plot outcome
        ax.plot(ts['period_start'], ts[outcome], 'o-', 
                color=colors.get(outcome, 'blue'), linewidth=2, markersize=4,
                label=f'Gaza {outcome}')
        
        # Shade pre-conflict
        ax.axvspan(ts['period_start'].min(), CONFLICT_DATE, 
                   alpha=0.1, color='green', label='Pre-conflict')
        
        # Conflict line
        ax.axvline(CONFLICT_DATE, color='red', linestyle='--', linewidth=2, label='Oct 7 Conflict')
        
        # UNDP intervention lines
        intervention_colors = plt.cm.Set2(np.linspace(0, 1, len(intervention_dates)))
        for (name, date), color in zip(intervention_dates.items(), intervention_colors):
            if pd.notna(date) and date > CONFLICT_DATE:
                ax.axvline(date, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.text(date, ax.get_ylim()[1], name, rotation=90, va='top', ha='right', fontsize=8)
        
        ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Date', fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle('Gaza Outcomes Timeline with UNDP Intervention Dates', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ {output_path.name}")


def main():
    print("\n" + "=" * 70)
    print("COMPREHENSIVE UNDP IMPACT ANALYSIS")
    print("Honest Assessment of What We Can Measure")
    print("=" * 70)
    
    # Load data
    panel, raster_panel, undp = load_all_data()
    
    # Analyze UNDP variation
    intervention_dates = analyze_undp_variation(undp)
    
    # Get outcomes
    outcomes = get_all_outcomes(panel, raster_panel)
    
    # Flatten outcomes for analysis
    all_outcomes = []
    for category, vars in outcomes.items():
        if category != 'raster':  # Handle raster separately
            all_outcomes.extend(vars)
    
    controls = ['rainfall'] if 'rainfall' in panel.columns else []
    
    # ==========================================================================
    # ANALYSIS 1: ITS for ALL outcomes
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: INTERRUPTED TIME SERIES (All Outcomes)")
    print("=" * 70)
    print("\nQuestion: Did outcomes change after UNDP intervention dates?")
    print("Caveat: Cannot separate UNDP from other concurrent interventions\n")
    
    its_results = []
    
    for outcome in all_outcomes:
        print(f"\n--- {outcome} ---")
        
        results = run_its_analysis(panel, outcome, intervention_dates, controls)
        
        if results:
            for intervention, res in results.items():
                sig_level = '***' if res['level_pval'] < 0.01 else ('**' if res['level_pval'] < 0.05 else ('*' if res['level_pval'] < 0.1 else ''))
                sig_trend = '***' if res['trend_pval'] < 0.01 else ('**' if res['trend_pval'] < 0.05 else ('*' if res['trend_pval'] < 0.1 else ''))
                
                print(f"  {intervention}:")
                print(f"    Level shift: {res['level_shift']:.4f} {sig_level}")
                print(f"    Trend change: {res['trend_change']:.4f} {sig_trend}")
                
                its_results.append({
                    'outcome': outcome,
                    'intervention': intervention,
                    'level_shift': res['level_shift'],
                    'level_pval': res['level_pval'],
                    'trend_change': res['trend_change'],
                    'trend_pval': res['trend_pval'],
                    'r2': res['r2'],
                    'n': res['n']
                })
    
    # ==========================================================================
    # ANALYSIS 2: DOSE-RESPONSE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: DOSE-RESPONSE")
    print("=" * 70)
    print("\nQuestion: Do more UNDP projects → better outcomes?")
    print("Caveat: Correlation, not causation (projects may target worst areas)\n")
    
    dose_results = []
    
    for outcome in all_outcomes[:5]:  # Top 5 outcomes
        print(f"\n--- {outcome} ---")
        
        results = run_dose_response(panel, outcome, undp, controls)
        
        for measure, res in results.items():
            sig = '***' if res['pval'] < 0.01 else ('**' if res['pval'] < 0.05 else ('*' if res['pval'] < 0.1 else ''))
            print(f"  {measure}: β = {res['coef']:.4f} {sig} (p={res['pval']:.3f})")
            
            dose_results.append({
                'outcome': outcome,
                'measure': measure,
                'coefficient': res['coef'],
                'pval': res['pval'],
                'r2': res['r2']
            })
    
    # ==========================================================================
    # CREATE VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_comprehensive_timeline_plot(panel, undp, outcomes, FIGURES_DIR / 'undp_timeline_comprehensive.png')
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if its_results:
        its_df = pd.DataFrame(its_results)
        its_df.to_csv(TABLES_DIR / 'its_all_outcomes.csv', index=False)
        print(f"  ✓ its_all_outcomes.csv ({len(its_df)} rows)")
    
    if dose_results:
        dose_df = pd.DataFrame(dose_results)
        dose_df.to_csv(TABLES_DIR / 'dose_response_results.csv', index=False)
        print(f"  ✓ dose_response_results.csv ({len(dose_df)} rows)")
    
    # ==========================================================================
    # SUMMARY AND INTERPRETATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: WHAT WE CAN AND CANNOT SAY")
    print("=" * 70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    WHAT THIS ANALYSIS MEASURES                       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  ✓ Whether Gaza's outcomes changed after UNDP intervention dates     ║
    ║  ✓ Whether more UNDP projects correlate with better outcomes         ║
    ║  ✓ The timing of changes relative to specific intervention types     ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                  WHAT THIS ANALYSIS CANNOT IDENTIFY                  ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  ✗ UNDP-specific effects isolated from other agencies                ║
    ║    (UNRWA, WFP, WHO, OCHA, NGOs all intervened simultaneously)       ║
    ║                                                                      ║
    ║  ✗ Causal UNDP impact without counterfactual                         ║
    ║    (No "what if UNDP hadn't intervened" scenario observable)         ║
    ║                                                                      ║
    ║  ✗ District-level UNDP effects (all Gaza districts got UNDP)         ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                      RECOMMENDED INTERPRETATION                       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  "Following the initiation of UNDP [intervention type] projects      ║
    ║   in [month], [outcome] showed a [direction] of [X units].           ║
    ║   This timing is consistent with UNDP contributions to recovery,     ║
    ║   though we cannot isolate UNDP effects from other concurrent        ║
    ║   humanitarian interventions."                                       ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Print significant results
    if its_results:
        print("\n--- STATISTICALLY SIGNIFICANT FINDINGS (p < 0.10) ---\n")
        its_df = pd.DataFrame(its_results)
        
        # Level shifts
        sig_level = its_df[its_df['level_pval'] < 0.10].copy()
        if len(sig_level) > 0:
            print("Level Shifts (immediate change after intervention):")
            for _, row in sig_level.iterrows():
                direction = "↑" if row['level_shift'] > 0 else "↓"
                print(f"  {row['outcome']} after {row['intervention']}: {direction} {abs(row['level_shift']):.3f} (p={row['level_pval']:.3f})")
        
        # Trend changes
        sig_trend = its_df[its_df['trend_pval'] < 0.10].copy()
        if len(sig_trend) > 0:
            print("\nTrend Changes (slope change after intervention):")
            for _, row in sig_trend.iterrows():
                direction = "accelerating" if row['trend_change'] > 0 else "decelerating"
                print(f"  {row['outcome']} after {row['intervention']}: {direction} by {abs(row['trend_change']):.3f}/month (p={row['trend_pval']:.3f})")
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
