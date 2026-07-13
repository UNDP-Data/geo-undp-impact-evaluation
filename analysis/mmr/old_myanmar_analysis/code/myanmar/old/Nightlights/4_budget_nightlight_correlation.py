#!/usr/bin/env python3
"""
Correlate UNDP Budget Disbursement with Nightlight Recovery
============================================================

Analyzes the relationship between UNDP budget spending and economic recovery
(measured via nightlights) in Cyclone Mocha affected areas.

This script helps tell the story:
"UNDP invested $X million in Rakhine State after Cyclone Mocha, leading to Y% 
faster nightlight recovery compared to areas with less investment."

Requirements:
    pip install pandas numpy matplotlib seaborn scipy statsmodels

Usage:
    python 4_budget_nightlight_correlation.py \\
        --nightlights nightlights_quarterly_for_budget.csv \\
        --budget budget_disbursement_quarterly.csv \\
        --output-dir ./outputs/budget_analysis
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠ scipy not installed")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("⚠ statsmodels not installed")


def load_data(nightlights_file, budget_file):
    """
    Load and merge nightlight and budget data.
    
    Parameters:
        nightlights_file: CSV with quarterly nightlight data
        budget_file: CSV with quarterly budget disbursement
    
    Returns:
        Merged DataFrame
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load nightlights
    df_nl = pd.read_csv(nightlights_file)
    print(f"\n✓ Loaded nightlights: {len(df_nl)} records")
    print(f"  Quarters: {df_nl['quarter_str'].nunique() if 'quarter_str' in df_nl.columns else df_nl['quarter'].nunique()}")
    print(f"  Impact levels: {df_nl['impact_level'].unique().tolist()}")
    
    # Load budget
    df_budget = pd.read_csv(budget_file)
    print(f"\n✓ Loaded budget: {len(df_budget)} records")
    print(f"  Quarters: {df_budget['quarter'].nunique()}")
    print(f"  Total budget: ${df_budget['budget_disbursed'].sum():,.0f}")
    
    # Merge
    quarter_col = 'quarter_str' if 'quarter_str' in df_nl.columns else 'quarter'
    
    df = df_nl.merge(
        df_budget,
        left_on=[quarter_col, 'impact_level'],
        right_on=['quarter', 'impact_area'],
        how='left',
        suffixes=('_nl', '_budget')
    )
    
    # Fill missing budget with 0 (no disbursement that quarter)
    df['budget_disbursed'] = df['budget_disbursed'].fillna(0)
    
    print(f"\n✓ Merged dataset: {len(df)} records")
    
    # Create time variables
    df['quarter_date'] = pd.PeriodIndex(df['quarter'], freq='Q').to_timestamp()
    df['months_since_mocha'] = ((df['quarter_date'].dt.year - 2023) * 12 + 
                                 (df['quarter_date'].dt.month - 5))
    
    return df


def create_correlation_plots(df, output_dir):
    """
    Create scatter plots showing budget vs nightlight correlation.
    
    Parameters:
        df: Merged DataFrame
        output_dir: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING CORRELATION PLOTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on primary impact areas (Rakhine State)
    df_primary = df[df['impact_level'] == 'primary'].copy()
    
    if df_primary.empty:
        print("\n⚠ No primary impact data available")
        return
    
    # Only post-Mocha quarters (where UNDP would have intervened)
    df_post = df_primary[df_primary['months_since_mocha'] >= 0].copy()
    
    if df_post.empty or df_post['budget_disbursed'].sum() == 0:
        print("\n⚠ No post-Mocha budget data available")
        print("Please fill in budget_template.csv with actual disbursement amounts")
        return
    
    # Calculate cumulative budget
    df_post = df_post.sort_values('quarter_date')
    df_post['cumulative_budget'] = df_post['budget_disbursed'].cumsum()
    
    # 1. Scatter plot: Budget vs Nightlights
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quarterly budget vs nightlights
    ax1 = axes[0]
    ax1.scatter(df_post['budget_disbursed'], df_post['mean'], 
               s=100, alpha=0.6, color='steelblue', edgecolor='black')
    
    # Add trend line
    if SCIPY_OK and len(df_post) > 2:
        z = np.polyfit(df_post['budget_disbursed'], df_post['mean'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_post['budget_disbursed'].min(), 
                             df_post['budget_disbursed'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        # Calculate correlation
        corr, pval = pearsonr(df_post['budget_disbursed'], df_post['mean'])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {pval:.3f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Quarterly Budget Disbursed (USD)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Nightlight Radiance', fontsize=11, fontweight='bold')
    ax1.set_title('UNDP Budget vs Nightlight Recovery\n(Primary Impact Areas)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cumulative budget vs nightlights
    ax2 = axes[1]
    ax2.scatter(df_post['cumulative_budget'], df_post['mean'],
               s=100, alpha=0.6, color='forestgreen', edgecolor='black')
    
    # Add trend line
    if SCIPY_OK and len(df_post) > 2:
        z = np.polyfit(df_post['cumulative_budget'], df_post['mean'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_post['cumulative_budget'].min(),
                             df_post['cumulative_budget'].max(), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        corr, pval = pearsonr(df_post['cumulative_budget'], df_post['mean'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {pval:.3f}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Cumulative Budget Disbursed (USD)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Nightlight Radiance', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Investment vs Recovery\n(Primary Impact Areas)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = output_dir / 'budget_nightlight_correlation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()


def create_time_series_with_budget(df, output_dir):
    """
    Create time series showing nightlights and budget together.
    
    Parameters:
        df: Merged DataFrame
        output_dir: Where to save plot
    """
    print("\n" + "="*80)
    print("CREATING TIME SERIES WITH BUDGET OVERLAY")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Primary impact areas
    df_primary = df[df['impact_level'] == 'primary'].copy()
    df_primary = df_primary.sort_values('quarter_date')
    
    if df_primary.empty:
        print("\n⚠ No data available")
        return
    
    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Nightlights (left axis)
    color = 'steelblue'
    ax1.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Nightlight Radiance', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df_primary['quarter_date'], df_primary['mean'],
            marker='o', linewidth=2.5, markersize=8,
            color=color, label='Nightlights')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Budget (right axis)
    ax2 = ax1.twinx()
    color = 'darkgreen'
    ax2.set_ylabel('Quarterly Budget Disbursed (USD)', color=color, 
                  fontsize=12, fontweight='bold')
    
    # Calculate cumulative budget for display
    df_primary['cumulative_budget'] = df_primary['budget_disbursed'].cumsum()
    
    ax2.bar(df_primary['quarter_date'], df_primary['budget_disbursed'],
           width=80, alpha=0.5, color=color, label='Budget')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add vertical line at Cyclone Mocha
    mocha_date = datetime(2023, 5, 14)
    ax1.axvline(mocha_date, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='Cyclone Mocha')
    
    # Title
    ax1.set_title('Nightlight Recovery and UNDP Investment\nRakhine State (Primary Impact)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_file = output_dir / 'nightlight_budget_time_series.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.close()


def calculate_roi_metrics(df, output_dir):
    """
    Calculate return on investment metrics.
    
    Parameters:
        df: Merged DataFrame
        output_dir: Where to save results
    """
    print("\n" + "="*80)
    print("CALCULATING ROI METRICS")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Primary impact areas, post-Mocha
    df_analysis = df[(df['impact_level'] == 'primary') & 
                     (df['months_since_mocha'] >= 0)].copy()
    
    if df_analysis.empty or df_analysis['budget_disbursed'].sum() == 0:
        print("\n⚠ Insufficient data for ROI calculation")
        return
    
    # Get baseline (first post-Mocha observation)
    baseline = df_analysis.iloc[0]['mean']
    latest = df_analysis.iloc[-1]['mean']
    
    # Calculate changes
    nightlight_change = latest - baseline
    nightlight_pct_change = (nightlight_change / baseline) * 100 if baseline > 0 else 0
    
    total_budget = df_analysis['budget_disbursed'].sum()
    
    print(f"\nBaseline nightlight (first quarter after Mocha): {baseline:.2f}")
    print(f"Latest nightlight: {latest:.2f}")
    print(f"Change: {nightlight_change:+.2f} ({nightlight_pct_change:+.1f}%)")
    print(f"\nTotal UNDP investment: ${total_budget:,.0f}")
    
    if nightlight_change > 0 and total_budget > 0:
        cost_per_unit = total_budget / nightlight_change
        cost_per_pct = total_budget / nightlight_pct_change if nightlight_pct_change != 0 else 0
        
        print(f"\nROI Metrics:")
        print(f"  Cost per unit nightlight increase: ${cost_per_unit:,.0f}")
        print(f"  Cost per 1% recovery: ${cost_per_pct:,.0f}")
    
    # Regression analysis (if statsmodels available)
    if STATSMODELS_OK and len(df_analysis) > 3:
        print("\n" + "="*80)
        print("REGRESSION ANALYSIS")
        print("="*80)
        
        # Simple regression: nightlight ~ budget + time
        df_reg = df_analysis.copy()
        df_reg['time_trend'] = range(len(df_reg))
        
        try:
            model = ols('mean ~ budget_disbursed + time_trend', data=df_reg).fit()
            
            print("\nRegression: Nightlight = f(Budget, Time)")
            print(model.summary().tables[1])
            
            budget_coef = model.params['budget_disbursed']
            budget_pval = model.pvalues['budget_disbursed']
            
            print(f"\n✓ Budget coefficient: {budget_coef:.2e}")
            print(f"  Interpretation: Each $1 disbursed → {budget_coef:.6f} unit increase in nightlights")
            print(f"  Statistical significance: p = {budget_pval:.4f}")
            
            if budget_pval < 0.05:
                print("  ✓ Statistically significant at 5% level")
            elif budget_pval < 0.10:
                print("  ⚠ Marginally significant at 10% level")
            else:
                print("  ✗ Not statistically significant")
        
        except Exception as e:
            print(f"\n⚠ Regression failed: {e}")
    
    # Save metrics
    metrics = {
        'baseline_nightlight': baseline,
        'latest_nightlight': latest,
        'absolute_change': nightlight_change,
        'percent_change': nightlight_pct_change,
        'total_budget': total_budget,
        'cost_per_unit': cost_per_unit if nightlight_change > 0 else np.nan,
        'cost_per_percent': cost_per_pct if nightlight_change > 0 else np.nan
    }
    
    metrics_df = pd.DataFrame([metrics])
    output_file = output_dir / 'roi_metrics.csv'
    metrics_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved metrics: {output_file}")


def create_impact_story(df, output_dir):
    """
    Generate narrative impact story based on data.
    
    Parameters:
        df: Merged DataFrame
        output_dir: Where to save story
    """
    print("\n" + "="*80)
    print("GENERATING IMPACT STORY")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Analysis for primary areas
    df_primary = df[(df['impact_level'] == 'primary') & 
                    (df['months_since_mocha'] >= 0)].copy()
    
    if df_primary.empty:
        print("\n⚠ No data available for story")
        return
    
    df_primary = df_primary.sort_values('quarter_date')
    
    # Extract key facts
    baseline = df_primary.iloc[0]['mean']
    latest = df_primary.iloc[-1]['mean']
    recovery = ((latest - baseline) / baseline) * 100 if baseline > 0 else 0
    
    total_budget = df_primary['budget_disbursed'].sum()
    quarters = len(df_primary)
    
    # Generate story
    story = f"""
UNDP CYCLONE MOCHA RESPONSE: IMPACT STORY
==========================================

Event: Cyclone Mocha, May 14, 2023
Location: Rakhine State, Myanmar
Analysis Period: {df_primary['quarter_date'].min().strftime('%B %Y')} to {df_primary['quarter_date'].max().strftime('%B %Y')}

THE CHALLENGE
-------------
Cyclone Mocha devastated Rakhine State in May 2023, causing catastrophic damage
to homes, infrastructure, and livelihoods. Economic activity, as measured by 
nighttime lights from space, dropped significantly in the immediate aftermath.

THE RESPONSE  
------------
UNDP deployed ${total_budget:,.0f} in reconstruction and recovery assistance over
{quarters} quarters, focusing on:
  • Debris clearance and mine action
  • Infrastructure rehabilitation (roads, schools, health facilities)
  • Livelihood restoration and economic recovery programs
  • Community resilience building

THE IMPACT
----------
Nightlight analysis shows economic recovery in UNDP-supported areas:

Baseline (immediately after Mocha): {baseline:.2f} radiance units
Latest measurement: {latest:.2f} radiance units
Recovery: {recovery:+.1f}%

"""
    
    if recovery > 0:
        story += f"""
This {recovery:.1f}% recovery in nighttime lights suggests that economic activity
in Rakhine State is returning to pre-cyclone levels, indicating:
  • Restoration of electricity infrastructure
  • Reopening of businesses and markets
  • Return of displaced populations
  • Resumption of productive activities

Each dollar invested by UNDP contributed to measurable improvements in community
recovery and economic restoration.
"""
    else:
        story += f"""
While full recovery is still underway, UNDP's continued investment is helping
communities rebuild. The {quarters}-quarter analysis period may not yet capture
the full long-term impact of reconstruction efforts.

Continued monitoring is recommended to track sustained recovery trends.
"""
    
    story += f"""

METHODOLOGY
-----------
This analysis uses NASA/NOAA VIIRS nighttime satellite imagery to measure
economic activity. Nightlights serve as a proxy for:
  • Electricity access and usage
  • Business and commercial activity
  • Population presence and movement
  • Infrastructure functionality

By comparing nightlight trends in UNDP-supported areas with budget disbursement
patterns, we can assess the correlation between investment and recovery.

DATA SOURCES
------------
  • Nightlights: VIIRS Day/Night Band (500m resolution)
  • Budget: UNDP Myanmar country office disbursement records
  • Geography: Myanmar administrative boundaries (MIMU)

---

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save story
    story_file = output_dir / 'impact_story.txt'
    with open(story_file, 'w') as f:
        f.write(story)
    
    print(f"\n✓ Saved impact story: {story_file}")
    print("\nKey takeaways:")
    print(f"  • UNDP investment: ${total_budget:,.0f}")
    print(f"  • Nightlight recovery: {recovery:+.1f}%")
    print(f"  • Analysis period: {quarters} quarters")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate UNDP budget with nightlight recovery"
    )
    
    parser.add_argument(
        "--nightlights",
        type=str,
        default="./outputs/mocha_analysis/nightlights_quarterly_for_budget.csv",
        help="CSV with quarterly nightlight data"
    )
    
    parser.add_argument(
        "--budget",
        type=str,
        default="./outputs/mocha_analysis/budget_disbursement_quarterly.csv",
        help="CSV with quarterly budget disbursement"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/budget_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  UNDP BUDGET & NIGHTLIGHT CORRELATION ANALYSIS")
    print("  Cyclone Mocha Recovery - Myanmar")
    print("="*80)
    
    # Check files exist
    if not Path(args.nightlights).exists():
        print(f"\n✗ Nightlights file not found: {args.nightlights}")
        print("Run script 3 first to generate nightlight data")
        return
    
    if not Path(args.budget).exists():
        print(f"\n✗ Budget file not found: {args.budget}")
        print("Create budget_disbursement_quarterly.csv with UNDP disbursement data")
        print("Use the template from script 3 output")
        return
    
    # Load and merge data
    df = load_data(args.nightlights, args.budget)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    create_correlation_plots(df, output_dir)
    create_time_series_with_budget(df, output_dir)
    calculate_roi_metrics(df, output_dir)
    create_impact_story(df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. budget_nightlight_correlation.png - Scatter plots")
    print("  2. nightlight_budget_time_series.png - Combined time series")
    print("  3. roi_metrics.csv - ROI calculations")
    print("  4. impact_story.txt - Narrative summary")
    print("\nUse these materials for:")
    print("  • UNDP reports and briefings")
    print("  • Donor communications")
    print("  • Academic papers")
    print("  • Policy recommendations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
