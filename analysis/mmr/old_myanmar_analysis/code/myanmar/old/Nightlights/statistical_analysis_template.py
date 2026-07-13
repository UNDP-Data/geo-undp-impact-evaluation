#!/usr/bin/env python3
"""
UNDP Myanmar - Statistical Analysis Template
==============================================

This script shows how to analyze the nightlights panel data 
using difference-in-differences (DiD) methodology.

Run this AFTER generating the panel data with undp_myanmar_nightlights_analysis.py

Requirements:
    pip install pandas numpy matplotlib statsmodels seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(data_file='outputs/data/panel_did_analysis.csv'):
    """Load the DiD panel dataset."""
    print("Loading panel data...")
    df = pd.read_csv(data_file)
    df['month'] = pd.to_datetime(df['month'])
    
    print(f"\nDataset summary:")
    print(f"  • Total observations: {len(df):,}")
    print(f"  • Unique cells: {df['cell_id'].nunique():,}")
    print(f"  • Months: {df['month'].nunique()}")
    print(f"  • Treated cells: {(df['treated']==1).sum() // df['month'].nunique():,}")
    print(f"  • Control cells: {(df['treated']==0).sum() // df['month'].nunique():,}")
    
    return df


def descriptive_stats(df):
    """Calculate descriptive statistics."""
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    
    # Overall stats
    print("\nOverall nightlight statistics:")
    print(df['nightlight'].describe())
    
    # By treatment status
    print("\nNightlight by treatment status:")
    print(df.groupby('treated')['nightlight'].describe())
    
    # By period
    print("\nNightlight by period:")
    print(df.groupby('post')['nightlight'].describe())
    
    # Treatment × Period
    print("\nNightlight by treatment × period:")
    print(df.groupby(['treated', 'post'])['log_nightlight'].mean())


def plot_trends(df, output_dir='outputs/figures'):
    """Plot nightlight trends for treated vs control groups."""
    print("\n" + "="*70)
    print("PLOTTING TRENDS")
    print("="*70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate mean by group and month
    trends = df.groupby(['month', 'treated'])['log_nightlight'].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for treatment in [0, 1]:
        data = trends[trends['treated'] == treatment]
        label = 'Treated (UNDP areas)' if treatment == 1 else 'Control'
        color = 'blue' if treatment == 1 else 'gray'
        ax.plot(data['month'], data['log_nightlight'], 
                label=label, marker='o', linewidth=2, color=color, alpha=0.7)
    
    # Add vertical line at event
    event_month = df[df['post'] == 1]['month'].min()
    ax.axvline(event_month, color='red', linestyle='--', linewidth=2, 
               label='Event', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log Nightlight Radiance', fontsize=12, fontweight='bold')
    ax.set_title('Nightlight Trends: Treated vs Control Areas', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    output_file = Path(output_dir) / 'trends_treated_vs_control.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # Plot with separate pre/post periods
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pre-period
    pre_data = trends[trends['month'] < event_month]
    for treatment in [0, 1]:
        data = pre_data[pre_data['treated'] == treatment]
        label = 'Treated' if treatment == 1 else 'Control'
        color = 'blue' if treatment == 1 else 'gray'
        ax1.plot(data['month'], data['log_nightlight'], 
                label=label, marker='o', linewidth=2, color=color, alpha=0.7)
    
    ax1.set_title('PRE-Event Trends (Parallel Trends Check)', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Log Nightlight Radiance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Post-period
    post_data = trends[trends['month'] >= event_month]
    for treatment in [0, 1]:
        data = post_data[post_data['treated'] == treatment]
        label = 'Treated' if treatment == 1 else 'Control'
        color = 'blue' if treatment == 1 else 'gray'
        ax2.plot(data['month'], data['log_nightlight'], 
                label=label, marker='o', linewidth=2, color=color, alpha=0.7)
    
    ax2.set_title('POST-Event Trends', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Log Nightlight Radiance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'trends_pre_post_comparison.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def basic_did_regression(df):
    """Run basic DiD regression."""
    print("\n" + "="*70)
    print("BASIC DID REGRESSION")
    print("="*70)
    
    try:
        import statsmodels.formula.api as smf
        
        # Basic DiD
        print("\nModel: log_nightlight ~ treated + post + treated_x_post")
        model = smf.ols('log_nightlight ~ treated + post + treated_x_post', 
                       data=df).fit()
        
        print(model.summary())
        
        # Extract DiD estimate
        did_estimate = model.params['treated_x_post']
        did_se = model.bse['treated_x_post']
        did_pval = model.pvalues['treated_x_post']
        
        print("\n" + "="*70)
        print("DiD ESTIMATE (Treatment Effect)")
        print("="*70)
        print(f"  Coefficient: {did_estimate:.4f}")
        print(f"  Std. Error:  {did_se:.4f}")
        print(f"  P-value:     {did_pval:.4f}")
        print(f"  Significant: {'Yes ***' if did_pval < 0.01 else 'Yes **' if did_pval < 0.05 else 'Yes *' if did_pval < 0.1 else 'No'}")
        
        # Interpretation
        pct_change = (np.exp(did_estimate) - 1) * 100
        print(f"\n  Interpretation:")
        print(f"  UNDP areas experienced a {pct_change:.2f}% {'increase' if pct_change > 0 else 'decrease'}")
        print(f"  in nightlight radiance compared to control areas after the event.")
        
        return model
        
    except ImportError:
        print("\nERROR: statsmodels not installed")
        print("Install with: pip install statsmodels")
        return None


def fixed_effects_did(df):
    """Run DiD with fixed effects."""
    print("\n" + "="*70)
    print("DID WITH FIXED EFFECTS")
    print("="*70)
    
    try:
        import statsmodels.formula.api as smf
        
        # Two-way fixed effects
        print("\nModel: log_nightlight ~ treated_x_post + C(cell_id) + C(month)")
        print("(This may take a minute with large datasets...)")
        
        model_fe = smf.ols(
            'log_nightlight ~ treated_x_post + C(cell_id) + C(month)', 
            data=df
        ).fit()
        
        # Just show the treatment effect
        did_estimate = model_fe.params['treated_x_post']
        did_se = model_fe.bse['treated_x_post']
        did_pval = model_fe.pvalues['treated_x_post']
        
        print("\n" + "="*70)
        print("FIXED EFFECTS DiD ESTIMATE")
        print("="*70)
        print(f"  Coefficient: {did_estimate:.4f}")
        print(f"  Std. Error:  {did_se:.4f}")
        print(f"  P-value:     {did_pval:.4f}")
        print(f"  R-squared:   {model_fe.rsquared:.4f}")
        
        pct_change = (np.exp(did_estimate) - 1) * 100
        print(f"\n  Interpretation (controlling for cell and time fixed effects):")
        print(f"  UNDP treatment associated with {pct_change:.2f}% {'increase' if pct_change > 0 else 'decrease'}")
        print(f"  in nightlight radiance.")
        
        return model_fe
        
    except ImportError:
        print("\nERROR: statsmodels not installed")
        print("Install with: pip install statsmodels")
        return None


def event_study_plot(df, output_dir='outputs/figures'):
    """Create event study plot."""
    print("\n" + "="*70)
    print("EVENT STUDY ANALYSIS")
    print("="*70)
    
    try:
        import statsmodels.formula.api as smf
        
        # Get time range
        time_range = df['months_from_event'].unique()
        time_range = sorted([t for t in time_range if -12 <= t <= 12])
        
        # Run regression with time interactions (excluding t=-1 as reference)
        formula = 'log_nightlight ~ treated + '
        formula += ' + '.join([
            f'I(months_from_event == {t}):treated' 
            for t in time_range if t != -1
        ])
        
        print(f"\nRunning event study regression...")
        model = smf.ols(formula, data=df).fit()
        
        # Extract coefficients
        coefs = []
        ses = []
        time_points = []
        
        for t in time_range:
            if t == -1:
                coefs.append(0)  # Reference period
                ses.append(0)
                time_points.append(t)
            else:
                param_name = f'I(months_from_event == {t}):treated[T.True]'
                if param_name in model.params:
                    coefs.append(model.params[param_name])
                    ses.append(model.bse[param_name])
                    time_points.append(t)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        coefs = np.array(coefs)
        ses = np.array(ses)
        ci_lower = coefs - 1.96 * ses
        ci_upper = coefs + 1.96 * ses
        
        ax.plot(time_points, coefs, 'o-', color='blue', linewidth=2, 
                markersize=8, label='DiD Estimate')
        ax.fill_between(time_points, ci_lower, ci_upper, 
                        alpha=0.3, color='blue', label='95% CI')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(-0.5, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Event')
        
        ax.set_xlabel('Months from Event', fontsize=12, fontweight='bold')
        ax.set_ylabel('Treatment Effect', fontsize=12, fontweight='bold')
        ax.set_title('Event Study: Dynamic Treatment Effects', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = Path(output_dir) / 'event_study.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
        
        print("\n  Event study plot shows:")
        print("  • Pre-trends (before event): Should be close to zero")
        print("  • Post-treatment effects: Changes after event")
        
    except ImportError:
        print("\nERROR: statsmodels not installed")
        print("Install with: pip install statsmodels")
    except Exception as e:
        print(f"\nEvent study failed: {e}")
        print("This is often due to insufficient time periods")


def robustness_checks(df):
    """Perform robustness checks."""
    print("\n" + "="*70)
    print("ROBUSTNESS CHECKS")
    print("="*70)
    
    try:
        import statsmodels.formula.api as smf
        
        # 1. Drop very first and last months
        print("\n1. Excluding first and last months:")
        months_sorted = sorted(df['month'].unique())
        df_robust1 = df[
            (df['month'] != months_sorted[0]) & 
            (df['month'] != months_sorted[-1])
        ]
        model1 = smf.ols('log_nightlight ~ treated + post + treated_x_post', 
                        data=df_robust1).fit()
        print(f"   DiD estimate: {model1.params['treated_x_post']:.4f} "
              f"(p={model1.pvalues['treated_x_post']:.4f})")
        
        # 2. Use raw nightlight (not log)
        print("\n2. Using raw nightlight (not log-transformed):")
        model2 = smf.ols('nightlight ~ treated + post + treated_x_post', 
                        data=df).fit()
        print(f"   DiD estimate: {model2.params['treated_x_post']:.4f} "
              f"(p={model2.pvalues['treated_x_post']:.4f})")
        
        # 3. Drop cells with zero nightlight
        print("\n3. Excluding cells with zero nightlight:")
        df_robust3 = df[df['nightlight'] > 0]
        model3 = smf.ols('log_nightlight ~ treated + post + treated_x_post', 
                        data=df_robust3).fit()
        print(f"   DiD estimate: {model3.params['treated_x_post']:.4f} "
              f"(p={model3.pvalues['treated_x_post']:.4f})")
        print(f"   (Used {len(df_robust3):,} of {len(df):,} observations)")
        
        print("\n  ✓ Check if results are consistent across specifications")
        
    except ImportError:
        print("\nERROR: statsmodels not installed")
    except Exception as e:
        print(f"\nRobustness checks failed: {e}")


def save_results_table(models, output_dir='outputs/data'):
    """Save regression results to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create results summary
    results = []
    for name, model in models.items():
        if model is not None:
            results.append({
                'Model': name,
                'DiD_Coefficient': model.params.get('treated_x_post', np.nan),
                'Std_Error': model.bse.get('treated_x_post', np.nan),
                'P_value': model.pvalues.get('treated_x_post', np.nan),
                'N_obs': int(model.nobs),
                'R_squared': model.rsquared
            })
    
    results_df = pd.DataFrame(results)
    output_file = Path(output_dir) / 'did_results_summary.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results table: {output_file}")


def main():
    """Run complete analysis pipeline."""
    
    print("="*70)
    print("  UNDP MYANMAR - STATISTICAL ANALYSIS")
    print("  Difference-in-Differences Analysis")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Descriptive statistics
    descriptive_stats(df)
    
    # Plot trends
    plot_trends(df)
    
    # Run regressions
    models = {}
    models['Basic DiD'] = basic_did_regression(df)
    models['Fixed Effects DiD'] = fixed_effects_did(df)
    
    # Event study
    event_study_plot(df)
    
    # Robustness
    robustness_checks(df)
    
    # Save results
    save_results_table(models)
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review figures in outputs/figures/")
    print("  2. Check did_results_summary.csv")
    print("  3. Consider additional robustness checks")
    print("  4. Write up results for report/paper")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
