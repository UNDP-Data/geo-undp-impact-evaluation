#!/usr/bin/env python3
"""
Comprehensive Landsat Robustness Checks
=======================================

Adapted from nightlights robustness script for Landsat satellite data.

Includes:
1. Two-Way Fixed Effects (TWFE) with township and time FE
2. Log specifications for % effects
3. Event study (leads/lags) for pre-trends testing
4. Interactive plots showing DiD slopes over time
5. Placebo tests
6. Dose-response models
7. Comparison table across specifications

Usage:
    python landsat_comprehensive_robustness.py \
        --landsat-data outputs/relaxed_threshold/landsat_did_relaxed_threshold.csv \
        --output-dir outputs/landsat_robustness_FINAL
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from linearmodels import PanelOLS
    DEPS_OK = True
except ImportError:
    DEPS_OK = False
    print("⚠ Missing dependencies. Install with:")
    print("  pip install statsmodels linearmodels --break-system-packages")

warnings.filterwarnings('ignore')

EARTHQUAKE_DATE = datetime(2025, 3, 28)


def load_and_prepare_data(csv_file):
    """Load Landsat data and prepare for panel analysis."""
    print("="*80)
    print("LOADING LANDSAT DATA")
    print("="*80)
    
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Validate March exclusion
    march_mask = (df['date'].dt.year == 2025) & (df['date'].dt.month == 3)
    if march_mask.sum() > 0:
        print(f"\n⚠️  WARNING: {march_mask.sum()} March observations found")
    else:
        print("\n✓ March 2025 properly excluded")
    
    # Calculate relative months
    df['month_int'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    def get_relative_month(row):
        if row['year'] != 2025:
            return np.nan
        month = row['month_int']
        if month <= 2:  # Jan-Feb (pre)
            return month - 3  # Jan=-2, Feb=-1
        elif month >= 4:  # Apr-Oct (post)
            return month - 4  # Apr=0, May=1, ..., Oct=6
        else:
            return np.nan
    
    df['relative_month'] = df.apply(get_relative_month, axis=1)
    
    # Create township_id if not exists
    if 'township_id' not in df.columns:
        township_ids = {name: idx for idx, name in enumerate(df['township_name'].unique())}
        df['township_id'] = df['township_name'].map(township_ids)
    
    print(f"\n✓ Loaded {len(df)} observations")
    print(f"  Townships: {df['township_name'].nunique()}")
    print(f"  Treated: {df[df['treated']==1]['township_name'].nunique()}")
    print(f"  Control: {df[df['treated']==0]['township_name'].nunique()}")
    print(f"  Months: {df['date'].nunique()}")
    
    # Show relative month distribution
    print("\n✓ Relative month distribution:")
    print(df.groupby('relative_month').size().sort_index())
    
    return df


def run_twfe_models(df, outcomes, output_dir):
    """
    Run Two-Way Fixed Effects models for each outcome.
    Township and time fixed effects with clustered standard errors.
    """
    print("\n" + "="*80)
    print("TWO-WAY FIXED EFFECTS (TWFE)")
    print("="*80)
    
    twfe_results = {}
    
    for outcome in outcomes:
        if outcome not in df.columns:
            print(f"\n⚠️  {outcome} not found, skipping")
            continue
        
        df_valid = df[df[outcome].notna()].copy()
        
        if len(df_valid) < 50:
            print(f"\n⚠️  {outcome}: insufficient data ({len(df_valid)} obs), skipping")
            continue
        
        print(f"\n{outcome.upper()}")
        print("-"*80)
        
        # Set panel structure
        df_valid = df_valid.set_index(['township_id', 'date'])
        
        try:
            # TWFE model: Y = β*treated_post + township_FE + time_FE
            formula = f'{outcome} ~ treated_post + EntityEffects + TimeEffects'
            
            model = PanelOLS.from_formula(
                formula,
                data=df_valid,
                drop_absorbed=True
            )
            
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            print(results.summary)
            
            # Extract key statistics
            coef = results.params.get('treated_post', np.nan)
            se = results.std_errors.get('treated_post', np.nan)
            tstat = results.tstats.get('treated_post', np.nan)
            pval = results.pvalues.get('treated_post', np.nan)
            
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            print(f"\nTWFE DiD Coefficient: {coef:+.4f}")
            print(f"  Std. Error: {se:.4f}")
            print(f"  T-statistic: {tstat:.2f}")
            print(f"  P-value: {pval:.4f}")
            print(f"  95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
            
            # Significance
            if pval < 0.01:
                sig = "***"
            elif pval < 0.05:
                sig = "**"
            elif pval < 0.10:
                sig = "*"
            else:
                sig = "ns"
            
            print(f"  Significance: {sig}")
            
            twfe_results[outcome] = {
                'coefficient': coef,
                'std_error': se,
                't_statistic': tstat,
                'p_value': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significance': sig,
                'n_obs': len(df_valid),
                'n_townships': df_valid.index.get_level_values('township_id').nunique()
            }
            
        except Exception as e:
            print(f"  ✗ TWFE failed: {e}")
            twfe_results[outcome] = None
    
    # Save TWFE results
    if twfe_results:
        twfe_df = pd.DataFrame(twfe_results).T
        twfe_file = output_dir / 'twfe_results_landsat.csv'
        twfe_df.to_csv(twfe_file)
        print(f"\n✓ Saved TWFE results: {twfe_file.name}")
    
    return twfe_results


def run_event_study(df, outcomes, output_dir):
    """
    Event study analysis: estimate treatment effects by relative month.
    Tests parallel pre-trends assumption.
    """
    print("\n" + "="*80)
    print("EVENT STUDY (PRE-TRENDS CHECK)")
    print("="*80)
    
    event_study_results = {}
    
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        
        df_valid = df[df[outcome].notna()].copy()
        
        if len(df_valid) < 50:
            print(f"\n⚠️  {outcome}: insufficient data, skipping")
            continue
        
        print(f"\n{outcome.upper()}")
        print("-"*80)
        
        # Create interaction terms for each relative month (omit -1 as reference)
        relative_months = sorted(df_valid['relative_month'].dropna().unique())
        
        # Omit month -1 (February) as reference period
        if -1 in relative_months:
            relative_months.remove(-1)
        
        print(f"Relative months in data: {relative_months}")
        print(f"Reference period: -1 (February)")
        
        # Create dummy variables for each relative month
        for month in relative_months:
            df_valid[f'month_{month}'] = (df_valid['relative_month'] == month).astype(int)
            df_valid[f'treated_x_month_{month}'] = df_valid['treated'] * df_valid[f'month_{month}']
        
        # Build formula
        month_dummies = ' + '.join([f'month_{m}' for m in relative_months])
        interaction_terms = ' + '.join([f'treated_x_month_{m}' for m in relative_months])
        
        formula = f'{outcome} ~ treated + {month_dummies} + {interaction_terms}'
        
        try:
            model = ols(formula, data=df_valid).fit(cov_type='HC3')
            
            print("\nEvent Study Coefficients:")
            print("-"*60)
            
            # Extract coefficients for each period
            event_coeffs = []
            
            for month in relative_months:
                var_name = f'treated_x_month_{month}'
                if var_name in model.params:
                    coef = model.params[var_name]
                    se = model.bse[var_name]
                    pval = model.pvalues[var_name]
                    ci_lower = model.conf_int().loc[var_name, 0]
                    ci_upper = model.conf_int().loc[var_name, 1]
                    
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    
                    period = "PRE" if month < 0 else "POST"
                    print(f"  Month {month:+2d} ({period:4s}): {coef:+.4f} {sig:3s} (SE={se:.4f}, p={pval:.3f})")
                    
                    event_coeffs.append({
                        'relative_month': month,
                        'coefficient': coef,
                        'std_error': se,
                        'p_value': pval,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'period': period
                    })
            
            # Add reference period (coefficient = 0)
            event_coeffs.append({
                'relative_month': -1,
                'coefficient': 0.0,
                'std_error': 0.0,
                'p_value': 1.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'period': 'PRE'
            })
            
            event_df = pd.DataFrame(event_coeffs).sort_values('relative_month')
            
            # Test pre-trends
            pre_periods = event_df[event_df['period'] == 'PRE']
            pre_sig = (pre_periods['p_value'] < 0.05).sum()
            
            print(f"\nPre-trends test:")
            print(f"  Pre-period coefficients: {len(pre_periods)}")
            print(f"  Significant at 5%: {pre_sig}")
            
            if pre_sig == 0:
                print("  ✓ No significant pre-trends (parallel trends assumption holds)")
            else:
                print("  ⚠️  Some pre-trends detected (may violate parallel trends)")
            
            # Save event study results
            event_file = output_dir / f'event_study_{outcome}.csv'
            event_df.to_csv(event_file, index=False)
            print(f"\n✓ Saved event study: {event_file.name}")
            
            event_study_results[outcome] = event_df
            
        except Exception as e:
            print(f"  ✗ Event study failed: {e}")
            event_study_results[outcome] = None
    
    return event_study_results


def create_event_study_plots(event_study_results, output_dir):
    """Create event study plots showing treatment effects over time."""
    print("\n" + "="*80)
    print("CREATING EVENT STUDY PLOTS")
    print("="*80)
    
    for outcome, event_df in event_study_results.items():
        if event_df is None:
            continue
        
        print(f"\n📊 Creating plot: {outcome}")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot coefficients
        ax.plot(event_df['relative_month'], event_df['coefficient'], 
               'o-', color='#2E86AB', linewidth=3, markersize=12, 
               label='Treatment Effect', zorder=3)
        
        # Add confidence intervals
        ax.fill_between(event_df['relative_month'],
                       event_df['ci_lower'],
                       event_df['ci_upper'],
                       alpha=0.3, color='#2E86AB', zorder=2)
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
        
        # Shade pre/post periods
        pre_months = event_df[event_df['period'] == 'PRE']['relative_month']
        post_months = event_df[event_df['period'] == 'POST']['relative_month']
        
        if len(pre_months) > 0:
            ax.axvspan(pre_months.min(), -0.5, alpha=0.1, color='blue', 
                      label='Pre-period', zorder=0)
        if len(post_months) > 0:
            ax.axvspan(-0.5, post_months.max(), alpha=0.1, color='green',
                      label='Post-period', zorder=0)
        
        # Mark earthquake month
        ax.axvline(-0.5, color='red', linestyle='--', linewidth=2,
                  label='Earthquake (Mar 28)', zorder=4)
        
        # Labels
        ax.set_xlabel('Months Relative to Earthquake', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Treatment Effect ({outcome.replace("_mean", "").upper()})', 
                     fontsize=14, fontweight='bold')
        
        outcome_names = {
            'ndbi_mean': 'Built-up Development (NDBI)',
            'ndvi_mean': 'Vegetation Recovery (NDVI)',
            'nbr_mean': 'Burn Ratio (NBR)',
            'ndmi_mean': 'Moisture Index (NDMI)'
        }
        
        title = outcome_names.get(outcome, outcome)
        ax.set_title(f'Event Study: {title}\n(Tests Parallel Pre-Trends Assumption)',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis ticks
        ax.set_xticks(event_df['relative_month'])
        
        plt.tight_layout()
        
        # Save
        plot_file = output_dir / f'event_study_plot_{outcome}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_file.name}")
        
        plt.close()


def run_log_specification(df, outcomes, output_dir):
    """Run log-log specification for percentage effects."""
    print("\n" + "="*80)
    print("LOG SPECIFICATION (For % Effects)")
    print("="*80)
    
    log_results = {}
    
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        
        df_valid = df[df[outcome].notna()].copy()
        
        if len(df_valid) < 50:
            continue
        
        print(f"\n{outcome.upper()}")
        print("-"*80)
        
        # Add constant to handle negative/zero values
        min_val = df_valid[outcome].min()
        
        if min_val <= 0:
            const = abs(min_val) + 0.05
            print(f"  Adding constant {const:.3f} to handle non-positive values")
            df_valid[f'log_{outcome}'] = np.log(df_valid[outcome] + const)
        else:
            df_valid[f'log_{outcome}'] = np.log(df_valid[outcome])
        
        # Run regression
        formula = f'log_{outcome} ~ treated + post + treated_post'
        
        try:
            model = ols(formula, data=df_valid).fit(cov_type='HC3')
            
            coef = model.params['treated_post']
            se = model.bse['treated_post']
            pval = model.pvalues['treated_post']
            
            # Convert to percentage effect
            pct_effect = (np.exp(coef) - 1) * 100
            
            print(f"  Log coefficient: {coef:.4f}")
            print(f"  Percentage effect: {pct_effect:+.2f}%")
            print(f"  P-value: {pval:.4f}")
            
            log_results[outcome] = {
                'log_coefficient': coef,
                'std_error': se,
                'p_value': pval,
                'pct_effect': pct_effect
            }
            
        except Exception as e:
            print(f"  ✗ Log specification failed: {e}")
            log_results[outcome] = None
    
    # Save
    if log_results:
        log_df = pd.DataFrame(log_results).T
        log_file = output_dir / 'log_specification_landsat.csv'
        log_df.to_csv(log_file)
        print(f"\n✓ Saved log results: {log_file.name}")
    
    return log_results


def run_placebo_tests(df, outcome, n_iterations=1000, output_dir=None):
    """Run placebo tests with random treatment assignment."""
    print("\n" + "="*80)
    print(f"PLACEBO TESTS: {outcome.upper()}")
    print("="*80)
    
    df_valid = df[df[outcome].notna()].copy()
    
    if len(df_valid) < 50:
        print(f"⚠️  Insufficient data, skipping")
        return None
    
    # Get actual DiD coefficient
    formula_actual = f'{outcome} ~ treated + post + treated_post'
    model_actual = ols(formula_actual, data=df_valid).fit(cov_type='HC3')
    actual_coef = model_actual.params['treated_post']
    actual_pval = model_actual.pvalues['treated_post']
    
    print(f"\nActual DiD coefficient: {actual_coef:+.4f} (p={actual_pval:.4f})")
    print(f"\nRunning {n_iterations} placebo iterations...")
    
    # Run placebo iterations
    placebo_coeffs = []
    
    for i in range(n_iterations):
        # Randomly assign treatment
        townships = df_valid['township_name'].unique()
        n_treated = (df_valid.groupby('township_name')['treated'].first() == 1).sum()
        
        random_treated = np.random.choice(townships, size=n_treated, replace=False)
        df_placebo = df_valid.copy()
        df_placebo['treated_placebo'] = df_placebo['township_name'].isin(random_treated).astype(int)
        df_placebo['treated_post_placebo'] = df_placebo['treated_placebo'] * df_placebo['post']
        
        # Run regression
        formula_placebo = f'{outcome} ~ treated_placebo + post + treated_post_placebo'
        
        try:
            model_placebo = ols(formula_placebo, data=df_placebo).fit(cov_type='HC3')
            placebo_coef = model_placebo.params['treated_post_placebo']
            placebo_coeffs.append(placebo_coef)
        except:
            continue
    
    placebo_coeffs = np.array(placebo_coeffs)
    
    # Calculate p-value
    if actual_coef > 0:
        placebo_pval = (placebo_coeffs >= actual_coef).sum() / len(placebo_coeffs)
    else:
        placebo_pval = (placebo_coeffs <= actual_coef).sum() / len(placebo_coeffs)
    
    print(f"\nPlacebo test results:")
    print(f"  Placebo mean: {placebo_coeffs.mean():+.4f}")
    print(f"  Placebo std: {placebo_coeffs.std():.4f}")
    print(f"  Placebo p-value: {placebo_pval:.4f}")
    
    if placebo_pval < 0.05:
        print(f"  ✓ Actual effect distinguishable from placebo")
    else:
        print(f"  ⚠️  Actual effect not distinguishable from placebo")
    
    # Create plot
    if output_dir:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(placebo_coeffs, bins=50, alpha=0.7, color='gray', edgecolor='black',
               label='Placebo distribution')
        ax.axvline(actual_coef, color='red', linestyle='--', linewidth=2,
                  label=f'Actual effect ({actual_coef:+.3f})')
        
        ax.set_xlabel('DiD Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Placebo Test: {outcome.upper()}\n(Random Treatment Assignment)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_file = output_dir / f'placebo_test_{outcome}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved placebo plot: {plot_file.name}")
        
        plt.close()
    
    # Save results
    if output_dir:
        placebo_df = pd.DataFrame({
            'iteration': range(len(placebo_coeffs)),
            'placebo_coefficient': placebo_coeffs
        })
        
        placebo_file = output_dir / f'placebo_results_{outcome}.csv'
        placebo_df.to_csv(placebo_file, index=False)
    
    return {
        'actual_coef': actual_coef,
        'placebo_mean': placebo_coeffs.mean(),
        'placebo_std': placebo_coeffs.std(),
        'placebo_pval': placebo_pval
    }


def run_dose_response(df, outcome, output_dir):
    """Dose-response: effect of number of UNDP interventions."""
    print("\n" + "="*80)
    print(f"DOSE-RESPONSE ANALYSIS: {outcome.upper()}")
    print("="*80)
    
    if 'n_interventions' not in df.columns:
        print("⚠️  n_interventions not found, skipping")
        return None
    
    df_valid = df[df[outcome].notna()].copy()
    
    # Run regression with continuous treatment
    formula = f'{outcome} ~ n_interventions + post + n_interventions:post'
    
    try:
        model = ols(formula, data=df_valid).fit(cov_type='HC3')
        
        print(model.summary())
        
        coef = model.params['n_interventions:post']
        se = model.bse['n_interventions:post']
        pval = model.pvalues['n_interventions:post']
        
        print(f"\nDose-response coefficient: {coef:+.4f}")
        print(f"  → Each additional UNDP intervention increased {outcome} by {coef:.4f}")
        print(f"  P-value: {pval:.4f}")
        
        # Save
        if output_dir:
            dose_df = pd.DataFrame({
                'coefficient': [coef],
                'std_error': [se],
                'p_value': [pval]
            })
            
            dose_file = output_dir / f'dose_response_{outcome}.csv'
            dose_df.to_csv(dose_file, index=False)
        
        return {'coefficient': coef, 'std_error': se, 'p_value': pval}
        
    except Exception as e:
        print(f"✗ Dose-response failed: {e}")
        return None


def create_comprehensive_comparison_table(twfe_results, log_results, basic_results, output_dir):
    """Create comparison table across specifications."""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    
    comparison_data = []
    
    for outcome in basic_results.keys():
        row = {'outcome': outcome}
        
        # Basic DiD
        if basic_results[outcome]:
            row['basic_coef'] = basic_results[outcome]['coefficient']
            row['basic_pval'] = basic_results[outcome]['p_value']
            row['basic_sig'] = basic_results[outcome]['significance']
        
        # TWFE
        if outcome in twfe_results and twfe_results[outcome]:
            row['twfe_coef'] = twfe_results[outcome]['coefficient']
            row['twfe_pval'] = twfe_results[outcome]['p_value']
            row['twfe_sig'] = twfe_results[outcome]['significance']
        
        # Log spec
        if outcome in log_results and log_results[outcome]:
            row['log_pct_effect'] = log_results[outcome]['pct_effect']
            row['log_pval'] = log_results[outcome]['p_value']
        
        comparison_data.append(row)
    
    comp_df = pd.DataFrame(comparison_data)
    
    print("\n")
    print(comp_df.to_string(index=False))
    
    # Save
    comp_file = output_dir / 'comprehensive_comparison_table.csv'
    comp_df.to_csv(comp_file, index=False)
    print(f"\n✓ Saved comparison table: {comp_file.name}")
    
    return comp_df


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Landsat robustness checks"
    )
    parser.add_argument("--landsat-data", type=str, required=True)
    parser.add_argument("--basic-results", type=str, required=True,
                       help="Path to landsat_did_results_CLEAR.csv")
    parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if not DEPS_OK:
        print("\n✗ Missing dependencies")
        return 1
    
    print("\n" + "="*80)
    print("COMPREHENSIVE LANDSAT ROBUSTNESS CHECKS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data(args.landsat_data)
    
    # Load basic results
    basic_results_df = pd.read_csv(args.basic_results)
    basic_results = {
        row['outcome'].replace(' (Vegetation)', '').replace(' (Built-up Index)', '').replace(' (Burn Ratio)', '').replace(' (Moisture)', ''): {
            'coefficient': row['did_coef'],
            'std_error': row['did_se'],
            'p_value': row['did_pval'],
            'significance': row['significance']
        }
        for _, row in basic_results_df.iterrows()
    }
    
    # Define main outcomes
    main_outcomes = ['ndbi_mean', 'ndvi_mean']
    all_outcomes = ['ndbi_mean', 'ndvi_mean', 'nbr_mean', 'ndmi_mean']
    
    # Run analyses
    print("\n" + "="*80)
    print("RUNNING ROBUSTNESS CHECKS")
    print("="*80)
    
    # 1. TWFE
    print("\n[1/6] Two-Way Fixed Effects...")
    twfe_results = run_twfe_models(df, all_outcomes, output_dir)
    
    # 2. Event Study
    print("\n[2/6] Event Study...")
    event_study_results = run_event_study(df, main_outcomes, output_dir)
    
    # 3. Event Study Plots
    print("\n[3/6] Event Study Plots...")
    create_event_study_plots(event_study_results, output_dir)
    
    # 4. Log Specification
    print("\n[4/6] Log Specification...")
    log_results = run_log_specification(df, main_outcomes, output_dir)
    
    # 5. Placebo Tests
    print("\n[5/6] Placebo Tests...")
    for outcome in main_outcomes:
        run_placebo_tests(df, outcome, n_iterations=1000, output_dir=output_dir)
    
    # 6. Dose-Response
    print("\n[6/6] Dose-Response...")
    for outcome in main_outcomes:
        run_dose_response(df, outcome, output_dir)
    
    # 7. Comparison Table
    print("\n[7/7] Comparison Table...")
    create_comprehensive_comparison_table(twfe_results, log_results, basic_results, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL ROBUSTNESS CHECKS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. twfe_results_landsat.csv")
    print("  2. event_study_*.csv (one per outcome)")
    print("  3. event_study_plot_*.png (pre-trends visualization)")
    print("  4. log_specification_landsat.csv")
    print("  5. placebo_test_*.png")
    print("  6. placebo_results_*.csv")
    print("  7. dose_response_*.csv")
    print("  8. comprehensive_comparison_table.csv")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
