#!/usr/bin/env python3
"""
Clear Comprehensive Landsat DiD Analysis
========================================

Properly analyzes ALL outcomes including NDBI.
Uses builtup_mean_FIXED if available.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import argparse


def load_and_check_data(csv_file):
    """Load data and report what outcomes are available."""
    print("="*80)
    print("LOADING AND CHECKING DATA")
    print("="*80)
    
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n✓ Loaded {len(df)} observations")
    print(f"  Townships: {df['township_name'].nunique()}")
    print(f"  Treated: {df[df['treated']==1]['township_name'].nunique()}")
    print(f"  Control: {df[df['treated']==0]['township_name'].nunique()}")
    
    # Check available outcomes
    print("\n" + "-"*80)
    print("AVAILABLE OUTCOMES:")
    print("-"*80)
    
    outcome_candidates = [
        ('ndvi_mean', 'NDVI - Vegetation Index'),
        ('ndbi_mean', 'NDBI - Built-up Index (PROPER)'),
        ('builtup_mean_FIXED', 'Built-up - Simple Threshold (FIXED)'),
        ('builtup_mean', 'Built-up - Simple Threshold (ORIGINAL)'),
        ('nbr_mean', 'NBR - Burn Ratio'),
        ('ndmi_mean', 'NDMI - Moisture Index'),
    ]
    
    available_outcomes = []
    
    for var, desc in outcome_candidates:
        if var in df.columns:
            n_valid = df[var].notna().sum()
            pct = 100 * n_valid / len(df)
            print(f"  ✓ {var:25s}: {n_valid:4d} valid ({pct:5.1f}%) - {desc}")
            available_outcomes.append(var)
        else:
            print(f"  ✗ {var:25s}: NOT FOUND")
    
    if 'builtup_mean_FIXED' in df.columns and 'builtup_mean' in df.columns:
        print("\n  💡 Both builtup_mean and builtup_mean_FIXED found")
        print("     → Will use builtup_mean_FIXED (proper NaN handling)")
        if 'builtup_mean' in available_outcomes:
            available_outcomes.remove('builtup_mean')
    
    return df, available_outcomes


def run_basic_did(df, outcome, outcome_name):
    """Run basic DiD regression for one outcome."""
    print("\n" + "="*80)
    print(f"ANALYZING: {outcome_name}")
    print("="*80)
    
    # Filter to valid data
    df_valid = df[df[outcome].notna()].copy()
    
    n_total = len(df_valid)
    n_treated = (df_valid['treated'] == 1).sum()
    n_control = (df_valid['treated'] == 0).sum()
    n_pre = (df_valid['post'] == 0).sum()
    n_post = (df_valid['post'] == 1).sum()
    
    print(f"\nSample:")
    print(f"  Total observations: {n_total}")
    print(f"  Treated: {n_treated}, Control: {n_control}")
    print(f"  Pre: {n_pre}, Post: {n_post}")
    
    if n_total < 50:
        print("  ⚠️  WARNING: Small sample size!")
    
    # Calculate means by group
    treated_pre = df_valid[(df_valid['treated'] == 1) & (df_valid['post'] == 0)][outcome].mean()
    treated_post = df_valid[(df_valid['treated'] == 1) & (df_valid['post'] == 1)][outcome].mean()
    control_pre = df_valid[(df_valid['treated'] == 0) & (df_valid['post'] == 0)][outcome].mean()
    control_post = df_valid[(df_valid['treated'] == 0) & (df_valid['post'] == 1)][outcome].mean()
    
    treated_change = treated_post - treated_pre
    control_change = control_post - control_pre
    did_simple = treated_change - control_change
    
    print("\n" + "-"*80)
    print("DESCRIPTIVE STATISTICS:")
    print("-"*80)
    print(f"  Treated Pre:    {treated_pre:.4f}")
    print(f"  Treated Post:   {treated_post:.4f}")
    print(f"  Treated Change: {treated_change:+.4f}")
    print(f"  ")
    print(f"  Control Pre:    {control_pre:.4f}")
    print(f"  Control Post:   {control_post:.4f}")
    print(f"  Control Change: {control_change:+.4f}")
    print(f"  ")
    print(f"  DiD (Simple):   {did_simple:+.4f}")
    
    # Run regression
    print("\n" + "-"*80)
    print("REGRESSION RESULTS:")
    print("-"*80)
    
    formula = f"{outcome} ~ treated + post + treated_post"
    
    try:
        model = smf.ols(formula, data=df_valid).fit(cov_type='HC3')
        
        print(model.summary())
        
        # Extract DiD coefficient
        did_coef = model.params['treated_post']
        did_se = model.bse['treated_post']
        did_pval = model.pvalues['treated_post']
        did_ci_lower = model.conf_int().loc['treated_post', 0]
        did_ci_upper = model.conf_int().loc['treated_post', 1]
        
        print("\n" + "-"*80)
        print("DiD ESTIMATE:")
        print("-"*80)
        print(f"  Coefficient:  {did_coef:+.4f}")
        print(f"  Std. Error:   {did_se:.4f}")
        print(f"  P-value:      {did_pval:.4f}")
        print(f"  95% CI:       [{did_ci_lower:+.4f}, {did_ci_upper:+.4f}]")
        
        if did_pval < 0.01:
            sig = "***"
            print(f"  ✓ Highly significant at 1% level")
        elif did_pval < 0.05:
            sig = "**"
            print(f"  ✓ Significant at 5% level")
        elif did_pval < 0.10:
            sig = "*"
            print(f"  ✓ Marginally significant at 10% level")
        else:
            sig = "ns"
            print(f"  ✗ Not statistically significant")
        
        # Interpretation
        if did_coef > 0:
            direction = "INCREASED"
        else:
            direction = "DECREASED"
        
        rel_effect = 100 * did_coef / abs(treated_pre) if treated_pre != 0 else 0
        
        print(f"\n  → UNDP {direction} {outcome_name} by {abs(did_coef):.4f} units")
        print(f"     ({abs(rel_effect):.1f}% relative to pre-treatment mean)")
        
        return {
            'outcome': outcome_name,
            'variable': outcome,
            'did_coef': did_coef,
            'did_se': did_se,
            'did_pval': did_pval,
            'did_ci_lower': did_ci_lower,
            'did_ci_upper': did_ci_upper,
            'significance': sig,
            'direction': direction,
            'rel_effect_pct': rel_effect,
            'treated_pre': treated_pre,
            'treated_post': treated_post,
            'control_pre': control_pre,
            'control_post': control_post,
            'n_obs': n_total
        }
    
    except Exception as e:
        print(f"\n  ✗ Regression failed: {e}")
        return None


def create_summary_table(results):
    """Create summary table of all results."""
    print("\n" + "="*80)
    print("SUMMARY OF ALL RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    print("\n")
    print(df_results[['outcome', 'did_coef', 'did_se', 'did_pval', 'significance']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("INTERPRETATION:")
    print("-"*80)
    
    for _, row in df_results.iterrows():
        sig_marker = row['significance']
        direction = "↑" if row['did_coef'] > 0 else "↓"
        
        print(f"  {direction} {row['outcome']:30s}: β = {row['did_coef']:+.4f} {sig_marker:3s} (p={row['did_pval']:.3f})")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Clear comprehensive Landsat DiD analysis"
    )
    parser.add_argument("--landsat-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLEAR COMPREHENSIVE LANDSAT DiD ANALYSIS")
    print("="*80)
    
    # Load data
    df, available_outcomes = load_and_check_data(args.landsat_data)
    
    # Check required columns
    required = ['treated', 'post', 'period', 'township_name']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"\n✗ ERROR: Missing required columns: {missing}")
        return
    
    # Run analysis for each outcome
    print("\n" + "="*80)
    print("RUNNING DiD ANALYSIS FOR ALL OUTCOMES")
    print("="*80)
    
    results = []
    
    outcome_names = {
        'ndvi_mean': 'NDVI (Vegetation)',
        'ndbi_mean': 'NDBI (Built-up Index)',
        'builtup_mean_FIXED': 'Built-up Threshold (Fixed)',
        'builtup_mean': 'Built-up Threshold',
        'nbr_mean': 'NBR (Burn Ratio)',
        'ndmi_mean': 'NDMI (Moisture)',
    }
    
    for outcome in available_outcomes:
        outcome_name = outcome_names.get(outcome, outcome)
        result = run_basic_did(df, outcome, outcome_name)
        if result:
            results.append(result)
    
    # Create summary
    if results:
        df_summary = create_summary_table(results)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'landsat_did_results_CLEAR.csv'
        df_summary.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved results: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Key recommendations
    print("\n" + "="*80)
    print("KEY RECOMMENDATIONS:")
    print("="*80)
    
    if 'ndbi_mean' in available_outcomes:
        ndbi_result = [r for r in results if r['variable'] == 'ndbi_mean']
        if ndbi_result:
            r = ndbi_result[0]
            if r['did_pval'] < 0.10:
                print("\n✓ NDBI (proper built-up index) shows significant effect")
                print(f"  Use this as your main Landsat outcome")
                print(f"  Effect: {r['did_coef']:+.4f} (p={r['did_pval']:.3f})")
    
    if 'ndvi_mean' in available_outcomes:
        ndvi_result = [r for r in results if r['variable'] == 'ndvi_mean']
        if ndvi_result:
            r = ndvi_result[0]
            if r['did_pval'] < 0.10:
                print("\n✓ NDVI shows significant effect")
                print(f"  Use as vegetation/environmental outcome")
                print(f"  Effect: {r['did_coef']:+.4f} (p={r['did_pval']:.3f})")
    
    if 'builtup_mean_FIXED' in available_outcomes or 'builtup_mean' in available_outcomes:
        print("\n⚠️  Simple built-up threshold:")
        print("  - Susceptible to seasonal/agricultural artifacts")
        print("  - Use NDBI instead (proper spectral index)")
        print("  - Report simple threshold only as secondary check")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
