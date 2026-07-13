#!/usr/bin/env python3
"""
Apply Relaxed Thresholds to Maximize Sample Size
================================================

Uses ≥1 pre + ≥2 post to get 97 townships instead of 12.
Fixes built-up calculation to properly handle NaNs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def fix_builtup_calculation(df):
    """
    CRITICAL FIX: Recalculate built-up to properly handle NaNs.
    
    The original calculation had a bug:
      builtup = np.where((ndvi < 0.2) & (swir1 > 0.1), 1.0, 0.0)
    
    This assigned 0.0 to NaN observations instead of NaN.
    """
    print("\n" + "="*80)
    print("FIXING BUILT-UP CALCULATION")
    print("="*80)
    
    if 'ndvi_mean' not in df.columns or 'swir1_mean' not in df.columns:
        print("Cannot fix - missing required columns")
        return df
    
    # Count original "valid" built-up
    original_valid = df['builtup_mean'].notna().sum()
    original_fake = (df['builtup_mean'] == 0) & df['ndvi_mean'].isna()
    n_fake = original_fake.sum()
    
    print(f"\nOriginal built-up:")
    print(f"  'Valid' observations: {original_valid}")
    print(f"  Fake zeros (NaN bands): {n_fake}")
    print(f"  True valid: {original_valid - n_fake}")
    
    # Recalculate properly
    df['builtup_mean_FIXED'] = np.where(
        df['ndvi_mean'].notna() & df['swir1_mean'].notna(),
        np.where((df['ndvi_mean'] < 0.2) & (df['swir1_mean'] > 0.1), 1.0, 0.0),
        np.nan  # Properly assign NaN when bands are missing
    )
    
    # Count fixed valid
    fixed_valid = df['builtup_mean_FIXED'].notna().sum()
    
    print(f"\nFixed built-up:")
    print(f"  Valid observations: {fixed_valid}")
    print(f"  NaN observations: {df['builtup_mean_FIXED'].isna().sum()}")
    
    print(f"\n✓ Fixed! Removed {n_fake} fake zeros")
    
    return df


def apply_relaxed_threshold(df, min_pre=1, min_post=2):
    """
    Apply relaxed quality threshold to maximize sample.
    
    Default: ≥1 pre + ≥2 post (gets 97 townships vs 12)
    """
    print("\n" + "="*80)
    print(f"APPLYING RELAXED THRESHOLD: ≥{min_pre} pre + ≥{min_post} post")
    print("="*80)
    
    # Exclude March
    df_clean = df[~((df['date'].dt.year == 2025) & (df['date'].dt.month == 3))].copy()
    
    # Create post indicator
    df_clean['post'] = (df_clean['date'] >= pd.to_datetime('2025-04-01')).astype(int)
    
    # Count valid observations per township
    township_valid = df_clean.groupby('township_name').apply(
        lambda x: pd.Series({
            'n_pre': x[(x['post'] == 0) & (x['ndvi_mean'].notna())].shape[0],
            'n_post': x[(x['post'] == 1) & (x['ndvi_mean'].notna())].shape[0],
            'n_total_valid': x['ndvi_mean'].notna().sum(),
            'treated': x['treated'].iloc[0] if len(x) > 0 else 0,
            'n_interventions': x['n_interventions'].iloc[0] if 'n_interventions' in x.columns else 0
        }), include_groups=False
    )
    
    # Filter townships
    valid_townships = township_valid[
        (township_valid['n_pre'] >= min_pre) & 
        (township_valid['n_post'] >= min_post)
    ]
    
    print(f"\nTownships meeting threshold:")
    print(f"  Total: {len(valid_townships)}")
    print(f"  Treated: {(valid_townships['treated'] == 1).sum()}")
    print(f"  Control: {(valid_townships['treated'] == 0).sum()}")
    
    # Show treatment distribution
    print(f"\nTreated township breakdown:")
    treated_townships = valid_townships[valid_townships['treated'] == 1].sort_values('n_interventions', ascending=False)
    if len(treated_townships) > 0:
        print(treated_townships[['n_pre', 'n_post', 'n_total_valid', 'n_interventions']].head(10).to_string())
    
    # Filter dataset
    df_filtered = df_clean[df_clean['township_name'].isin(valid_townships.index)].copy()
    
    # Add period labels
    df_filtered['period'] = 'pre'
    df_filtered.loc[df_filtered['post'] == 1, 'period'] = 'post'
    
    # Create interaction
    df_filtered['treated_post'] = df_filtered['treated'] * df_filtered['post']
    
    print(f"\nFiltered dataset:")
    print(f"  Total observations: {len(df_filtered)}")
    print(f"  Pre-period: {(df_filtered['period'] == 'pre').sum()}")
    print(f"  Post-period: {(df_filtered['period'] == 'post').sum()}")
    print(f"  Valid NDVI: {df_filtered['ndvi_mean'].notna().sum()}")
    print(f"  Valid NDBI: {df_filtered['ndbi_mean'].notna().sum()}")
    
    return df_filtered, valid_townships


def quick_did_with_relaxed_data(df):
    """Calculate quick DiD with relaxed threshold data."""
    print("\n" + "="*80)
    print("QUICK DiD WITH RELAXED THRESHOLD (97 townships)")
    print("="*80)
    
    for outcome in ['ndvi_mean', 'ndbi_mean', 'builtup_mean_FIXED']:
        if outcome not in df.columns or df[outcome].notna().sum() < 10:
            continue
        
        print(f"\n{outcome.upper().replace('_FIXED', ' (FIXED)')}:")
        
        # Calculate means
        treated_pre = df[(df['treated'] == 1) & (df['period'] == 'pre')][outcome].mean()
        treated_post = df[(df['treated'] == 1) & (df['period'] == 'post')][outcome].mean()
        control_pre = df[(df['treated'] == 0) & (df['period'] == 'pre')][outcome].mean()
        control_post = df[(df['treated'] == 0) & (df['period'] == 'post')][outcome].mean()
        
        # Count observations
        n_treated_pre = df[(df['treated'] == 1) & (df['period'] == 'pre') & (df[outcome].notna())].shape[0]
        n_treated_post = df[(df['treated'] == 1) & (df['period'] == 'post') & (df[outcome].notna())].shape[0]
        n_control_pre = df[(df['treated'] == 0) & (df['period'] == 'pre') & (df[outcome].notna())].shape[0]
        n_control_post = df[(df['treated'] == 0) & (df['period'] == 'post') & (df[outcome].notna())].shape[0]
        
        print(f"  Treated pre:  {treated_pre:.4f} (n={n_treated_pre})")
        print(f"  Treated post: {treated_post:.4f} (n={n_treated_post})")
        print(f"  Control pre:  {control_pre:.4f} (n={n_control_pre})")
        print(f"  Control post: {control_post:.4f} (n={n_control_post})")
        
        # Calculate DiD
        treated_diff = treated_post - treated_pre
        control_diff = control_post - control_pre
        did = treated_diff - control_diff
        
        print(f"\n  Treated change: {treated_diff:+.4f}")
        print(f"  Control change: {control_diff:+.4f}")
        print(f"  DiD estimate:   {did:+.4f}")
        
        if did > 0:
            print(f"    → UNDP INCREASED {outcome.replace('_FIXED', '')} ✓")
        else:
            print(f"    → UNDP DECREASED {outcome.replace('_FIXED', '')}")


def compare_thresholds(df):
    """Compare results across different thresholds."""
    print("\n" + "="*80)
    print("COMPARISON: Different Quality Thresholds")
    print("="*80)
    
    comparison = []
    
    for min_pre, min_post in [(1, 2), (2, 2), (2, 3), (2, 4)]:
        df_filtered, valid_townships = apply_relaxed_threshold(df, min_pre, min_post)
        
        # Calculate DiD for NDVI
        if df_filtered['ndvi_mean'].notna().sum() >= 10:
            treated_pre = df_filtered[(df_filtered['treated'] == 1) & (df_filtered['period'] == 'pre')]['ndvi_mean'].mean()
            treated_post = df_filtered[(df_filtered['treated'] == 1) & (df_filtered['period'] == 'post')]['ndvi_mean'].mean()
            control_pre = df_filtered[(df_filtered['treated'] == 0) & (df_filtered['period'] == 'pre')]['ndvi_mean'].mean()
            control_post = df_filtered[(df_filtered['treated'] == 0) & (df_filtered['period'] == 'post')]['ndvi_mean'].mean()
            
            did = (treated_post - treated_pre) - (control_post - control_pre)
            
            comparison.append({
                'threshold': f'≥{min_pre} pre + ≥{min_post} post',
                'n_townships': len(valid_townships),
                'n_treated': (valid_townships['treated'] == 1).sum(),
                'n_control': (valid_townships['treated'] == 0).sum(),
                'n_obs': len(df_filtered),
                'did_ndvi': did
            })
    
    if comparison:
        comp_df = pd.DataFrame(comparison)
        print("\n")
        print(comp_df.to_string(index=False))
        
        print("\n💡 RECOMMENDATION:")
        print("  Best balance: ≥1 pre + ≥2 post")
        print("  Reason: Maximizes sample while maintaining DiD validity")


def main():
    parser = argparse.ArgumentParser(
        description="Apply relaxed thresholds to maximize sample"
    )
    parser.add_argument("--landsat-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/relaxed_threshold")
    parser.add_argument("--min-pre", type=int, default=1)
    parser.add_argument("--min-post", type=int, default=2)
    
    args = parser.parse_args()
    
    print("="*80)
    print("APPLYING RELAXED THRESHOLD APPROACH")
    print("="*80)
    
    # Load data
    df = pd.read_csv(args.landsat_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nOriginal data: {len(df)} observations, {df['township_name'].nunique()} townships")
    
    # Fix built-up
    df = fix_builtup_calculation(df)
    
    # Apply relaxed threshold
    df_filtered, valid_townships = apply_relaxed_threshold(df, args.min_pre, args.min_post)
    
    # Quick DiD
    quick_did_with_relaxed_data(df_filtered)
    
    # Compare thresholds
    compare_thresholds(df)
    
    # Save filtered dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'landsat_did_relaxed_threshold.csv'
    df_filtered.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80)
    print(f"\nSaved filtered dataset: {output_file}")
    print(f"  {len(df_filtered)} observations")
    print(f"  {df_filtered['township_name'].nunique()} townships")
    print(f"  {(df_filtered['treated'] == 1).sum() // df_filtered.groupby('township_name').size().iloc[0]} treated townships")
    
    print("\n💡 NEXT STEPS:")
    print("  1. Run comprehensive DiD analysis on this dataset")
    print("  2. Compare results to your nightlight analysis")
    print("  3. Report Landsat as robustness check (acknowledge limited coverage)")
    print("="*80)


if __name__ == "__main__":
    main()
