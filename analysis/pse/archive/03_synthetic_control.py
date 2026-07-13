#!/usr/bin/env python3
"""
analysis/pse/03_synthetic_control.py
====================================
Synthetic Control Method for UNDP Gaza Impact Evaluation.

FIXED VERSION:
- Proper array alignment in fit() method
- Better handling of missing data
- Improved donor matching

Usage:
    python analysis/pse/03_synthetic_control.py
    python analysis/pse/03_synthetic_control.py --outcome nightlights
    python analysis/pse/03_synthetic_control.py --placebo
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import argparse
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Files
GAZA_PANEL = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_TREATMENT_START = pd.Timestamp("2022-01-01")
PRE_TREATMENT_END = pd.Timestamp("2023-09-01")

# Outcomes
OUTCOMES = ['nightlights', 'ndvi', 'vh']


class SyntheticControl:
    """Synthetic Control Method implementation with proper alignment."""
    
    def __init__(self, outcome: str):
        self.outcome = outcome
        self.weights = None
        self.donor_names = None
        self.treated_name = "Gaza"
        self.pre_times = None
        self.post_times = None
        
    def prepare_data(self, treated_df: pd.DataFrame, donor_df: pd.DataFrame,
                     unit_col: str, time_col: str, 
                     pre_start: pd.Timestamp, pre_end: pd.Timestamp):
        """
        Prepare aligned time series for synthetic control.
        FIXED: Ensures all arrays have same time alignment.
        """
        
        # Get common time periods
        treated_df = treated_df.copy()
        donor_df = donor_df.copy()
        
        treated_df[time_col] = pd.to_datetime(treated_df[time_col])
        donor_df[time_col] = pd.to_datetime(donor_df[time_col])
        
        # Aggregate treated to single series
        treated_agg = treated_df.groupby(time_col)[self.outcome].mean().reset_index()
        treated_agg = treated_agg.sort_values(time_col)
        
        # Get all unique time periods from treated
        all_times = treated_agg[time_col].unique()
        pre_times = [t for t in all_times if pd.Timestamp(t) >= pre_start and pd.Timestamp(t) <= pre_end]
        post_times = [t for t in all_times if pd.Timestamp(t) > pre_end]
        
        pre_times = sorted(pre_times)
        post_times = sorted(post_times)
        
        self.pre_times = np.array(pre_times)
        self.post_times = np.array(post_times)
        
        # Extract treated outcomes
        treated_pre = treated_agg[treated_agg[time_col].isin(pre_times)].sort_values(time_col)
        treated_post = treated_agg[treated_agg[time_col].isin(post_times)].sort_values(time_col)
        
        Y1_pre = treated_pre[self.outcome].values
        Y1_post = treated_post[self.outcome].values
        
        # Build donor matrix - each column is a donor unit's time series
        donor_units = donor_df[unit_col].unique()
        
        valid_donors = []
        Y0_pre_list = []
        Y0_post_list = []
        
        for unit in donor_units:
            unit_df = donor_df[donor_df[unit_col] == unit].copy()
            unit_df = unit_df.sort_values(time_col)
            
            # Get pre-treatment values for this donor
            unit_pre = unit_df[unit_df[time_col].isin(pre_times)]
            unit_post = unit_df[unit_df[time_col].isin(post_times)]
            
            # Skip if insufficient data or too many missing values
            if len(unit_pre) < len(pre_times) * 0.7:
                continue
            
            if unit_pre[self.outcome].isna().sum() > len(pre_times) * 0.3:
                continue
            
            # Align to common time periods and interpolate missing
            unit_pre_aligned = []
            for t in pre_times:
                val = unit_pre[unit_pre[time_col] == t][self.outcome].values
                if len(val) > 0 and not np.isnan(val[0]):
                    unit_pre_aligned.append(val[0])
                else:
                    # Use mean imputation for missing
                    unit_pre_aligned.append(unit_pre[self.outcome].mean())
            
            unit_post_aligned = []
            for t in post_times:
                val = unit_post[unit_post[time_col] == t][self.outcome].values
                if len(val) > 0 and not np.isnan(val[0]):
                    unit_post_aligned.append(val[0])
                else:
                    unit_post_aligned.append(unit_post[self.outcome].mean() if len(unit_post) > 0 else np.nan)
            
            # Skip if still has issues
            if np.any(np.isnan(unit_pre_aligned)):
                continue
            
            valid_donors.append(unit)
            Y0_pre_list.append(unit_pre_aligned)
            Y0_post_list.append(unit_post_aligned)
        
        if len(valid_donors) < 2:
            print(f"  ⚠ Only {len(valid_donors)} valid donors found")
            return None, None, None, None, None, None
        
        self.donor_names = valid_donors
        
        # Convert to arrays - rows are time, columns are donors
        Y0_pre = np.array(Y0_pre_list).T  # (T_pre x J)
        Y0_post = np.array(Y0_post_list).T  # (T_post x J)
        
        # Also create X matrices for predictor matching
        # Use mean of pre-treatment outcomes as predictors
        X1 = np.array([Y1_pre.mean()]).reshape(-1, 1)
        X0 = np.array([Y0_pre.mean(axis=0)])  # (1 x J)
        
        print(f"  Treated pre-treatment periods: {len(Y1_pre)}")
        print(f"  Treated post-treatment periods: {len(Y1_post)}")
        print(f"  Valid donors: {len(valid_donors)}")
        
        return X1, X0, Y1_pre, Y0_pre, Y1_post, Y0_post
    
    def fit(self, X1, X0, Y1_pre, Y0_pre):
        """
        Fit synthetic control weights.
        FIXED: Proper array shapes and optimization.
        """
        if Y0_pre is None or len(Y0_pre) == 0:
            print("  ⚠ No valid donor data")
            return False
        
        J = Y0_pre.shape[1]  # Number of donors
        T_pre = Y0_pre.shape[0]  # Pre-treatment periods
        
        # Objective: minimize squared prediction error in pre-treatment
        def objective(W):
            # Synthetic = weighted average of donors
            Y_synthetic = Y0_pre @ W
            # MSE
            return np.mean((Y1_pre - Y_synthetic) ** 2)
        
        # Constraints: weights sum to 1, non-negative
        constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
        bounds = [(0, 1) for _ in range(J)]
        
        # Initial guess: equal weights
        W0 = np.ones(J) / J
        
        # Optimize
        result = minimize(
            objective, W0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.weights = result.x
        else:
            print(f"  ⚠ Optimization did not converge: {result.message}")
            self.weights = W0
        
        # Compute pre-treatment fit
        Y_synthetic_pre = Y0_pre @ self.weights
        mspe = np.mean((Y1_pre - Y_synthetic_pre) ** 2)
        rmse = np.sqrt(mspe)
        
        # R-squared
        ss_tot = np.sum((Y1_pre - Y1_pre.mean()) ** 2)
        ss_res = np.sum((Y1_pre - Y_synthetic_pre) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        print(f"  Pre-treatment RMSE: {rmse:.4f}")
        print(f"  Pre-treatment R²: {r2:.4f}")
        
        # Show top donors
        print(f"  Top donor weights:")
        sorted_idx = np.argsort(self.weights)[::-1]
        for i in sorted_idx[:5]:
            if self.weights[i] > 0.01:
                print(f"    {self.donor_names[i]}: {self.weights[i]:.3f}")
        
        return True
    
    def predict(self, Y0):
        """Generate synthetic control prediction."""
        if self.weights is None:
            raise ValueError("Must call fit() first")
        return Y0 @ self.weights
    
    def estimate_effect(self, Y1_post, Y0_post):
        """Estimate treatment effect as gap between actual and synthetic."""
        Y_synthetic = self.predict(Y0_post)
        gaps = Y1_post - Y_synthetic
        att = np.mean(gaps)
        return gaps, att, Y_synthetic


def create_synthetic_control_plot(sc, Y1_pre, Y1_post, Y0_pre, Y0_post, 
                                  outcome, output_path):
    """Create visualization of synthetic control results."""
    
    Y_synthetic_pre = sc.predict(Y0_pre)
    Y_synthetic_post = sc.predict(Y0_post)
    
    # Combine
    Y_actual = np.concatenate([Y1_pre, Y1_post])
    Y_synthetic = np.concatenate([Y_synthetic_pre, Y_synthetic_post])
    all_times = np.concatenate([sc.pre_times, sc.post_times])
    all_times = pd.to_datetime(all_times)
    
    n_pre = len(Y1_pre)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Panel A: Actual vs Synthetic
    ax1 = axes[0]
    
    ax1.plot(all_times, Y_actual, 'b-', linewidth=2.5, label='Gaza (Actual)', marker='o', markersize=4)
    ax1.plot(all_times, Y_synthetic, 'r--', linewidth=2, label='Synthetic Gaza', marker='s', markersize=4)
    
    # Shade pre/post
    ax1.axvspan(all_times.min(), CONFLICT_DATE, alpha=0.1, color='green', label='Pre-conflict')
    ax1.axvspan(CONFLICT_DATE, all_times.max(), alpha=0.1, color='red', label='Post-conflict')
    
    # Conflict line
    ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    ax1.text(CONFLICT_DATE, ax1.get_ylim()[1] * 0.95, 'Oct 7, 2023', 
             ha='center', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_ylabel(outcome.replace('_', ' ').title(), fontsize=12)
    ax1.set_title(f'Synthetic Control: Gaza vs Synthetic Gaza ({outcome})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Panel B: Gap
    ax2 = axes[1]
    
    gaps = Y_actual - Y_synthetic
    
    colors = ['gray' if i < n_pre else 'red' for i in range(len(gaps))]
    ax2.bar(all_times, gaps, width=20, color=colors, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
    
    # ATT line
    att = np.mean(gaps[n_pre:])
    ax2.axhline(y=att, color='red', linestyle=':', linewidth=2, label=f'ATT = {att:.3f}')
    
    ax2.set_ylabel('Gap (Actual - Synthetic)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ {output_path.name}")
    return True


def run_placebo_tests(treated_df, donor_df, outcome, unit_col, time_col,
                      pre_start, pre_end, n_placebos=10):
    """Run placebo tests using donors as pseudo-treated."""
    
    print("\n--- Running Placebo Tests ---")
    
    # First get actual effect
    sc_actual = SyntheticControl(outcome)
    result = sc_actual.prepare_data(treated_df, donor_df, unit_col, time_col, pre_start, pre_end)
    
    if result[0] is None:
        return None
    
    X1, X0, Y1_pre, Y0_pre, Y1_post, Y0_post = result
    
    if not sc_actual.fit(X1, X0, Y1_pre, Y0_pre):
        return None
    
    gaps_actual, att_actual, _ = sc_actual.estimate_effect(Y1_post, Y0_post)
    
    print(f"  Actual ATT: {att_actual:.4f}")
    
    # Placebo tests
    donor_units = donor_df[unit_col].unique()[:n_placebos]
    
    placebo_effects = []
    placebo_gaps = []
    
    for placebo_unit in donor_units:
        placebo_treated = donor_df[donor_df[unit_col] == placebo_unit].copy()
        placebo_donors = donor_df[donor_df[unit_col] != placebo_unit].copy()
        
        try:
            sc_p = SyntheticControl(outcome)
            result_p = sc_p.prepare_data(placebo_treated, placebo_donors, unit_col, time_col, pre_start, pre_end)
            
            if result_p[0] is None:
                continue
            
            X1_p, X0_p, Y1_pre_p, Y0_pre_p, Y1_post_p, Y0_post_p = result_p
            
            if sc_p.fit(X1_p, X0_p, Y1_pre_p, Y0_pre_p):
                gaps_p, att_p, _ = sc_p.estimate_effect(Y1_post_p, Y0_post_p)
                placebo_effects.append(att_p)
                placebo_gaps.append(gaps_p)
                print(f"  Placebo {placebo_unit}: ATT = {att_p:.4f}")
        except Exception as e:
            continue
    
    if placebo_effects:
        # P-value
        if att_actual < 0:
            p_value = np.mean([p <= att_actual for p in placebo_effects])
        else:
            p_value = np.mean([p >= att_actual for p in placebo_effects])
        
        print(f"\n  P-value: {p_value:.3f}")
        
        return {
            'actual_att': att_actual,
            'actual_gaps': gaps_actual,
            'placebo_effects': placebo_effects,
            'placebo_gaps': placebo_gaps,
            'p_value': p_value
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Synthetic Control Analysis')
    parser.add_argument('--outcome', '-o', default='nightlights')
    parser.add_argument('--placebo', '-p', action='store_true')
    parser.add_argument('--all-outcomes', '-a', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL ANALYSIS (FIXED)")
    print("=" * 70)
    
    # Check for donor data
    if not DONOR_POOL.exists():
        print(f"\n⚠ Donor pool not found: {DONOR_POOL}")
        print("Run: python -m pipeline.gee_donor_pool")
        print("\nUsing West Bank as pseudo-donor (for demonstration only)")
        use_wb = True
    else:
        use_wb = False
    
    # Load Gaza data
    print("\nLoading data...")
    panel = pd.read_csv(GAZA_PANEL)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    westbank_df = panel[panel['unit_id'].str.startswith('PS01')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    print(f"  West Bank: {len(westbank_df)} obs")
    
    # Load donors
    if use_wb:
        donor_df = westbank_df.rename(columns={'unit_id': 'admin_id'})
        donor_df['country'] = 'West Bank'
        unit_col = 'admin_id'
    else:
        donor_df = pd.read_csv(DONOR_POOL)
        donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
        unit_col = 'admin_id'
        print(f"  Donors: {len(donor_df)} obs from {donor_df['country'].nunique()} countries")
    
    # Outcomes
    outcomes = OUTCOMES if args.all_outcomes else [args.outcome]
    
    results_summary = []
    
    for outcome in outcomes:
        if outcome not in gaza_df.columns:
            print(f"\n⚠ {outcome} not in data")
            continue
        
        print(f"\n{'='*60}")
        print(f"OUTCOME: {outcome.upper()}")
        print(f"{'='*60}")
        
        sc = SyntheticControl(outcome)
        
        result = sc.prepare_data(
            gaza_df, donor_df, unit_col, 'period_start',
            PRE_TREATMENT_START, PRE_TREATMENT_END
        )
        
        if result[0] is None:
            continue
        
        X1, X0, Y1_pre, Y0_pre, Y1_post, Y0_post = result
        
        print("\nFitting synthetic control...")
        if not sc.fit(X1, X0, Y1_pre, Y0_pre):
            continue
        
        gaps, att, Y_synthetic = sc.estimate_effect(Y1_post, Y0_post)
        
        print(f"\n  Average Treatment Effect: {att:.4f}")
        print(f"  Effect range: [{gaps.min():.4f}, {gaps.max():.4f}]")
        
        results_summary.append({
            'Outcome': outcome,
            'ATT': att,
            'ATT_pct': (att / Y1_pre.mean()) * 100 if Y1_pre.mean() != 0 else np.nan,
            'N_donors': len(sc.donor_names),
            'Pre_RMSE': np.sqrt(np.mean((Y1_pre - sc.predict(Y0_pre))**2))
        })
        
        # Plot
        create_synthetic_control_plot(
            sc, Y1_pre, Y1_post, Y0_pre, Y0_post, outcome,
            FIGURES_DIR / f'synthetic_control_{outcome}.png'
        )
        
        # Save weights
        weights_df = pd.DataFrame({
            'Donor': sc.donor_names,
            'Weight': sc.weights
        }).sort_values('Weight', ascending=False)
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        weights_df.to_csv(TABLES_DIR / f'sc_weights_{outcome}.csv', index=False)
        print(f"  ✓ sc_weights_{outcome}.csv")
        
        # Placebo tests
        if args.placebo:
            placebo_result = run_placebo_tests(
                gaza_df, donor_df, outcome, unit_col, 'period_start',
                PRE_TREATMENT_START, PRE_TREATMENT_END, n_placebos=8
            )
            
            if placebo_result:
                results_summary[-1]['p_value'] = placebo_result['p_value']
    
    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(TABLES_DIR / 'synthetic_control_summary.csv', index=False)
        print(f"\n✓ {TABLES_DIR / 'synthetic_control_summary.csv'}")
        
        print("\n--- SUMMARY ---")
        print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ SYNTHETIC CONTROL COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
