#!/usr/bin/env python3
"""
analysis/pse/06_synthetic_control_full.py
==========================================
Comprehensive Synthetic Control Analysis with Placebo Tests.

Generates publication-quality visualizations for ALL outcomes showing:
1. Gaza vs Synthetic Gaza time series
2. Gap plot with ATT
3. Placebo test (Gaza gap vs all donor gaps)

Usage:
    python analysis/pse/06_synthetic_control_full.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "synthetic_control"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Files
GAZA_PANEL = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")

# All outcomes to analyze
OUTCOMES = ['nightlights', 'ndvi', 'vh', 'vv', 'rvi']

# Outcome labels for plots
OUTCOME_LABELS = {
    'nightlights': 'Nighttime Lights (VIIRS)',
    'ndvi': 'Vegetation Index (NDVI)',
    'ndbi': 'Built-up Index (NDBI)',
    'bsi': 'Bare Soil Index (BSI)',
    'vh': 'SAR Backscatter (VH)',
    'vv': 'SAR Backscatter (VV)',
    'rvi': 'Radar Vegetation Index (RVI)',
}


class SyntheticControlAnalysis:
    """Full synthetic control analysis with placebo tests."""
    
    def __init__(self, outcome: str):
        self.outcome = outcome
        self.weights = None
        self.donor_ids = None
        self.pre_times = None
        self.post_times = None
        self.all_times = None
        
        # Store results
        self.Y1_pre = None
        self.Y1_post = None
        self.Y0_pre = None
        self.Y0_post = None
        self.Y_synthetic_pre = None
        self.Y_synthetic_post = None
        self.pre_rmse = None
        self.pre_r2 = None
        self.att = None
        self.gaps = None
        
        # Placebo results
        self.placebo_gaps = {}
        self.placebo_atts = []
        self.p_value = None
    
    def prepare_data(self, treated_df: pd.DataFrame, donor_df: pd.DataFrame,
                     treated_unit_col: str, donor_unit_col: str, time_col: str):
        """Prepare aligned time series data."""
        
        treated_df = treated_df.copy()
        donor_df = donor_df.copy()
        
        treated_df[time_col] = pd.to_datetime(treated_df[time_col])
        donor_df[time_col] = pd.to_datetime(donor_df[time_col])
        
        # Aggregate treated (Gaza) to single time series
        treated_agg = treated_df.groupby(time_col)[self.outcome].mean().reset_index()
        treated_agg = treated_agg.sort_values(time_col)
        
        # Get time periods
        all_times = sorted(treated_agg[time_col].unique())
        pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
        post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
        
        self.pre_times = pre_times
        self.post_times = post_times
        self.all_times = all_times
        
        # Extract treated outcomes
        self.Y1_pre = treated_agg[treated_agg[time_col].isin(pre_times)].sort_values(time_col)[self.outcome].values
        self.Y1_post = treated_agg[treated_agg[time_col].isin(post_times)].sort_values(time_col)[self.outcome].values
        
        # Build donor matrix
        donor_units = donor_df[donor_unit_col].unique()
        
        valid_donors = []
        Y0_pre_list = []
        Y0_post_list = []
        
        for unit in donor_units:
            unit_df = donor_df[donor_df[donor_unit_col] == unit].sort_values(time_col)
            
            # Get values for this donor
            unit_pre = []
            unit_post = []
            
            for t in pre_times:
                val = unit_df[unit_df[time_col] == t][self.outcome].values
                if len(val) > 0 and not np.isnan(val[0]):
                    unit_pre.append(val[0])
                else:
                    unit_pre.append(np.nan)
            
            for t in post_times:
                val = unit_df[unit_df[time_col] == t][self.outcome].values
                if len(val) > 0 and not np.isnan(val[0]):
                    unit_post.append(val[0])
                else:
                    unit_post.append(np.nan)
            
            # Skip if too much missing
            if np.sum(np.isnan(unit_pre)) > len(pre_times) * 0.3:
                continue
            
            # Impute remaining missing with mean
            unit_pre = np.array(unit_pre)
            unit_post = np.array(unit_post)
            
            if np.any(np.isnan(unit_pre)):
                unit_pre[np.isnan(unit_pre)] = np.nanmean(unit_pre)
            if np.any(np.isnan(unit_post)):
                unit_post[np.isnan(unit_post)] = np.nanmean(unit_post) if not np.all(np.isnan(unit_post)) else 0
            
            valid_donors.append(unit)
            Y0_pre_list.append(unit_pre)
            Y0_post_list.append(unit_post)
        
        if len(valid_donors) < 3:
            print(f"  ⚠ Only {len(valid_donors)} valid donors")
            return False
        
        self.donor_ids = valid_donors
        self.Y0_pre = np.array(Y0_pre_list).T  # (T_pre x J)
        self.Y0_post = np.array(Y0_post_list).T  # (T_post x J)
        
        return True
    
    def fit(self):
        """Fit synthetic control weights."""
        
        if self.Y0_pre is None:
            return False
        
        J = self.Y0_pre.shape[1]
        
        def objective(W):
            Y_synth = self.Y0_pre @ W
            return np.mean((self.Y1_pre - Y_synth) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
        bounds = [(0, 1) for _ in range(J)]
        W0 = np.ones(J) / J
        
        result = minimize(objective, W0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000})
        
        self.weights = result.x if result.success else W0
        
        # Compute fit metrics
        self.Y_synthetic_pre = self.Y0_pre @ self.weights
        self.Y_synthetic_post = self.Y0_post @ self.weights
        
        self.pre_rmse = np.sqrt(np.mean((self.Y1_pre - self.Y_synthetic_pre) ** 2))
        
        ss_tot = np.sum((self.Y1_pre - self.Y1_pre.mean()) ** 2)
        ss_res = np.sum((self.Y1_pre - self.Y_synthetic_pre) ** 2)
        self.pre_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Compute ATT
        self.gaps = np.concatenate([
            self.Y1_pre - self.Y_synthetic_pre,
            self.Y1_post - self.Y_synthetic_post
        ])
        self.att = np.mean(self.Y1_post - self.Y_synthetic_post)
        
        return True
    
    def run_placebo_tests(self, donor_df: pd.DataFrame, donor_unit_col: str, time_col: str):
        """Run placebo tests treating each donor as if it were treated."""
        
        donor_df = donor_df.copy()
        donor_df[time_col] = pd.to_datetime(donor_df[time_col])
        
        self.placebo_gaps = {}
        self.placebo_atts = []
        
        for placebo_unit in self.donor_ids[:15]:  # Limit to 15 for speed
            # This donor becomes "treated"
            placebo_treated = donor_df[donor_df[donor_unit_col] == placebo_unit].copy()
            placebo_donors = donor_df[donor_df[donor_unit_col] != placebo_unit].copy()
            
            # Aggregate placebo treated
            placebo_agg = placebo_treated.groupby(time_col)[self.outcome].mean().reset_index()
            
            if len(placebo_agg) < len(self.pre_times) * 0.5:
                continue
            
            # Get Y1 for placebo
            Y1_pre_p = []
            Y1_post_p = []
            
            for t in self.pre_times:
                val = placebo_agg[placebo_agg[time_col] == t][self.outcome].values
                Y1_pre_p.append(val[0] if len(val) > 0 else np.nan)
            
            for t in self.post_times:
                val = placebo_agg[placebo_agg[time_col] == t][self.outcome].values
                Y1_post_p.append(val[0] if len(val) > 0 else np.nan)
            
            Y1_pre_p = np.array(Y1_pre_p)
            Y1_post_p = np.array(Y1_post_p)
            
            if np.sum(np.isnan(Y1_pre_p)) > len(self.pre_times) * 0.3:
                continue
            
            # Impute
            if np.any(np.isnan(Y1_pre_p)):
                Y1_pre_p[np.isnan(Y1_pre_p)] = np.nanmean(Y1_pre_p)
            if np.any(np.isnan(Y1_post_p)):
                Y1_post_p[np.isnan(Y1_post_p)] = np.nanmean(Y1_post_p) if not np.all(np.isnan(Y1_post_p)) else 0
            
            # Build donor matrix for this placebo (excluding placebo unit)
            other_donors = [d for d in self.donor_ids if d != placebo_unit]
            
            Y0_pre_p = []
            Y0_post_p = []
            
            for donor in other_donors:
                donor_data = donor_df[donor_df[donor_unit_col] == donor]
                
                d_pre = []
                d_post = []
                
                for t in self.pre_times:
                    val = donor_data[donor_data[time_col] == t][self.outcome].values
                    d_pre.append(val[0] if len(val) > 0 and not np.isnan(val[0]) else np.nan)
                
                for t in self.post_times:
                    val = donor_data[donor_data[time_col] == t][self.outcome].values
                    d_post.append(val[0] if len(val) > 0 and not np.isnan(val[0]) else np.nan)
                
                d_pre = np.array(d_pre)
                d_post = np.array(d_post)
                
                if np.sum(np.isnan(d_pre)) <= len(self.pre_times) * 0.3:
                    if np.any(np.isnan(d_pre)):
                        d_pre[np.isnan(d_pre)] = np.nanmean(d_pre)
                    if np.any(np.isnan(d_post)):
                        d_post[np.isnan(d_post)] = np.nanmean(d_post) if not np.all(np.isnan(d_post)) else 0
                    
                    Y0_pre_p.append(d_pre)
                    Y0_post_p.append(d_post)
            
            if len(Y0_pre_p) < 3:
                continue
            
            Y0_pre_p = np.array(Y0_pre_p).T
            Y0_post_p = np.array(Y0_post_p).T
            
            # Fit SC for placebo
            J_p = Y0_pre_p.shape[1]
            
            def obj_p(W):
                return np.mean((Y1_pre_p - Y0_pre_p @ W) ** 2)
            
            constraints_p = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
            bounds_p = [(0, 1) for _ in range(J_p)]
            W0_p = np.ones(J_p) / J_p
            
            try:
                result_p = minimize(obj_p, W0_p, method='SLSQP', bounds=bounds_p,
                                   constraints=constraints_p, options={'maxiter': 500})
                W_p = result_p.x if result_p.success else W0_p
            except:
                continue
            
            # Compute placebo gaps
            Y_synth_pre_p = Y0_pre_p @ W_p
            Y_synth_post_p = Y0_post_p @ W_p
            
            gaps_p = np.concatenate([
                Y1_pre_p - Y_synth_pre_p,
                Y1_post_p - Y_synth_post_p
            ])
            
            att_p = np.mean(Y1_post_p - Y_synth_post_p)
            
            self.placebo_gaps[placebo_unit] = gaps_p
            self.placebo_atts.append(att_p)
        
        # Compute p-value
        if self.placebo_atts:
            if self.att < 0:
                self.p_value = np.mean([p <= self.att for p in self.placebo_atts])
            else:
                self.p_value = np.mean([p >= self.att for p in self.placebo_atts])
        
        return len(self.placebo_gaps) > 0
    
    def create_visualization(self, output_path: Path):
        """Create comprehensive 3-panel visualization."""
        
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.25)
        
        # Convert times to datetime for plotting
        all_times_dt = pd.to_datetime(self.all_times)
        pre_end_idx = len(self.pre_times)
        
        # Combine actual and synthetic for full time series
        Y_actual = np.concatenate([self.Y1_pre, self.Y1_post])
        Y_synthetic = np.concatenate([self.Y_synthetic_pre, self.Y_synthetic_post])
        
        # =======================================================================
        # PANEL 1: Gaza vs Synthetic Gaza
        # =======================================================================
        ax1 = fig.add_subplot(gs[0])
        
        # Shading
        ax1.axvspan(all_times_dt[0], CONFLICT_DATE, alpha=0.15, color='green', label='Pre-conflict')
        ax1.axvspan(CONFLICT_DATE, all_times_dt[-1], alpha=0.15, color='red', label='Post-conflict')
        
        # Lines
        ax1.plot(all_times_dt, Y_actual, 'b-', linewidth=2.5, marker='o', markersize=5, 
                label='Gaza (Actual)')
        ax1.plot(all_times_dt, Y_synthetic, 'r--', linewidth=2, marker='s', markersize=4,
                label='Synthetic Gaza')
        
        # Conflict line
        ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
        ax1.text(CONFLICT_DATE, ax1.get_ylim()[1] * 0.95, 'Oct 7, 2023', 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax1.set_ylabel(OUTCOME_LABELS.get(self.outcome, self.outcome), fontsize=12)
        ax1.set_title(f'Synthetic Control: Gaza vs Synthetic Gaza ({self.outcome})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Add fit metrics
        ax1.text(0.98, 0.98, f'Pre-treatment R² = {self.pre_r2:.3f}\nPre-treatment RMSE = {self.pre_rmse:.3f}',
                transform=ax1.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # =======================================================================
        # PANEL 2: Gap (Actual - Synthetic)
        # =======================================================================
        ax2 = fig.add_subplot(gs[1])
        
        # Color bars by pre/post
        colors = ['gray'] * pre_end_idx + ['red'] * (len(self.gaps) - pre_end_idx)
        ax2.bar(all_times_dt, self.gaps, width=20, color=colors, alpha=0.7, edgecolor='none')
        
        # ATT line
        ax2.axhline(y=self.att, color='darkred', linestyle=':', linewidth=2.5, 
                   label=f'ATT = {self.att:.3f}')
        
        # Reference lines
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
        
        ax2.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
        ax2.legend(loc='upper right' if self.att > 0 else 'lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # =======================================================================
        # PANEL 3: Placebo Test
        # =======================================================================
        ax3 = fig.add_subplot(gs[2])
        
        # Plot all placebo gaps in light gray
        for unit_id, p_gaps in self.placebo_gaps.items():
            if len(p_gaps) == len(all_times_dt):
                ax3.plot(all_times_dt, p_gaps, color='lightgray', alpha=0.5, linewidth=1)
        
        # Plot Gaza gap in bold blue
        ax3.plot(all_times_dt, self.gaps, color='blue', linewidth=2.5, label='Gaza')
        
        # Reference lines
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=2)
        
        # Shading
        ax3.axvspan(all_times_dt[0], CONFLICT_DATE, alpha=0.1, color='green')
        ax3.axvspan(CONFLICT_DATE, all_times_dt[-1], alpha=0.1, color='red')
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Gap (Actual - Synthetic)', fontsize=11)
        ax3.set_title('Placebo Test: Gaza vs Donor Placebos', fontsize=12, fontweight='bold')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add interpretation
        if self.p_value is not None:
            interp = "✓ Effect is extreme" if self.p_value < 0.1 else "Effect not distinguishable from placebos"
            ax3.text(0.98, 0.02, f'P-value = {self.p_value:.3f}\n{interp}',
                    transform=ax3.transAxes, fontsize=10, va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2.5, label='Gaza'),
            Line2D([0], [0], color='lightgray', linewidth=1, label=f'Placebos (n={len(self.placebo_gaps)})'),
        ]
        ax3.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True


def create_summary_figure(results: dict, output_path: Path):
    """Create a summary figure comparing all outcomes."""
    
    n_outcomes = len(results)
    fig, axes = plt.subplots(n_outcomes, 2, figsize=(14, 4 * n_outcomes))
    
    if n_outcomes == 1:
        axes = axes.reshape(1, -1)
    
    for i, (outcome, sc) in enumerate(results.items()):
        # Left: Time series
        ax1 = axes[i, 0]
        
        all_times_dt = pd.to_datetime(sc.all_times)
        Y_actual = np.concatenate([sc.Y1_pre, sc.Y1_post])
        Y_synthetic = np.concatenate([sc.Y_synthetic_pre, sc.Y_synthetic_post])
        
        ax1.axvspan(all_times_dt[0], CONFLICT_DATE, alpha=0.15, color='green')
        ax1.axvspan(CONFLICT_DATE, all_times_dt[-1], alpha=0.15, color='red')
        
        ax1.plot(all_times_dt, Y_actual, 'b-', linewidth=2, marker='o', markersize=3, label='Gaza')
        ax1.plot(all_times_dt, Y_synthetic, 'r--', linewidth=1.5, label='Synthetic')
        ax1.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=1.5)
        
        ax1.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=10)
        ax1.set_title(f'{outcome.upper()}: R²={sc.pre_r2:.2f}', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Right: Placebo test
        ax2 = axes[i, 1]
        
        for unit_id, p_gaps in sc.placebo_gaps.items():
            if len(p_gaps) == len(all_times_dt):
                ax2.plot(all_times_dt, p_gaps, color='lightgray', alpha=0.5, linewidth=0.8)
        
        ax2.plot(all_times_dt, sc.gaps, color='blue', linewidth=2, label='Gaza')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=CONFLICT_DATE, color='black', linestyle='--', linewidth=1.5)
        
        pval_text = f'p={sc.p_value:.2f}' if sc.p_value is not None else ''
        ax2.set_title(f'Placebo Test: ATT={sc.att:.2f} {pval_text}', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Gap', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    axes[-1, 0].set_xlabel('Date', fontsize=11)
    axes[-1, 1].set_xlabel('Date', fontsize=11)
    
    plt.suptitle('Synthetic Control Analysis: All Outcomes with Placebo Tests', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Run comprehensive synthetic control analysis."""
    
    print("\n" + "=" * 70)
    print("SYNTHETIC CONTROL ANALYSIS WITH PLACEBO TESTS")
    print("=" * 70)
    
    # Load Gaza data
    print("\nLoading Gaza data...")
    panel = pd.read_csv(GAZA_PANEL)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza_df = panel[panel['unit_id'].str.startswith('PS02')].copy()
    westbank_df = panel[panel['unit_id'].str.startswith('PS01')].copy()
    
    print(f"  Gaza: {len(gaza_df)} obs")
    print(f"  West Bank: {len(westbank_df)} obs")
    
    # Load donors
    if DONOR_POOL.exists():
        donor_df = pd.read_csv(DONOR_POOL)
        donor_df['period_start'] = pd.to_datetime(donor_df['period_start'])
        donor_unit_col = 'admin_id'
        print(f"  Donors: {len(donor_df)} obs from {donor_df['country'].nunique()} countries")
    else:
        print(f"\n⚠ Donor pool not found at {DONOR_POOL}")
        print("  Using West Bank districts as pseudo-donors (for demonstration)")
        donor_df = westbank_df.rename(columns={'unit_id': 'admin_id'})
        donor_unit_col = 'admin_id'
    
    # Determine available outcomes
    available_outcomes = []
    for outcome in OUTCOMES:
        if outcome in gaza_df.columns and outcome in donor_df.columns:
            gaza_coverage = gaza_df[outcome].notna().sum() / len(gaza_df)
            donor_coverage = donor_df[outcome].notna().sum() / len(donor_df)
            if gaza_coverage > 0.5 and donor_coverage > 0.3:
                available_outcomes.append(outcome)
                print(f"  {outcome}: Gaza {gaza_coverage:.0%}, Donors {donor_coverage:.0%} ✓")
            else:
                print(f"  {outcome}: Insufficient coverage (Gaza {gaza_coverage:.0%}, Donors {donor_coverage:.0%})")
        else:
            print(f"  {outcome}: Not in data")
    
    if not available_outcomes:
        print("\n❌ No outcomes with sufficient data")
        return
    
    # Run analysis for each outcome
    results = {}
    summary_data = []
    
    for outcome in available_outcomes:
        print(f"\n{'='*60}")
        print(f"OUTCOME: {outcome.upper()}")
        print(f"{'='*60}")
        
        sc = SyntheticControlAnalysis(outcome)
        
        # Prepare data
        print("\nPreparing data...")
        if not sc.prepare_data(gaza_df, donor_df, 'unit_id', donor_unit_col, 'period_start'):
            print(f"  ❌ Data preparation failed")
            continue
        
        print(f"  Pre-treatment periods: {len(sc.pre_times)}")
        print(f"  Post-treatment periods: {len(sc.post_times)}")
        print(f"  Donors: {len(sc.donor_ids)}")
        
        # Fit synthetic control
        print("\nFitting synthetic control...")
        if not sc.fit():
            print(f"  ❌ Fitting failed")
            continue
        
        print(f"  Pre-treatment R²: {sc.pre_r2:.4f}")
        print(f"  Pre-treatment RMSE: {sc.pre_rmse:.4f}")
        print(f"  ATT: {sc.att:.4f}")
        
        # Top weights
        print(f"  Top donor weights:")
        sorted_idx = np.argsort(sc.weights)[::-1]
        for j in sorted_idx[:5]:
            if sc.weights[j] > 0.01:
                print(f"    {sc.donor_ids[j]}: {sc.weights[j]:.3f}")
        
        # Run placebo tests
        print("\nRunning placebo tests...")
        sc.run_placebo_tests(donor_df, donor_unit_col, 'period_start')
        
        if sc.p_value is not None:
            print(f"  Placebo units: {len(sc.placebo_gaps)}")
            print(f"  P-value: {sc.p_value:.3f}")
        
        # Store results
        results[outcome] = sc
        
        summary_data.append({
            'Outcome': outcome,
            'Pre_R2': sc.pre_r2,
            'Pre_RMSE': sc.pre_rmse,
            'ATT': sc.att,
            'ATT_pct': (sc.att / np.mean(sc.Y1_pre)) * 100 if np.mean(sc.Y1_pre) != 0 else np.nan,
            'P_value': sc.p_value,
            'N_donors': len(sc.donor_ids),
            'N_placebos': len(sc.placebo_gaps),
            'Valid': 'Yes' if sc.pre_r2 > 0.5 else 'Caution' if sc.pre_r2 > 0.2 else 'No'
        })
        
        # Create individual visualization
        print("\nCreating visualization...")
        sc.create_visualization(FIGURES_DIR / f'sc_{outcome}_full.png')
        print(f"  ✓ sc_{outcome}_full.png")
        
        # Save weights
        weights_df = pd.DataFrame({
            'Donor': sc.donor_ids,
            'Weight': sc.weights
        }).sort_values('Weight', ascending=False)
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        weights_df.to_csv(TABLES_DIR / f'sc_weights_{outcome}.csv', index=False)
    
    # Create summary visualization
    if results:
        print("\n" + "=" * 60)
        print("CREATING SUMMARY OUTPUTS")
        print("=" * 60)
        
        create_summary_figure(results, FIGURES_DIR / 'sc_all_outcomes_summary.png')
        print(f"  ✓ sc_all_outcomes_summary.png")
        
        # Save summary table
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(TABLES_DIR / 'synthetic_control_full_summary.csv', index=False)
        print(f"  ✓ synthetic_control_full_summary.csv")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
        # Interpretation guide
        print("\n" + "=" * 60)
        print("INTERPRETATION GUIDE")
        print("=" * 60)
        print("""
    Pre-R² indicates how well the synthetic control matches Gaza BEFORE conflict:
      > 0.8: Excellent fit - results are credible
      0.5-0.8: Good fit - results are reasonably credible  
      0.2-0.5: Fair fit - interpret with caution
      < 0.2: Poor fit - synthetic control INVALID for this outcome
    
    ATT (Average Treatment Effect on Treated):
      Negative: Gaza declined relative to counterfactual
      Positive: Gaza improved relative to counterfactual
    
    P-value (from placebo test):
      < 0.10: Gaza's effect is extreme compared to placebos
      > 0.10: Cannot distinguish Gaza from random variation
    
    For nightlights with poor fit: Use ITS/DiD instead of SC
        """)
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
