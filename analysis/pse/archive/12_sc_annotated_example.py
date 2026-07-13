#!/usr/bin/env python3
"""
analysis/pse/12_sc_annotated_example.py
=======================================
Creates annotated example plots to help interpret SC results.
Uses VH as the example since that's what the user uploaded.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs/pse")
DATA_DIR = Path("data")
FIGURES_DIR = OUTPUT_DIR / "figures" / "synthetic_control"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
DONOR_POOL = DATA_DIR / "synthetic_control" / "donor_pool_combined.csv"
UNDP_FILE = DATA_DIR / "pse" / "external" / "undp_projects.csv"

CONFLICT_DATE = pd.Timestamp("2023-10-07")
PRE_START = pd.Timestamp("2023-01-01")
PRE_END = pd.Timestamp("2023-09-01")


def fit_synthetic_control(Y1_pre, Y0_pre, Y1_post, Y0_post):
    J = Y0_pre.shape[1]
    def objective(W):
        return np.mean((Y1_pre - Y0_pre @ W) ** 2)
    constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
    bounds = [(0, 1) for _ in range(J)]
    W0 = np.ones(J) / J
    result = minimize(objective, W0, method='SLSQP', bounds=bounds,
                     constraints=constraints, options={'maxiter': 1000})
    weights = result.x if result.success else W0
    Y_synth_pre = Y0_pre @ weights
    Y_synth_post = Y0_post @ weights
    return {
        'Y_synth_pre': Y_synth_pre,
        'Y_synth_post': Y_synth_post,
        'gaps_pre': Y1_pre - Y_synth_pre,
        'gaps_post': Y1_post - Y_synth_post,
        'att': np.mean(Y1_post - Y_synth_post),
    }


def prepare_data(df, donor_df, outcome, pre_times, post_times):
    agg = df.groupby('period_start')[outcome].mean().reset_index()
    agg = agg.sort_values('period_start')
    Y1_pre = agg[agg['period_start'].isin(pre_times)][outcome].values
    Y1_post = agg[agg['period_start'].isin(post_times)][outcome].values
    
    if len(Y1_pre) == 0:
        return None
    
    valid_donors, Y0_pre_list, Y0_post_list = [], [], []
    for unit in donor_df['admin_id'].unique():
        udf = donor_df[donor_df['admin_id'] == unit]
        upre = [udf[udf['period_start'] == t][outcome].values[0] if len(udf[udf['period_start'] == t]) > 0 else np.nan for t in pre_times]
        upost = [udf[udf['period_start'] == t][outcome].values[0] if len(udf[udf['period_start'] == t]) > 0 else np.nan for t in post_times]
        upre, upost = np.array(upre), np.array(upost)
        if np.sum(np.isnan(upre)) > len(pre_times) * 0.3:
            continue
        upre[np.isnan(upre)] = np.nanmean(upre)
        upost[np.isnan(upost)] = np.nanmean(upost) if not np.all(np.isnan(upost)) else 0
        valid_donors.append(unit)
        Y0_pre_list.append(upre)
        Y0_post_list.append(upost)
    
    if len(valid_donors) < 3:
        return None
    
    return {
        'Y1_pre': Y1_pre, 'Y1_post': Y1_post,
        'Y0_pre': np.array(Y0_pre_list).T, 'Y0_post': np.array(Y0_post_list).T
    }


def create_annotated_interpretation_plot(outcome='vh'):
    """Create fully annotated plot explaining how to interpret SC gaps."""
    
    print("Loading data...")
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    panel['unit_id'] = panel['unit_id'].astype(str)
    
    gaza = panel[panel['unit_id'].str.startswith('PS02')]
    
    donor = pd.read_csv(DONOR_POOL)
    donor['period_start'] = pd.to_datetime(donor['period_start'])
    
    # UNDP dates
    undp = pd.read_csv(UNDP_FILE)
    undp['start_date'] = pd.to_datetime(undp['start_date'], errors='coerce')
    if 'is_likely_umbrella' in undp.columns:
        undp = undp[undp['is_likely_umbrella'] != True]
    undp = undp[undp['start_date'] >= CONFLICT_DATE]
    
    int_dates = {}
    if 'ml_category' in undp.columns:
        for cat in undp['ml_category'].dropna().unique():
            date = undp[undp['ml_category'] == cat]['start_date'].min()
            if pd.notna(date):
                int_dates[cat] = date
    int_dates['Any UNDP'] = undp['start_date'].min()
    
    # Time periods
    all_times = sorted(panel['period_start'].unique())
    pre_times = [t for t in all_times if pd.Timestamp(t) >= PRE_START and pd.Timestamp(t) <= PRE_END]
    post_times = [t for t in all_times if pd.Timestamp(t) > PRE_END]
    
    # Fit SC
    data = prepare_data(gaza, donor, outcome, pre_times, post_times)
    if data is None:
        print(f"Could not prepare data for {outcome}")
        return
    
    sc = fit_synthetic_control(data['Y1_pre'], data['Y0_pre'], data['Y1_post'], data['Y0_post'])
    
    all_gaps = np.concatenate([sc['gaps_pre'], sc['gaps_post']])
    all_times = pre_times + post_times
    times_dt = pd.to_datetime(all_times)
    
    # Create annotated plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Post-conflict only
    post_mask = np.array([t > CONFLICT_DATE for t in times_dt])
    times_post = times_dt[post_mask]
    gaps_post = all_gaps[post_mask]
    
    # Plot gap
    ax.plot(times_post, gaps_post, 'o-', color='darkred', linewidth=3, markersize=10, 
            label='Gaza Gap (Actual - Synthetic)', zorder=5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Counterfactual (Gap = 0)')
    
    # Shade interpretation zones
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5, alpha=0.1, color='green')
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -1.5, 0, alpha=0.1, color='red')
    
    # Add zone labels
    ax.text(times_post[0], 0.15, 'ABOVE COUNTERFACTUAL\n(Better than expected)', 
            fontsize=11, color='darkgreen', fontweight='bold', va='bottom')
    ax.text(times_post[0], -0.15, 'BELOW COUNTERFACTUAL\n(Worse than expected = Damage)', 
            fontsize=11, color='darkred', fontweight='bold', va='top')
    
    # Key interventions for VH
    key_interventions = {
        'Any UNDP': int_dates.get('Any UNDP'),
        'Debris & Solid Waste': int_dates.get('Debris & Solid Waste'),
        'Infrastructure & Housing': int_dates.get('Infrastructure & Housing'),
    }
    
    colors = {'Any UNDP': 'gray', 'Debris & Solid Waste': 'brown', 'Infrastructure & Housing': 'orange'}
    
    for cat, date in key_interventions.items():
        if date and date >= times_post[0] and date <= times_post[-1]:
            ax.axvline(x=date, color=colors.get(cat, 'purple'), linestyle='--', linewidth=2.5,
                      label=f'{cat} ({date.strftime("%b %Y")})', zorder=3)
    
    # Add annotations for key points
    # Find lowest point
    min_idx = np.argmin(gaps_post)
    min_date = times_post[min_idx]
    min_gap = gaps_post[min_idx]
    
    ax.annotate(f'LOWEST POINT\nGap = {min_gap:.2f}', 
                xy=(min_date, min_gap), 
                xytext=(min_date - pd.Timedelta(days=60), min_gap - 0.2),
                fontsize=10, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred'))
    
    # Find recovery point (if any)
    max_post_min = np.argmax(gaps_post[min_idx:]) + min_idx
    if max_post_min > min_idx and gaps_post[max_post_min] > min_gap + 0.3:
        recovery_date = times_post[max_post_min]
        recovery_gap = gaps_post[max_post_min]
        ax.annotate(f'RECOVERY\nGap = {recovery_gap:.2f}', 
                    xy=(recovery_date, recovery_gap), 
                    xytext=(recovery_date + pd.Timedelta(days=30), recovery_gap + 0.2),
                    fontsize=10, fontweight='bold', color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkgreen'))
    
    # Add interpretation guide
    guide_text = """
HOW TO READ THIS PLOT:

1. GAP = Actual Gaza - Synthetic Gaza
   • Synthetic = "What Gaza would look like without conflict"

2. GAP < 0 (Below zero line)
   → Gaza is BELOW its counterfactual
   → This represents DAMAGE from the conflict

3. GAP MOVING UP (toward 0)
   → RECOVERY toward counterfactual
   → If coincides with UNDP → possible intervention effect

4. VERTICAL LINES = UNDP intervention start dates
   • Check if gap improves AFTER these lines
   • VH should respond to: Debris, Infrastructure
    """
    
    ax.text(0.98, 0.98, guide_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray'))
    
    # Labels
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gap (Actual VH - Synthetic VH)', fontsize=12)
    ax.set_title('SAR Backscatter (VH): Annotated Gap Interpretation\n'
                 'How to Read the Synthetic Control Results', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add arrows showing "recovery direction"
    ax.annotate('', xy=(times_post[-1], 0), xytext=(times_post[-1], -0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3, ls='--'))
    ax.text(times_post[-1] + pd.Timedelta(days=10), -0.25, 'Recovery\nDirection', 
            fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sc_vh_annotated_interpretation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: sc_vh_annotated_interpretation.png")
    
    # Create a summary figure explaining SC methodology
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: What is Synthetic Control?
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('1. What is Synthetic Control?', fontsize=14, fontweight='bold')
    
    sc_text = """
    SYNTHETIC CONTROL creates a "fake Gaza"
    from weighted average of donor countries:
    
    Synthetic Gaza = 
        40% × Jordan +
        35% × Lebanon +
        25% × Egypt
    
    This represents:
    "What Gaza WOULD have looked like
     if the conflict never happened"
    
    The WEIGHTS are chosen to match
    Gaza's PRE-conflict trajectory.
    """
    ax1.text(0.1, 0.9, sc_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Panel 2: What is the Gap?
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('2. What is the Gap?', fontsize=14, fontweight='bold')
    
    gap_text = """
    GAP = Actual Gaza - Synthetic Gaza
    
    ┌─────────────────────────────┐
    │ If Gap = 0                  │
    │ → Gaza equals counterfactual│
    │ → No conflict impact        │
    ├─────────────────────────────┤
    │ If Gap < 0 (NEGATIVE)       │
    │ → Gaza BELOW counterfactual │
    │ → DAMAGE from conflict      │
    ├─────────────────────────────┤
    │ If Gap > 0 (POSITIVE)       │
    │ → Gaza ABOVE counterfactual │
    │ → Better than expected      │
    └─────────────────────────────┘
    
    RECOVERY = Gap becoming LESS negative
               (moving toward zero)
    """
    ax2.text(0.1, 0.9, gap_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Panel 3: How to Test UNDP Effect?
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title('3. Testing UNDP Effect', fontsize=14, fontweight='bold')
    
    undp_text = """
    We test if the Gap IMPROVES after
    UNDP interventions start:
    
    Model:
    Gap = α + β₁×time + β₂×Post_UNDP 
              + β₃×time_since_UNDP
    
    ┌─────────────────────────────┐
    │ β₂ > 0 (Level shift)        │
    │ → Gap jumps UP at UNDP start│
    │ → Immediate improvement     │
    ├─────────────────────────────┤
    │ β₃ > 0 (Trend change)       │
    │ → Recovery ACCELERATES      │
    │   after UNDP starts         │
    └─────────────────────────────┘
    
    WEST BANK PLACEBO:
    Run same model on WB (not bombed)
    If WB Gap ≈ 0 → Model is VALID
    """
    ax3.text(0.1, 0.9, undp_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sc_methodology_guide.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: sc_methodology_guide.png")


if __name__ == "__main__":
    create_annotated_interpretation_plot()
