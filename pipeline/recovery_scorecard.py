#!/usr/bin/env python3
"""
recovery_scorecard.py — Generate Recovery Scorecard
====================================================
Computes recovery metrics for each satellite indicator:
  - Overall Recovery %: how much of the disaster drop has been recovered
  - Recovery since UNDP %: how much of the remaining gap closed after UNDP arrived

Usage:
    python pipeline/recovery_scorecard.py --config config/myanmar.yaml

Outputs:
    outputs/{code}/figures/scorecard_data.csv
    outputs/{code}/figures/scorecard.png
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Indicators to include in scorecard
# higher_is_better: True = recovery means values going UP (nightlights, NDVI)
#                   False = recovery means values going DOWN (BSI = debris)
INDICATORS = {
    'nightlights': {
        'name': 'Nighttime Lights',
        'short': 'Nightlights',
        'icon': '💡',
        'higher_is_better': True,
        'description': 'Economic activity & energy access',
    },
    'ndvi': {
        'name': 'Vegetation (NDVI)',
        'short': 'Vegetation',
        'icon': '🌿',
        'higher_is_better': True,
        'description': 'Agricultural & ecosystem recovery',
    },
    'vh': {
        'name': 'SAR VH Backscatter',
        'short': 'Radar VH',
        'icon': '📡',
        'higher_is_better': True,
        'description': 'Infrastructure & building presence',
    },
    'vv': {
        'name': 'SAR VV Backscatter',
        'short': 'Radar VV',
        'icon': '📡',
        'higher_is_better': True,
        'description': 'Surface roughness & structures',
    },
}

# Additional indicators added if present in data
OPTIONAL_INDICATORS = {
    'ndbi': {
        'name': 'Built-Up Index (NDBI)',
        'short': 'Built-Up',
        'icon': '🏗️',
        'higher_is_better': True,
        'description': 'Urban & built-up area density',
    },
    'bsi': {
        'name': 'Bare Soil Index (BSI)',
        'short': 'Bare Soil',
        'icon': '🏚️',
        'higher_is_better': False,  # high BSI = more debris/bare soil = bad
        'description': 'Debris & exposed ground (lower is better)',
    },
}


def compute_recovery(series_monthly, disaster_date, intervention_date, higher_is_better=True):
    """
    Compute recovery metrics for a single indicator time series.
    
    Parameters
    ----------
    series_monthly : pd.Series with DatetimeIndex (monthly averages)
    disaster_date : datetime
    intervention_date : datetime
    higher_is_better : bool
    
    Returns
    -------
    dict with recovery metrics
    """
    # Split into periods
    pre = series_monthly[series_monthly.index < disaster_date]
    post = series_monthly[series_monthly.index >= disaster_date]
    post_undp = series_monthly[series_monthly.index >= intervention_date]
    
    if len(pre) < 3 or len(post) < 2:
        return None
    
    # Baseline: average of last 6 months before disaster
    baseline = pre.tail(6).mean()
    
    # Post-disaster: worst 3 months (depends on direction)
    if higher_is_better:
        trough = post.head(6).nsmallest(3).mean()  # lowest values = worst
    else:
        trough = post.head(6).nlargest(3).mean()   # highest values = worst
    
    # Current: average of most recent 3 months
    current = post.tail(3).mean()
    
    # UNDP arrival: average of 2 months around intervention start
    if len(post_undp) >= 2:
        undp_arrival = post_undp.head(2).mean()
    else:
        undp_arrival = current  # fallback
    
    # Overall recovery %
    disaster_drop = abs(baseline - trough)
    drop_pct = (disaster_drop / abs(baseline) * 100) if abs(baseline) > 1e-10 else 0
    
    # Check if the "trough" is actually worse than baseline
    if higher_is_better:
        real_drop = baseline - trough  # positive = actual drop
    else:
        real_drop = trough - baseline  # positive = actual increase (bad)
    
    # Minimum threshold: if drop < 5% of baseline, mark as insufficient
    if drop_pct < 5.0:
        overall_pct = -1.0  # sentinel: no significant impact
        undp_pct = -1.0
    else:
        if higher_is_better:
            overall_pct = max(0, min(100, (current - trough) / (baseline - trough) * 100))
        else:
            overall_pct = max(0, min(100, (trough - current) / (trough - baseline) * 100))
        
        # Recovery since UNDP
        remaining_gap_at_undp = abs(baseline - undp_arrival)
        if remaining_gap_at_undp < 1e-10 or drop_pct < 5.0:
            undp_pct = -1.0
        else:
            if higher_is_better:
                undp_pct = max(0, min(150, (current - undp_arrival) / (baseline - undp_arrival) * 100))
            else:
                undp_pct = max(0, min(150, (undp_arrival - current) / (undp_arrival - baseline) * 100))
    
    return {
        'baseline': baseline,
        'trough': trough,
        'current': current,
        'undp_arrival': undp_arrival,
        'disaster_drop_pct': round(drop_pct, 1),
        'overall_recovery_pct': round(overall_pct, 1),
        'undp_recovery_pct': round(undp_pct, 1),
    }


def generate_scorecard_plot(scorecard_df, country_name, disaster_date, intervention_date, output_path):
    """
    Generate a visual scorecard as a horizontal bar chart.
    """
    df = scorecard_df.copy()
    n = len(df)
    
    fig, ax = plt.subplots(figsize=(12, max(4, n * 1.2 + 2)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    y_positions = np.arange(n)[::-1]  # reverse so first indicator is at top
    bar_height = 0.35
    
    # Colors
    color_overall = '#0468B1'   # UNDP blue
    color_undp = '#00A5E5'      # lighter blue
    color_bg = '#e8ecf0'        # background track
    
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        
        # Background track (100% mark)
        ax.barh(y, 100, height=bar_height * 2.2, color=color_bg, zorder=1)
        
        # Overall recovery bar
        ax.barh(y + bar_height/2, min(row['overall_recovery_pct'], 100),
                height=bar_height, color=color_overall, zorder=2, label='Overall Recovery' if i == 0 else '')
        
        # UNDP recovery bar
        ax.barh(y - bar_height/2, min(row['undp_recovery_pct'], 120),
                height=bar_height, color=color_undp, zorder=2, label='Since UNDP Arrival' if i == 0 else '')
        
        # Labels
        ax.text(-2, y, f"{row['icon']}  {row['short_name']}", ha='right', va='center',
                fontsize=11, fontweight='600', color='#1a2332')
        
        # Percentage labels
        ax.text(min(row['overall_recovery_pct'], 100) + 1.5, y + bar_height/2,
                f"{row['overall_recovery_pct']:.0f}%", va='center', fontsize=10,
                fontweight='600', color=color_overall)
        ax.text(min(row['undp_recovery_pct'], 120) + 1.5, y - bar_height/2,
                f"{row['undp_recovery_pct']:.0f}%", va='center', fontsize=10,
                fontweight='600', color=color_undp)
    
    # 100% reference line
    ax.axvline(x=100, color='#2e7d32', linestyle='--', linewidth=1, alpha=0.7, zorder=3)
    ax.text(101, y_positions[0] + 0.8, 'Pre-disaster\nlevel', fontsize=8, color='#2e7d32',
            va='bottom', ha='left')
    
    # Formatting
    ax.set_xlim(-35, 130)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color='#666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    # Title
    fig.suptitle(f'Recovery Scorecard — {country_name}',
                fontsize=16, fontweight='700', color='#1a2332', y=0.98,
                fontfamily='serif')
    ax.set_title(f'Disaster: {disaster_date.strftime("%B %d, %Y")}  |  '
                f'UNDP Intervention: {intervention_date.strftime("%B %Y")}',
                fontsize=10, color='#666', pad=12)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=9, frameon=True,
                      facecolor='white', edgecolor='#ddd')
    
    # Footnote
    fig.text(0.5, 0.01,
            'Overall Recovery: % of disaster drop recovered to date  |  '
            'Since UNDP Arrival: % of remaining gap closed after UNDP interventions began',
            ha='center', fontsize=8, color='#999', style='italic')
    
    plt.tight_layout(rect=[0.15, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {output_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    code = config['country_code'].lower()
    country_name = config['country_name']
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    intervention_date = pd.Timestamp(config['events'].get('intervention_start', config['events']['disaster_date']))
    
    panel_path = Path(f"outputs/{code}/panel_data.csv")
    figures_dir = Path(f"outputs/{code}/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RECOVERY SCORECARD — {country_name.upper()}")
    print(f"{'='*60}")
    print(f"  Disaster: {disaster_date.date()}")
    print(f"  UNDP intervention: {intervention_date.date()}")
    
    if not panel_path.exists():
        print(f"  ✗ Panel data not found: {panel_path}")
        return
    
    panel = pd.read_csv(panel_path)
    # Filter units if specified in config
    unit_filter = config.get("unit_filter", None)
    if unit_filter:
        panel = panel[panel["unit_id"].isin(unit_filter)]
        print(f"  Filtered to {panel["unit_id"].nunique()} units: {unit_filter}")
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    print(f"  Panel: {panel.shape[0]} rows, {panel['unit_id'].nunique()} units")
    
    # Build indicator list based on what's in the data
    active_indicators = {}
    for col, meta in {**INDICATORS, **OPTIONAL_INDICATORS}.items():
        if col in panel.columns and panel[col].notna().sum() > 10:
            active_indicators[col] = meta
    
    print(f"  Indicators found: {len(active_indicators)}")
    
    # Compute monthly national averages
    monthly = panel.groupby('period_start')[list(active_indicators.keys())].mean()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    
    # Compute recovery for each indicator
    results = []
    for col, meta in active_indicators.items():
        series = monthly[col].dropna()
        if len(series) < 6:
            print(f"  ⚠ Skipping {meta['name']}: insufficient data")
            continue
        
        recovery = compute_recovery(series, disaster_date, intervention_date, meta['higher_is_better'])
        if recovery is None:
            print(f"  ⚠ Skipping {meta['name']}: could not compute recovery")
            continue
        
        results.append({
            'indicator': col,
            'name': meta['name'],
            'short_name': meta['short'],
            'icon': meta['icon'],
            'description': meta['description'],
            'higher_is_better': meta['higher_is_better'],
            **recovery,
        })
        
        direction = "↑" if meta['higher_is_better'] else "↓"
        print(f"  {meta['icon']} {meta['short']:15s}  "
              f"Overall: {recovery['overall_recovery_pct']:5.1f}%  "
              f"Since UNDP: {recovery['undp_recovery_pct']:5.1f}%  "
              f"(drop: {recovery['disaster_drop_pct']:.1f}% {direction})")
    
    if not results:
        print("  ✗ No indicators could be computed")
        return
    
    scorecard_df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = figures_dir / "scorecard_data.csv"
    scorecard_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ {csv_path.name}")
    
    # Generate plot
    plot_path = figures_dir / "scorecard.png"
    generate_scorecard_plot(scorecard_df, country_name, disaster_date, intervention_date, plot_path)
    
    print(f"\n{'='*60}")
    print(f"✓ SCORECARD COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
