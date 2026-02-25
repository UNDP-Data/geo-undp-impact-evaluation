#!/usr/bin/env python3
"""
recovery_scorecard.py — Generate Recovery Scorecard (v4)
========================================================
Key methodology:
  - Unit-level filtering: only admin units with >10% drop are included
  - Seasonal adjustment (YoY) for NDVI AND nightlights
  - Intervention formula: (Current - Intervention) / (Baseline - Trough) x 100
    i.e. intervention progress as share of TOTAL disaster impact
  - Both metrics capped at 100%
  - Sentinel value -1.0 for indicators with <10% drop ("Stable")

Usage:
    python pipeline/recovery_scorecard.py --config config/myanmar.yaml
"""

import argparse, yaml
import pandas as pd, numpy as np
from pathlib import Path

INDICATORS = {
    'nightlights': {'name': 'Nighttime Lights (VIIRS)', 'short': 'Nightlights', 'icon': '💡',
        'higher_is_better': True, 'description': 'Economic activity & energy access', 'seasonal': True},
    'ndvi': {'name': 'Vegetation (NDVI)', 'short': 'Vegetation', 'icon': '🌿',
        'higher_is_better': True, 'description': 'Agricultural & ecosystem health', 'seasonal': True},
    'vh': {'name': 'SAR VH Backscatter', 'short': 'Radar VH', 'icon': '📡',
        'higher_is_better': True, 'description': 'Infrastructure & building presence', 'seasonal': False, 'db_scale': True},
    'vv': {'name': 'SAR VV Backscatter', 'short': 'Radar VV', 'icon': '📡',
        'higher_is_better': True, 'description': 'Surface roughness & structures', 'seasonal': False, 'db_scale': True},
}
OPTIONAL = {
    'ndbi': {'name': 'Built-Up Index (NDBI)', 'short': 'Built-Up', 'icon': '🏗️',
        'higher_is_better': True, 'description': 'Urban & built-up area density', 'seasonal': False},
    'bsi': {'name': 'Bare Soil Index (BSI)', 'short': 'Bare Soil', 'icon': '🏚️',
        'higher_is_better': False, 'description': 'Debris & exposed ground', 'seasonal': True},
}

MIN_DROP_PCT = 10.0


def compute_unit_level_recovery(panel, col, event_date, intervention_date, higher_is_better, seasonal, db_scale=False):
    """
    Compute recovery at the UNIT level, then aggregate only units with >10% drop.
    This prevents national averaging from diluting localized shocks.
    """
    units = panel['unit_id'].unique()
    unit_results = []
    unit_trends = []

    for uid in units:
        udata = panel[panel['unit_id'] == uid].set_index('period_start')[col].dropna().sort_index()
        if len(udata) < 6:
            continue

        pre = udata[udata.index < event_date]
        post = udata[udata.index >= event_date]
        if len(pre) < 3 or len(post) < 2:
            continue

        if seasonal:
            # Year-over-year: build monthly baseline from pre-event
            pre_df = pre.to_frame()
            pre_df['month'] = pre_df.index.month
            monthly_baseline = pre_df.groupby('month')[col].mean()

            post_df = post.to_frame()
            post_df['month'] = post_df.index.month
            post_df['expected'] = post_df['month'].map(monthly_baseline)
            post_df['deviation'] = post_df[col] - post_df['expected']
            post_df = post_df.dropna(subset=['expected'])
            if len(post_df) < 2:
                continue

            baseline_val = pre.mean()
            # Trough = baseline + worst deviation
            if higher_is_better:
                worst_dev = post_df['deviation'].head(6).nsmallest(3).mean()
            else:
                worst_dev = post_df['deviation'].head(6).nlargest(3).mean()
            trough_val = baseline_val + worst_dev

            # Current = baseline + recent deviation
            recent_dev = post_df['deviation'].tail(3).mean()
            current_val = baseline_val + recent_dev

            # Intervention point
            post_int = post_df[post_df.index >= intervention_date]
            if len(post_int) >= 2:
                int_dev = post_int['deviation'].head(2).mean()
            else:
                int_dev = recent_dev
            intervention_val = baseline_val + int_dev
        else:
            baseline_val = pre.tail(6).mean()
            if higher_is_better:
                trough_val = post.head(6).nsmallest(3).mean()
            else:
                trough_val = post.head(6).nlargest(3).mean()
            current_val = post.tail(3).mean()
            post_int = udata[udata.index >= intervention_date]
            intervention_val = post_int.head(2).mean() if len(post_int) >= 2 else current_val

        # Check unit-level drop
        # For dB-scale indicators (SAR), convert to linear power before computing %
        if db_scale:
            b_lin = 10 ** (baseline_val / 10)
            t_lin = 10 ** (trough_val / 10)
            drop = abs(b_lin - t_lin)
            drop_pct = (drop / abs(b_lin) * 100) if abs(b_lin) > 1e-10 else 0
        else:
            drop = abs(baseline_val - trough_val)
            drop_pct = (drop / abs(baseline_val) * 100) if abs(baseline_val) > 1e-10 else 0

        # SAR uses 5% threshold (small dB changes = significant physical change)
        threshold = 5.0 if db_scale else MIN_DROP_PCT
        if drop_pct >= threshold:
            unit_results.append({
                'unit_id': uid,
                'baseline': baseline_val,
                'trough': trough_val,
                'current': current_val,
                'intervention': intervention_val,
                'drop_pct': drop_pct,
            })
            # Trend from last 3 months (use deviations for seasonal indicators)
            if seasonal and len(post_df) >= 3:
                last3_dev = post_df['deviation'].tail(3).values
                if len(last3_dev) >= 2:
                    unit_trends.append(float(last3_dev[-1]) - float(last3_dev[0]))
            elif not seasonal and len(post) >= 3:
                last3_raw = post.tail(3).values
                unit_trends.append(float(last3_raw[-1]) - float(last3_raw[0]))

    if not unit_results:
        return None

    # Aggregate: average across affected units
    df = pd.DataFrame(unit_results)
    n_affected = len(df)
    n_total = len(units)
    baseline = df['baseline'].mean()
    trough = df['trough'].mean()
    current = df['current'].mean()
    intervention_val = df['intervention'].mean()
    avg_drop = df['drop_pct'].mean()

    # For dB-scale indicators, compute recovery in linear power space
    if db_scale:
        bl = 10 ** (baseline / 10)
        tl = 10 ** (trough / 10)
        cl = 10 ** (current / 10)
        il = 10 ** (intervention_val / 10)
    else:
        bl, tl, cl, il = baseline, trough, current, intervention_val

    # Overall recovery: (Current - Trough) / (Baseline - Trough) * 100
    total_impact = abs(bl - tl)
    if total_impact < 1e-10:
        overall = 0.0
    else:
        if higher_is_better:
            overall = max(0, min(100, (cl - tl) / (bl - tl) * 100))
        else:
            overall = max(0, min(100, (tl - cl) / (tl - bl) * 100))

    # Since intervention: (Current - Intervention) / (Baseline - Trough) * 100
    # This measures intervention progress as % of TOTAL disaster impact
    if total_impact < 1e-10:
        undp = 0.0
    else:
        if higher_is_better:
            undp = max(0, min(100, (cl - il) / (bl - tl) * 100))
        else:
            undp = max(0, min(100, (il - cl) / (tl - bl) * 100))

    # Trend direction

    _avg = sum(unit_trends) / len(unit_trends) if unit_trends else 0
    if higher_is_better:
        if _avg > 0.005: _trend = 'improving'
        elif _avg < -0.005: _trend = 'declining'
        else: _trend = 'stagnant'
    else:
        if _avg < -0.005: _trend = 'improving'
        elif _avg > 0.005: _trend = 'declining'
        else: _trend = 'stagnant'

    return {
        'baseline': baseline,
        'trough': trough,
        'current': current,
        'undp_arrival': intervention_val,
        'disaster_drop_pct': round(avg_drop, 1),
        'overall_recovery_pct': round(overall, 1),
        'undp_recovery_pct': round(undp, 1),
        'n_affected_units': n_affected,
        'n_total_units': n_total,
        'trend': _trend,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    code = config['country_code'].lower()
    country_name = config['country_name']
    event_date = pd.Timestamp(config['events']['disaster_date'])
    intervention_date = pd.Timestamp(config['events'].get('intervention_start', config['events']['disaster_date']))

    panel_path = Path(f"outputs/{code}/panel_data.csv")
    figures_dir = Path(f"outputs/{code}/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RECOVERY SCORECARD — {country_name.upper()}")
    print(f"{'='*60}")
    print(f"  Event: {event_date.date()}")
    print(f"  Intervention: {intervention_date.date()}")

    if not panel_path.exists():
        print(f"  ✗ Panel not found"); return

    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])

    # Filter units if specified in config (e.g., Gaza only)
    unit_filter = config.get('unit_filter', None)
    if unit_filter:
        panel = panel[panel['unit_id'].isin(unit_filter)]
        print(f"  Filtered to {panel['unit_id'].nunique()} units (config)")

    print(f"  Panel: {panel.shape[0]} rows, {panel['unit_id'].nunique()} units")

    all_ind = {**INDICATORS, **OPTIONAL}
    active = {c: m for c, m in all_ind.items() if c in panel.columns and panel[c].notna().sum() > 10}
    print(f"  Indicators: {len(active)}")

    results = []
    for col, meta in active.items():
        r = compute_unit_level_recovery(
            panel, col, event_date, intervention_date,
            meta['higher_is_better'], meta.get('seasonal', False),
            db_scale=meta.get('db_scale', False)
        )

        if r is None:
            # No units had >10% drop
            results.append({
                'indicator': col, 'name': meta['name'], 'short_name': meta['short'],
                'icon': meta['icon'], 'description': meta['description'],
                'higher_is_better': meta['higher_is_better'],
                'baseline': 0, 'trough': 0, 'current': 0, 'undp_arrival': 0,
                'disaster_drop_pct': 0,
                'overall_recovery_pct': -1.0, 'undp_recovery_pct': -1.0,
                'n_affected_units': 0, 'n_total_units': panel['unit_id'].nunique(), 'trend': 'stable',
            })
            print(f"  {meta['icon']} {meta['short']:15s}  Stable (no units with >{MIN_DROP_PCT:.0f}% drop)")
            continue

        results.append({
            'indicator': col, 'name': meta['name'], 'short_name': meta['short'],
            'icon': meta['icon'], 'description': meta['description'],
            'higher_is_better': meta['higher_is_better'],
            **r,
        })

        print(f"  {meta['icon']} {meta['short']:15s}  "
              f"Overall: {r['overall_recovery_pct']:5.1f}%  "
              f"Since intervention: {r['undp_recovery_pct']:5.1f}%  "
              f"(drop: {r['disaster_drop_pct']:.1f}%, "
              f"{r['n_affected_units']}/{r['n_total_units']} units affected)")

    if not results:
        print("  ✗ No indicators"); return

    df = pd.DataFrame(results)
    df.to_csv(figures_dir / "scorecard_data.csv", index=False)
    print(f"\n  ✓ scorecard_data.csv")
    print(f"{'='*60}")
    print(f"✓ COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
