#!/usr/bin/env python3
"""
pipeline/visualizations.py
==========================
Time series and choropleth visualizations with UNDP strip.

Usage:
    python -m pipeline.visualizations --config config/myanmar.yaml
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm, Normalize
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


CATEGORY_COLORS = {
    'Education': '#3498DB', 'Debris & Solid Waste': '#795548',
    'Water & Sanitation (WASH)': '#00BCD4', 'Energy & Power': '#FFC107',
    'Health & Emergency': '#F44336', 'Infrastructure & Housing': '#9C27B0',
    'Livelihoods & Economy': '#4CAF50', 'Governance & Rule of Law': '#607D8B',
    'Social Inclusion': '#E91E63', 'Environment': '#8BC34A',
    'Other / Unspecified': '#9E9E9E'
}

# Core outcomes to plot (not derived variables)
CORE_OUTCOMES = {
    'nightlights': {'name': 'Nighttime Lights', 'line': 'darkblue', 'fill': 'blue', 'unit': 'nW/cm²/sr', 'cmap': 'hot_r'},
    'ndvi': {'name': 'Vegetation (NDVI)', 'line': 'forestgreen', 'fill': 'lightgreen', 'unit': 'Index', 'cmap': 'RdYlGn'},
    'rvi': {'name': 'Radar Vegetation (RVI)', 'line': 'darkgreen', 'fill': 'palegreen', 'unit': 'Index', 'cmap': 'Greens'},
    'conflict_events': {'name': 'Conflict Events', 'line': 'crimson', 'fill': 'lightcoral', 'unit': 'Count', 'cmap': 'Reds'},
    'conflict_fatalities': {'name': 'Conflict Fatalities', 'line': 'darkred', 'fill': 'salmon', 'unit': 'Count', 'cmap': 'Reds'},
    'rainfall': {'name': 'Rainfall', 'line': 'steelblue', 'fill': 'lightblue', 'unit': 'mm', 'cmap': 'Blues'},
    'vh': {'name': 'SAR VH (Radar)', 'line': 'slategray', 'fill': 'silver', 'unit': 'dB', 'cmap': 'viridis'},
    'vv': {'name': 'SAR VV (Radar)', 'line': 'dimgray', 'fill': 'lightgray', 'unit': 'dB', 'cmap': 'viridis'},
}

# Columns to EXCLUDE from time series
EXCLUDE_PATTERNS = [
    'log_', 'post', 'treated', 'months_since', 'gt_', 'trends_', 
    'structure_proxy', 'flood_proxy', 'time_index', 'year_month',
    'unit_id', 'period_start', 'period_end', 'admin'
]


def should_plot(col):
    col_lower = col.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False
    return True


def find_latest_valid_month(panel, outcome, dates):
    """Find the latest month that has non-NaN data for this outcome."""
    for d in reversed(sorted(dates)):
        df_date = panel[panel['period_start'] == d]
        vals = df_date[outcome].dropna()
        if len(vals) > 0 and vals.abs().sum() > 0:
            return pd.Timestamp(d)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config['country_name']
    country_code = config['country_code'].lower()
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    disaster_type = config.get('disaster_label', config['events'].get('disaster_type', 'Disaster').title())
    
    study_start = pd.Timestamp(config.get('study_period_start', disaster_date - pd.Timedelta(days=365)))
    study_end = pd.Timestamp(config.get('study_period_end', pd.Timestamp.now()))
    
    data_dir = Path(f"data/{country_code}")
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"VISUALIZATIONS: {country_name}")
    print(f"{'='*70}")
    print(f"Disaster: {disaster_date.strftime('%B %d, %Y')} ({disaster_type})")
    print(f"Study period: {study_start.strftime('%Y-%m')} to {study_end.strftime('%Y-%m')}")
    
    # Load panel
    panel_path = output_dir / "panel_data.csv"
    if not panel_path.exists():
        print(f"⚠ Panel not found: {panel_path}")
        return
    
    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    print(f"Panel: {len(panel):,} obs, {panel['unit_id'].nunique()} units")
    
    # Load UNDP projects
    undp_path = data_dir / "external" / "undp_projects.csv"
    undp_df = None
    if undp_path.exists():
        undp_df = pd.read_csv(undp_path)
        if len(undp_df) > 0:
            undp_df['start_date'] = pd.to_datetime(undp_df['start_date'], errors='coerce')
            undp_df = undp_df.dropna(subset=['start_date'])
            if 'is_likely_umbrella' in undp_df.columns:
                undp_df = undp_df[undp_df['is_likely_umbrella'] != True]
            print(f"UNDP Projects: {len(undp_df)}")
        else:
            undp_df = None
    
    # Load boundaries for maps
    gdf = None
    id_col = config['boundaries']['id_column']
    if Path(config['boundaries']['file_path']).exists():
        gdf = gpd.read_file(config['boundaries']['file_path'])
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    
    # Find CORE outcomes only
    outcomes = []
    for col in panel.columns:
        if col in CORE_OUTCOMES:
            if panel[col].notna().any():
                outcomes.append(col)
        elif should_plot(col) and panel[col].dtype in ['float64', 'int64']:
            if panel[col].notna().any() and col not in CORE_OUTCOMES:
                CORE_OUTCOMES[col] = {'name': col.replace('_', ' ').title(), 
                                      'line': 'gray', 'fill': 'lightgray', 
                                      'unit': '', 'cmap': 'viridis'}
                outcomes.append(col)
    
    print(f"Outcomes: {', '.join(outcomes)}")
    
    # Time series
    print("\n--- Time Series ---")
    for outcome in outcomes:
        cfg = CORE_OUTCOMES.get(outcome, {'name': outcome, 'line': 'gray', 'fill': 'lightgray', 'unit': ''})
        create_timeseries(panel, undp_df, outcome, cfg, disaster_date, disaster_type, 
                         country_name, study_start, study_end, figures_dir)
        print(f"  ✓ ts_{outcome}.png")
    
    # Choropleth maps
    if gdf is not None:
        print("\n--- Choropleth Maps ---")
        for outcome in outcomes:
            cfg = CORE_OUTCOMES.get(outcome, {'name': outcome, 'cmap': 'viridis', 'unit': ''})
            success = create_map(panel, gdf, id_col, outcome, cfg, disaster_date, country_name, figures_dir)
            if success:
                print(f"  ✓ map_{outcome}.png")
    
    # Summary
    print("\n--- Summary ---")
    create_summary(panel, outcomes, disaster_date, figures_dir)
    print(f"  ✓ summary_statistics.csv")
    
    n_figures = len(list(figures_dir.glob('*.png')))
    print(f"\n{'='*70}")
    print(f"✓ Complete: {n_figures} figures in {figures_dir}")
    print(f"{'='*70}")


def create_timeseries(panel, undp_df, outcome, cfg, disaster_date, disaster_type, 
                     country_name, study_start, study_end, figures_dir):
    """Time series with UNDP strip, auto y-axis scaling."""
    
    # Aggregate
    if 'conflict' in outcome or 'events' in outcome or 'fatalities' in outcome:
        monthly = panel.groupby('period_start')[outcome].sum().reset_index()
    else:
        agg = panel.groupby('period_start').agg({outcome: ['mean', 'std', 'count']}).reset_index()
        agg.columns = ['period_start', 'mean', 'std', 'count']
        agg['se'] = agg['std'] / np.sqrt(agg['count'])
        monthly = agg.rename(columns={'mean': outcome})
    
    monthly = monthly.sort_values('period_start')
    if monthly[outcome].isna().all():
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    xmin = study_start - timedelta(days=15)
    xmax = study_end + timedelta(days=15)
    
    # Confidence ribbon
    if 'se' in monthly.columns:
        ax.fill_between(monthly['period_start'], 
                       monthly[outcome] - 1.96*monthly['se'],
                       monthly[outcome] + 1.96*monthly['se'], 
                       alpha=0.3, color=cfg['fill'])
    
    # Main line
    ax.plot(monthly['period_start'], monthly[outcome], 
           color=cfg['line'], linewidth=2.5, marker='o', markersize=5)
    
    # Pre/post means
    pre = monthly[monthly['period_start'] < disaster_date][outcome]
    post = monthly[monthly['period_start'] >= disaster_date][outcome]
    pre_mean, post_mean = pre.mean(), post.mean()
    
    if not pd.isna(pre_mean):
        ax.axhline(pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax.axhline(post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Disaster line
    ax.axvline(disaster_date, color='red', linestyle='--', linewidth=2.5)
    
    # Shading
    if xmin < disaster_date:
        ax.axvspan(xmin, disaster_date, alpha=0.1, color='green')
    if xmax > disaster_date:
        ax.axvspan(disaster_date, xmax, alpha=0.08, color='red')
    
    ax.set_xlim(xmin, xmax)
    
    # ── AUTO Y-AXIS SCALING ──────────────────────────────────────────────
    # Use the actual data range (line values) with 10% padding
    # Don't let CI ribbons push the axis too far
    valid_vals = monthly[outcome].dropna()
    if len(valid_vals) > 0:
        data_min = valid_vals.min()
        data_max = valid_vals.max()
        data_range = data_max - data_min
        
        if data_range > 0:
            padding = data_range * 0.15  # 15% padding
            y_bottom = data_min - padding
            y_top = data_max + padding
            
            # Include CI bounds but cap at 2x data range
            if 'se' in monthly.columns:
                ci_low = (monthly[outcome] - 1.96 * monthly['se']).min()
                ci_high = (monthly[outcome] + 1.96 * monthly['se']).max()
                y_bottom = max(y_bottom, min(ci_low, data_min - data_range * 0.5))
                y_top = min(y_top, max(ci_high, data_max + data_range * 0.5))
                # Ensure we don't clip the actual data
                y_bottom = min(y_bottom, data_min - padding * 0.5)
                y_top = max(y_top, data_max + padding * 0.5)
            
            # Reserve space for disaster label at top (~12%)
            label_space = (y_top - y_bottom) * 0.15
            y_top += label_space
            
            ax.set_ylim(y_bottom, y_top)
    
    # Disaster label at top
    y_lim = ax.get_ylim()
    label_y = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.02
    ax.annotate(f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
               xy=(disaster_date, label_y),
               ha='center', va='top',
               fontsize=9, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor='red'))
    
    # Title with % change
    pct = ((post_mean - pre_mean) / abs(pre_mean) * 100) if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean) else None
    title_suffix = f" ({pct:+.1f}%)" if pct else ""
    
    ax.set_title(f'{country_name}: {cfg["name"]}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_xticklabels([])
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend = [Line2D([0], [0], color=cfg['line'], linewidth=2.5, marker='o', label=country_name)]
    if not pd.isna(pre_mean):
        legend.append(Line2D([0], [0], color='green', linestyle=':', label=f'Pre: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend.append(Line2D([0], [0], color='red', linestyle=':', label=f'Post: {post_mean:.2f}'))
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    
    # UNDP strip
    ax_strip.set_xlim(xmin, xmax)
    ax_strip.set_ylim(0, 1)
    
    if xmin < disaster_date:
        ax_strip.axvspan(xmin, disaster_date, alpha=0.15, color='green')
    if xmax > disaster_date:
        ax_strip.axvspan(disaster_date, xmax, alpha=0.1, color='red')
    ax_strip.axvline(disaster_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    cats_shown = set()
    if undp_df is not None and len(undp_df) > 0:
        markers = undp_df[(undp_df['start_date'] >= xmin) & (undp_df['start_date'] <= xmax)]
        if len(markers) > 0:
            monthly_markers = defaultdict(list)
            for _, row in markers.iterrows():
                m = row['start_date'].to_period('M').to_timestamp()
                cat = row.get('ml_category', 'Other / Unspecified')
                monthly_markers[m].append(cat)
            
            for month, cats in monthly_markers.items():
                for i, cat in enumerate(cats[:3]):
                    color = CATEGORY_COLORS.get(cat, '#9E9E9E')
                    y = max(0.15, min(0.85, 0.5 + (i - len(cats)/2 + 0.5) * 0.25))
                    ax_strip.scatter([month], [y], color=color, s=80, marker='s',
                                    edgecolor='black', linewidth=0.5, zorder=5)
                    cats_shown.add(cat)
    
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    if cats_shown:
        patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, '#9E9E9E'), label=c[:25]) for c in sorted(cats_shown)]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(5, len(patches)), fontsize=7, frameon=False)
    
    plt.savefig(figures_dir / f'ts_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_map(panel, gdf, id_col, outcome, cfg, disaster_date, country_name, figures_dir):
    """Three-panel choropleth: pre / post / latest available month."""
    
    dates = sorted(panel['period_start'].unique())
    pre_dates = [d for d in dates if pd.Timestamp(d) < disaster_date]
    post_dates = [d for d in dates if pd.Timestamp(d) >= disaster_date]
    
    if not pre_dates or not post_dates:
        return False
    
    # ── SMART DATE SELECTION ─────────────────────────────────────────────
    # Pre: at least one full month before the disaster month
    # (e.g., if disaster is March 28, pick February, not March)
    disaster_month_start = disaster_date.to_period('M').to_timestamp()
    pre_dates_clean = [d for d in pre_dates if pd.Timestamp(d) < disaster_month_start]
    if not pre_dates_clean:
        # Fall back to all pre_dates if filtering removes everything
        pre_dates_clean = pre_dates
    date1 = find_latest_valid_month(panel, outcome, pre_dates_clean)
    
    # Post: first month after disaster with valid data
    date2 = None
    for d in post_dates:
        df_check = panel[panel['period_start'] == d]
        vals = df_check[outcome].dropna()
        if len(vals) > 0 and vals.abs().sum() > 0:
            date2 = pd.Timestamp(d)
            break
    # Recent: latest month with valid data (NOT hardcoded study_period_end)
    date3 = find_latest_valid_month(panel, outcome, dates)
    
    if date1 is None or date2 is None or date3 is None:
        return False
    
    # If date3 == date2, try to find a later one
    if date3 <= date2 and len(post_dates) > 1:
        later_dates = [d for d in post_dates if pd.Timestamp(d) > date2]
        if later_dates:
            date3 = find_latest_valid_month(panel, outcome, later_dates) or date3
    
    def get_data(date):
        df_date = panel[panel['period_start'] == date]
        if 'conflict' in outcome or 'events' in outcome:
            return df_date.groupby('unit_id')[outcome].sum().reset_index()
        return df_date.groupby('unit_id')[outcome].mean().reset_index()
    
    gdf1 = gdf.merge(get_data(date1), left_on=id_col, right_on='unit_id', how='left')
    gdf2 = gdf.merge(get_data(date2), left_on=id_col, right_on='unit_id', how='left')
    gdf3 = gdf.merge(get_data(date3), left_on=id_col, right_on='unit_id', how='left')
    
    all_vals = pd.concat([gdf1[[outcome]], gdf2[[outcome]], gdf3[[outcome]]]).dropna()
    if len(all_vals) == 0:
        return False
    
    vmin = all_vals[outcome].quantile(0.02)
    vmax = all_vals[outcome].quantile(0.98)
    if vmin == vmax:
        vmax = vmin + 1
    
    # ── COLORMAP & NORMALIZATION ─────────────────────────────────────────
    # Nightlights: use PowerNorm for better contrast (black → yellow → red)
    cmap_name = cfg.get('cmap', 'viridis')
    if outcome == 'nightlights':
        norm = PowerNorm(gamma=0.4, vmin=max(vmin, 0.01), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    titles = [f"Pre-Event\n{date1.strftime('%B %Y')}", 
              f"Post-Event\n{date2.strftime('%B %Y')}", 
              f"Most Recent\n{date3.strftime('%B %Y')}"]
    
    for ax, g, title in zip(axes, [gdf1, gdf2, gdf3], titles):
        g.plot(column=outcome, ax=ax, cmap=cmap_name, norm=norm,
              edgecolor='black', linewidth=0.1, legend=False, 
              missing_kwds={'color': 'lightgray'})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', 
                label=f'{cfg["name"]} ({cfg.get("unit", "")})')
    
    fig.suptitle(f'{country_name}: {cfg["name"]}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(figures_dir / f'map_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True


def create_summary(panel, outcomes, disaster_date, figures_dir):
    """Summary statistics."""
    rows = []
    for o in outcomes:
        cfg = CORE_OUTCOMES.get(o, {'name': o})
        pre = panel[panel['period_start'] < disaster_date][o]
        post = panel[panel['period_start'] >= disaster_date][o]
        pre_m, post_m = pre.mean(), post.mean()
        pct = ((post_m - pre_m) / abs(pre_m) * 100) if pre_m != 0 and not pd.isna(pre_m) and not pd.isna(post_m) else None
        
        rows.append({
            'Outcome': cfg['name'],
            'Pre Mean': f"{pre_m:.3f}" if not pd.isna(pre_m) else "N/A",
            'Post Mean': f"{post_m:.3f}" if not pd.isna(post_m) else "N/A",
            'Change (%)': f"{pct:+.1f}%" if pct else "N/A",
            'N Pre': len(pre.dropna()),
            'N Post': len(post.dropna()),
        })
    
    pd.DataFrame(rows).to_csv(figures_dir / 'summary_statistics.csv', index=False)


if __name__ == "__main__":
    main()
