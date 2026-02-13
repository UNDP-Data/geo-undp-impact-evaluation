#!/usr/bin/env python3
"""
pipeline/visualizations_v4.py
=============================
COMPLETE visualization pipeline including:

1. TIME SERIES (Gaza format) with UNDP strip
2. CHOROPLETH MAPS from PANEL DATA (no GeoTIFFs needed!)
3. GOOGLE TRENDS plots
4. UNDP PROJECT TIMELINE (detailed with dates)
5. CONFLICT TIMELINE
6. SUMMARY STATISTICS

The choropleth maps use the satellite data already in panel_data.csv,
so NO separate raster downloads are needed!

Usage:
    python -m pipeline.visualizations_v4 --config config/myanmar.yaml
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


CATEGORY_COLORS = {
    'Education': '#3498DB',
    'Debris & Solid Waste': '#795548',
    'Water & Sanitation (WASH)': '#00BCD4',
    'Energy & Power': '#FFC107',
    'Health & Emergency': '#F44336',
    'Infrastructure & Housing': '#9C27B0',
    'Livelihoods & Economy': '#4CAF50',
    'Governance & Rule of Law': '#607D8B',
    'Social Inclusion': '#E91E63',
    'Environment': '#8BC34A',
    'Other / Unspecified': '#9E9E9E'
}

OUTCOME_CONFIG = {
    'nightlights': {'name': 'Nighttime Lights', 'line': 'darkblue', 'fill': 'blue', 'unit': 'nW/cm²/sr', 'cmap': 'YlOrRd'},
    'ndvi': {'name': 'Vegetation (NDVI)', 'line': 'forestgreen', 'fill': 'lightgreen', 'unit': 'Index', 'cmap': 'RdYlGn'},
    'conflict_events': {'name': 'Conflict Events', 'line': 'crimson', 'fill': 'lightcoral', 'unit': 'Count', 'cmap': 'Reds'},
    'conflict_fatalities': {'name': 'Conflict Fatalities', 'line': 'darkred', 'fill': 'salmon', 'unit': 'Count', 'cmap': 'Reds'},
    'rainfall': {'name': 'Rainfall', 'line': 'steelblue', 'fill': 'lightblue', 'unit': 'mm', 'cmap': 'Blues'},
    'price_rice': {'name': 'Rice Price', 'line': 'olive', 'fill': 'khaki', 'unit': 'MMK/kg', 'cmap': 'YlOrBr'},
    'price_oil': {'name': 'Oil Price', 'line': 'darkorange', 'fill': 'peachpuff', 'unit': 'MMK/L', 'cmap': 'Oranges'},
    'vh': {'name': 'SAR VH (Radar)', 'line': 'slategray', 'fill': 'silver', 'unit': 'dB', 'cmap': 'viridis'},
    'vv': {'name': 'SAR VV (Radar)', 'line': 'dimgray', 'fill': 'lightgray', 'unit': 'dB', 'cmap': 'viridis'},
    'rvi': {'name': 'Radar Vegetation', 'line': 'darkgreen', 'fill': 'palegreen', 'unit': 'Index', 'cmap': 'Greens'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').lower()
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    intervention_date = pd.Timestamp(config['events']['intervention_start'])
    disaster_type = config['events'].get('disaster_type', 'Disaster').title()
    
    data_dir = Path(f"data/{country_code}")
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE VISUALIZATIONS: {country_name}")
    print(f"{'='*70}")
    print(f"Disaster: {disaster_date.strftime('%B %d, %Y')} ({disaster_type})")
    print(f"Intervention: {intervention_date.strftime('%B %Y')}")
    
    # Load panel
    panel_path = output_dir / "panel_data.csv"
    if not panel_path.exists():
        print(f"⚠ Panel not found: {panel_path}")
        return
    
    panel = pd.read_csv(panel_path)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    print(f"Data: {len(panel):,} obs, {panel['unit_id'].nunique()} units")
    
    # Load UNDP projects
    undp_path = data_dir / "external" / "undp_projects.csv"
    undp_df = None
    if undp_path.exists():
        undp_df = pd.read_csv(undp_path)
        undp_df['start_date'] = pd.to_datetime(undp_df['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp_df.columns:
            undp_df = undp_df[undp_df['is_likely_umbrella'] != True]
        print(f"UNDP Projects: {len(undp_df)}")
    
    # Load boundaries
    boundary_file = config['boundaries']['file_path']
    gdf = None
    id_col = config['boundaries']['id_column']
    if Path(boundary_file).exists():
        gdf = gpd.read_file(boundary_file)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        print(f"Boundaries: {len(gdf)} units")
    
    # Load Google Trends
    trends_path = data_dir / "external" / "google_trends.csv"
    trends_df = None
    if trends_path.exists():
        trends_df = pd.read_csv(trends_path)
        print(f"Google Trends: loaded")
    
    outcomes = [o for o in OUTCOME_CONFIG.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"Outcomes: {', '.join(outcomes)}")
    
    # =========================================================================
    # 1. TIME SERIES (Gaza Format)
    # =========================================================================
    print("\n--- Time Series (Gaza Format) ---")
    for outcome in outcomes:
        cfg = OUTCOME_CONFIG[outcome]
        create_timeseries(panel, undp_df, outcome, cfg, disaster_date, disaster_type, country_name, figures_dir)
        print(f"  ✓ ts_{outcome}.png")
    
    # =========================================================================
    # 2. CHOROPLETH MAPS FROM PANEL DATA
    # =========================================================================
    if gdf is not None:
        print("\n--- Choropleth Maps (from Panel Data) ---")
        for outcome in outcomes[:6]:
            cfg = OUTCOME_CONFIG[outcome]
            create_choropleth_panel(panel, gdf, id_col, outcome, cfg, disaster_date, disaster_type, country_name, figures_dir)
            print(f"  ✓ map_{outcome}.png")
    
    # =========================================================================
    # 3. GOOGLE TRENDS
    # =========================================================================
    if trends_df is not None:
        print("\n--- Google Trends ---")
        create_google_trends(trends_df, disaster_date, disaster_type, country_name, figures_dir)
        print(f"  ✓ google_trends.png")
    
    # =========================================================================
    # 4. UNDP PROJECT TIMELINE (DETAILED)
    # =========================================================================
    if undp_df is not None and len(undp_df) > 0:
        print("\n--- UNDP Project Timeline ---")
        create_undp_timeline_detailed(undp_df, disaster_date, disaster_type, country_name, figures_dir)
        print(f"  ✓ undp_timeline_detailed.png")
    
    # =========================================================================
    # 5. CONFLICT TIMELINE
    # =========================================================================
    if 'conflict_events' in panel.columns:
        print("\n--- Conflict Timeline ---")
        create_conflict_timeline(panel, disaster_date, disaster_type, country_name, figures_dir)
        print(f"  ✓ conflict_timeline.png")
    
    # =========================================================================
    # 6. SUMMARY
    # =========================================================================
    print("\n--- Summary Statistics ---")
    create_summary(panel, outcomes, disaster_date, disaster_type, figures_dir)
    print(f"  ✓ summary_statistics.csv")
    
    n_figures = len(list(figures_dir.glob("*.png")))
    print(f"\n{'='*70}")
    print(f"✓ Complete: {n_figures} figures in {figures_dir}")
    print(f"{'='*70}")


def create_timeseries(panel, undp_df, outcome, cfg, disaster_date, disaster_type, country_name, figures_dir):
    """Gaza-format time series with UNDP strip."""
    
    if 'conflict' in outcome or 'events' in outcome:
        monthly = panel.groupby('period_start')[outcome].sum().reset_index()
    else:
        monthly = panel.groupby('period_start').agg({outcome: ['mean', 'std', 'count']}).reset_index()
        monthly.columns = ['period_start', 'mean', 'std', 'count']
        monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
        monthly = monthly.rename(columns={'mean': outcome})
    
    monthly = monthly.sort_values('period_start')
    if monthly[outcome].isna().all():
        return
    
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # Fill ribbon
    if 'se' in monthly.columns:
        ax_main.fill_between(monthly['period_start'],
                            monthly[outcome] - 1.96 * monthly['se'],
                            monthly[outcome] + 1.96 * monthly['se'],
                            alpha=0.3, color=cfg['fill'])
    
    ax_main.plot(monthly['period_start'], monthly[outcome],
                color=cfg['line'], linewidth=2.5, marker='o', markersize=5)
    
    # Pre/post means
    pre = monthly[monthly['period_start'] < disaster_date][outcome]
    post = monthly[monthly['period_start'] >= disaster_date][outcome]
    pre_mean = pre.mean() if len(pre) > 0 else np.nan
    post_mean = post.mean() if len(post) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax_main.axvline(x=disaster_date, color='red', linestyle='--', linewidth=2.5)
    
    xmin, xmax = monthly['period_start'].min(), monthly['period_start'].max()
    if xmin < disaster_date:
        ax_main.axvspan(xmin, min(disaster_date, xmax), alpha=0.1, color='green')
    if xmax > disaster_date:
        ax_main.axvspan(max(disaster_date, xmin), xmax, alpha=0.08, color='red')
    
    y_top = ax_main.get_ylim()[1]
    ax_main.text(disaster_date, y_top * 0.95, f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct:+.1f}% After {disaster_type})"
    else:
        title_suffix = ""
    
    ax_main.set_title(f'{country_name}: {cfg["name"]}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3)
    
    legend_elements = [Line2D([0], [0], color=cfg['line'], linewidth=2.5, marker='o', label=country_name)]
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label=f'Pre Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label=f'Post Mean: {post_mean:.2f}'))
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # UNDP strip
    ax_strip.set_xlim(ax_main.get_xlim())
    ax_strip.set_ylim(0, 1)
    
    if xmin < disaster_date:
        ax_strip.axvspan(xmin, min(disaster_date, xmax), alpha=0.15, color='green')
    if xmax > disaster_date:
        ax_strip.axvspan(max(disaster_date, xmin), xmax, alpha=0.1, color='red')
    ax_strip.axvline(x=disaster_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    categories_shown = set()
    if undp_df is not None:
        markers_df = undp_df[(undp_df['start_date'] >= xmin) & (undp_df['start_date'] <= xmax)]
        if len(markers_df) > 0:
            monthly_markers = defaultdict(list)
            for _, row in markers_df.iterrows():
                month_key = row['start_date'].to_period('M').to_timestamp()
                cat = row.get('ml_category', 'Other / Unspecified')
                monthly_markers[month_key].append(cat)
            
            for month, cats in sorted(monthly_markers.items()):
                for i, cat in enumerate(cats[:3]):
                    color = CATEGORY_COLORS.get(cat, '#9E9E9E')
                    y_pos = 0.5 + (i - len(cats)/2 + 0.5) * 0.25
                    y_pos = max(0.15, min(0.85, y_pos))
                    ax_strip.scatter([month], [y_pos], color=color, s=80, marker='s', edgecolor='black', linewidth=0.5, zorder=5)
                    categories_shown.add(cat)
    
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    if categories_shown:
        patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, '#9E9E9E'), label=c[:25]) for c in sorted(categories_shown)]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(patches)), fontsize=7, frameon=False)
    
    plt.savefig(figures_dir / f'ts_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_choropleth_panel(panel, gdf, id_col, outcome, cfg, disaster_date, disaster_type, country_name, figures_dir):
    """Create PRE/POST choropleth maps from PANEL DATA."""
    
    pre = panel[panel['period_start'] < disaster_date].groupby('unit_id')[outcome].mean().reset_index()
    post = panel[panel['period_start'] >= disaster_date].groupby('unit_id')[outcome].mean().reset_index()
    
    gdf_pre = gdf.merge(pre, left_on=id_col, right_on='unit_id', how='left')
    gdf_post = gdf.merge(post, left_on=id_col, right_on='unit_id', how='left')
    
    all_vals = pd.concat([gdf_pre[[outcome]], gdf_post[[outcome]]]).dropna()
    if len(all_vals) == 0:
        return
    
    vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    gdf_pre.plot(column=outcome, ax=axes[0], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
    axes[0].set_title(f'Pre-{disaster_type}', fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')
    
    gdf_post.plot(column=outcome, ax=axes[1], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                 edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
    axes[1].set_title(f'Post-{disaster_type}', fontsize=12, fontweight='bold', color='red')
    axes[1].axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    
    fig.suptitle(f'{country_name}: {cfg["name"]}', fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig(figures_dir / f'map_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_google_trends(trends_df, disaster_date, disaster_type, country_name, figures_dir):
    """Create Google Trends plot."""
    
    # Find date column
    date_col = next((c for c in trends_df.columns if 'date' in c.lower()), trends_df.columns[0])
    trends_df['date'] = pd.to_datetime(trends_df[date_col], errors='coerce')
    
    # Find value columns (exclude date-like columns)
    value_cols = [c for c in trends_df.columns if c != date_col and c != 'date' and 'partial' not in c.lower()]
    
    if len(value_cols) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(value_cols), 10)))
    
    for col, color in zip(value_cols[:10], colors):
        data = trends_df[['date', col]].dropna().sort_values('date')
        if len(data) > 0:
            ax.plot(data['date'], data[col], label=col, color=color, linewidth=2)
    
    ax.axvline(x=disaster_date, color='red', linestyle='--', linewidth=2.5)
    ax.text(disaster_date, ax.get_ylim()[1] * 0.95, f'{disaster_type}',
            ha='center', va='top', fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax.set_title(f'{country_name}: Google Trends Search Interest', fontsize=14, fontweight='bold')
    ax.set_ylabel('Search Interest (0-100)', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'google_trends.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_undp_timeline_detailed(undp_df, disaster_date, disaster_type, country_name, figures_dir):
    """Create DETAILED UNDP project timeline showing each project with dates."""
    
    df = undp_df.sort_values('start_date')
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(18, max(8, len(df) * 0.6)))
    
    # Disaster line
    ax.axvline(x=disaster_date, color='red', linestyle='--', linewidth=3, zorder=5)
    ax.text(disaster_date, len(df) + 0.5, f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Plot each project as a bar
    for i, (_, row) in enumerate(df.iterrows()):
        start = row['start_date']
        cat = row.get('ml_category', 'Other / Unspecified')
        color = CATEGORY_COLORS.get(cat, '#9E9E9E')
        
        # Bar
        ax.barh(i, width=30, left=start, height=0.7, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Project info
        title = str(row.get('title', 'Unknown'))[:60]
        budget = row.get('total_budget', 0)
        
        if budget > 0:
            if budget >= 1e6:
                budget_str = f" [${budget/1e6:.1f}M]"
            elif budget >= 1e3:
                budget_str = f" [${budget/1e3:.0f}K]"
            else:
                budget_str = f" [${budget:.0f}]"
        else:
            budget_str = ""
        
        # Text
        ax.text(start + timedelta(days=35), i, f"{title}{budget_str}", va='center', fontsize=9)
        ax.text(start - timedelta(days=3), i, start.strftime('%Y-%m-%d'), va='center', ha='right', fontsize=8, alpha=0.7)
    
    # Shading
    xmin = df['start_date'].min() - timedelta(days=90)
    xmax = df['start_date'].max() + timedelta(days=150)
    
    if xmin < disaster_date:
        ax.axvspan(xmin, disaster_date, alpha=0.1, color='green')
    if xmax > disaster_date:
        ax.axvspan(disaster_date, xmax, alpha=0.08, color='red')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, len(df) + 1)
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title(f'{country_name}: UNDP Projects Timeline (Detailed)', fontsize=14, fontweight='bold', pad=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend
    cats = df['ml_category'].unique() if 'ml_category' in df.columns else []
    patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, '#9E9E9E'), label=c) for c in sorted(cats) if c in CATEGORY_COLORS]
    if patches:
        ax.legend(handles=patches, loc='upper left', fontsize=9, ncol=2, title='Project Category')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'undp_timeline_detailed.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_conflict_timeline(panel, disaster_date, disaster_type, country_name, figures_dir):
    """Create conflict timeline."""
    
    monthly = panel.groupby('period_start')['conflict_events'].sum().reset_index()
    monthly = monthly.sort_values('period_start')
    
    if monthly['conflict_events'].sum() == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for _, row in monthly.iterrows():
        date = row['period_start']
        val = row['conflict_events']
        color = 'forestgreen' if date < disaster_date else 'crimson'
        ax.bar(date, val, width=25, color=color, alpha=0.7, edgecolor='white')
    
    ax.axvline(x=disaster_date, color='black', linestyle='--', linewidth=2)
    ax.text(disaster_date, ax.get_ylim()[1]*0.95, f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
           ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'{country_name}: Conflict Events (ACLED)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Events', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'conflict_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary(panel, outcomes, disaster_date, disaster_type, figures_dir):
    """Create summary statistics."""
    
    rows = []
    for outcome in outcomes:
        cfg = OUTCOME_CONFIG.get(outcome, {'name': outcome})
        pre = panel[panel['period_start'] < disaster_date][outcome]
        post = panel[panel['period_start'] >= disaster_date][outcome]
        
        pre_mean = pre.mean()
        post_mean = post.mean()
        
        rows.append({
            'Outcome': cfg['name'],
            f'Pre-{disaster_type} Mean': f"{pre_mean:.3f}" if not pd.isna(pre_mean) else "N/A",
            f'Pre-{disaster_type} SD': f"{pre.std():.3f}" if not pd.isna(pre.std()) else "N/A",
            f'Post-{disaster_type} Mean': f"{post_mean:.3f}" if not pd.isna(post_mean) else "N/A",
            f'Post-{disaster_type} SD': f"{post.std():.3f}" if not pd.isna(post.std()) else "N/A",
            'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%" if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
        })
    
    pd.DataFrame(rows).to_csv(figures_dir / 'summary_statistics.csv', index=False)


if __name__ == "__main__":
    main()
