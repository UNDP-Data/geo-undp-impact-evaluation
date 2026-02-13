#!/usr/bin/env python3
"""
analysis/mmr/03_visualizations_panel.py
=======================================
Myanmar Earthquake Visualization Suite - Gaza Format

Creates TIME SERIES with UNDP PROJECT STRIP at bottom (matching Gaza style):
- Main panel: Continuous line plot with 95% CI ribbon
- Bottom strip: UNDP project markers by date and category
- Vertical lines: Earthquake date, UNDP intervention start
- Pre/Post period shading

Usage:
    python analysis/mmr/03_visualizations_panel.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "mmr_admin3.geojson"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
INTERVENTION_DATE = pd.Timestamp("2025-04-01")

# Affected states for filtering
AFFECTED_STATES = ['Sagaing', 'Mandalay', 'Naypyitaw', 'Bago', 'Magway']

# Outcome configurations
OUTCOMES = {
    'nightlights': {
        'name': 'Nighttime Lights (VIIRS)',
        'cmap': 'YlOrRd',
        'line_color': 'darkblue',
        'unit': 'nW/cm²/sr'
    },
    'ndvi': {
        'name': 'Vegetation Index (NDVI)',
        'cmap': 'RdYlGn',
        'line_color': 'forestgreen',
        'unit': 'Index'
    },
    'conflict_events': {
        'name': 'Conflict Events',
        'cmap': 'Reds',
        'line_color': 'crimson',
        'unit': 'Count'
    },
    'rainfall': {
        'name': 'Rainfall',
        'cmap': 'Blues',
        'line_color': 'steelblue',
        'unit': 'mm'
    },
    'price_rice': {
        'name': 'Rice Price',
        'cmap': 'YlGn',
        'line_color': 'olivedrab',
        'unit': 'MMK'
    },
    'vv': {
        'name': 'SAR Backscatter (VV)',
        'cmap': 'Greys',
        'line_color': 'dimgray',
        'unit': 'dB'
    },
    'vh': {
        'name': 'SAR Backscatter (VH)',
        'cmap': 'Greys',
        'line_color': 'slategray',
        'unit': 'dB'
    },
    'rvi': {
        'name': 'Radar Vegetation Index',
        'cmap': 'YlGn',
        'line_color': 'olive',
        'unit': 'Index'
    },
}

# Category colors for intervention types
CATEGORY_COLORS = {
    'Community leader': '#3498DB',
    'Community action board': '#795548',
    'Women community leader': '#E91E63',
    'Youth': '#4CAF50',
    'Monastic': '#9C27B0',
    'Other': '#9E9E9E'
}


def load_data():
    """Load panel data and boundaries."""
    # Panel
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel not found: {PANEL_FILE}")
        return None, None, None
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    # Boundaries
    gdf = None
    for path in [BOUNDARIES_FILE, 
                 DATA_DIR / "myanmar_boundaries" / "mmr_admin3.geojson",
                 DATA_DIR / "boundaries" / "mmr_admbnda_adm3_mimu_250201.shp"]:
        if path.exists():
            gdf = gpd.read_file(path).to_crs("EPSG:4326")
            break
    
    # UNDP sites
    undp_sites = None
    if UNDP_SITES_FILE.exists():
        undp_sites = pd.read_csv(UNDP_SITES_FILE)
    
    return panel, gdf, undp_sites


def create_time_series_with_undp(panel, undp_sites, outcome, cfg):
    """Create time series plot with UNDP intervention markers."""
    
    monthly = panel.groupby('period_start')[outcome].mean().reset_index()
    monthly = monthly.sort_values('period_start')
    
    if monthly[outcome].isna().all():
        return False
    
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # Main plot
    ax_main.fill_between(monthly['period_start'], monthly[outcome],
                        alpha=0.3, color=cfg['line_color'])
    ax_main.plot(monthly['period_start'], monthly[outcome],
                color=cfg['line_color'], linewidth=2.5, marker='o', markersize=5,
                label='Myanmar (National Average)')
    
    # Pre/post means
    pre_data = monthly[monthly['period_start'] < EARTHQUAKE_DATE][outcome]
    post_data = monthly[monthly['period_start'] >= EARTHQUAKE_DATE][outcome]
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Earthquake line
    ax_main.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=2.5)
    
    # Shade periods
    xmin = monthly['period_start'].min()
    xmax = monthly['period_start'].max()
    
    if xmin < EARTHQUAKE_DATE:
        ax_main.axvspan(xmin, min(EARTHQUAKE_DATE, xmax), alpha=0.1, color='green')
    if xmax > EARTHQUAKE_DATE:
        ax_main.axvspan(max(EARTHQUAKE_DATE, xmin), xmax, alpha=0.08, color='red')
    
    # Earthquake label
    y_top = ax_main.get_ylim()[1]
    ax_main.text(EARTHQUAKE_DATE, y_top * 0.95, 'Earthquake\nMar 28, 2025',
                ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # % change
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct_change:+.1f}% After Earthquake)"
    else:
        title_suffix = ""
    
    ax_main.set_title(f'Myanmar: {cfg["name"]}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Myanmar'),
    ]
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, 
                                     label=f'Pre-Earthquake Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, 
                                     label=f'Post-Earthquake Mean: {post_mean:.2f}'))
    
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # UNDP Project Strip
    ax_strip.set_xlim(ax_main.get_xlim())
    ax_strip.set_ylim(0, 1)
    
    if xmin < EARTHQUAKE_DATE:
        ax_strip.axvspan(xmin, min(EARTHQUAKE_DATE, xmax), alpha=0.15, color='green')
    if xmax > EARTHQUAKE_DATE:
        ax_strip.axvspan(max(EARTHQUAKE_DATE, xmin), xmax, alpha=0.1, color='red')
    
    ax_strip.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot UNDP site markers (simplified - just show intervention period)
    if undp_sites is not None and len(undp_sites) > 0:
        # Show intervention start
        ax_strip.scatter([INTERVENTION_DATE], [0.5], color='blue', s=100,
                        marker='s', edgecolor='black', linewidth=0.5, zorder=5)
        ax_strip.text(INTERVENTION_DATE, 0.8, f'{len(undp_sites)} UNDP Sites',
                     ha='center', fontsize=8, color='blue')
    
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    plt.savefig(FIGURES_DIR / f'ts_{outcome}_with_undp.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_choropleth_maps(panel, gdf, outcome, cfg):
    """Create pre/post earthquake choropleth maps."""
    
    id_col = None
    for col in ['adm3_pcode', 'TS_PCODE', 'admin_id', 'ADM3_PCODE']:
        if col in gdf.columns:
            id_col = col
            break
    
    if id_col is None:
        print(f"    ⚠ Could not find ID column")
        return False
    
    # Aggregate by admin unit and period
    pre_data = panel[panel['period_start'] < EARTHQUAKE_DATE].groupby('unit_id')[outcome].mean().reset_index()
    post_data = panel[panel['period_start'] >= EARTHQUAKE_DATE].groupby('unit_id')[outcome].mean().reset_index()
    
    # Merge with geometry
    gdf_pre = gdf.merge(pre_data, left_on=id_col, right_on='unit_id', how='left')
    gdf_post = gdf.merge(post_data, left_on=id_col, right_on='unit_id', how='left')
    
    if gdf_pre[outcome].isna().all() and gdf_post[outcome].isna().all():
        return False
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Common color scale
    all_vals = pd.concat([gdf_pre[[outcome]], gdf_post[[outcome]]]).dropna()
    if len(all_vals) == 0:
        plt.close()
        return False
    
    vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
    
    # Pre-earthquake
    gdf_pre.plot(column=outcome, ax=axes[0], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
    axes[0].set_title('Pre-Earthquake\n(Oct 2023 - Mar 2025)', fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Post-earthquake
    gdf_post.plot(column=outcome, ax=axes[1], cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                 edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
    axes[1].set_title('Post-Earthquake\n(Apr 2025 - Present)', fontsize=12, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    
    fig.suptitle(f'Myanmar: {cfg["name"]} - Before vs After Earthquake',
                fontsize=14, fontweight='bold', y=0.95)
    
    plt.savefig(FIGURES_DIR / f'map_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_conflict_timeline(panel):
    """Create conflict events timeline."""
    
    if 'conflict_events' not in panel.columns:
        return False
    
    monthly = panel.groupby('period_start')['conflict_events'].sum().reset_index()
    monthly = monthly.sort_values('period_start')
    
    if monthly['conflict_events'].sum() == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for _, row in monthly.iterrows():
        date = row['period_start']
        val = row['conflict_events']
        
        if date < EARTHQUAKE_DATE:
            color = 'forestgreen'
        else:
            color = 'crimson'
        
        ax.bar(date, val, width=25, color=color, alpha=0.7, edgecolor='white')
    
    ax.axvline(x=EARTHQUAKE_DATE, color='black', linestyle='--', linewidth=2)
    
    ax.text(EARTHQUAKE_DATE, ax.get_ylim()[1]*0.95, 
           EARTHQUAKE_DATE.strftime('%b %d, %Y'),
           ha='center', fontsize=10, fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='forestgreen', alpha=0.7, label='Pre-Earthquake'),
        mpatches.Patch(facecolor='crimson', alpha=0.7, label='Post-Earthquake'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('Myanmar: Conflict Events (ACLED)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Events', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'conflict_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_summary_statistics(panel):
    """Create summary statistics table."""
    
    available = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
    
    summary_rows = []
    for outcome in available:
        cfg = OUTCOMES[outcome]
        
        pre = panel[panel['period_start'] < EARTHQUAKE_DATE][outcome]
        post = panel[panel['period_start'] >= EARTHQUAKE_DATE][outcome]
        
        pre_mean = pre.mean()
        post_mean = post.mean()
        
        summary_rows.append({
            'Outcome': cfg['name'],
            'Pre-Earthquake Mean': f"{pre_mean:.2f}" if not pd.isna(pre_mean) else "N/A",
            'Pre-Earthquake SD': f"{pre.std():.2f}" if not pd.isna(pre.std()) else "N/A",
            'Post-Earthquake Mean': f"{post_mean:.2f}" if not pd.isna(post_mean) else "N/A",
            'Post-Earthquake SD': f"{post.std():.2f}" if not pd.isna(post.std()) else "N/A",
            'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%" 
                         if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(FIGURES_DIR / 'summary_statistics.csv', index=False)
    
    return summary_df


def create_undp_sites_map(gdf, undp_sites):
    """Create map showing UNDP intervention sites."""
    
    if undp_sites is None or len(undp_sites) == 0:
        return False
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Plot boundaries
    gdf.plot(ax=ax, facecolor='#f0f0f0', edgecolor='#999999', linewidth=0.3)
    
    # Plot sites
    sites_gdf = gpd.GeoDataFrame(
        undp_sites,
        geometry=gpd.points_from_xy(undp_sites.longitude, undp_sites.latitude),
        crs="EPSG:4326"
    )
    
    sites_gdf.plot(ax=ax, markersize=30, color='#e74c3c', alpha=0.7,
                  edgecolor='black', linewidth=0.5)
    
    # Epicenter
    ax.scatter([95.925], [22.001], marker='*', s=500, c='yellow',
              edgecolor='black', linewidth=2, zorder=10, label='Epicenter')
    
    ax.set_title(f'Myanmar: UNDP Intervention Sites\nn = {len(undp_sites)} sites',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Focus on affected area
    ax.set_xlim([94, 98.5])
    ax.set_ylim([19, 24.5])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'undp_sites_map.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: VISUALIZATION SUITE")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading Data ---")
    panel, gdf, undp_sites = load_data()
    
    if panel is None:
        print("  ⚠ No panel data. Run preprocessing first.")
        return
    
    print(f"  Panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")
    print(f"  Period: {panel['period_start'].min()} to {panel['period_start'].max()}")
    
    if gdf is not None:
        print(f"  Boundaries: {len(gdf)} admin units")
    
    if undp_sites is not None:
        print(f"  UNDP sites: {len(undp_sites)}")
    
    # Available outcomes
    available = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"  Outcomes: {', '.join(available)}")
    
    # Create visualizations
    print("\n--- Time Series ---")
    for outcome in available:
        cfg = OUTCOMES[outcome]
        success = create_time_series_with_undp(panel, undp_sites, outcome, cfg)
        status = "✓" if success else "✗"
        print(f"  {status} ts_{outcome}_with_undp.png")
    
    if gdf is not None:
        print("\n--- Choropleth Maps ---")
        for outcome in available:
            cfg = OUTCOMES[outcome]
            success = create_choropleth_maps(panel, gdf, outcome, cfg)
            status = "✓" if success else "✗"
            print(f"  {status} map_{outcome}.png")
    
    print("\n--- Conflict Timeline ---")
    success = create_conflict_timeline(panel)
    status = "✓" if success else "✗"
    print(f"  {status} conflict_timeline.png")
    
    print("\n--- Summary Statistics ---")
    summary_df = create_summary_statistics(panel)
    print("  ✓ summary_statistics.csv")
    print(summary_df.to_string(index=False))
    
    if gdf is not None and undp_sites is not None:
        print("\n--- UNDP Sites Map ---")
        success = create_undp_sites_map(gdf, undp_sites)
        status = "✓" if success else "✗"
        print(f"  {status} undp_sites_map.png")
    
    # Count outputs
    n_figures = len(list(FIGURES_DIR.glob("*.png")))
    n_csv = len(list(FIGURES_DIR.glob("*.csv")))
    
    print(f"\n{'='*70}")
    print(f"✓ VISUALIZATIONS COMPLETE")
    print(f"  Output: {FIGURES_DIR}")
    print(f"  Figures: {n_figures}")
    print(f"  CSVs: {n_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
