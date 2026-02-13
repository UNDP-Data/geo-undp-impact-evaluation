#!/usr/bin/env python3
"""
analysis/mmr/03_timeseries_panel.py
===================================
Myanmar Time Series using PANEL DATA (not separate rasters).

Uses the already-downloaded satellite panel data from GEE and creates
Gaza-style time series plots with UNDP project strip at bottom.

The UNDP categories come from undp_projects.csv (ML-classified) NOT from
undp_sites.csv (which only has raw site types like "Community Leader").

Usage:
    python analysis/mmr/03_timeseries_panel.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/mmr")
OUTPUT_DIR = Path("outputs/mmr")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Panel data (already has all satellite data!)
PANEL_FILE = OUTPUT_DIR / "panel_data.csv"

# UNDP files
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"  # Has ml_category
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"  # Has coordinates

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
INTERVENTION_START = pd.Timestamp("2025-04-01")

# ML Category colors (from UNDP projects classification)
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

# Dataset configurations (matching column names in panel_data.csv)
DATASETS = {
    'nightlights': {
        'name': 'Nighttime Lights (VIIRS)',
        'line_color': 'darkblue',
        'fill_color': 'blue',
        'unit': 'nW/cm²/sr'
    },
    'ndvi': {
        'name': 'Vegetation Index (NDVI)',
        'line_color': 'forestgreen',
        'fill_color': 'green',
        'unit': 'Index'
    },
    'vh': {
        'name': 'SAR VH Backscatter',
        'line_color': 'indigo',
        'fill_color': 'slateblue',
        'unit': 'dB'
    },
    'vv': {
        'name': 'SAR VV Backscatter',
        'line_color': 'purple',
        'fill_color': 'mediumpurple',
        'unit': 'dB'
    },
    'rvi': {
        'name': 'Radar Vegetation Index',
        'line_color': 'darkgreen',
        'fill_color': 'palegreen',
        'unit': 'Index'
    },
    'conflict_events': {
        'name': 'Conflict Events (ACLED)',
        'line_color': 'crimson',
        'fill_color': 'lightcoral',
        'unit': 'Count'
    },
    'rainfall': {
        'name': 'Rainfall (CHIRPS)',
        'line_color': 'steelblue',
        'fill_color': 'lightblue',
        'unit': 'mm'
    },
    'price_rice': {
        'name': 'Rice Price',
        'line_color': 'olive',
        'fill_color': 'khaki',
        'unit': 'MMK/kg'
    },
}


def load_panel_data():
    """Load panel data with satellite values."""
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel data not found: {PANEL_FILE}")
        return None
    
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    
    # Also try 'date' column if period_start doesn't exist
    if 'date' in df.columns and 'period_start' not in df.columns:
        df['period_start'] = pd.to_datetime(df['date'])
    
    return df


def load_undp_projects():
    """
    Load UNDP PROJECTS (not sites) for ML categories and dates.
    This is the file with ml_category from web scraping.
    """
    if not UNDP_PROJECTS_FILE.exists():
        print(f"  ⚠ UNDP projects not found: {UNDP_PROJECTS_FILE}")
        print(f"    Run: python -m pipeline.undp_projects --config config/myanmar.yaml")
        return None
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter out umbrella projects
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    print(f"  Loaded {len(df)} UNDP projects with ML categories")
    
    # Show category distribution
    if 'ml_category' in df.columns:
        print(f"  Categories: {df['ml_category'].value_counts().to_dict()}")
    
    return df


def get_undp_markers(undp_df, min_date, max_date):
    """
    Get UNDP project markers using ML categories.
    Returns list of (date, ml_category, title) tuples.
    """
    if undp_df is None:
        return []
    
    df = undp_df[
        (undp_df['start_date'] >= min_date) & 
        (undp_df['start_date'] <= max_date)
    ].copy()
    
    if df.empty:
        return []
    
    markers = []
    for _, row in df.iterrows():
        category = row.get('ml_category', 'Other / Unspecified')
        title = str(row.get('title', ''))[:30]
        markers.append((row['start_date'], category, title))
    
    return sorted(markers, key=lambda x: x[0])


def create_timeseries_plot(panel, dataset, cfg, undp_df, output_path):
    """
    Create Gaza-style time series plot using panel data.
    
    - Main panel: Line plot with fill ribbon
    - Bottom strip: UNDP projects by date and ML category
    """
    
    # Check if column exists
    if dataset not in panel.columns:
        return False
    
    # Aggregate monthly - use mean for most, sum for counts
    if 'conflict' in dataset or 'events' in dataset:
        monthly = panel.groupby('period_start')[dataset].sum().reset_index()
    else:
        monthly = panel.groupby('period_start').agg({
            dataset: ['mean', 'std', 'count']
        }).reset_index()
        monthly.columns = ['period_start', 'mean', 'std', 'count']
        monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
        monthly = monthly.rename(columns={'mean': dataset})
    
    monthly = monthly.sort_values('period_start')
    
    if monthly[dataset].isna().all():
        return False
    
    # Create figure with main panel + UNDP strip
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # === MAIN TIME SERIES ===
    
    # Fill ribbon
    if 'se' in monthly.columns:
        ax_main.fill_between(
            monthly['period_start'],
            monthly[dataset] - 1.96 * monthly['se'],
            monthly[dataset] + 1.96 * monthly['se'],
            alpha=0.3, color=cfg['fill_color']
        )
    else:
        ax_main.fill_between(monthly['period_start'], monthly[dataset],
                            alpha=0.3, color=cfg['fill_color'])
    
    # Main line with markers
    ax_main.plot(monthly['period_start'], monthly[dataset],
                color=cfg['line_color'], linewidth=2.5, marker='o', markersize=5,
                label='Myanmar')
    
    # Pre/post means
    pre_data = monthly[monthly['period_start'] < EARTHQUAKE_DATE][dataset]
    post_data = monthly[monthly['period_start'] >= EARTHQUAKE_DATE][dataset]
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    # Dotted horizontal lines for means
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Vertical dashed line at earthquake
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
    
    # Calculate % change
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct_change:+.1f}% After Earthquake)"
    else:
        title_suffix = ""
    
    ax_main.set_title(f'Myanmar: {cfg["name"]}{title_suffix}',
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])  # Labels on strip
    ax_main.grid(True, alpha=0.3)
    
    # Legend for main plot
    legend_elements = [
        Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Myanmar'),
    ]
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':',
                                     linewidth=1.5, label=f'Pre-Earthquake Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':',
                                     linewidth=1.5, label=f'Post-Earthquake Mean: {post_mean:.2f}'))
    
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # === UNDP PROJECT STRIP ===
    ax_strip.set_xlim(ax_main.get_xlim())
    ax_strip.set_ylim(0, 1)
    
    # Period shading on strip
    if xmin < EARTHQUAKE_DATE:
        ax_strip.axvspan(xmin, min(EARTHQUAKE_DATE, xmax), alpha=0.15, color='green')
    if xmax > EARTHQUAKE_DATE:
        ax_strip.axvspan(max(EARTHQUAKE_DATE, xmin), xmax, alpha=0.1, color='red')
    
    ax_strip.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot UNDP project markers using ML CATEGORIES
    categories_shown = set()
    
    if undp_df is not None:
        markers = get_undp_markers(undp_df, xmin, xmax)
        
        if markers:
            # Group by month
            monthly_markers = defaultdict(list)
            for date, category, title in markers:
                month_key = date.to_period('M').to_timestamp()
                monthly_markers[month_key].append((category, title))
            
            for month, projects in sorted(monthly_markers.items()):
                for i, (category, title) in enumerate(projects[:3]):  # Max 3 per month
                    color = CATEGORY_COLORS.get(category, '#9E9E9E')
                    y_pos = 0.5 + (i - len(projects)/2 + 0.5) * 0.25
                    y_pos = max(0.15, min(0.85, y_pos))
                    
                    ax_strip.scatter([month], [y_pos], color=color, s=80,
                                    marker='s', edgecolor='black', linewidth=0.5, zorder=5)
                    categories_shown.add(category)
    
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    # Category legend at bottom (ML categories, not site types!)
    if categories_shown:
        legend_patches = [
            mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:25])
            for cat in sorted(categories_shown)
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)),
                   fontsize=8, frameon=False)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("=" * 70)
    print(" MYANMAR: TIME SERIES FROM PANEL DATA (Gaza Format)")
    print("=" * 70)
    
    # Load panel data
    print("\n--- Loading Panel Data ---")
    panel = load_panel_data()
    
    if panel is None:
        print("  ⚠ Run preprocessing first:")
        print("    python -m pipeline.preprocessing --config config/myanmar.yaml")
        return
    
    print(f"  Panel: {len(panel):,} rows, {panel['unit_id'].nunique()} units")
    print(f"  Period: {panel['period_start'].min().date()} to {panel['period_start'].max().date()}")
    print(f"  Columns: {list(panel.columns)}")
    
    # Load UNDP PROJECTS (with ML categories)
    print("\n--- Loading UNDP Projects ---")
    undp_df = load_undp_projects()
    
    # Find available datasets
    available = [d for d in DATASETS.keys() if d in panel.columns and panel[d].notna().any()]
    print(f"\n  Available datasets: {', '.join(available)}")
    
    # Create time series plots
    print("\n--- Creating Time Series Plots ---")
    
    for dataset in available:
        cfg = DATASETS[dataset]
        output_path = FIGURES_DIR / f'ts_{dataset}_panel.png'
        
        success = create_timeseries_plot(panel, dataset, cfg, undp_df, output_path)
        
        status = "✓" if success else "✗"
        print(f"  {status} ts_{dataset}_panel.png")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✓ TIME SERIES COMPLETE")
    print(f"  Output: {FIGURES_DIR}")
    print(f"  Note: Using panel data (already downloaded), not separate rasters")
    print(f"  UNDP categories from ml_category (ML-classified projects)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
