#!/usr/bin/env python3
"""
analysis/mmr/03_timeseries_gaza_format.py
==========================================
Myanmar Earthquake Time Series - EXACT Gaza Format

Creates time series plots with:
- Main panel: Continuous line with fill ribbon (not bars!)
- Dotted horizontal lines for pre/post means
- Vertical dashed lines at earthquake + UNDP start
- Bottom strip showing UNDP projects by date and category
- Pre/post period background shading
- % change annotation in title

This EXACTLY replicates the Gaza visualization format.

Usage:
    python analysis/mmr/03_timeseries_gaza_format.py
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

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"
UNDP_SITES_FILE = DATA_DIR / "external" / "undp_sites.csv"

# Key dates
EARTHQUAKE_DATE = pd.Timestamp("2025-03-28")
INTERVENTION_START = pd.Timestamp("2025-04-01")

# Category colors for UNDP projects (matching Gaza)
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

# Outcome configurations
OUTCOMES = {
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
    'rvi': {
        'name': 'Radar Vegetation Index',
        'line_color': 'darkgreen',
        'fill_color': 'palegreen',
        'unit': 'Index'
    },
}


def load_panel_data():
    """Load panel data."""
    if not PANEL_FILE.exists():
        print(f"  ⚠ Panel not found: {PANEL_FILE}")
        return None
    
    df = pd.read_csv(PANEL_FILE)
    df['period_start'] = pd.to_datetime(df['period_start'])
    return df


def load_undp_projects():
    """Load UNDP projects with dates and ML categories."""
    if not UNDP_PROJECTS_FILE.exists():
        print(f"  ⚠ UNDP projects not found: {UNDP_PROJECTS_FILE}")
        print(f"    Run: python -m pipeline.undp_projects --config config/myanmar.yaml")
        return None
    
    df = pd.read_csv(UNDP_PROJECTS_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Filter out umbrella projects if column exists
    if 'is_likely_umbrella' in df.columns:
        df = df[df['is_likely_umbrella'] != True]
    
    return df


def get_undp_markers(undp_df, min_date, max_date):
    """
    Get UNDP project markers for plotting.
    Returns list of (date, category, short_title) tuples.
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


def create_timeseries_with_undp_strip(panel, undp_df, outcome, cfg, output_path):
    """
    Create time series plot EXACTLY matching Gaza format:
    - Main panel with line + fill ribbon
    - Pre/post horizontal dotted lines
    - UNDP project strip at bottom
    - Category legend
    """
    
    # Aggregate monthly
    monthly = panel.groupby('period_start').agg({
        outcome: ['mean', 'std', 'count']
    }).reset_index()
    monthly.columns = ['period_start', 'mean', 'std', 'count']
    monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
    monthly = monthly.sort_values('period_start')
    
    if monthly['mean'].isna().all():
        return False
    
    # Create figure with main panel + UNDP strip
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # === MAIN TIME SERIES ===
    
    # Fill ribbon (95% CI)
    ax_main.fill_between(
        monthly['period_start'],
        monthly['mean'] - 1.96 * monthly['se'],
        monthly['mean'] + 1.96 * monthly['se'],
        alpha=0.3, color=cfg['fill_color']
    )
    
    # Main line with markers
    ax_main.plot(
        monthly['period_start'], monthly['mean'],
        color=cfg['line_color'], linewidth=2.5, marker='o', markersize=5,
        label='Myanmar (National Average)'
    )
    
    # Pre/Post means
    pre_data = monthly[monthly['period_start'] < EARTHQUAKE_DATE]['mean']
    post_data = monthly[monthly['period_start'] >= EARTHQUAKE_DATE]['mean']
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    # Dotted horizontal lines for means
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Vertical dashed lines for key dates
    ax_main.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=2.5)
    ax_main.axvline(x=INTERVENTION_START, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    # Period shading
    xmin = monthly['period_start'].min()
    xmax = monthly['period_start'].max()
    
    if xmin < EARTHQUAKE_DATE:
        ax_main.axvspan(xmin, min(EARTHQUAKE_DATE, xmax), alpha=0.1, color='green')
    if xmax > EARTHQUAKE_DATE:
        ax_main.axvspan(max(EARTHQUAKE_DATE, xmin), xmax, alpha=0.08, color='red')
    
    # Event labels
    y_top = ax_main.get_ylim()[1]
    ax_main.text(EARTHQUAKE_DATE, y_top * 0.95, 'Earthquake\nMar 28, 2025',
                ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax_main.text(INTERVENTION_START, y_top * 0.82, 'UNDP\nResponse',
                ha='center', va='top', fontsize=8, fontweight='bold', color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
    
    # Calculate % change for title
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
    ax_main.set_xticklabels([])  # Labels on strip instead
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
    
    # Vertical lines on strip
    ax_strip.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_strip.axvline(x=INTERVENTION_START, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot UNDP project markers
    categories_shown = set()
    
    if undp_df is not None:
        markers = get_undp_markers(undp_df, xmin, xmax)
        
        if markers:
            # Group by month to avoid overlapping
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
    ax_strip.set_ylabel('UNDP\nProjects', fontsize=9, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    # Category legend at bottom
    if categories_shown:
        legend_patches = [
            mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:25])
            for cat in sorted(categories_shown)
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)),
                   fontsize=7, frameon=False)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def create_undp_project_timeline(undp_df, output_path):
    """Create detailed UNDP project timeline visualization."""
    
    if undp_df is None or len(undp_df) == 0:
        return False
    
    # Filter to post-earthquake
    df = undp_df[undp_df['start_date'] >= datetime(2024, 1, 1)].copy()
    df = df.sort_values('start_date')
    
    if df.empty:
        return False
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.3)))
    
    # Plot earthquake line
    ax.axvline(x=EARTHQUAKE_DATE, color='red', linestyle='--', linewidth=2.5, zorder=5)
    ax.text(EARTHQUAKE_DATE, 1.02, 'Earthquake\nMar 28, 2025',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    
    # Plot each project as horizontal bar
    for i, (_, row) in enumerate(df.iterrows()):
        start = row['start_date']
        category = row.get('ml_category', 'Other / Unspecified')
        color = CATEGORY_COLORS.get(category, '#9E9E9E')
        
        # Bar for project
        ax.barh(i, width=30, left=start, height=0.6, color=color,
               alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Project title (truncated)
        title = str(row.get('title', ''))[:50]
        if len(str(row.get('title', ''))) > 50:
            title += '...'
        ax.text(start + timedelta(days=35), i, title, va='center', fontsize=8, alpha=0.9)
    
    # Formatting
    ax.set_xlim(datetime(2024, 1, 1), datetime(2026, 6, 1))
    ax.set_ylim(-1, len(df) + 1)
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Myanmar: UNDP Projects Timeline\n(Post-2024 Projects)',
                fontsize=14, fontweight='bold', pad=25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend
    categories_in_data = df['ml_category'].unique()
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat)
        for cat in sorted(categories_in_data)
        if cat in CATEGORY_COLORS
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    
    # Shade regions
    ax.axvspan(datetime(2024, 1, 1), EARTHQUAKE_DATE, alpha=0.1, color='green')
    ax.axvspan(EARTHQUAKE_DATE, datetime(2026, 6, 1), alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


def main():
    print("=" * 70)
    print(" MYANMAR EARTHQUAKE: TIME SERIES (GAZA FORMAT)")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading Data ---")
    
    panel = load_panel_data()
    if panel is None:
        print("  ⚠ No panel data. Run preprocessing first:")
        print("    python -m pipeline.preprocessing --config config/myanmar.yaml")
        return
    
    print(f"  Panel: {len(panel):,} rows, {panel['unit_id'].nunique()} units")
    print(f"  Period: {panel['period_start'].min().date()} to {panel['period_start'].max().date()}")
    
    undp_df = load_undp_projects()
    if undp_df is not None:
        print(f"  UNDP projects: {len(undp_df)}")
        if 'ml_category' in undp_df.columns:
            print(f"    Categories: {undp_df['ml_category'].nunique()}")
    else:
        print("  ⚠ No UNDP projects. Run:")
        print("    python -m pipeline.undp_projects --config config/myanmar.yaml")
    
    # Available outcomes
    available = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"\n  Available outcomes: {', '.join(available)}")
    
    # Create time series with UNDP strip (Gaza format)
    print("\n--- Creating Time Series (Gaza Format) ---")
    
    for outcome in available:
        cfg = OUTCOMES[outcome]
        output_path = FIGURES_DIR / f'ts_{outcome}_gaza_format.png'
        
        success = create_timeseries_with_undp_strip(panel, undp_df, outcome, cfg, output_path)
        
        status = "✓" if success else "✗"
        print(f"  {status} ts_{outcome}_gaza_format.png")
    
    # Create UNDP project timeline
    if undp_df is not None:
        print("\n--- Creating UNDP Timeline ---")
        success = create_undp_project_timeline(undp_df, FIGURES_DIR / 'undp_project_timeline.png')
        status = "✓" if success else "✗"
        print(f"  {status} undp_project_timeline.png")
    
    # Summary
    n_figures = len(list(FIGURES_DIR.glob("*gaza_format*.png"))) + \
                len(list(FIGURES_DIR.glob("*timeline*.png")))
    
    print(f"\n{'='*70}")
    print(f"✓ VISUALIZATIONS COMPLETE (Gaza Format)")
    print(f"  Output: {FIGURES_DIR}")
    print(f"  Figures: {n_figures}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
