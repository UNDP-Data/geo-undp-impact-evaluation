#!/usr/bin/env python3
"""
analysis/pse/gaza_viz_v3.py
===========================
Gaza Strip Visualization Suite - Version 3

Updates from v2:
- UNDP project timeline annotations (from scraped data)
- Study design map (treatment vs control)
- Added NDBI, BSI, SAR indicators
- Fixed data availability messages

Outputs:
- Time series with UNDP project markers
- Choropleth maps
- Treatment/Control study design map
- Gaza vs West Bank comparison charts
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
DATA_DIR = Path("data/pse")
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_FILE = OUTPUT_DIR / "panel_data.csv"
BOUNDARIES_FILE = DATA_DIR / "boundaries" / "gaza_admin2.geojson"

# UNDP Projects file
UNDP_PROJECTS_FILE = DATA_DIR / "external" / "undp_projects.csv"
UNDP_PROJECTS_ALT = DATA_DIR / "pal_undp_projects_ml_classified.csv"

# Key dates
CONFLICT_DATE = pd.Timestamp("2023-10-07")

# Outcome configurations - EXPANDED
OUTCOMES = {
    # Nightlights
    'nightlights': {
        'name': 'Nighttime Lights (Radiance)',
        'cmap': 'YlOrRd',
        'line_color': 'darkblue',
        'unit': 'nW/cm²/sr'
    },
    # Vegetation
    'ndvi': {
        'name': 'Vegetation Index (NDVI)',
        'cmap': 'RdYlGn',
        'line_color': 'forestgreen',
        'unit': 'Index'
    },
    # Built-up / Construction
    'ndbi': {
        'name': 'Built-up Index (NDBI)',
        'cmap': 'RdYlBu_r',
        'line_color': 'brown',
        'unit': 'Index'
    },
    # Bare Soil Index
    'bsi': {
        'name': 'Bare Soil Index (BSI)',
        'cmap': 'YlOrBr',
        'line_color': 'sienna',
        'unit': 'Index'
    },
    # Conflict
    'conflict_events': {
        'name': 'Conflict Events',
        'cmap': 'Reds',
        'line_color': 'crimson',
        'unit': 'Count'
    },
    # Rainfall
    'rainfall': {
        'name': 'Rainfall',
        'cmap': 'Blues',
        'line_color': 'steelblue',
        'unit': 'mm'
    },
    # Food prices
    'price_bread': {
        'name': 'Bread Price',
        'cmap': 'Oranges',
        'line_color': 'darkorange',
        'unit': 'ILS'
    },
    'price_wheat': {
        'name': 'Wheat Price',
        'cmap': 'YlOrBr',
        'line_color': 'goldenrod',
        'unit': 'ILS'
    },
    'price_rice': {
        'name': 'Rice Price',
        'cmap': 'YlGn',
        'line_color': 'olivedrab',
        'unit': 'ILS'
    },
    'price_flour': {
        'name': 'Flour Price',
        'cmap': 'Oranges',
        'line_color': 'peru',
        'unit': 'ILS'
    },
    'price_sugar': {
        'name': 'Sugar Price',
        'cmap': 'Purples',
        'line_color': 'mediumpurple',
        'unit': 'ILS'
    },
    'price_oil': {
        'name': 'Oil Price',
        'cmap': 'YlOrRd',
        'line_color': 'olive',
        'unit': 'ILS'
    },
    'price_milk': {
        'name': 'Milk Price',
        'cmap': 'Blues',
        'line_color': 'steelblue',
        'unit': 'ILS'
    },
    'price_chicken': {
        'name': 'Chicken Price',
        'cmap': 'Reds',
        'line_color': 'firebrick',
        'unit': 'ILS'
    },
    # Population
    'population': {
        'name': 'Population',
        'cmap': 'YlOrRd',
        'line_color': 'darkred',
        'unit': 'persons'
    },
    # SAR indicators
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
    # Google Trends
    'gt_water': {
        'name': 'Google Trends: Water (ماء)',
        'cmap': 'Blues',
        'line_color': 'royalblue',
        'unit': 'Interest'
    },
    'gt_food': {
        'name': 'Google Trends: Food (طعام)',
        'cmap': 'Oranges',
        'line_color': 'coral',
        'unit': 'Interest'
    },
    'gt_passport': {
        'name': 'Google Trends: Passport (جواز سفر)',
        'cmap': 'Purples',
        'line_color': 'purple',
        'unit': 'Interest'
    },
    'gt_hospital': {
        'name': 'Google Trends: Hospital (مستشفى)',
        'cmap': 'Reds',
        'line_color': 'darkred',
        'unit': 'Interest'
    },
    'gt_help': {
        'name': 'Google Trends: Help (مساعدة)',
        'cmap': 'Greens',
        'line_color': 'seagreen',
        'unit': 'Interest'
    },
}

# =============================================================================
# LOAD UNDP PROJECTS DATA
# =============================================================================

def load_undp_projects():
    """Load UNDP projects data for timeline annotations."""
    for path in [UNDP_PROJECTS_FILE, UNDP_PROJECTS_ALT]:
        if path.exists():
            df = pd.read_csv(path)
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            # Filter to non-umbrella projects
            if 'is_likely_umbrella' in df.columns:
                df = df[df['is_likely_umbrella'] != True]
            return df
    return None


def get_undp_markers(undp_df, min_date, max_date):
    """
    Get UNDP project markers for a date range.
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
        short_title = str(row['title'])[:30] + '...' if len(str(row['title'])) > 30 else str(row['title'])
        markers.append((
            row['start_date'],
            row['ml_category'],
            short_title
        ))
    
    return sorted(markers, key=lambda x: x[0])


# =============================================================================
# STUDY DESIGN MAP
# =============================================================================

def create_study_design_map(gdf, id_col, name_col):
    """Create map showing treatment (Gaza) vs control (West Bank)."""
    
    print("\n--- Study Design Map ---")
    
    # Identify Gaza vs West Bank
    gdf = gdf.copy()
    gdf['is_gaza'] = gdf[id_col].str.startswith('PS02')
    gdf['group'] = gdf['is_gaza'].map({True: 'Treatment (Gaza)', False: 'Control (West Bank)'})
    
    # Colors
    colors = {
        'Treatment (Gaza)': '#E74C3C',      # Red
        'Control (West Bank)': '#3498DB'    # Blue
    }
    
    # Create figure with timeline
    fig = plt.figure(figsize=(14, 12))
    
    # Main map
    ax_map = fig.add_axes([0.05, 0.25, 0.9, 0.68])
    
    # Plot regions
    for group, color in colors.items():
        subset = gdf[gdf['group'] == group]
        subset.plot(ax=ax_map, color=color, edgecolor='black', linewidth=0.8, alpha=0.7)
    
    # Add district labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[name_col]
        fontsize = 8 if row['is_gaza'] else 9
        fontweight = 'bold' if row['is_gaza'] else 'normal'
        ax_map.annotate(
            name, 
            xy=(centroid.x, centroid.y),
            ha='center', va='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color='white' if row['is_gaza'] else 'black'
        )
    
    # Legend
    treatment_patch = mpatches.Patch(color=colors['Treatment (Gaza)'], label='Treatment: Gaza Strip (n=5)', alpha=0.7)
    control_patch = mpatches.Patch(color=colors['Control (West Bank)'], label='Control: West Bank (n=11)', alpha=0.7)
    ax_map.legend(handles=[treatment_patch, control_patch], loc='lower left', fontsize=11, framealpha=0.95)
    
    ax_map.set_title('Palestine: Difference-in-Differences Study Design\n(Treatment vs Control Regions)', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_map.set_axis_off()
    
    # Info box
    info_text = """Study Design:
• Treatment: Gaza Strip (5 districts)
• Control: West Bank (11 districts)
• Event: Oct 7, 2023 conflict onset
• Method: Two-Way Fixed Effects (TWFE)
• Clustered SEs at district level"""
    
    ax_map.text(0.02, 0.98, info_text, transform=ax_map.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    
    # Timeline at bottom
    ax_timeline = fig.add_axes([0.1, 0.06, 0.8, 0.14])
    
    # Timeline setup
    dates = [datetime(2023, 1, 1), datetime(2023, 10, 7), datetime(2024, 10, 1), datetime(2026, 1, 1)]
    ax_timeline.set_xlim(dates[0], dates[-1])
    ax_timeline.set_ylim(0, 1)
    
    # Draw timeline
    ax_timeline.axhline(y=0.5, color='black', linewidth=2)
    
    # Conflict onset
    ax_timeline.axvline(x=CONFLICT_DATE, color='red', linewidth=3, linestyle='--')
    ax_timeline.scatter([CONFLICT_DATE], [0.5], color='red', s=150, zorder=5)
    ax_timeline.text(CONFLICT_DATE, 0.75, 'Conflict Onset\nOct 7, 2023', ha='center', fontsize=10, 
                     fontweight='bold', color='red')
    
    # UNDP projects (if available)
    undp_df = load_undp_projects()
    if undp_df is not None:
        # Get post-conflict projects
        post_conflict = undp_df[undp_df['start_date'] > CONFLICT_DATE].head(5)
        for i, (_, row) in enumerate(post_conflict.iterrows()):
            x_pos = row['start_date']
            ax_timeline.axvline(x=x_pos, color='green', linewidth=1.5, linestyle=':', alpha=0.7)
            ax_timeline.scatter([x_pos], [0.5], color='green', s=80, zorder=5, alpha=0.8)
            
            # Stagger labels
            y_offset = 0.2 + (i % 2) * 0.15
            ax_timeline.annotate(
                f"{row['ml_category'][:15]}",
                xy=(x_pos, 0.5),
                xytext=(x_pos, y_offset),
                fontsize=7,
                ha='center',
                color='green',
                arrowprops=dict(arrowstyle='-', color='green', alpha=0.5)
            )
    
    # Period labels
    ax_timeline.text(datetime(2023, 5, 1), 0.85, 'Pre-Conflict', ha='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax_timeline.text(datetime(2024, 6, 1), 0.85, 'Post-Conflict', ha='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    ax_timeline.text(datetime(2025, 6, 1), 0.85, 'UNDP Response', ha='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    ax_timeline.xaxis.set_major_locator(mdates.YearLocator())
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_timeline.set_yticks([])
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)
    ax_timeline.spines['left'].set_visible(False)
    ax_timeline.set_xlabel('Timeline', fontsize=11)
    
    # Save
    output_path = FIGURES_DIR / "study_design_map.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ study_design_map.png")
    
    return gdf


def create_time_series_with_undp(gaza_monthly, wb_monthly, outcome, cfg, undp_df):
    """Create time series plot with UNDP project markers (clean design)."""
    
    if len(gaza_monthly) == 0:
        return False
    
    # Category colors for UNDP projects
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
    
    # Create figure with project strip at bottom
    fig = plt.figure(figsize=(16, 9))
    
    # Main time series plot (takes most of the space) - moved down slightly
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    
    # Project strip at bottom (thin strip showing UNDP project categories)
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # === MAIN TIME SERIES ===
    # Plot Gaza with shaded area
    ax_main.fill_between(gaza_monthly['period_start'], gaza_monthly[outcome], 
                         alpha=0.3, color=cfg['line_color'], label='_nolegend_')
    ax_main.plot(gaza_monthly['period_start'], gaza_monthly[outcome], 
                 color=cfg['line_color'], linewidth=2.5, marker='o', markersize=4,
                 label='Gaza Strip')
    
    # Plot West Bank for comparison (if available)
    if len(wb_monthly) > 0 and outcome in wb_monthly.columns and not wb_monthly[outcome].isna().all():
        ax_main.plot(wb_monthly['period_start'], wb_monthly[outcome], 
                     color='gray', linewidth=2, linestyle='--', alpha=0.7,
                     label='West Bank (Control)')
    
    # Pre/post means for Gaza
    pre_data = gaza_monthly[gaza_monthly['period_start'] < CONFLICT_DATE][outcome]
    post_data = gaza_monthly[gaza_monthly['period_start'] >= CONFLICT_DATE][outcome]
    
    pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
    post_mean = post_data.mean() if len(post_data) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Conflict line
    ax_main.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=2.5)
    
    # Shade pre-conflict period
    xmin = gaza_monthly['period_start'].min()
    xmax = gaza_monthly['period_start'].max()
    
    if xmin < CONFLICT_DATE:
        ax_main.axvspan(xmin, min(CONFLICT_DATE, xmax), alpha=0.1, color='green')
    if xmax > CONFLICT_DATE:
        ax_main.axvspan(max(CONFLICT_DATE, xmin), xmax, alpha=0.08, color='red')
    
    # Conflict label - positioned inside the plot area at the top
    y_top = ax_main.get_ylim()[1]
    ax_main.text(CONFLICT_DATE, y_top * 0.95, 'Conflict\nOct 7, 2023', 
                 ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Calculate % change
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct_change:+.1f}% After Conflict)"
    else:
        title_suffix = ""
    
    # Title with more padding
    ax_main.set_title(f'Gaza: {cfg["name"]}{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])  # Hide x labels on main plot (shown on strip)
    ax_main.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=cfg['line_color'], linewidth=2.5, marker='o', label='Gaza Strip'),
    ]
    if len(wb_monthly) > 0 and outcome in wb_monthly.columns:
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='West Bank'))
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label=f'Pre-Conflict Mean: {pre_mean:.2f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label=f'Post-Conflict Mean: {post_mean:.2f}'))
    
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # === UNDP PROJECT STRIP ===
    ax_strip.set_xlim(ax_main.get_xlim())
    ax_strip.set_ylim(0, 1)
    
    # Shade conflict period on strip too
    if xmin < CONFLICT_DATE:
        ax_strip.axvspan(xmin, min(CONFLICT_DATE, xmax), alpha=0.15, color='green')
    if xmax > CONFLICT_DATE:
        ax_strip.axvspan(max(CONFLICT_DATE, xmin), xmax, alpha=0.1, color='red')
    
    ax_strip.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot UNDP projects as colored markers
    categories_shown = set()
    if undp_df is not None:
        markers = get_undp_markers(undp_df, xmin, xmax)
        
        if markers:
            from collections import defaultdict
            monthly_markers = defaultdict(list)
            for date, category, title in markers:
                month_key = date.to_period('M').to_timestamp()
                monthly_markers[month_key].append((category, title))
            
            for month, projects in sorted(monthly_markers.items()):
                # Stack multiple projects vertically if same month
                for i, (category, title) in enumerate(projects[:3]):  # Max 3 per month
                    color = CATEGORY_COLORS.get(category, '#9E9E9E')
                    y_pos = 0.5 + (i - len(projects)/2 + 0.5) * 0.25
                    y_pos = max(0.15, min(0.85, y_pos))  # Keep within bounds
                    
                    ax_strip.scatter([month], [y_pos], color=color, s=80, 
                                    marker='s', edgecolor='black', linewidth=0.5, zorder=5)
                    categories_shown.add(category)
    
    # Format strip x-axis
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_strip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_strip.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('UNDP', fontsize=9, rotation=0, ha='right', va='center')
    
    # Remove top/right spines on strip
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['left'].set_visible(False)
    
    # Add category legend at the very bottom of the figure (below everything)
    if categories_shown:
        legend_patches = [mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:20]) 
                          for cat in sorted(categories_shown)]
        fig.legend(handles=legend_patches, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)), 
                   fontsize=7, frameon=False)
    
    plt.savefig(FIGURES_DIR / f'ts_{outcome}_line.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("Loading data...")
    
    # Load panel data
    if not PANEL_FILE.exists():
        print(f"✗ Panel file not found: {PANEL_FILE}")
        print("  Run: make preprocess COUNTRY=gaza")
        return
    
    panel = pd.read_csv(PANEL_FILE)
    panel['period_start'] = pd.to_datetime(panel['period_start'])
    
    # Load boundaries
    if not BOUNDARIES_FILE.exists():
        print(f"✗ Boundaries file not found: {BOUNDARIES_FILE}")
        return
    
    gdf = gpd.read_file(BOUNDARIES_FILE)
    
    # Determine ID column
    id_col = 'adm2_pcode' if 'adm2_pcode' in gdf.columns else gdf.columns[gdf.columns.str.contains('pcode', case=False)][0]
    name_col = 'adm2_name' if 'adm2_name' in gdf.columns else gdf.columns[gdf.columns.str.contains('name', case=False)][0]
    
    print(f"Using ID column: {id_col}")
    
    # Identify Gaza vs West Bank
    GAZA_CODES = [c for c in gdf[id_col].unique() if str(c).startswith('PS02')]
    WESTBANK_CODES = [c for c in gdf[id_col].unique() if str(c).startswith('PS01')]
    
    print(f"\nDetected Gaza codes ({len(GAZA_CODES)}): {GAZA_CODES}")
    print(f"Detected West Bank codes ({len(WESTBANK_CODES)}): {WESTBANK_CODES[:5]}...")
    
    # Split panel
    panel['is_gaza'] = panel['unit_id'].isin(GAZA_CODES)
    gaza_panel = panel[panel['is_gaza']].copy()
    westbank_panel = panel[~panel['is_gaza']].copy()
    
    print(f"\n  Gaza: {gaza_panel['unit_id'].nunique()} districts, {len(gaza_panel)} obs")
    print(f"  West Bank: {westbank_panel['unit_id'].nunique()} districts, {len(westbank_panel)} obs")
    
    # Load UNDP projects
    undp_df = load_undp_projects()
    if undp_df is not None:
        post_conflict_projects = undp_df[undp_df['start_date'] > CONFLICT_DATE]
        print(f"\n  UNDP Projects loaded: {len(undp_df)} total, {len(post_conflict_projects)} post-conflict")
    else:
        print("\n  ⚠ No UNDP projects file found")
    
    # Available outcomes
    available = [o for o in OUTCOMES.keys() if o in panel.columns and panel[o].notna().any()]
    print(f"  Available outcomes: {', '.join(available)}")
    
    # ==========================================================================
    # 1. STUDY DESIGN MAP
    # ==========================================================================
    gdf = create_study_design_map(gdf, id_col, name_col)
    
    # ==========================================================================
    # 2. TIME SERIES WITH UNDP MARKERS
    # ==========================================================================
    print("\n--- Time Series (with UNDP markers) ---")
    
    for outcome in available:
        cfg = OUTCOMES[outcome]
        
        # Aggregate by month
        gaza_monthly = gaza_panel.groupby('period_start')[outcome].mean().reset_index()
        gaza_monthly = gaza_monthly.sort_values('period_start').dropna(subset=[outcome])
        
        if len(westbank_panel) > 0 and outcome in westbank_panel.columns:
            wb_monthly = westbank_panel.groupby('period_start')[outcome].mean().reset_index()
            wb_monthly = wb_monthly.sort_values('period_start').dropna(subset=[outcome])
        else:
            wb_monthly = pd.DataFrame()
        
        success = create_time_series_with_undp(gaza_monthly, wb_monthly, outcome, cfg, undp_df)
        if success:
            print(f"  ✓ ts_{outcome}_line.png")
        else:
            print(f"  ✗ ts_{outcome}_line.png (no data)")
    
    # ==========================================================================
    # 3. MAPS - SPECIFIC MONTHS
    # ==========================================================================
    print("\n--- Maps (Specific Months) ---")
    
    # Define months to compare
    MAP_MONTHS = ['2023-09', '2024-09', '2025-09']
    
    # Add year_month column if not present
    if 'year_month' not in panel.columns:
        panel['year_month'] = panel['period_start'].dt.strftime('%Y-%m')
    
    # Separate GDF for Gaza and West Bank
    gdf['is_gaza'] = gdf[id_col].str.startswith('PS02')
    gdf_gaza = gdf[gdf['is_gaza']].copy()
    gdf_westbank = gdf[~gdf['is_gaza']].copy()
    
    # Get bounds for each region
    gaza_bounds = gdf_gaza.total_bounds if len(gdf_gaza) > 0 else gdf.total_bounds
    wb_bounds = gdf_westbank.total_bounds if len(gdf_westbank) > 0 else gdf.total_bounds
    
    for outcome in available:
        cfg = OUTCOMES[outcome]
        
        # Get data for ALL regions (Gaza + West Bank) for each month
        month_data = {}
        for month in MAP_MONTHS:
            # Use FULL panel (both regions)
            month_df = panel[panel['year_month'] == month]
            if not month_df.empty and not month_df[outcome].isna().all():
                agg = month_df.groupby('unit_id')[outcome].mean().reset_index()
                # Merge with FULL boundary file (both regions)
                merged = gdf.merge(agg, left_on=id_col, right_on='unit_id', how='left')
                month_data[month] = merged
        
        if len(month_data) < 2:
            print(f"  ✗ map_{outcome}_months.png (insufficient data)")
            continue
        
        # Create figure with 2 rows: West Bank (top), Gaza (bottom)
        n_months = len(month_data)
        fig, axes = plt.subplots(2, n_months, figsize=(4*n_months + 1, 10),
                                 gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.05, 'wspace': 0.05})
        
        # Handle case of single column
        if n_months == 1:
            axes = axes.reshape(2, 1)
        
        # Determine color scale across all months (for comparability)
        all_vals = pd.concat([df[[outcome]] for df in month_data.values() if outcome in df.columns]).dropna()
        if len(all_vals) == 0:
            plt.close()
            continue
        vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
        
        # Plot each month
        for col_idx, (month, gdf_month) in enumerate(month_data.items()):
            if outcome not in gdf_month.columns:
                axes[0, col_idx].set_visible(False)
                axes[1, col_idx].set_visible(False)
                continue
            
            # Separate Gaza and West Bank data
            gdf_wb = gdf_month[gdf_month[id_col].str.startswith('PS01', na=False)]
            gdf_gz = gdf_month[gdf_month[id_col].str.startswith('PS02', na=False)]
            
            # --- TOP ROW: West Bank ---
            ax_wb = axes[0, col_idx]
            if len(gdf_wb) > 0:
                gdf_wb.plot(column=outcome, ax=ax_wb, cmap=cfg['cmap'], 
                           vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.5,
                           legend=False, missing_kwds={'color': 'lightgray'})
            ax_wb.set_xlim(wb_bounds[0] - 0.03, wb_bounds[2] + 0.03)
            ax_wb.set_ylim(wb_bounds[1] - 0.03, wb_bounds[3] + 0.03)
            ax_wb.axis('off')
            
            # Title only on top row
            if month < "2023-10":
                period_label = "(Pre-Conflict)"
                title_color = 'green'
            else:
                period_label = "(Post-Conflict)"
                title_color = 'red'
            ax_wb.set_title(f'{month}\n{period_label}', fontsize=11, fontweight='bold', color=title_color, pad=2)
            
            # --- BOTTOM ROW: Gaza ---
            ax_gz = axes[1, col_idx]
            if len(gdf_gz) > 0:
                gdf_gz.plot(column=outcome, ax=ax_gz, cmap=cfg['cmap'], 
                           vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.5,
                           legend=False, missing_kwds={'color': 'lightgray'})
                # Add value labels
                for idx, row in gdf_gz.iterrows():
                    if outcome in row and pd.notna(row[outcome]):
                        c = row.geometry.centroid
                        val_str = f'{row[outcome]:.1f}' if abs(row[outcome]) < 100 else f'{row[outcome]:.0f}'
                        ax_gz.annotate(val_str, xy=(c.x, c.y), ha='center', fontsize=8, fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            ax_gz.set_xlim(gaza_bounds[0] - 0.02, gaza_bounds[2] + 0.02)
            ax_gz.set_ylim(gaza_bounds[1] - 0.02, gaza_bounds[3] + 0.02)
            ax_gz.axis('off')
        
        # Row labels
        fig.text(0.08, 0.68, 'West Bank\n(Control)', ha='center', va='center', fontsize=10, 
                 fontweight='bold', rotation=90)
        fig.text(0.08, 0.22, 'Gaza\n(Treatment)', ha='center', va='center', fontsize=10, 
                 fontweight='bold', rotation=90, color='darkred')
        
        # Colorbar at bottom
        sm = plt.cm.ScalarMappable(cmap=cfg['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
        
        fig.suptitle(f'Palestine: {cfg["name"]} - Monthly Comparison', 
                    fontsize=13, fontweight='bold', y=0.97)
        
        plt.savefig(FIGURES_DIR / f'map_{outcome}_months.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ map_{outcome}_months.png")
    
    # ==========================================================================
    # 4. SUMMARY STATISTICS TABLE
    # ==========================================================================
    print("\n--- Summary Statistics ---")
    
    summary_rows = []
    for outcome in available:
        cfg = OUTCOMES[outcome]
        
        pre = gaza_panel[gaza_panel['period_start'] < CONFLICT_DATE][outcome]
        post = gaza_panel[gaza_panel['period_start'] >= CONFLICT_DATE][outcome]
        
        pre_mean = pre.mean()
        post_mean = post.mean()
        
        summary_rows.append({
            'Outcome': cfg['name'],
            'Pre-Conflict Mean': f"{pre_mean:.2f}" if not pd.isna(pre_mean) else "N/A",
            'Pre-Conflict SD': f"{pre.std():.2f}" if not pd.isna(pre.std()) else "N/A",
            'Post-Conflict Mean': f"{post_mean:.2f}" if not pd.isna(post_mean) else "N/A",
            'Post-Conflict SD': f"{post.std():.2f}" if not pd.isna(post.std()) else "N/A",
            'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%" if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(FIGURES_DIR / 'summary_statistics.csv', index=False)
    print("  ✓ summary_statistics.csv")
    
    # ==========================================================================
    # 5. UNDP PROJECT TIMELINE
    # ==========================================================================
    if undp_df is not None:
        print("\n--- UNDP Project Timeline ---")
        create_undp_timeline(undp_df)
    
    # ==========================================================================
    # COMPLETE
    # ==========================================================================
    n_figures = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"✓ Visualizations complete: {FIGURES_DIR}")
    print(f"  Generated {n_figures} figures")
    print(f"{'='*60}")


def create_undp_timeline(undp_df):
    """Create detailed UNDP project timeline visualization."""
    
    # Filter to study period (post-conflict)
    df = undp_df[undp_df['start_date'] >= datetime(2023, 1, 1)].copy()
    df = df.sort_values('start_date')
    
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Category colors
    category_colors = {
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
    
    # Plot conflict line
    ax.axvline(x=CONFLICT_DATE, color='red', linestyle='--', linewidth=2, zorder=5)
    ax.text(CONFLICT_DATE, 1.02, 'Oct 7, 2023\nConflict Onset', 
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    
    # Plot each project as a horizontal bar
    for i, (_, row) in enumerate(df.iterrows()):
        start = row['start_date']
        category = row['ml_category']
        color = category_colors.get(category, '#9E9E9E')
        
        # Bar for project
        ax.barh(i, width=30, left=start, height=0.6, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Project name (truncated)
        title = str(row['title'])[:45] + '...' if len(str(row['title'])) > 45 else str(row['title'])
        ax.text(start + timedelta(days=35), i, title, va='center', fontsize=8, alpha=0.9)
    
    # Formatting
    ax.set_xlim(datetime(2023, 1, 1), datetime(2026, 1, 1))
    ax.set_ylim(-1, len(df) + 1)
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('UNDP Projects Timeline: Palestine\n(Non-umbrella projects since 2023)', 
                 fontsize=14, fontweight='bold', pad=25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legend
    legend_patches = [mpatches.Patch(color=color, label=cat) 
                      for cat, color in category_colors.items() 
                      if cat in df['ml_category'].values]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    
    # Shade regions
    ax.axvspan(datetime(2023, 1, 1), CONFLICT_DATE, alpha=0.1, color='green')
    ax.axvspan(CONFLICT_DATE, datetime(2026, 1, 1), alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Leave room for conflict label above plot
    plt.savefig(FIGURES_DIR / 'undp_project_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ undp_project_timeline.png")


if __name__ == "__main__":
    main()
