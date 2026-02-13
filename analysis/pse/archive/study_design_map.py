#!/usr/bin/env python3
"""
analysis/pse/study_design_map.py
================================
Creates a map showing the study design:
- Treatment: Gaza Strip (PS02* districts)
- Control: West Bank (PS01* districts)
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
BOUNDARIES_FILE = "data/pse/boundaries/gaza_admin2.geojson"
OUTPUT_DIR = Path("outputs/pse/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_study_design_map():
    """Create a map showing treatment vs control regions."""
    
    # Load boundaries
    gdf = gpd.read_file(BOUNDARIES_FILE)
    
    # Identify Gaza vs West Bank
    id_col = 'adm2_pcode' if 'adm2_pcode' in gdf.columns else gdf.columns[gdf.columns.str.contains('pcode', case=False)][0]
    name_col = 'adm2_name' if 'adm2_name' in gdf.columns else gdf.columns[gdf.columns.str.contains('name', case=False)][0]
    
    gdf['is_gaza'] = gdf[id_col].str.startswith('PS02')
    gdf['group'] = gdf['is_gaza'].map({True: 'Treatment (Gaza)', False: 'Control (West Bank)'})
    
    # Colors
    colors = {
        'Treatment (Gaza)': '#E74C3C',      # Red
        'Control (West Bank)': '#3498DB'    # Blue
    }
    gdf['color'] = gdf['group'].map(colors)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    # Plot regions
    for group, color in colors.items():
        subset = gdf[gdf['group'] == group]
        subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.8, alpha=0.7)
    
    # Add district labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[name_col]
        
        # Adjust font size based on region
        fontsize = 7 if row['is_gaza'] else 8
        fontweight = 'bold' if row['is_gaza'] else 'normal'
        
        ax.annotate(
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
    ax.legend(handles=[treatment_patch, control_patch], loc='lower left', fontsize=10, framealpha=0.9)
    
    # Title and labels
    ax.set_title('Palestine: Difference-in-Differences Study Design\n(Treatment vs Control Regions)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add annotation box with study info
    info_text = """Study Design:
• Treatment: Gaza Strip (5 districts)
• Control: West Bank (11 districts)
• Event: Oct 7, 2023 conflict onset
• Intervention: UNDP (June 2025)
• Method: Two-Way Fixed Effects (TWFE)"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_axis_off()
    
    # Save
    output_path = OUTPUT_DIR / "study_design_map.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Study design map saved: {output_path}")
    
    # Also create a version with conflict timeline
    create_timeline_map(gdf, id_col, name_col, colors)


def create_timeline_map(gdf, id_col, name_col, colors):
    """Create a map with timeline showing key events."""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Main map
    ax_map = fig.add_axes([0.05, 0.25, 0.9, 0.7])
    
    # Plot regions
    for group, color in colors.items():
        subset = gdf[gdf['group'] == group]
        subset.plot(ax=ax_map, color=color, edgecolor='black', linewidth=0.8, alpha=0.7)
    
    # Add district labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[name_col]
        fontsize = 7 if row['is_gaza'] else 8
        ax_map.annotate(name, xy=(centroid.x, centroid.y), ha='center', va='center',
                       fontsize=fontsize, color='white' if row['is_gaza'] else 'black')
    
    # Legend
    treatment_patch = mpatches.Patch(color=colors['Treatment (Gaza)'], label='Treatment: Gaza', alpha=0.7)
    control_patch = mpatches.Patch(color=colors['Control (West Bank)'], label='Control: West Bank', alpha=0.7)
    ax_map.legend(handles=[treatment_patch, control_patch], loc='lower left', fontsize=10)
    
    ax_map.set_title('Palestine: Impact Evaluation Study Design', fontsize=14, fontweight='bold')
    ax_map.set_axis_off()
    
    # Timeline at bottom
    ax_timeline = fig.add_axes([0.1, 0.08, 0.8, 0.12])
    
    # Timeline setup
    import matplotlib.dates as mdates
    from datetime import datetime
    
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 10, 7),
        datetime(2025, 6, 1),
        datetime(2026, 1, 1)
    ]
    
    ax_timeline.set_xlim(dates[0], dates[-1])
    ax_timeline.set_ylim(0, 1)
    
    # Draw timeline
    ax_timeline.axhline(y=0.5, color='black', linewidth=2)
    
    # Events
    events = [
        (datetime(2023, 10, 7), 'Conflict Onset\nOct 7, 2023', 'red', 0.8),
        (datetime(2025, 6, 1), 'UNDP Intervention\nJun 2025', 'green', 0.8),
    ]
    
    for date, label, color, y in events:
        ax_timeline.axvline(x=date, color=color, linewidth=2, linestyle='--')
        ax_timeline.scatter([date], [0.5], color=color, s=100, zorder=5)
        ax_timeline.text(date, y, label, ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
    
    # Period labels
    ax_timeline.text(datetime(2023, 5, 1), 0.2, 'Pre-Conflict', ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax_timeline.text(datetime(2024, 8, 1), 0.2, 'Post-Conflict', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax_timeline.text(datetime(2025, 9, 1), 0.2, 'Post-UNDP', ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax_timeline.xaxis.set_major_locator(mdates.YearLocator())
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_timeline.set_yticks([])
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)
    ax_timeline.spines['left'].set_visible(False)
    
    # Save
    output_path = OUTPUT_DIR / "study_design_with_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Study design with timeline saved: {output_path}")


if __name__ == "__main__":
    create_study_design_map()
