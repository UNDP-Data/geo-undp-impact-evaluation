#!/usr/bin/env python3
"""
pipeline/google_trends_plots.py
===============================
Google Trends Visualization - ONE PLOT PER KEYWORD

Data format expected:
    date, keyword, term, score
    2023-10-01, earthquake, ငလျင်, 0
    
Where:
- keyword = English keyword name
- term = Local language term (Burmese)
- score = Search interest (0-100)

Usage:
    python -m pipeline.google_trends_plots --config config/myanmar.yaml
"""

import pandas as pd
import numpy as np
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


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').lower()
    disaster_date = pd.Timestamp(config['events']['disaster_date'])
    disaster_type = config['events'].get('disaster_type', 'Disaster').title()
    
    data_dir = Path(f"data/{country_code}")
    output_dir = Path(f"outputs/{country_code}")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load UNDP projects
    undp_path = data_dir / "external" / "undp_projects.csv"
    undp_df = None
    if undp_path.exists():
        undp_df = pd.read_csv(undp_path)
        undp_df['start_date'] = pd.to_datetime(undp_df['start_date'], errors='coerce')
        if 'is_likely_umbrella' in undp_df.columns:
            undp_df = undp_df[undp_df['is_likely_umbrella'] != True]
    
    # Load Google Trends
    trends_path = data_dir / "external" / "google_trends.csv"
    if not trends_path.exists():
        print(f"⚠ Google Trends not found: {trends_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"GOOGLE TRENDS: {country_name}")
    print(f"{'='*70}")
    
    trends_df = pd.read_csv(trends_path)
    print(f"\nData: {trends_df.shape[0]} rows, {trends_df.shape[1]} columns")
    print(f"Columns: {list(trends_df.columns)}")
    
    # Parse dates
    date_col = 'date' if 'date' in trends_df.columns else trends_df.columns[0]
    trends_df['parsed_date'] = pd.to_datetime(trends_df[date_col], errors='coerce')
    
    # Identify format based on columns
    if 'keyword' in trends_df.columns and 'score' in trends_df.columns:
        # LONG FORMAT: date, keyword, term, score
        print(f"\nFormat: Long (keyword + score columns)")
        keywords = trends_df['keyword'].unique()
        print(f"Keywords: {list(keywords)}")
        
        n_plots = 0
        for keyword in keywords:
            subset = trends_df[trends_df['keyword'] == keyword][['parsed_date', 'score']].copy()
            subset = subset.dropna()
            subset = subset.rename(columns={'parsed_date': 'date', 'score': 'value'})
            
            # Ensure numeric
            subset['value'] = pd.to_numeric(subset['value'], errors='coerce')
            subset = subset.dropna()
            
            if len(subset) >= 2:
                clean_name = str(keyword).replace(' ', '_').replace('/', '_')[:30]
                create_trends_plot(subset, keyword, undp_df, disaster_date, disaster_type,
                                  country_name, figures_dir / f'trends_{clean_name}.png')
                n_plots += 1
                print(f"  ✓ trends_{clean_name}.png")
        
        print(f"\n{'='*70}")
        print(f"✓ Created {n_plots} Google Trends plots")
        print(f"{'='*70}")
        return
    
    # WIDE FORMAT: date, keyword1, keyword2, ...
    print(f"\nFormat: Wide (each column is a keyword)")
    
    exclude_cols = ['date', 'parsed_date', 'ispartial', 'geo', 'term', 'keyword']
    value_cols = [c for c in trends_df.columns if c.lower() not in [x.lower() for x in exclude_cols]]
    
    n_plots = 0
    for col in value_cols:
        subset = trends_df[['parsed_date', col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors='coerce')
        subset = subset.dropna()
        subset = subset.rename(columns={'parsed_date': 'date', col: 'value'})
        
        if len(subset) >= 2:
            clean_name = str(col).replace(' ', '_').replace('/', '_')[:30]
            create_trends_plot(subset, col, undp_df, disaster_date, disaster_type,
                              country_name, figures_dir / f'trends_{clean_name}.png')
            n_plots += 1
            print(f"  ✓ trends_{clean_name}.png")
    
    print(f"\n{'='*70}")
    print(f"✓ Created {n_plots} Google Trends plots")
    print(f"{'='*70}")


def create_trends_plot(data, keyword, undp_df, disaster_date, disaster_type, 
                       country_name, output_path):
    """Create Gaza-format time series for a single keyword."""
    
    data = data.sort_values('date')
    
    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
    ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
    
    # Main line with fill
    ax_main.fill_between(data['date'], data['value'], alpha=0.3, color='steelblue')
    ax_main.plot(data['date'], data['value'], color='steelblue', linewidth=2.5, 
                 marker='o', markersize=4)
    
    # Pre/post means
    pre = data[data['date'] < disaster_date]['value']
    post = data[data['date'] >= disaster_date]['value']
    pre_mean = pre.mean() if len(pre) > 0 else np.nan
    post_mean = post.mean() if len(post) > 0 else np.nan
    
    if not pd.isna(pre_mean):
        ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    if not pd.isna(post_mean):
        ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Disaster line
    ax_main.axvline(x=disaster_date, color='red', linestyle='--', linewidth=2.5)
    
    # Period shading
    xmin, xmax = data['date'].min(), data['date'].max()
    if xmin < disaster_date:
        ax_main.axvspan(xmin, min(disaster_date, xmax), alpha=0.1, color='green')
    if xmax > disaster_date:
        ax_main.axvspan(max(disaster_date, xmin), xmax, alpha=0.08, color='red')
    
    # Event label
    y_top = ax_main.get_ylim()[1]
    ax_main.text(disaster_date, y_top * 0.95, 
                f'{disaster_type}\n{disaster_date.strftime("%b %d, %Y")}',
                ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # % change in title
    if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
        pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
        title_suffix = f" ({pct:+.1f}% After {disaster_type})"
    else:
        title_suffix = ""
    
    ax_main.set_title(f'{country_name}: Google Trends - "{keyword}"{title_suffix}', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylabel('Search Interest (0-100)', fontsize=11)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(bottom=0)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2.5, marker='o', label=f'"{keyword}"')
    ]
    if not pd.isna(pre_mean):
        legend_elements.append(Line2D([0], [0], color='green', linestyle=':', 
                                     linewidth=1.5, label=f'Pre Mean: {pre_mean:.1f}'))
    if not pd.isna(post_mean):
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', 
                                     linewidth=1.5, label=f'Post Mean: {post_mean:.1f}'))
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
                    ax_strip.scatter([month], [y_pos], color=color, s=80, marker='s', 
                                    edgecolor='black', linewidth=0.5, zorder=5)
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
        patches = [mpatches.Patch(color=CATEGORY_COLORS.get(c, '#9E9E9E'), label=c[:25]) 
                   for c in sorted(categories_shown)]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(5, len(patches)), fontsize=7, frameon=False)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
