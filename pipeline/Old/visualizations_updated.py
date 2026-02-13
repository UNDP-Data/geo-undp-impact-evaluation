#!/usr/bin/env python3
"""
pipeline/visualizations.py
==========================
Universal visualization module for impact evaluation.
Creates Gaza-style time series with UNDP project strip.

Works for ANY country using config-driven parameters.

Usage:
    python -m pipeline.visualizations --config config/myanmar.yaml
    python -m pipeline.visualizations --config config/gaza.yaml
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
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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

# Outcome configurations
OUTCOME_CONFIG = {
    'nightlights': {'name': 'Nighttime Lights (VIIRS)', 'line': 'darkblue', 'fill': 'blue', 'unit': 'nW/cm²/sr', 'agg': 'mean'},
    'ndvi': {'name': 'Vegetation Index (NDVI)', 'line': 'forestgreen', 'fill': 'lightgreen', 'unit': 'Index', 'agg': 'mean'},
    'conflict_events': {'name': 'Conflict Events', 'line': 'crimson', 'fill': 'lightcoral', 'unit': 'Count', 'agg': 'sum'},
    'conflict_fatalities': {'name': 'Conflict Fatalities', 'line': 'darkred', 'fill': 'salmon', 'unit': 'Count', 'agg': 'sum'},
    'rainfall': {'name': 'Rainfall', 'line': 'steelblue', 'fill': 'lightblue', 'unit': 'mm', 'agg': 'mean'},
    'price_bread': {'name': 'Bread Price', 'line': 'purple', 'fill': 'lavender', 'unit': 'Local Currency', 'agg': 'mean'},
    'price_rice': {'name': 'Rice Price', 'line': 'olive', 'fill': 'khaki', 'unit': 'Local Currency', 'agg': 'mean'},
    'vv': {'name': 'SAR Backscatter (VV)', 'line': 'dimgray', 'fill': 'lightgray', 'unit': 'dB', 'agg': 'mean'},
    'vh': {'name': 'SAR Backscatter (VH)', 'line': 'slategray', 'fill': 'silver', 'unit': 'dB', 'agg': 'mean'},
    'rvi': {'name': 'Radar Vegetation Index', 'line': 'darkgreen', 'fill': 'palegreen', 'unit': 'Index', 'agg': 'mean'},
}


class ImpactVisualizer:
    """Generate Gaza-style visualizations for any country."""
    
    def __init__(self, config_path: str, outcomes: list = None):
        """Initialize visualizer with config."""
        
        self.config = load_config(config_path)
        
        # Extract config values
        self.country_name = self.config.get('country_name', 'Unknown')
        self.country_code = self.config.get('country_code', 'XXX').lower()
        self.disaster_date = pd.Timestamp(self.config['events']['disaster_date'])
        self.intervention_date = pd.Timestamp(self.config['events']['intervention_start'])
        self.disaster_type = self.config['events'].get('disaster_type', 'Disaster').title()
        
        # Paths
        self.data_dir = Path(f"data/{self.country_code}")
        self.output_dir = Path(f"outputs/{self.country_code}")
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load panel data
        panel_path = self.output_dir / "panel_data.csv"
        if not panel_path.exists():
            raise FileNotFoundError(f"Panel data not found: {panel_path}")
        
        self.panel = pd.read_csv(panel_path)
        self.panel['period_start'] = pd.to_datetime(self.panel['period_start'])
        
        # Load UNDP projects (for timeline)
        self.undp_df = self._load_undp_projects()
        
        # Load boundaries
        boundary_path = Path(self.config['boundaries']['file_path'])
        if boundary_path.exists():
            self.gdf = gpd.read_file(boundary_path)
            self.id_col = self.config['boundaries']['id_column']
            self.name_col = self.config['boundaries']['name_column']
        else:
            self.gdf = None
        
        # Determine available outcomes
        potential_outcomes = list(OUTCOME_CONFIG.keys())
        self.available_outcomes = [o for o in potential_outcomes if o in self.panel.columns 
                                   and self.panel[o].notna().any()]
        
        if outcomes:
            self.outcomes = [o for o in outcomes if o in self.available_outcomes]
        else:
            self.outcomes = self.available_outcomes
        
        print(f"\n{'='*60}")
        print(f"VISUALIZATIONS: {self.country_name}")
        print(f"{'='*60}")
        print(f"Disaster: {self.disaster_date.strftime('%B %d, %Y')} ({self.disaster_type})")
        print(f"Intervention: {self.intervention_date.strftime('%B %Y')}")
        print(f"Data: {len(self.panel)} obs, {self.panel['unit_id'].nunique()} units")
        print(f"Period: {self.panel['period_start'].min().date()} to {self.panel['period_start'].max().date()}")
        print(f"Outcomes: {', '.join(self.outcomes)}")
        if self.undp_df is not None:
            print(f"UNDP Projects: {len(self.undp_df)}")
    
    def _load_undp_projects(self):
        """Load UNDP projects for timeline."""
        undp_path = self.data_dir / "external" / "undp_projects.csv"
        if not undp_path.exists():
            print(f"  ⚠ UNDP projects not found: {undp_path}")
            return None
        
        df = pd.read_csv(undp_path)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        
        # Filter out umbrella projects
        if 'is_likely_umbrella' in df.columns:
            df = df[df['is_likely_umbrella'] != True]
        
        return df
    
    def _get_undp_markers(self, min_date, max_date):
        """Get UNDP project markers for date range."""
        if self.undp_df is None:
            return []
        
        df = self.undp_df[
            (self.undp_df['start_date'] >= min_date) & 
            (self.undp_df['start_date'] <= max_date)
        ].copy()
        
        markers = []
        for _, row in df.iterrows():
            category = row.get('ml_category', 'Other / Unspecified')
            title = str(row.get('title', ''))[:30]
            markers.append((row['start_date'], category, title))
        
        return sorted(markers, key=lambda x: x[0])
    
    def generate_all(self):
        """Generate all visualizations."""
        
        print("\n--- Time Series (Gaza Format) ---")
        self._time_series_gaza_format()
        
        print("\n--- Choropleth Maps ---")
        self._choropleth_maps()
        
        print("\n--- Conflict Timeline ---")
        self._conflict_timeline()
        
        print("\n--- UNDP Project Timeline ---")
        self._undp_timeline()
        
        print("\n--- Summary Statistics ---")
        self._summary_statistics()
        
        n_figures = len(list(self.figures_dir.glob("*.png")))
        print(f"\n{'='*60}")
        print(f"✓ All visualizations saved to: {self.figures_dir}")
        print(f"  Generated {n_figures} figures")
        print(f"{'='*60}")
    
    def _time_series_gaza_format(self):
        """Generate Gaza-style time series with UNDP strip at bottom."""
        
        for outcome in self.outcomes:
            cfg = OUTCOME_CONFIG.get(outcome, {'name': outcome, 'line': 'blue', 'fill': 'lightblue', 'unit': '', 'agg': 'mean'})
            
            # Aggregate monthly
            if cfg['agg'] == 'sum':
                monthly = self.panel.groupby('period_start')[outcome].sum().reset_index()
            else:
                monthly = self.panel.groupby('period_start').agg({
                    outcome: ['mean', 'std', 'count']
                }).reset_index()
                monthly.columns = ['period_start', 'mean', 'std', 'count']
                monthly['se'] = monthly['std'] / np.sqrt(monthly['count'])
                monthly = monthly.rename(columns={'mean': outcome})
            
            monthly = monthly.sort_values('period_start')
            
            if monthly[outcome].isna().all():
                continue
            
            # Create figure with main panel + UNDP strip
            fig = plt.figure(figsize=(16, 9))
            ax_main = fig.add_axes([0.08, 0.22, 0.85, 0.68])
            ax_strip = fig.add_axes([0.08, 0.12, 0.85, 0.06])
            
            # === MAIN TIME SERIES ===
            
            # Fill ribbon (if we have SE)
            if 'se' in monthly.columns:
                ax_main.fill_between(
                    monthly['period_start'],
                    monthly[outcome] - 1.96 * monthly['se'],
                    monthly[outcome] + 1.96 * monthly['se'],
                    alpha=0.3, color=cfg['fill']
                )
            else:
                ax_main.fill_between(monthly['period_start'], monthly[outcome],
                                    alpha=0.3, color=cfg['fill'])
            
            # Main line
            ax_main.plot(monthly['period_start'], monthly[outcome],
                        color=cfg['line'], linewidth=2.5, marker='o', markersize=5,
                        label=self.country_name)
            
            # Pre/Post means
            pre_data = monthly[monthly['period_start'] < self.disaster_date][outcome]
            post_data = monthly[monthly['period_start'] >= self.disaster_date][outcome]
            
            pre_mean = pre_data.mean() if len(pre_data) > 0 else np.nan
            post_mean = post_data.mean() if len(post_data) > 0 else np.nan
            
            # Dotted horizontal lines
            if not pd.isna(pre_mean):
                ax_main.axhline(y=pre_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
            if not pd.isna(post_mean):
                ax_main.axhline(y=post_mean, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # Vertical dashed lines
            ax_main.axvline(x=self.disaster_date, color='red', linestyle='--', linewidth=2.5)
            ax_main.axvline(x=self.intervention_date, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            
            # Period shading
            xmin = monthly['period_start'].min()
            xmax = monthly['period_start'].max()
            
            if xmin < self.disaster_date:
                ax_main.axvspan(xmin, min(self.disaster_date, xmax), alpha=0.1, color='green')
            if xmax > self.disaster_date:
                ax_main.axvspan(max(self.disaster_date, xmin), xmax, alpha=0.08, color='red')
            
            # Event labels
            y_top = ax_main.get_ylim()[1]
            ax_main.text(self.disaster_date, y_top * 0.95, 
                        f'{self.disaster_type}\n{self.disaster_date.strftime("%b %d, %Y")}',
                        ha='center', va='top', fontsize=9, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
            
            ax_main.text(self.intervention_date, y_top * 0.82, 'UNDP\nResponse',
                        ha='center', va='top', fontsize=8, fontweight='bold', color='blue',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue'))
            
            # Calculate % change
            if pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean):
                pct_change = ((post_mean - pre_mean) / abs(pre_mean)) * 100
                title_suffix = f" ({pct_change:+.1f}% After {self.disaster_type})"
            else:
                title_suffix = ""
            
            ax_main.set_title(f'{self.country_name}: {cfg["name"]}{title_suffix}',
                             fontsize=14, fontweight='bold', pad=20)
            ax_main.set_ylabel(f'{cfg["name"]} ({cfg["unit"]})', fontsize=11)
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_main.set_xticklabels([])
            ax_main.grid(True, alpha=0.3)
            
            # Legend
            legend_elements = [
                Line2D([0], [0], color=cfg['line'], linewidth=2.5, marker='o', label=self.country_name),
            ]
            if not pd.isna(pre_mean):
                legend_elements.append(Line2D([0], [0], color='green', linestyle=':',
                                             linewidth=1.5, label=f'Pre-{self.disaster_type} Mean: {pre_mean:.2f}'))
            if not pd.isna(post_mean):
                legend_elements.append(Line2D([0], [0], color='red', linestyle=':',
                                             linewidth=1.5, label=f'Post-{self.disaster_type} Mean: {post_mean:.2f}'))
            
            ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            # === UNDP PROJECT STRIP ===
            ax_strip.set_xlim(ax_main.get_xlim())
            ax_strip.set_ylim(0, 1)
            
            # Period shading on strip
            if xmin < self.disaster_date:
                ax_strip.axvspan(xmin, min(self.disaster_date, xmax), alpha=0.15, color='green')
            if xmax > self.disaster_date:
                ax_strip.axvspan(max(self.disaster_date, xmin), xmax, alpha=0.1, color='red')
            
            ax_strip.axvline(x=self.disaster_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_strip.axvline(x=self.intervention_date, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Plot UNDP project markers
            categories_shown = set()
            markers = self._get_undp_markers(xmin, xmax)
            
            if markers:
                monthly_markers = defaultdict(list)
                for date, category, title in markers:
                    month_key = date.to_period('M').to_timestamp()
                    monthly_markers[month_key].append((category, title))
                
                for month, projects in sorted(monthly_markers.items()):
                    for i, (category, title) in enumerate(projects[:3]):
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
            
            # Category legend
            if categories_shown:
                legend_patches = [
                    mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat[:25])
                    for cat in sorted(categories_shown)
                ]
                fig.legend(handles=legend_patches, loc='lower center',
                           bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(legend_patches)),
                           fontsize=7, frameon=False)
            
            plt.savefig(self.figures_dir / f'ts_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✓ ts_{outcome}.png")
    
    def _choropleth_maps(self):
        """Generate pre/post choropleth maps."""
        
        if self.gdf is None:
            print("  ⚠ No boundaries loaded, skipping maps")
            return
        
        for outcome in self.outcomes[:4]:  # Limit to 4 outcomes for maps
            cfg = OUTCOME_CONFIG.get(outcome, {'name': outcome, 'agg': 'mean'})
            
            # Aggregate by admin unit and period
            agg_func = 'sum' if cfg['agg'] == 'sum' else 'mean'
            
            pre_data = self.panel[self.panel['period_start'] < self.disaster_date].groupby('unit_id')[outcome].agg(agg_func).reset_index()
            post_data = self.panel[self.panel['period_start'] >= self.disaster_date].groupby('unit_id')[outcome].agg(agg_func).reset_index()
            
            # Merge with geometry
            gdf_pre = self.gdf.merge(pre_data, left_on=self.id_col, right_on='unit_id', how='left')
            gdf_post = self.gdf.merge(post_data, left_on=self.id_col, right_on='unit_id', how='left')
            
            if gdf_pre[outcome].isna().all() and gdf_post[outcome].isna().all():
                continue
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            
            # Common color scale
            all_vals = pd.concat([gdf_pre[[outcome]], gdf_post[[outcome]]]).dropna()
            if len(all_vals) == 0:
                plt.close()
                continue
            
            vmin, vmax = all_vals[outcome].quantile([0.02, 0.98])
            cmap = 'YlOrRd' if 'conflict' in outcome else 'viridis'
            
            # Pre-disaster
            gdf_pre.plot(column=outcome, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
            axes[0].set_title(f'Pre-{self.disaster_type}', fontsize=12, fontweight='bold', color='green')
            axes[0].axis('off')
            
            # Post-disaster
            gdf_post.plot(column=outcome, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                         edgecolor='black', linewidth=0.3, legend=False, missing_kwds={'color': 'lightgray'})
            axes[1].set_title(f'Post-{self.disaster_type}', fontsize=12, fontweight='bold', color='red')
            axes[1].axis('off')
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08)
            cbar.set_label(f'{cfg["name"]}', fontsize=11)
            
            fig.suptitle(f'{self.country_name}: {cfg["name"]}',
                        fontsize=14, fontweight='bold', y=0.95)
            
            plt.savefig(self.figures_dir / f'map_{outcome}.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✓ map_{outcome}.png")
    
    def _conflict_timeline(self):
        """Generate conflict timeline if data available."""
        
        if 'conflict_events' not in self.panel.columns:
            return
        
        monthly = self.panel.groupby('period_start')['conflict_events'].sum().reset_index()
        monthly = monthly.sort_values('period_start')
        
        if monthly['conflict_events'].sum() == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for _, row in monthly.iterrows():
            date = row['period_start']
            val = row['conflict_events']
            
            color = 'forestgreen' if date < self.disaster_date else 'crimson'
            ax.bar(date, val, width=25, color=color, alpha=0.7, edgecolor='white')
        
        ax.axvline(x=self.disaster_date, color='black', linestyle='--', linewidth=2)
        
        ax.text(self.disaster_date, ax.get_ylim()[1]*0.95,
               f'{self.disaster_type}\n{self.disaster_date.strftime("%b %d, %Y")}',
               ha='center', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{self.country_name}: Conflict Events (ACLED)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Monthly Events', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'conflict_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("  ✓ conflict_timeline.png")
    
    def _undp_timeline(self):
        """Generate UNDP project timeline."""
        
        if self.undp_df is None or len(self.undp_df) == 0:
            return
        
        df = self.undp_df.sort_values('start_date')
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.3)))
        
        # Disaster line
        ax.axvline(x=self.disaster_date, color='red', linestyle='--', linewidth=2.5, zorder=5)
        ax.text(self.disaster_date, 1.02, f'{self.disaster_type}\n{self.disaster_date.strftime("%b %d, %Y")}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
        
        # Plot projects
        for i, (_, row) in enumerate(df.iterrows()):
            start = row['start_date']
            category = row.get('ml_category', 'Other / Unspecified')
            color = CATEGORY_COLORS.get(category, '#9E9E9E')
            
            ax.barh(i, width=30, left=start, height=0.6, color=color,
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            
            title = str(row.get('title', ''))[:50]
            if len(str(row.get('title', ''))) > 50:
                title += '...'
            ax.text(start + pd.Timedelta(days=35), i, title, va='center', fontsize=8, alpha=0.9)
        
        ax.set_ylim(-1, len(df) + 1)
        ax.set_yticks([])
        ax.set_xlabel('Date', fontsize=11)
        ax.set_title(f'{self.country_name}: UNDP Projects Timeline', fontsize=14, fontweight='bold', pad=25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Legend
        categories_in_data = df['ml_category'].unique() if 'ml_category' in df.columns else []
        legend_patches = [
            mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#9E9E9E'), label=cat)
            for cat in sorted(categories_in_data) if cat in CATEGORY_COLORS
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(self.figures_dir / 'undp_project_timeline.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("  ✓ undp_project_timeline.png")
    
    def _summary_statistics(self):
        """Generate summary statistics table."""
        
        summary_rows = []
        for outcome in self.outcomes:
            cfg = OUTCOME_CONFIG.get(outcome, {'name': outcome})
            
            pre = self.panel[self.panel['period_start'] < self.disaster_date][outcome]
            post = self.panel[self.panel['period_start'] >= self.disaster_date][outcome]
            
            pre_mean = pre.mean()
            post_mean = post.mean()
            
            summary_rows.append({
                'Outcome': cfg['name'],
                f'Pre-{self.disaster_type} Mean': f"{pre_mean:.2f}" if not pd.isna(pre_mean) else "N/A",
                f'Pre-{self.disaster_type} SD': f"{pre.std():.2f}" if not pd.isna(pre.std()) else "N/A",
                f'Post-{self.disaster_type} Mean': f"{post_mean:.2f}" if not pd.isna(post_mean) else "N/A",
                f'Post-{self.disaster_type} SD': f"{post.std():.2f}" if not pd.isna(post.std()) else "N/A",
                'Change (%)': f"{((post_mean - pre_mean) / abs(pre_mean) * 100):+.1f}%"
                             if (pre_mean != 0 and not pd.isna(pre_mean) and not pd.isna(post_mean)) else "N/A",
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.figures_dir / 'summary_statistics.csv', index=False)
        
        print("  ✓ summary_statistics.csv")


def main():
    parser = argparse.ArgumentParser(description='Generate impact evaluation visualizations')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--outcomes', '-o', nargs='+', help='Specific outcomes to visualize')
    
    args = parser.parse_args()
    
    viz = ImpactVisualizer(args.config, outcomes=args.outcomes)
    viz.generate_all()


if __name__ == "__main__":
    main()
