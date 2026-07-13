#!/usr/bin/env python3
"""
analysis/pse/08_google_trends_proxy.py
======================================
Google Trends as Displacement/Activity Proxy.

Uses Google Trends data to track:
1. Search interest in displacement-related terms
2. Economic activity indicators (job searches, business queries)
3. Recovery indicators (reconstruction, rebuilding)

Limitations:
- Internet access in Gaza is limited and disrupted
- Biased toward younger, urban, educated population
- Search volume ≠ actual activity
- Best used as supplementary indicator, not primary measure

Usage:
    python analysis/pse/08_google_trends_proxy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Try to import pytrends (Google Trends API)
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False
    print("⚠ pytrends not installed. Run: pip install pytrends --break-system-packages")

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("outputs/pse")
FIGURES_DIR = OUTPUT_DIR / "figures" / "google_trends"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Geographic targeting
# Note: Google Trends has limited data for Palestine/Gaza specifically
# We may need to use broader Middle East or compare with related regions
GEO_CODES = {
    'palestine': 'PS',      # Palestine (limited data)
    'israel': 'IL',         # Israel (for comparison)
    'worldwide': '',        # Global baseline
}

# Search terms for different indicators
SEARCH_TERMS = {
    'displacement': [
        'evacuation gaza',
        'shelter gaza',
        'refugee camp',
        'displacement',
        'نزوح',  # Arabic: displacement
    ],
    'economic_activity': [
        'jobs gaza',
        'work gaza',
        'business gaza',
        'عمل غزة',  # Arabic: work Gaza
    ],
    'reconstruction': [
        'rebuilding gaza',
        'reconstruction gaza',
        'إعادة الإعمار',  # Arabic: reconstruction
    ],
    'humanitarian': [
        'food aid gaza',
        'water gaza',
        'UNRWA',
        'humanitarian aid',
    ],
    'conflict_intensity': [
        'airstrike gaza',
        'bombing gaza',
        'ceasefire gaza',
    ]
}

# Time periods
CONFLICT_START = '2023-10-07'
PRE_CONFLICT_START = '2023-01-01'
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')

# =============================================================================
# GOOGLE TRENDS DATA COLLECTION
# =============================================================================

def fetch_google_trends(keywords, geo='', timeframe='today 12-m'):
    """
    Fetch Google Trends data for given keywords.
    
    Args:
        keywords: List of search terms (max 5 per query)
        geo: Geographic code ('' for worldwide)
        timeframe: Time range (e.g., 'today 12-m', '2023-01-01 2024-01-01')
    
    Returns:
        DataFrame with interest over time
    """
    if not HAS_PYTRENDS:
        return None
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    try:
        pytrends.build_payload(keywords[:5], cat=0, timeframe=timeframe, geo=geo)
        interest_over_time = pytrends.interest_over_time()
        
        if not interest_over_time.empty:
            interest_over_time = interest_over_time.drop('isPartial', axis=1, errors='ignore')
            return interest_over_time
        
    except Exception as e:
        print(f"    ⚠ Error fetching trends: {e}")
    
    return None


def fetch_trends_for_category(category_name, terms, geo=''):
    """Fetch trends for a category of search terms."""
    print(f"\n  Fetching {category_name} trends...")
    
    # Split into batches of 5 (Google Trends limit)
    all_data = []
    
    for i in range(0, len(terms), 5):
        batch = terms[i:i+5]
        
        # Try different timeframes
        timeframe = f'{PRE_CONFLICT_START} {CURRENT_DATE}'
        
        data = fetch_google_trends(batch, geo=geo, timeframe=timeframe)
        
        if data is not None:
            all_data.append(data)
            print(f"    ✓ Got data for: {batch}")
        else:
            print(f"    ⚠ No data for: {batch}")
    
    if all_data:
        # Combine all batches
        combined = pd.concat(all_data, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
    
    return None


def create_composite_index(trends_data, category_name):
    """Create a composite index from multiple search terms."""
    if trends_data is None or trends_data.empty:
        return None
    
    # Normalize each column to 0-100
    normalized = trends_data.copy()
    for col in normalized.columns:
        max_val = normalized[col].max()
        if max_val > 0:
            normalized[col] = normalized[col] / max_val * 100
    
    # Create composite (average)
    composite = normalized.mean(axis=1)
    composite.name = f'{category_name}_index'
    
    return composite


# =============================================================================
# ALTERNATIVE: MANUAL DATA INPUT
# =============================================================================

def load_manual_trends_data():
    """
    If Google Trends API doesn't work well for Gaza,
    use manually collected data from trends.google.com.
    
    Instructions:
    1. Go to https://trends.google.com/trends/
    2. Search for relevant terms
    3. Set location to Palestine or Worldwide
    4. Download CSV and place in data/pse/external/google_trends/
    """
    print("\n  Manual Google Trends data:")
    print("    To collect manually:")
    print("    1. Visit https://trends.google.com/trends/")
    print("    2. Search for: 'gaza reconstruction', 'UNRWA', 'gaza aid'")
    print("    3. Set timeframe: Jan 2023 - Present")
    print("    4. Download CSV files")
    print("    5. Place in: data/pse/external/google_trends/")
    
    # Check for existing manual data
    manual_dir = Path("data/pse/external/google_trends")
    
    if manual_dir.exists():
        csv_files = list(manual_dir.glob("*.csv"))
        if csv_files:
            print(f"    Found {len(csv_files)} manual data files")
            
            # Try to load and combine
            all_data = []
            for f in csv_files:
                try:
                    df = pd.read_csv(f, skiprows=1)  # Skip header row
                    df.columns = ['date', f.stem]
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    all_data.append(df)
                except:
                    continue
            
            if all_data:
                return pd.concat(all_data, axis=1)
    
    return None


# =============================================================================
# SYNTHETIC TRENDS DATA (FOR DEMONSTRATION)
# =============================================================================

def generate_synthetic_trends():
    """
    Generate synthetic trends data for demonstration.
    Based on known patterns from the conflict timeline.
    
    This is for illustration only - replace with real data!
    """
    print("\n  Generating synthetic trends (for demonstration)...")
    
    # Create date range
    dates = pd.date_range(start=PRE_CONFLICT_START, end='2025-01-31', freq='W')
    
    conflict_start = pd.Timestamp(CONFLICT_START)
    
    data = {
        'displacement_searches': [],
        'humanitarian_searches': [],
        'reconstruction_searches': [],
        'economic_searches': [],
    }
    
    for date in dates:
        # Days since conflict start
        days_since = (date - conflict_start).days
        
        # Displacement: Spike at conflict, gradual decline
        if days_since < 0:
            disp = 10 + np.random.normal(0, 5)
        elif days_since < 30:
            disp = 100 - days_since * 0.5 + np.random.normal(0, 10)
        elif days_since < 180:
            disp = 85 - days_since * 0.2 + np.random.normal(0, 8)
        else:
            disp = 50 + np.random.normal(0, 10)
        
        # Humanitarian: High and sustained after conflict
        if days_since < 0:
            hum = 15 + np.random.normal(0, 5)
        elif days_since < 60:
            hum = 100 + np.random.normal(0, 10)
        else:
            hum = 80 + np.sin(days_since/30) * 10 + np.random.normal(0, 8)
        
        # Reconstruction: Gradual increase after ~6 months
        if days_since < 180:
            recon = 10 + max(0, days_since/20) + np.random.normal(0, 5)
        else:
            recon = 30 + (days_since - 180) * 0.1 + np.random.normal(0, 8)
        
        # Economic: Drop at conflict, slow recovery
        if days_since < 0:
            econ = 50 + np.random.normal(0, 5)
        elif days_since < 30:
            econ = 20 + np.random.normal(0, 8)
        elif days_since < 180:
            econ = 25 + days_since * 0.05 + np.random.normal(0, 5)
        else:
            econ = 35 + (days_since - 180) * 0.02 + np.random.normal(0, 5)
        
        data['displacement_searches'].append(max(0, min(100, disp)))
        data['humanitarian_searches'].append(max(0, min(100, hum)))
        data['reconstruction_searches'].append(max(0, min(100, recon)))
        data['economic_searches'].append(max(0, min(100, econ)))
    
    df = pd.DataFrame(data, index=dates)
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_trends_analysis(trends_data, output_path):
    """Create comprehensive trends visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    conflict_start = pd.Timestamp(CONFLICT_START)
    
    # Plot 1: All trends over time
    ax1 = axes[0, 0]
    for col in trends_data.columns:
        ax1.plot(trends_data.index, trends_data[col], linewidth=2, label=col.replace('_', ' ').title())
    
    ax1.axvline(conflict_start, color='red', linestyle='--', linewidth=2, label='Conflict Start')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Search Interest Index', fontsize=11)
    ax1.set_title('Google Trends Indicators Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    # Plot 2: Pre vs Post comparison
    ax2 = axes[0, 1]
    
    pre_conflict = trends_data[trends_data.index < conflict_start]
    post_conflict = trends_data[trends_data.index >= conflict_start]
    
    x = np.arange(len(trends_data.columns))
    width = 0.35
    
    pre_means = pre_conflict.mean()
    post_means = post_conflict.mean()
    
    ax2.bar(x - width/2, pre_means, width, label='Pre-Conflict', color='#3498db', edgecolor='black')
    ax2.bar(x + width/2, post_means, width, label='Post-Conflict', color='#e74c3c', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_', '\n') for c in trends_data.columns], fontsize=9)
    ax2.set_ylabel('Average Search Interest', fontsize=11)
    ax2.set_title('Pre vs Post Conflict Search Interest', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Recovery indicators (reconstruction + economic)
    ax3 = axes[1, 0]
    
    if 'reconstruction_searches' in trends_data.columns:
        recon = trends_data['reconstruction_searches']
        ax3.fill_between(recon.index, 0, recon, alpha=0.3, color='#27ae60')
        ax3.plot(recon.index, recon, color='#27ae60', linewidth=2, label='Reconstruction')
    
    if 'economic_searches' in trends_data.columns:
        econ = trends_data['economic_searches']
        ax3.plot(econ.index, econ, color='#3498db', linewidth=2, label='Economic Activity')
    
    ax3.axvline(conflict_start, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Search Interest Index', fontsize=11)
    ax3.set_title('Recovery Indicators\n(Reconstruction & Economic Searches)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interpretation notes
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    notes = """
    INTERPRETATION NOTES
    ════════════════════════════════════════
    
    Google Trends as Displacement Proxy:
    
    STRENGTHS:
    • Near real-time data availability
    • Reflects population's information needs
    • Can capture recovery patterns
    
    LIMITATIONS:
    • Internet access in Gaza severely disrupted
    • Biased toward urban, younger population
    • Power outages limit access
    • VPN usage may distort geographic data
    • Search volume ≠ actual behavior
    
    RECOMMENDED USE:
    • Supplementary indicator only
    • Compare trends, not absolute levels
    • Validate against ground-truth data
    • Use in conjunction with VIIRS, SAR
    
    KEY PATTERNS TO LOOK FOR:
    • Spike in displacement searches → mass movement
    • Rise in reconstruction searches → recovery phase
    • Economic searches recovering → normalization
    
    NOTE: This data may be synthetic for demonstration.
    Replace with actual Google Trends downloads.
    """
    
    ax4.text(0.05, 0.95, notes, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_trends_vs_satellite(trends_data, satellite_data_path, output_path):
    """
    Compare Google Trends with satellite-based indicators.
    Shows correlation between search behavior and physical changes.
    """
    if not Path(satellite_data_path).exists():
        print(f"    ⚠ Satellite panel data not found: {satellite_data_path}")
        return
    
    # Load satellite panel data
    sat_df = pd.read_csv(satellite_data_path)
    
    # Check for required columns - try different possible names
    date_col = None
    for col in ['period_start', 'date', 'Date', 'time']:
        if col in sat_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"    ⚠ No date column found in satellite data. Columns: {list(sat_df.columns)}")
        return
    
    sat_df[date_col] = pd.to_datetime(sat_df[date_col], errors='coerce')
    
    # Check for indicator columns - try multiple possible names
    viirs_col = None
    sar_col = None
    
    for col in sat_df.columns:
        col_lower = col.lower()
        # VIIRS / nightlights
        if viirs_col is None:
            if 'viirs' in col_lower or col_lower == 'nightlights' or col_lower == 'log_nightlights':
                viirs_col = col
        # SAR VH
        if sar_col is None:
            if ('sar' in col_lower and 'vh' in col_lower) or col_lower == 'vh' or col_lower == 'structure_proxy_vh':
                sar_col = col
    
    if viirs_col is None and sar_col is None:
        print(f"    ⚠ No VIIRS or SAR columns found. Available: {list(sat_df.columns)}")
        return
    
    # Build aggregation dict with available columns
    agg_dict = {}
    if viirs_col:
        agg_dict[viirs_col] = 'mean'
    if sar_col:
        agg_dict[sar_col] = 'mean'
    
    if not agg_dict:
        return
    
    # Aggregate satellite data monthly
    sat_monthly = sat_df.groupby(sat_df[date_col].dt.to_period('M')).agg(agg_dict).reset_index()
    sat_monthly['date'] = sat_monthly[date_col].dt.to_timestamp()
    
    # Rename columns for consistency
    if viirs_col and viirs_col != 'viirs':
        sat_monthly['viirs'] = sat_monthly[viirs_col]
    if sar_col and sar_col != 'sar_vh':
        sat_monthly['sar_vh'] = sat_monthly[sar_col]
    
    # Resample trends to monthly
    trends_monthly = trends_data.resample('M').mean()
    
    # Merge
    merged = pd.merge(
        trends_monthly.reset_index().rename(columns={'index': 'date'}),
        sat_monthly[['date', 'viirs', 'sar_vh']],
        on='date',
        how='inner'
    )
    
    if merged.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reconstruction searches vs SAR
    ax1 = axes[0]
    if 'reconstruction_searches' in merged.columns and 'sar_vh' in merged.columns:
        ax1.scatter(merged['sar_vh'], merged['reconstruction_searches'], 
                   s=80, alpha=0.7, c=range(len(merged)), cmap='viridis')
        
        # Correlation
        corr = merged['sar_vh'].corr(merged['reconstruction_searches'])
        ax1.set_xlabel('SAR VH (Structures)', fontsize=11)
        ax1.set_ylabel('Reconstruction Searches', fontsize=11)
        ax1.set_title(f'Reconstruction Interest vs Physical Structures\nCorrelation: {corr:.2f}',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Economic searches vs VIIRS
    ax2 = axes[1]
    if 'economic_searches' in merged.columns and 'viirs' in merged.columns:
        ax2.scatter(merged['viirs'], merged['economic_searches'],
                   s=80, alpha=0.7, c=range(len(merged)), cmap='viridis')
        
        corr = merged['viirs'].corr(merged['economic_searches'])
        ax2.set_xlabel('VIIRS Nightlights', fontsize=11)
        ax2.set_ylabel('Economic Searches', fontsize=11)
        ax2.set_title(f'Economic Interest vs Nightlights\nCorrelation: {corr:.2f}',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" GOOGLE TRENDS AS DISPLACEMENT/ACTIVITY PROXY")
    print("=" * 70)
    
    # Try to get real Google Trends data
    trends_data = None
    
    if HAS_PYTRENDS:
        print("\n--- Attempting to Fetch Google Trends Data ---")
        
        # Try fetching for each category
        all_trends = {}
        
        for category, terms in SEARCH_TERMS.items():
            data = fetch_trends_for_category(category, terms, geo='')
            if data is not None:
                composite = create_composite_index(data, category)
                if composite is not None:
                    all_trends[f'{category}_searches'] = composite
        
        if all_trends:
            trends_data = pd.DataFrame(all_trends)
    
    # Try loading manual data
    if trends_data is None:
        manual_data = load_manual_trends_data()
        if manual_data is not None:
            trends_data = manual_data
    
    # Fall back to synthetic data for demonstration
    if trends_data is None:
        print("\n  ⚠ Could not get real Google Trends data.")
        print("    Using synthetic data for demonstration.")
        trends_data = generate_synthetic_trends()
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    plot_trends_analysis(trends_data, FIGURES_DIR / "google_trends_analysis.png")
    
    # Compare with satellite data if available
    satellite_panel = OUTPUT_DIR / "panel_data.csv"
    if satellite_panel.exists():
        plot_trends_vs_satellite(trends_data, satellite_panel, 
                                 FIGURES_DIR / "trends_vs_satellite.png")
    
    # Save trends data
    trends_data.to_csv(TABLES_DIR / "google_trends_proxy.csv")
    print(f"  ✓ Saved: google_trends_proxy.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    print("""
  GOOGLE TRENDS AS SUPPLEMENTARY INDICATOR
  ════════════════════════════════════════
  
  WHAT IT CAN SHOW:
  • Relative changes in search interest over time
  • Spikes during displacement events
  • Recovery patterns (reconstruction searches)
  • Population information-seeking behavior
  
  RECOMMENDED SEARCHES TO TRACK:
  • "gaza reconstruction" - recovery indicator
  • "UNRWA" - humanitarian need
  • "gaza shelter" - displacement indicator
  • "jobs gaza" - economic recovery
  
  HOW TO USE:
  1. Download from trends.google.com manually
  2. Compare trends with satellite indicators
  3. Use for narrative support, not primary analysis
  4. Always note limitations (internet access, bias)
  
  CAVEATS FOR GAZA:
  • Internet connectivity severely disrupted
  • Power outages limit access
  • ~40-50% had internet pre-conflict
  • Current access likely much lower
  • Data may reflect diaspora searches
    """)
    
    print("=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
