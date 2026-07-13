#!/usr/bin/env python3
"""
pipeline/donor_data_download.py
===============================
Download data for SC donor countries (Jordan, Lebanon, Egypt)

This script downloads:
1. WFP food prices (for price outcomes)
2. Google Trends (for GT outcomes)  
3. ACLED conflict data (for conflict outcomes)

Enables Synthetic Control analysis for ALL outcomes, not just satellite data.

Usage:
    python -m pipeline.donor_data_download --data wfp
    python -m pipeline.donor_data_download --data trends
    python -m pipeline.donor_data_download --data acled
    python -m pipeline.donor_data_download --data all
    
    Or via Makefile:
    make donors
"""

import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("data/synthetic_control")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Donor countries
DONOR_COUNTRIES = {
    'JOR': {'name': 'Jordan', 'iso3': 'JOR', 'iso2': 'JO', 'wfp_id': 112},
    'LBN': {'name': 'Lebanon', 'iso3': 'LBN', 'iso2': 'LB', 'wfp_id': 124},
    'EGY': {'name': 'Egypt', 'iso3': 'EGY', 'iso2': 'EG', 'wfp_id': 59},
}

# Date range
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Commodities mapping
COMMODITIES = {
    'Bread': 'price_bread',
    'Wheat flour': 'price_flour',
    'Flour (wheat)': 'price_flour',
    'Rice': 'price_rice',
    'Milk': 'price_milk',
    'Oil (vegetable)': 'price_oil',
    'Oil': 'price_oil',
    'Sugar': 'price_sugar',
    'Meat (chicken)': 'price_chicken',
    'Chicken': 'price_chicken',
}

# Google Trends keywords (matching Gaza panel)
GT_KEYWORDS = [
    'food', 'water', 'hospital', 'help', 'electricity',
    'shelter', 'medicine', 'bread', 'flour'
]


# =============================================================================
# WFP DOWNLOAD
# =============================================================================
def download_wfp_prices():
    """Download WFP food prices for donor countries."""
    print("\n" + "=" * 70)
    print("WFP FOOD PRICES - DONOR COUNTRIES")
    print("=" * 70)
    
    all_data = []
    
    for country_code, info in DONOR_COUNTRIES.items():
        print(f"\n--- {info['name']} ({country_code}) ---")
        
        # Try HDX (Humanitarian Data Exchange) - most reliable source
        df = download_hdx_wfp(country_code, info['name'])
        
        if df is not None:
            processed = process_wfp_data(df, country_code)
            if processed is not None and len(processed) > 0:
                all_data.append(processed)
                print(f"  ✓ {len(processed)} monthly records")
        else:
            print(f"  ⚠ No data available")
        
        time.sleep(1)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = OUTPUT_DIR / "donor_wfp_prices.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file} ({len(combined)} records)")
        return combined
    else:
        print("\n⚠ No WFP data retrieved")
        print("  Manual download: https://data.humdata.org/dataset?q=wfp+food+prices")
        return None


def download_hdx_wfp(country_code, country_name):
    """Download WFP data from HDX."""
    try:
        # Search HDX for WFP food prices
        search_url = "https://data.humdata.org/api/3/action/package_search"
        params = {
            'q': f'wfp food prices {country_name}',
            'rows': 5
        }
        
        response = requests.get(search_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success') and data.get('result', {}).get('results'):
                for dataset in data['result']['results']:
                    # Look for CSV resources
                    for resource in dataset.get('resources', []):
                        if resource.get('format', '').upper() == 'CSV':
                            csv_url = resource.get('url')
                            if csv_url and 'wfp' in csv_url.lower():
                                print(f"  Downloading from HDX...")
                                csv_response = requests.get(csv_url, timeout=120)
                                if csv_response.status_code == 200:
                                    from io import StringIO
                                    df = pd.read_csv(StringIO(csv_response.text))
                                    return df
    except Exception as e:
        print(f"  HDX error: {e}")
    
    return None


def process_wfp_data(df, country_code):
    """Process WFP data into standardized format."""
    if df is None or len(df) == 0:
        return None
    
    # Common WFP column names
    date_col = None
    for col in ['date', 'Date', 'mp_year', 'year']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        # Try to find date from year/month columns
        if 'mp_year' in df.columns and 'mp_month' in df.columns:
            df['period_start'] = pd.to_datetime(
                df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str).str.zfill(2) + '-01'
            )
        elif 'year' in df.columns and 'month' in df.columns:
            df['period_start'] = pd.to_datetime(
                df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
            )
        else:
            print(f"    ⚠ No date column found. Columns: {list(df.columns)[:5]}")
            return None
    else:
        df['period_start'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Find commodity and price columns
    commodity_col = None
    for col in ['cm_name', 'commodity', 'Commodity', 'item', 'cm_name']:
        if col in df.columns:
            commodity_col = col
            break
    
    price_col = None
    for col in ['mp_price', 'price', 'Price', 'value']:
        if col in df.columns:
            price_col = col
            break
    
    if commodity_col is None or price_col is None:
        print(f"    ⚠ Missing columns. Have: {list(df.columns)[:8]}")
        return None
    
    # Filter to date range
    df = df[df['period_start'] >= START_DATE]
    
    # Aggregate to monthly means by commodity
    df['period_month'] = df['period_start'].dt.to_period('M').dt.to_timestamp()
    
    result_rows = []
    for period in df['period_month'].unique():
        period_df = df[df['period_month'] == period]
        
        row = {
            'period_start': period,
            'admin_id': country_code,
            'country': country_code,
        }
        
        # Match commodities
        for wfp_pattern, panel_name in COMMODITIES.items():
            matches = period_df[
                period_df[commodity_col].str.lower().str.contains(
                    wfp_pattern.lower().split()[0], na=False
                )
            ]
            if len(matches) > 0:
                # Convert to numeric and take mean
                price_values = pd.to_numeric(matches[price_col], errors='coerce')
                if price_values.notna().sum() > 0:
                    row[panel_name] = price_values.mean()
        
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


# =============================================================================
# GOOGLE TRENDS DOWNLOAD
# =============================================================================
def download_google_trends():
    """Download Google Trends data for donor countries."""
    print("\n" + "=" * 70)
    print("GOOGLE TRENDS - DONOR COUNTRIES")
    print("=" * 70)
    
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("\n⚠ pytrends not installed. Install with:")
        print("  pip install pytrends --break-system-packages")
        return None
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    all_data = []
    
    for country_code, info in DONOR_COUNTRIES.items():
        print(f"\n--- {info['name']} ({country_code}) ---")
        
        country_data = []
        
        for keyword in GT_KEYWORDS:
            print(f"  {keyword}...", end=" ")
            
            try:
                # Build payload
                pytrends.build_payload(
                    [keyword],
                    cat=0,
                    timeframe=f'{START_DATE} {END_DATE}',
                    geo=info['iso2'],
                    gprop=''
                )
                
                # Get interest over time
                df = pytrends.interest_over_time()
                
                if df is not None and len(df) > 0:
                    df = df.reset_index()
                    df['keyword'] = keyword
                    df['country'] = country_code
                    df = df.rename(columns={keyword: 'value', 'date': 'period_start'})
                    country_data.append(df[['period_start', 'country', 'keyword', 'value']])
                    print(f"✓ {len(df)} records")
                else:
                    print("no data")
                
                time.sleep(2)  # Rate limiting - Google is strict
                
            except Exception as e:
                print(f"error: {str(e)[:30]}")
                time.sleep(5)
        
        if country_data:
            country_df = pd.concat(country_data, ignore_index=True)
            all_data.append(country_df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Pivot to wide format
        pivot = combined.pivot_table(
            index=['period_start', 'country'],
            columns='keyword',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Rename columns to match panel
        pivot.columns = [f'gt_{c}' if c not in ['period_start', 'country'] else c 
                        for c in pivot.columns]
        pivot['admin_id'] = pivot['country']
        
        # Resample to monthly - only numeric columns
        pivot['period_start'] = pd.to_datetime(pivot['period_start'])
        numeric_cols = pivot.select_dtypes(include=[np.number]).columns.tolist()
        
        # Group by country and resample
        resampled_dfs = []
        for country in pivot['country'].unique():
            country_df = pivot[pivot['country'] == country].copy()
            country_df = country_df.set_index('period_start')
            country_resampled = country_df[numeric_cols].resample('MS').mean().reset_index()
            country_resampled['country'] = country
            country_resampled['admin_id'] = country
            resampled_dfs.append(country_resampled)
        
        pivot = pd.concat(resampled_dfs, ignore_index=True)
        
        output_file = OUTPUT_DIR / "donor_google_trends.csv"
        pivot.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file} ({len(pivot)} records)")
        return pivot
    else:
        print("\n⚠ No Google Trends data retrieved")
        return None


# =============================================================================
# ACLED DOWNLOAD
# =============================================================================
def download_acled_conflict():
    """Download ACLED conflict data for donor countries."""
    print("\n" + "=" * 70)
    print("ACLED CONFLICT DATA - DONOR COUNTRIES")
    print("=" * 70)
    
    # Check for credentials
    email = os.environ.get('ACLED_EMAIL')
    key = os.environ.get('ACLED_KEY') or os.environ.get('ACLED_PASSWORD')
    
    if not email or not key:
        print("\n⚠ ACLED credentials not found in environment variables")
        print("  Set ACLED_EMAIL and ACLED_KEY, then re-run")
        print("  Or download manually from: https://acleddata.com/data-export-tool/")
        return None
    
    all_data = []
    
    for country_code, info in DONOR_COUNTRIES.items():
        print(f"\n--- {info['name']} ({country_code}) ---")
        
        try:
            # ACLED API endpoint
            url = "https://api.acleddata.com/acled/read"
            
            params = {
                'key': key,
                'email': email,
                'country': info['name'],
                'event_date': f'{START_DATE}|{END_DATE}',
                'event_date_where': 'BETWEEN',
                'limit': 0  # No limit
            }
            
            response = requests.get(url, params=params, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success') and data.get('data'):
                    df = pd.DataFrame(data['data'])
                    print(f"  ✓ {len(df)} events")
                    
                    # Process to monthly aggregates
                    df['event_date'] = pd.to_datetime(df['event_date'])
                    df['period_start'] = df['event_date'].dt.to_period('M').dt.to_timestamp()
                    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce')
                    
                    monthly = df.groupby('period_start').agg({
                        'event_id_cnty': 'count',
                        'fatalities': 'sum'
                    }).reset_index()
                    
                    monthly.columns = ['period_start', 'conflict_events', 'conflict_fatalities']
                    monthly['country'] = country_code
                    monthly['admin_id'] = country_code
                    
                    all_data.append(monthly)
                else:
                    print(f"  ⚠ No data or API error")
            else:
                print(f"  ⚠ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        
        time.sleep(1)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = OUTPUT_DIR / "donor_acled_conflict.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✓ Saved: {output_file} ({len(combined)} records)")
        return combined
    else:
        print("\n⚠ No ACLED data retrieved")
        return None


# =============================================================================
# COMBINE ALL DONOR DATA
# =============================================================================
def combine_donor_data():
    """Combine all donor data sources into unified donor pool."""
    print("\n" + "=" * 70)
    print("COMBINING DONOR DATA")
    print("=" * 70)
    
    # Load existing satellite donor pool
    satellite_file = OUTPUT_DIR / "donor_pool_combined.csv"
    
    if satellite_file.exists():
        satellite = pd.read_csv(satellite_file)
        satellite['period_start'] = pd.to_datetime(satellite['period_start'])
        print(f"  Satellite data: {len(satellite)} records")
    else:
        print("  ⚠ No satellite donor pool found")
        satellite = None
    
    # Load WFP prices
    wfp_file = OUTPUT_DIR / "donor_wfp_prices.csv"
    if wfp_file.exists():
        wfp = pd.read_csv(wfp_file)
        wfp['period_start'] = pd.to_datetime(wfp['period_start'])
        print(f"  WFP prices: {len(wfp)} records")
    else:
        wfp = None
        print("  ⚠ No WFP donor data")
    
    # Load Google Trends
    gt_file = OUTPUT_DIR / "donor_google_trends.csv"
    if gt_file.exists():
        gt = pd.read_csv(gt_file)
        gt['period_start'] = pd.to_datetime(gt['period_start'])
        print(f"  Google Trends: {len(gt)} records")
    else:
        gt = None
        print("  ⚠ No Google Trends donor data")
    
    # Load ACLED
    acled_file = OUTPUT_DIR / "donor_acled_conflict.csv"
    if acled_file.exists():
        acled = pd.read_csv(acled_file)
        acled['period_start'] = pd.to_datetime(acled['period_start'])
        print(f"  ACLED conflict: {len(acled)} records")
    else:
        acled = None
        print("  ⚠ No ACLED donor data")
    
    # Start with satellite or create empty frame
    if satellite is not None:
        combined = satellite.copy()
    else:
        # Create base from available data
        all_periods = set()
        all_countries = set()
        
        for df in [wfp, gt, acled]:
            if df is not None:
                all_periods.update(df['period_start'].unique())
                if 'country' in df.columns:
                    all_countries.update(df['country'].unique())
                elif 'admin_id' in df.columns:
                    all_countries.update(df['admin_id'].unique())
        
        if not all_periods or not all_countries:
            print("\n⚠ No donor data available to combine")
            return None
        
        # Create base dataframe
        rows = []
        for country in all_countries:
            for period in all_periods:
                rows.append({
                    'period_start': period,
                    'admin_id': country,
                    'country': country
                })
        combined = pd.DataFrame(rows)
    
    # Merge WFP prices
    if wfp is not None:
        price_cols = [c for c in wfp.columns if c.startswith('price_')]
        if price_cols:
            merge_cols = ['period_start', 'admin_id'] if 'admin_id' in wfp.columns else ['period_start', 'country']
            combined = combined.merge(
                wfp[merge_cols + price_cols],
                on=['period_start'],
                how='left',
                suffixes=('', '_wfp')
            )
    
    # Merge Google Trends
    if gt is not None:
        gt_cols = [c for c in gt.columns if c.startswith('gt_')]
        if gt_cols:
            combined = combined.merge(
                gt[['period_start', 'country'] + gt_cols],
                on=['period_start', 'country'],
                how='left',
                suffixes=('', '_gt')
            )
    
    # Merge ACLED
    if acled is not None:
        conflict_cols = [c for c in acled.columns if 'conflict' in c]
        if conflict_cols:
            combined = combined.merge(
                acled[['period_start', 'country'] + conflict_cols],
                on=['period_start', 'country'],
                how='left',
                suffixes=('', '_acled')
            )
    
    # Save combined
    output_file = OUTPUT_DIR / "donor_pool_full.csv"
    combined.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Records: {len(combined)}")
    print(f"  Columns: {len(combined.columns)}")
    print(f"  Date range: {combined['period_start'].min()} to {combined['period_start'].max()}")
    
    # Show coverage
    print("\nColumn coverage:")
    for col in combined.columns:
        if col not in ['period_start', 'admin_id', 'country']:
            coverage = combined[col].notna().sum() / len(combined) * 100
            print(f"  {col}: {coverage:.0f}%")
    
    return combined


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Download donor country data for SC analysis')
    parser.add_argument('--data', choices=['wfp', 'trends', 'acled', 'all', 'combine'],
                       default='all', help='Data source to download')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DONOR DATA DOWNLOAD FOR SYNTHETIC CONTROL")
    print("=" * 70)
    print(f"\nDonor countries: Jordan, Lebanon, Egypt")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    if args.data == 'wfp' or args.data == 'all':
        download_wfp_prices()
    
    if args.data == 'trends' or args.data == 'all':
        download_google_trends()
    
    if args.data == 'acled' or args.data == 'all':
        download_acled_conflict()
    
    if args.data == 'all' or args.data == 'combine':
        combine_donor_data()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python analysis/pse/00_synthetic_control.py")
    print("  2. SC analysis will now include price/GT/conflict outcomes")


if __name__ == "__main__":
    main()
