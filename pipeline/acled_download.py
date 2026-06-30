#!/usr/bin/env python3
"""
pipeline/acled_download.py
==========================
Download ACLED conflict data using OAuth authentication.
Works with yearly chunks to handle large datasets.

Usage:
    python -m pipeline.acled_download --config config/gaza.yaml
    python -m pipeline.acled_download --config config/myanmar.yaml
    
Environment variables:
    ACLED_EMAIL - Your ACLED account email
    ACLED_PASSWORD - Your ACLED account password
"""

import pandas as pd
import requests
from pathlib import Path
import time
import os
import urllib3
import yaml
import argparse
from datetime import datetime

# Disable SSL warnings for troubleshooting environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "https://acleddata.com/api/acled/read"
TOKEN_URL = "https://acleddata.com/oauth/token"

# Default credentials from environment
DEFAULT_EMAIL = os.environ.get('ACLED_EMAIL', '')
DEFAULT_PASSWORD = os.environ.get('ACLED_PASSWORD', '')


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_access_token(email: str, password: str):
    """Requests an OAuth access token valid for 24 hours."""
    print("--- Authenticating with ACLED ---")
    payload = {
        'username': email,
        'password': password,
        'grant_type': 'password',
        'client_id': 'acled',
        'scope': 'authenticated'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(TOKEN_URL, headers=headers, data=payload, timeout=30)
        if response.status_code == 200:
            token = response.json().get('access_token')
            print("  ✓ Authentication successful")
            return token
        else:
            print(f"  ✗ Auth Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"  ✗ Connection error during auth: {e}")
        return None


def download_chunk(token: str, country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads a data chunk using the Bearer token in headers."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    params = {
        'country': country,
        'event_date': f"{start_date}|{end_date}",
        'event_date_where': 'BETWEEN',
        'limit': 0,
        '_format': 'json'
    }
    
    print(f"  {start_date} to {end_date}...", end=" ", flush=True)
    
    try:
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=120)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                df = pd.DataFrame(data['data'])
                print(f"✓ {len(df)} events")
                return df
            else:
                print("✓ No events found")
                return pd.DataFrame()
        else:
            print(f"✗ HTTP {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Download ACLED conflict data')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--email', help='ACLED email (overrides env var)')
    parser.add_argument('--password', help='ACLED password (overrides env var)')
    parser.add_argument('--start-year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end-year', type=int, default=2026, help='End year (default: 2026)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    country_name = config.get('country_name', 'Unknown')
    country_code = config.get('country_code', 'XXX').lower()
    acled_country = config.get('acled_country_name', country_name.split(' - ')[0])
    
    # Output path
    output_file = Path(f"data/{country_code}/external/acled_conflict.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"ACLED OAUTH DOWNLOADER: {acled_country}")
    print("="*60)
    print(f"Output: {output_file}")
    
    # Get credentials
    email = args.email or DEFAULT_EMAIL
    password = args.password or DEFAULT_PASSWORD
    
    if not email:
        email = input("Enter your ACLED email: ").strip()
    if not password:
        password = input("Enter your ACLED password: ").strip()
    
    # Get token
    token = get_access_token(email, password)
    if not token:
        print("!! Could not obtain access token. Check credentials or ACLED server status.")
        return
    
    # Build date ranges
    date_ranges = []
    current_year = datetime.now().year
    today = datetime.now().strftime('%Y-%m-%d')
    
    for year in range(args.start_year, args.end_year + 1):
        if year > current_year:
            continue  # Skip future years
        elif year == current_year:
            # Only get current year data up to today
            date_ranges.append((f"{year}-01-01", today))
        else:
            date_ranges.append((f"{year}-01-01", f"{year}-12-31"))
    
    # Download data
    print("\n--- Downloading Data ---")
    all_chunks = []
    for start, end in date_ranges:
        df = download_chunk(token, acled_country, start, end)
        if not df.empty:
            all_chunks.append(df)
        time.sleep(2)  # Prevent rate limiting
    
    # Process and save
    if all_chunks:
        combined = pd.concat(all_chunks, ignore_index=True)
        
        # Deduplicate
        if 'event_id_cnty' in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset=['event_id_cnty'])
            if before > len(combined):
                print(f"\nRemoved {before - len(combined)} duplicate records.")
        
        # Sort and format
        if 'event_date' in combined.columns:
            combined['event_date'] = pd.to_datetime(combined['event_date'])
            combined = combined.sort_values('event_date')
        
        combined.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: {len(combined)} total events saved")
        print(f"  File: {output_file}")
        if 'event_date' in combined.columns:
            print(f"  Date range: {combined['event_date'].min().date()} to {combined['event_date'].max().date()}")
        print(f"{'='*60}")
    else:
        print("\n!! No data was retrieved. Please check your account permissions.")


if __name__ == "__main__":
    main()
