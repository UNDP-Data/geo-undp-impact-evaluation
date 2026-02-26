#!/usr/bin/env python3
"""
pipeline/google_trends.py
=========================
Download Google Trends data for disaster/recovery analysis.
Uses universal keywords applicable across different countries/contexts.

Usage:
    python -m pipeline.google_trends --config config/gaza.yaml
    python -m pipeline.google_trends --config config/myanmar.yaml
    
Note: Google Trends aggressively rate-limits. If you get RetryErrors,
      wait 10-15 minutes and try again, or use --slow flag.
"""

import pandas as pd
import time
import random
import warnings
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Fix for pytrends compatibility
import urllib3.util.retry
_old_init = urllib3.util.retry.Retry.__init__
def _new_init(self, *args, **kwargs):
    if 'method_whitelist' in kwargs:
        kwargs['allowed_methods'] = kwargs.pop('method_whitelist')
    _old_init(self, *args, **kwargs)
urllib3.util.retry.Retry.__init__ = _new_init

from pytrends.request import TrendReq

warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# COUNTRY-SPECIFIC KEYWORDS
# =============================================================================
# For best results, use LOCAL LANGUAGE keywords - English terms often have
# low search volume in non-English speaking countries.

COUNTRY_KEYWORDS = {
    'pse': {  # Palestine - Arabic terms
        # Construction & Rebuilding
        'cement': 'اسمنت',
        'construction': 'بناء',
        'building_materials': 'مواد بناء',
        'rebuild': 'إعادة بناء',
        'iron': 'حديد',
        'concrete': 'خرسانة',
        
        # Basic Needs
        'water': 'ماء',
        'food': 'طعام',
        'bread': 'خبز',
        'flour': 'طحين',
        'shelter': 'مأوى',
        'tent': 'خيمة',
        'mattress': 'فرشة',
        'blanket': 'بطانية',
        
        # Economic Activity  
        'jobs': 'وظائف',
        'work': 'عمل',
        'market': 'سوق',
        'price': 'سعر',
        'money': 'مال',
        'salary': 'راتب',
        
        # Safety & Crisis
        'safety': 'أمان',
        'help': 'مساعدة',
        'emergency': 'طوارئ',
        'hospital': 'مستشفى',
        'medicine': 'دواء',
        'ambulance': 'إسعاف',
        'aid': 'إغاثة',
        
        # Migration & Displacement
        'passport': 'جواز سفر',
        'visa': 'تأشيرة',
        'travel': 'سفر',
        'refugee': 'لاجئ',
        'egypt': 'مصر',
        'rafah': 'رفح',
        'crossing': 'معبر',
        
        # Conflict-specific
        'ceasefire': 'وقف إطلاق النار',
        'bombing': 'قصف',
        'martyrs': 'شهداء',
        'destruction': 'دمار',
        'electricity': 'كهرباء',
        'fuel': 'وقود',
        'gas': 'غاز',
    },
    
    'mmr': {  # Myanmar - Burmese terms
        # Disaster
        'earthquake': 'ငလျင်',
        'aftershock': 'ငလျင်ငယ်',
        
        # Relief
        'donation': 'လှူဒါန်း',
        'aid': 'အကူအညီ',
        'relief': 'ထောက်ပံ့ရေး',
        'shelter': 'အမိုးအကာ',
        
        # Construction
        'cement': 'ဘိလပ်မြေ',
        'rebuild': 'ပြန်လည်တည်ဆောက်',
        'construction': 'ဆောက်လုပ်ရေး',
        
        # Basic needs
        'water': 'ရေ',
        'food': 'အစားအစာ',
        'medicine': 'ဆေး',
        
        # Migration
        'passport': 'နိုင်ငံကူးလက်မှတ်',
        'thailand': 'ထိုင်း',
        
        # Mental health
        'anxiety': 'စိုးရိမ်ပူပန်',
        'ptsd': 'စိတ်ဒဏ်ရာ',
    },
    
    'ukr': {  # Ukraine - Ukrainian terms
        'bomb_shelter': 'бомбосховище',
        'evacuation': 'евакуація',
        'humanitarian_aid': 'гуманітарна допомога',
        'electricity': 'електрика',
        'generator': 'генератор',
        'refugee': 'біженець',
        'poland': 'польща',
    }
}

# Fallback English keywords (for countries without local language config)
ENGLISH_KEYWORDS = {
    'cement': 'cement',
    'construction': 'construction', 
    'water': 'water',
    'food': 'food',
    'shelter': 'shelter',
    'jobs': 'jobs',
    'hospital': 'hospital',
    'passport': 'passport',
    'refugee': 'refugee',
    'aid': 'humanitarian aid',
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_geo_code(country_code: str) -> str:
    """Map country code to Google Trends geo code."""
    geo_mapping = {
        'pse': 'PS',  # Palestine
        'mmr': 'MM',  # Myanmar
        'ukr': 'UA',  # Ukraine
        'syr': 'SY',  # Syria
        'yem': 'YE',  # Yemen
        'afg': 'AF',  # Afghanistan
        'eth': 'ET',  # Ethiopia
        'sdn': 'SD',  # Sudan
    }
    return geo_mapping.get(country_code.lower(), country_code.upper()[:2])


def download_trends(config_path: str, slow_mode: bool = False, english_only: bool = False):
    """
    Download Google Trends data for a country.
    
    Args:
        config_path: Path to country config YAML
        slow_mode: If True, use longer delays between requests (helps avoid rate limits)
        english_only: If True, use only English keywords
    """
    # Load config
    config = load_config(config_path)
    country_code = config.get('country_code', 'XXX').lower()
    country_name = config.get('country_name', 'Unknown')
    study_start = config.get('study_period_start', '2023-01-01')
    study_end = config.get('study_period_end', '2025-12-31')
    
    # Check for custom keywords in config
    custom_keywords = config.get('google_trends_keywords', None)
    
    # Output path
    output_dir = Path(f"data/{country_code}/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "google_trends.csv"
    
    print("=" * 60)
    print(f"GOOGLE TRENDS DOWNLOAD: {country_name}")
    print("=" * 60)
    print(f"Period: {study_start} to {study_end}")
    print(f"Output: {output_file}")
    if slow_mode:
        print("Mode: SLOW (longer delays to avoid rate limits)")
    
    # Get geo code
    geo = get_geo_code(country_code)
    print(f"Geo code: {geo}")
    
    # Initialize pytrends with longer timeout
    try:
        pytrends = TrendReq(
            hl='en-US', 
            tz=0, 
            timeout=(15, 45),  # Longer timeout
            retries=2,
            backoff_factor=1.0
        )
    except Exception as e:
        print(f"✗ Failed to initialize pytrends: {e}")
        return
    
    # Select keywords
    if custom_keywords:
        keywords = custom_keywords
        print(f"\nUsing {len(keywords)} custom keywords from config")
    elif english_only:
        keywords = ENGLISH_KEYWORDS
        print(f"\nUsing {len(keywords)} English keywords")
    elif country_code in COUNTRY_KEYWORDS:
        keywords = COUNTRY_KEYWORDS[country_code]
        print(f"\nUsing {len(keywords)} local language keywords for {country_code.upper()}")
    else:
        keywords = ENGLISH_KEYWORDS
        print(f"\nNo local keywords for {country_code}, using {len(keywords)} English keywords")
    
    # Timeframe
    timeframe = f"{study_start} {study_end}"
    print(f"Timeframe: {timeframe}")
    
    # Delays
    min_delay = 8 if slow_mode else 4
    max_delay = 15 if slow_mode else 8
    
    # Download each keyword
    print("\n--- Downloading ---")
    all_data = []
    success_count = 0
    fail_count = 0
    
    # Handle both dict and list formats
    if isinstance(keywords, list):
        keywords = {w: w for w in keywords}
    for key, term in keywords.items():
        print(f"  > {key}: '{term}'...", end=" ", flush=True)
        
        try:
            pytrends.build_payload([term], cat=0, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()
            
            if not data.empty:
                data = data.reset_index()
                data = data.rename(columns={term: 'score', 'date': 'date'})
                data['keyword'] = key
                data['term'] = term
                data = data[['date', 'keyword', 'term', 'score']]
                all_data.append(data)
                print(f"✓ {len(data)} records")
                success_count += 1
            else:
                print("no data (term may have low search volume)")
                fail_count += 1
            
            # Rate limiting - random delay
            delay = random.uniform(min_delay, max_delay)
            time.sleep(delay)
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"✗ {error_type}")
            fail_count += 1
            
            # On error, wait longer before next request
            if 'Retry' in error_type or '429' in str(e):
                print("    (rate limited - waiting 30s)")
                time.sleep(30)
            else:
                time.sleep(10)
    
    # Combine and save
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(output_file, index=False)
        
        print(f"\n{'=' * 60}")
        print(f"✓ SUCCESS: {len(combined)} records saved")
        print(f"  Keywords succeeded: {success_count}/{len(keywords)}")
        print(f"  Keywords failed: {fail_count}/{len(keywords)}")
        print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"  File: {output_file}")
        
        if fail_count > success_count:
            print(f"\n  TIP: Many failures. Try again with --slow flag:")
            print(f"       python -m pipeline.google_trends --config {config_path} --slow")
        
        print(f"{'=' * 60}")
    else:
        print("\n✗ No data retrieved")
        print("  This usually means rate limiting. Wait 15 minutes and try:")
        print(f"  python -m pipeline.google_trends --config {config_path} --slow")


def main():
    parser = argparse.ArgumentParser(description='Download Google Trends data')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    parser.add_argument('--slow', action='store_true', 
                        help='Use longer delays between requests (helps avoid rate limits)')
    parser.add_argument('--english-only', action='store_true', 
                        help='Use only English keywords (skip local language)')
    
    args = parser.parse_args()
    
    download_trends(args.config, slow_mode=args.slow, english_only=args.english_only)


if __name__ == "__main__":
    main()
