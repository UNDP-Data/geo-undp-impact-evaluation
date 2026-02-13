#!/usr/bin/env python3
"""
pipeline/undp_projects.py
=========================
Scrape & classify UNDP projects for any country.

Strategy:
  1. Try UNDP Open Portal API (primary - like the working Gaza scraper)
  2. Fall back to local CSV if API fails
  3. Classify projects using TF-IDF + cosine similarity (ML) with keyword fallback

Usage:
    python -m pipeline.undp_projects --config config/myanmar.yaml
    python -m pipeline.undp_projects --config config/gaza.yaml
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import argparse
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ─── API Configuration ───────────────────────────────────────────────────────
SUMMARY_API = "https://api.open.undp.org/api/project_summary_{year}.json"
PROJECT_DETAIL_API = "https://api.open.undp.org/api/projects/{project_id}.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko)"
}
API_DELAY = 0.15  # seconds between requests (be polite)

# ─── Country code mapping (ISO3 → UNDP operating unit) ───────────────────────
COUNTRY_TO_UNIT = {
    'MMR': 'MMR',  # Myanmar
    'PSE': 'PAL',  # Palestine → PAL in UNDP system
    'AFG': 'AFG',  # Afghanistan
    'UKR': 'UKR',  # Ukraine
    'SYR': 'SYR',  # Syria
    'YEM': 'YEM',  # Yemen
    'ETH': 'ETH',  # Ethiopia
    'SDN': 'SDN',  # Sudan
    'SOM': 'SOM',  # Somalia
    'HTI': 'HTI',  # Haiti
    'TUR': 'TUR',  # Türkiye
    'NPL': 'NPL',  # Nepal
    'PAK': 'PAK',  # Pakistan
    'BGD': 'BGD',  # Bangladesh
    'MOZ': 'MOZ',  # Mozambique
}

# ─── ML Classification Definitions (TF-IDF centroids) ────────────────────────
CATEGORY_DEFINITIONS = {
    "Infrastructure & Housing": (
        "infrastructure housing construction rehabilitation buildings roads "
        "shelter reconstruction civil works structural cement bridge repair "
        "building rehabilitation urban planning settlement"
    ),
    "Debris & Solid Waste": (
        "rubble debris solid waste removal clearing cleaning dumping landfill "
        "hazardous demolition clearance cleanup disposal"
    ),
    "Water & Sanitation (WASH)": (
        "water sanitation hygiene sewage wastewater treatment desalination "
        "wells aquifer pipes potable latrine drainage"
    ),
    "Energy & Power": (
        "energy solar electricity power renewable grid photovoltaic panel "
        "fuel generator battery transmission"
    ),
    "Health & Emergency": (
        "health medical hospital clinic medicine doctor nurse psychosocial "
        "trauma emergency mental surgery ambulance pandemic response disaster "
        "earthquake flood cyclone typhoon hurricane crisis relief recovery "
        "humanitarian aid rescue early warning preparedness"
    ),
    "Education": (
        "education school students teachers university learning scholarship "
        "training vocational academic curriculum classroom"
    ),
    "Livelihoods & Economy": (
        "cash employment jobs business microfinance economic market livelihood "
        "grants msme private sector salary agriculture farming income"
    ),
    "Governance & Rule of Law": (
        "governance justice human rights democracy rule of law civil society "
        "legal institution policy parliament peace security accountability "
        "election capacity building"
    ),
    "Environment": (
        "environment climate change biodiversity nature conservation pollution "
        "green ecosystem forest adaptation mitigation"
    ),
    "Social Inclusion": (
        "youth gender women disability empowerment community social cohesion "
        "culture sports refugee displaced idp protection inclusion vulnerable"
    ),
}

# ─── Keyword fallback (used when sklearn not available or few projects) ───────
CATEGORY_KEYWORDS = {
    'Infrastructure & Housing': [
        'infrastructure', 'housing', 'shelter', 'building', 'construction',
        'rehabilitation', 'reconstruction', 'road', 'bridge'],
    'Debris & Solid Waste': [
        'debris', 'waste', 'rubble', 'clearance', 'cleanup', 'disposal'],
    'Water & Sanitation (WASH)': [
        'water', 'sanitation', 'wash', 'hygiene', 'sewage', 'drainage', 'latrine'],
    'Energy & Power': [
        'energy', 'power', 'electricity', 'solar', 'renewable', 'generator', 'grid'],
    'Health & Emergency': [
        'health', 'medical', 'hospital', 'emergency', 'covid', 'pandemic',
        'clinic', 'ambulance', 'response', 'disaster', 'earthquake', 'flood',
        'cyclone', 'crisis', 'relief', 'recovery', 'humanitarian', 'rescue'],
    'Education': [
        'education', 'school', 'training', 'learning', 'teacher', 'student',
        'scholarship'],
    'Livelihoods & Economy': [
        'livelihood', 'economic', 'employment', 'business', 'enterprise',
        'market', 'income', 'agriculture', 'farming'],
    'Governance & Rule of Law': [
        'governance', 'rule of law', 'justice', 'institution', 'democracy',
        'election', 'parliament', 'capacity building'],
    'Environment': [
        'environment', 'climate', 'forest', 'biodiversity', 'green',
        'conservation', 'ecosystem'],
    'Social Inclusion': [
        'gender', 'women', 'youth', 'disability', 'refugee', 'displaced',
        'idp', 'protection', 'inclusion', 'vulnerable'],
}


# =============================================================================
#  Classification
# =============================================================================

def classify_keyword(title, desc=''):
    """Fallback: classify using simple keyword matching."""
    text = f"{title} {desc}".lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    return max(scores, key=scores.get) if any(scores.values()) else 'Other / Unspecified'


def classify_ml(df, text_col='text_blob'):
    """ML classification using TF-IDF + cosine similarity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  ⚠ sklearn not installed, using keyword fallback")
        return df.apply(
            lambda x: classify_keyword(
                str(x.get('title', '')), str(x.get('description', ''))),
            axis=1
        )

    cat_names = list(CATEGORY_DEFINITIONS.keys())
    cat_texts = [CATEGORY_DEFINITIONS[c] for c in cat_names]
    project_texts = df[text_col].fillna('').astype(str).tolist()

    all_texts = cat_texts + project_texts
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_texts)

    cat_vectors = tfidf_matrix[:len(cat_names)]
    proj_vectors = tfidf_matrix[len(cat_names):]
    sim_matrix = cosine_similarity(proj_vectors, cat_vectors)

    categories = []
    for i in range(len(df)):
        sims = sim_matrix[i]
        best_idx = np.argmax(sims)
        if sims[best_idx] > 0.05:
            categories.append(cat_names[best_idx])
        else:
            categories.append('Other / Unspecified')
    return pd.Series(categories, index=df.index)


# =============================================================================
#  API Scraping (from the working Gaza scraper)
# =============================================================================

def scrape_undp_api(operating_unit, years, verbose=True):
    """
    Scrape UNDP Open Portal API for projects.
    
    This replicates the approach from the working Gaza scraper:
    1. Get project summaries by year
    2. Fetch detailed info for each project
    """
    all_projects = {}
    
    if verbose:
        print(f"\n  Scraping UNDP API for unit: {operating_unit}")
        print(f"  Years: {years}")
    
    for year in years:
        if verbose:
            print(f"  → Scanning {year}...", end=" ", flush=True)
        
        try:
            resp = requests.get(
                SUMMARY_API.format(year=year),
                headers=HEADERS,
                timeout=30
            )
            if resp.status_code != 200:
                if verbose:
                    print(f"HTTP {resp.status_code}")
                continue
            
            data = resp.json()
            target = [
                p for p in data
                if p.get('operating_unit') == operating_unit
            ]
            
            if verbose:
                print(f"found {len(target)} projects", end="", flush=True)
            
            new_count = 0
            for proj in target:
                p_id = proj.get('project_id') or proj.get('id')
                if not p_id or p_id in all_projects:
                    continue
                
                time.sleep(API_DELAY)
                try:
                    d_resp = requests.get(
                        PROJECT_DETAIL_API.format(project_id=p_id),
                        headers=HEADERS,
                        timeout=15
                    )
                    if d_resp.status_code != 200:
                        continue
                    
                    details = d_resp.json()
                    
                    # Extract title
                    title = (details.get('title')
                             or details.get('project_title')
                             or "No Title")
                    
                    # Extract description (including outputs)
                    desc_parts = [
                        details.get('description', '')
                        or details.get('project_descr', '')
                    ]
                    for out in details.get('outputs', []):
                        t = out.get('title', '')
                        d = out.get('descr', '')
                        if t or d:
                            desc_parts.append(f"[Output: {t} - {d}]")
                    full_text = " || ".join([str(d) for d in desc_parts if d])
                    
                    # Extract dates
                    start = details.get('start', '')
                    end = details.get('end', '')
                    
                    # Umbrella check (projects > 5 years old)
                    is_umbrella = False
                    if start:
                        try:
                            days = (datetime.now() 
                                    - datetime.strptime(start, '%Y-%m-%d')).days
                            if days > 1825:
                                is_umbrella = True
                        except (ValueError, TypeError):
                            pass
                    
                    all_projects[p_id] = {
                        'project_id': p_id,
                        'title': title,
                        'start_date': start,
                        'end_date': end,
                        'total_budget': details.get('budget', 0),
                        'is_likely_umbrella': is_umbrella,
                        'url': f"https://open.undp.org/projects/{p_id}",
                        'description': full_text,
                    }
                    new_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"\n    Error fetching {p_id}: {e}")
            
            if verbose:
                print(f" ({new_count} new)")
        
        except requests.exceptions.Timeout:
            if verbose:
                print("TIMEOUT")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
    
    if all_projects:
        df = pd.DataFrame(list(all_projects.values()))
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['total_budget'] = pd.to_numeric(df['total_budget'], errors='coerce')
        return df
    
    return None


# =============================================================================
#  CSV Fallback
# =============================================================================

def load_csv_fallback(data_dir, country_code):
    """Load from local CSV if API fails."""
    candidates = [
        data_dir / "undp_projects_raw.csv",
        data_dir / "undp_projects.csv",
        data_dir / f"{country_code}_projects.csv",
        data_dir / "projects.csv",
        # Also check the UNDP-specific data folder
        Path(f"data/{country_code}/undp") / "project_data.xlsx",
    ]
    
    for f in candidates:
        if f.exists():
            print(f"  → Loading local file: {f}")
            if f.suffix == '.xlsx':
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f)
            
            if len(df) == 0:
                print(f"    ⚠ File exists but has 0 rows, skipping")
                continue
            
            return df, f
    
    return None, None


def standardize_columns(df):
    """Standardize column names from various CSV formats."""
    col_map = {
        'project_title': 'title',
        'name': 'title',
        'project_id': 'project_id',
        'id': 'project_id',
        'start': 'start_date',
        'budget': 'total_budget',
        'project_description': 'description',
        'full_description_text': 'description',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Find title column if missing
    if 'title' not in df.columns:
        for col in df.columns:
            if 'title' in col.lower() or 'name' in col.lower():
                df['title'] = df[col]
                break
        else:
            df['title'] = 'Unknown Project'
    
    # Find/parse start_date
    if 'start_date' not in df.columns:
        for col in df.columns:
            if 'start' in col.lower() or 'date' in col.lower():
                df['start_date'] = pd.to_datetime(df[col], errors='coerce')
                break
    else:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # Parse budget
    if 'total_budget' in df.columns:
        df['total_budget'] = pd.to_numeric(
            df['total_budget'].astype(str).str.replace('[,$]', '', regex=True),
            errors='coerce'
        )
    else:
        df['total_budget'] = 0
    
    # Ensure description
    if 'description' not in df.columns:
        df['description'] = ''
    
    return df


# =============================================================================
#  Main
# =============================================================================

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape & classify UNDP projects for a country"
    )
    parser.add_argument('--config', '-c', required=True,
                        help="Path to country config YAML")
    parser.add_argument('--force-api', action='store_true',
                        help="Force API scraping even if CSV exists")
    parser.add_argument('--force-csv', action='store_true',
                        help="Skip API, only use local CSV")
    parser.add_argument('--years', nargs='+', type=int, default=None,
                        help="Years to scrape (default: auto from config)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    country_name = config['country_name']
    country_code = config['country_code'].upper()
    disaster_date = config['events']['disaster_date']
    disaster_dt = pd.Timestamp(disaster_date)
    
    # Determine years to scrape
    if args.years:
        years = args.years
    else:
        # Scrape from 2 years before disaster to current year
        start_year = disaster_dt.year - 2
        end_year = datetime.now().year
        years = list(range(start_year, end_year + 1))
    
    # UNDP operating unit code
    operating_unit = COUNTRY_TO_UNIT.get(country_code, country_code)
    
    data_dir = Path(f"data/{country_code.lower()}/external")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "undp_projects.csv"
    
    print(f"\n{'='*70}")
    print(f"  UNDP PROJECTS: {country_name} ({country_code})")
    print(f"  Operating Unit: {operating_unit}")
    print(f"  Disaster Date: {disaster_date}")
    print(f"  Years: {years}")
    print(f"{'='*70}")
    
    df = None
    source = None
    
    # ── Strategy 1: API Scraping ──────────────────────────────────────────
    if not args.force_csv:
        print(f"\n[1/2] Trying UNDP API...")
        try:
            df = scrape_undp_api(operating_unit, years)
            if df is not None and len(df) > 0:
                source = 'API'
                print(f"  ✓ API returned {len(df)} projects")
            else:
                print(f"  ⚠ API returned no projects")
                df = None
        except Exception as e:
            print(f"  ✗ API failed: {e}")
            df = None
    
    # ── Strategy 2: CSV Fallback ──────────────────────────────────────────
    if df is None or args.force_csv:
        print(f"\n[2/2] Trying local CSV fallback...")
        df, csv_path = load_csv_fallback(data_dir, country_code.lower())
        if df is not None and len(df) > 0:
            source = f'CSV ({csv_path})'
            print(f"  ✓ Loaded {len(df)} rows from {csv_path}")
            df = standardize_columns(df)
        else:
            print(f"  ✗ No local CSV found")
    
    # ── No data at all ────────────────────────────────────────────────────
    if df is None or len(df) == 0:
        print(f"\n{'='*70}")
        print(f"  ⚠ NO UNDP PROJECT DATA AVAILABLE")
        print(f"{'='*70}")
        print(f"\n  To add UNDP projects manually:")
        print(f"  1. Go to: https://open.undp.org/projects")
        print(f"  2. Filter by Operating Unit: {operating_unit}")
        print(f"  3. Export as CSV")
        print(f"  4. Save to: {data_dir / 'undp_projects_raw.csv'}")
        print(f"  5. Re-run: python -m pipeline.undp_projects -c {args.config}")
        
        # Create empty file so downstream pipeline doesn't break
        empty = pd.DataFrame(columns=[
            'project_id', 'title', 'start_date', 'end_date',
            'total_budget', 'ml_category', 'is_likely_umbrella',
            'is_post_disaster', 'description', 'url'
        ])
        empty.to_csv(output_file, index=False)
        print(f"\n  Created empty placeholder: {output_file}")
        return
    
    # ── Classify projects ─────────────────────────────────────────────────
    print(f"\n  Classifying projects...")
    
    # Build text blob for ML classification
    df['text_blob'] = (
        df['title'].fillna('').astype(str) + " " +
        df['description'].fillna('').astype(str)
    ).str.lower()
    
    # Use ML if enough projects, keyword otherwise
    if len(df) >= 5:
        df['ml_category'] = classify_ml(df, text_col='text_blob')
    else:
        df['ml_category'] = df.apply(
            lambda x: classify_keyword(
                str(x.get('title', '')),
                str(x.get('description', ''))),
            axis=1
        )
    
    # ── Add disaster context ──────────────────────────────────────────────
    if 'start_date' in df.columns:
        df['is_post_disaster'] = df['start_date'] >= disaster_dt
    
    # ── Flag umbrella projects ────────────────────────────────────────────
    if 'is_likely_umbrella' not in df.columns:
        umbrella_kw = [
            'framework', 'umbrella', 'programme support',
            'cost recovery', 'management cost'
        ]
        df['is_likely_umbrella'] = df['title'].str.lower().apply(
            lambda x: any(kw in str(x) for kw in umbrella_kw)
        )
    
    # ── Filter to study period ────────────────────────────────────────────
    # Uses study_period_start from config (covers full analysis window)
    if 'start_date' in df.columns:
        study_start = config.get('study_period_start')
        if study_start:
            cutoff = pd.Timestamp(study_start)
        else:
            # Fallback: 1 year before disaster
            cutoff = disaster_dt - pd.Timedelta(days=365)
        
        n_before = len(df)
        df_filtered = df[
            df['start_date'].isna() | (df['start_date'] >= cutoff)
        ].copy()
        n_after = len(df_filtered)
        if n_before != n_after:
            print(f"  Filtered: {n_before} → {n_after} "
                  f"(removed {n_before - n_after} projects before {cutoff.date()})")
    else:
        df_filtered = df.copy()
    
    # ── Ensure standard output columns ────────────────────────────────────
    output_cols = [
        'project_id', 'title', 'start_date', 'end_date',
        'total_budget', 'ml_category', 'is_likely_umbrella',
        'is_post_disaster', 'description', 'url'
    ]
    for col in output_cols:
        if col not in df_filtered.columns:
            df_filtered[col] = '' if col in ('url', 'description') else None
    
    # ── Save ──────────────────────────────────────────────────────────────
    df_filtered[output_cols].to_csv(output_file, index=False)
    
    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✓ SAVED: {output_file}")
    print(f"  Source: {source}")
    print(f"{'='*70}")
    print(f"  Total projects: {len(df_filtered)}")
    
    if 'start_date' in df_filtered.columns:
        valid = df_filtered['start_date'].notna().sum()
        print(f"  With valid dates: {valid}")
    
    # Category breakdown
    print(f"\n  By category:")
    for cat, count in df_filtered['ml_category'].value_counts().items():
        print(f"    {cat}: {count}")
    
    # Pre/post breakdown
    if 'is_post_disaster' in df_filtered.columns:
        n_post = df_filtered['is_post_disaster'].sum()
        n_pre = len(df_filtered) - n_post
        print(f"\n  Pre-disaster: {n_pre}")
        print(f"  Post-disaster: {n_post}")
    
    # Umbrella projects
    n_umb = df_filtered['is_likely_umbrella'].sum()
    if n_umb:
        print(f"  Umbrella (excluded from viz): {n_umb}")
    
    # Show recent projects
    recent = df_filtered.nlargest(5, 'start_date')[
        ['title', 'start_date', 'ml_category', 'total_budget']
    ]
    if len(recent) > 0:
        print(f"\n  Most recent projects:")
        for _, row in recent.iterrows():
            budget_str = f"${row['total_budget']:,.0f}" if pd.notna(row['total_budget']) and row['total_budget'] > 0 else "N/A"
            print(f"    [{row['start_date'].strftime('%Y-%m-%d') if pd.notna(row['start_date']) else '?'}] "
                  f"{row['title'][:60]} ({row['ml_category']}) [{budget_str}]")


if __name__ == "__main__":
    main()
