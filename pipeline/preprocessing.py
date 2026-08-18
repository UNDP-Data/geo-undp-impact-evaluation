#!/usr/bin/env python3
"""
pipeline/preprocessing.py
=========================
Generic panel data builder for impact evaluation.
Works with any country by using config-driven name matching.

This module:
1. Loads all GEE satellite data parts
2. Merges ACLED conflict data (with fuzzy name matching)
3. Merges WFP price data (with fuzzy name matching)
4. Creates derived variables (log transforms, time indicators)
5. Saves the final panel dataset

Usage:
    python -m pipeline.preprocessing --config config/gaza.yaml
    python -m pipeline.preprocessing --config config/myanmar.yaml
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import yaml
import argparse
import re
import sys

# Try to import unidecode, fall back to simple lowercase
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s):
        return s

# Try to import from schema
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.schema import load_config
    USE_SCHEMA = True
except ImportError:
    USE_SCHEMA = False


def load_config_simple(config_path: str) -> dict:
    """Simple YAML loader as fallback."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_name(name: str) -> str:
    """Normalize admin names for fuzzy matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower()
    name = unidecode(name)  # Remove accents
    name = re.sub(r'[^a-z0-9]', '', name)  # Remove non-alphanumeric
    return name


# Common name variations by country
# Add more as needed for new countries
NAME_ALIASES = {
    # Gaza / Palestine - WFP data variations
    'khanyunis': 'khanyounis',
    'kanyounis': 'khanyounis',     # WFP uses "Kan Younis"
    'kanyunis': 'khanyounis',
    'khanyounis': 'khanyounis',
    'gazacity': 'gaza',
    'gazastrip': 'gaza',
    'deirelbalah': 'deiralbalah',
    'deiralbalah': 'deiralbalah',
    'deirelbalah': 'deiralbalah',
    'middlearea': 'deiralbalah',
    'northgaza': 'northgaza',
    'gazanorth': 'northgaza',
    'alquds': 'jerusalem',
    'ramallahandalbireh': 'ramallah',
    'ramallah': 'ramallahandalbireh',
    
    # Myanmar (add as needed)
    'yangoneast': 'yangon',
    'yangonwest': 'yangon',
    'yangonnorth': 'yangon',
    'yangonsouth': 'yangon',
    
    # Generic variations
    'north': '',
    'south': '',
    'east': '',
    'west': '',
    'central': '',
}


class PanelDataBuilder:
    """Build panel data from satellite and external sources."""
    
    def __init__(self, config_path: str):
        """Initialize with country configuration."""
        
        # Load config
        if USE_SCHEMA:
            config_obj = load_config(config_path)
            self.config = {
                'country_name': config_obj.country_name,
                'country_code': config_obj.country_code.lower(),
                'study_period_start': config_obj.study_period_start,
                'study_period_end': config_obj.study_period_end,
                'boundaries': {
                    'file_path': config_obj.boundaries.file_path,
                    'id_column': config_obj.boundaries.id_column,
                    'name_column': config_obj.boundaries.name_column,
                },
                'events': {
                    'disaster_date': config_obj.events.disaster_date,
                    'intervention_start': config_obj.events.intervention_start,
                },
                'wfp_commodity': getattr(config_obj, 'wfp_commodity', 'Bread'),
            }
        else:
            self.config = load_config_simple(config_path)
            self.config['country_code'] = self.config.get('country_code', 'XXX').lower()
        
        self.country_code = self.config['country_code']
        self.country_name = self.config.get('country_name', 'Unknown')
        
        # Key dates
        self.disaster_date = pd.Timestamp(self.config['events']['disaster_date'])
        self.intervention_date = pd.Timestamp(self.config['events']['intervention_start'])
        
        # Paths
        self.data_dir = Path(f"data/{self.country_code}")
        self.output_dir = Path(f"outputs/{self.country_code}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load boundaries
        boundary_config = self.config.get('boundaries', {})
        boundary_path = Path(boundary_config.get('file_path', ''))
        
        if boundary_path.exists():
            self.gdf = gpd.read_file(boundary_path)
            self.id_col = boundary_config.get('id_column', 'admin_id')
            self.name_col = boundary_config.get('name_column', 'admin_name')
        else:
            raise FileNotFoundError(f"Boundaries not found: {boundary_path}")
        
        # Build name-to-code mapping
        self.name_to_code = {}
        self.code_to_name = {}
        for _, row in self.gdf.iterrows():
            code = str(row[self.id_col])
            name = str(row[self.name_col])
            cleaned = clean_name(name)
            self.name_to_code[cleaned] = code
            self.code_to_name[code] = name
        
        print(f"\n{'='*60}")
        print(f"PANEL DATA BUILDER: {self.country_name}")
        print(f"{'='*60}")
        print(f"Disaster: {self.disaster_date.strftime('%Y-%m-%d')}")
        print(f"Intervention: {self.intervention_date.strftime('%Y-%m-%d')}")
        print(f"Admin units: {len(self.name_to_code)}")
    
    def _match_admin_name(self, name: str) -> str:
        """Match an admin name to a unit code using fuzzy matching."""
        if pd.isna(name):
            return None
        
        cleaned = clean_name(name)
        
        # Direct match
        if cleaned in self.name_to_code:
            return self.name_to_code[cleaned]
        
        # Try aliases
        if cleaned in NAME_ALIASES:
            aliased = NAME_ALIASES[cleaned]
            if aliased in self.name_to_code:
                return self.name_to_code[aliased]
        
        # Try partial match (name contains or is contained by)
        for ref_name, code in self.name_to_code.items():
            if cleaned in ref_name or ref_name in cleaned:
                return code
        
        return None
    
    def build(self) -> pd.DataFrame:
        """Build the complete panel dataset."""
        
        # ==========================================================================
        # LOAD GEE DATA
        # ==========================================================================
        print("\n--- Loading Satellite Data ---")
        
        gee_files = sorted(self.data_dir.glob("gee_panel_*.csv"))
        
        if not gee_files:
            raise FileNotFoundError(f"No GEE files found in {self.data_dir}")
        
        print(f"  Found {len(gee_files)} GEE files")
        
        dfs = []
        for f in gee_files:
            df = pd.read_csv(f)
            dfs.append(df)
        
        panel = pd.concat(dfs, ignore_index=True)
        panel['period_start'] = pd.to_datetime(panel['period_start'])
        
        # Standardize column names
        if 'admin_id' in panel.columns:
            panel = panel.rename(columns={'admin_id': 'unit_id', 'admin_name': 'unit_name'})
        
        # Deduplicate (batches may overlap)
        panel = panel.drop_duplicates(subset=['unit_id', 'period_start'], keep='first')
        panel = panel.sort_values(['unit_id', 'period_start']).reset_index(drop=True)
        
        # Add clean name for matching
        panel['name_clean'] = panel['unit_name'].apply(clean_name)
        
        print(f"  Loaded {len(panel)} rows")
        print(f"  Period: {panel['period_start'].min().strftime('%Y-%m-%d')} to {panel['period_start'].max().strftime('%Y-%m-%d')}")
        print(f"  Units: {panel['unit_id'].nunique()}")
        
        # ==========================================================================
        # TAG PARENT ADMIN1 REGION (for dashboard admin-level filters)
        # ==========================================================================
        try:
            adm1_path = self.data_dir / "boundaries" / f"{self.country_code.lower()}_admin1.geojson"
            b_path = Path(self.config.get('boundaries', {}).get('file_path', ''))
            if adm1_path.exists() and b_path.exists():
                import geopandas as gpd
                units = gpd.read_file(b_path)
                id_col = self.config['boundaries'].get('id_column', 'shapeID')
                units = units[[id_col, 'geometry']].copy()
                units['geometry'] = units.geometry.centroid
                adm1 = gpd.read_file(adm1_path)[['shapeName', 'geometry']].rename(
                    columns={'shapeName': 'admin1_name'})
                joined = gpd.sjoin(units, adm1, how='left', predicate='within')
                lut = dict(zip(joined[id_col].astype(str), joined['admin1_name']))
                panel['admin1_name'] = panel['unit_id'].astype(str).map(lut)
                print(f"  ✓ Tagged admin1 regions ({panel['admin1_name'].nunique()} states)")
        except Exception as _e:
            print(f"  ⚠ Admin1 tagging skipped: {_e}")

        # ==========================================================================
        # MERGE BLACK MARBLE DAILY NTL (near-real-time, separate indicator)
        # ==========================================================================
        bm_path = self.data_dir / "external" / "blackmarble_ntl.csv"
        if bm_path.exists():
            try:
                bm = pd.read_csv(bm_path)
                bm['period_start'] = pd.to_datetime(bm['period_start'])
                bm['unit_id'] = bm['unit_id'].astype(str)
                panel['unit_id'] = panel['unit_id'].astype(str)
                panel = panel.merge(bm[['unit_id', 'period_start', 'bm_nightlights']],
                                    on=['unit_id', 'period_start'], how='left')
                n_bm = panel['bm_nightlights'].notna().sum()
                print(f"  ✓ Merged Black Marble daily NTL ({n_bm} unit-months)")
            except Exception as e:
                print(f"  ⚠ Black Marble merge failed: {e}")

        # ==========================================================================
        # MERGE ACLED CONFLICT DATA
        # ==========================================================================
        acled_path = self.data_dir / "external" / "acled_conflict.csv"
        
        if acled_path.exists():
            print("\n--- Merging ACLED Data ---")
            panel = self._merge_acled(panel, acled_path)
        else:
            print("\n--- ACLED Data ---")
            print("  ⚠ File not found, skipping")
            panel['conflict_events'] = 0
            panel['conflict_fatalities'] = 0
        
        # ==========================================================================
        # MERGE WFP PRICE DATA
        # ==========================================================================
        wfp_path = self.data_dir / "external" / "wfp_prices.csv"
        
        if wfp_path.exists():
            print("\n--- Merging WFP Data ---")
            panel = self._merge_wfp(panel, wfp_path)
        else:
            print("\n--- WFP Data ---")
            print("  ⚠ File not found, skipping")
        
        # ==========================================================================
        # MERGE GOOGLE TRENDS DATA
        # ==========================================================================
        gt_path = self.data_dir / "external" / "google_trends.csv"
        
        if gt_path.exists():
            print("\n--- Merging Google Trends ---")
            panel = self._merge_google_trends(panel, gt_path)
        else:
            print("\n--- Google Trends ---")
            print("  ⚠ File not found, skipping (run: python -m pipeline.google_trends --config ...)")
        
        # ==========================================================================
        # CREATE DERIVED VARIABLES
        # ==========================================================================
        print("\n--- Creating Derived Variables ---")
        panel = self._create_derived_variables(panel)
        
        # ==========================================================================
        # SAVE
        # ==========================================================================
        output_path = self.output_dir / "panel_data.csv"
        panel.to_csv(output_path, index=False)
        
        self._print_summary(panel, output_path)
        
        return panel
    
    def _merge_acled(self, panel: pd.DataFrame, acled_path: Path) -> pd.DataFrame:
        """Merge ACLED conflict data."""
        
        acled = pd.read_csv(acled_path)
        acled['event_date'] = pd.to_datetime(acled['event_date'])
        
        print(f"  Events: {len(acled)}")
        print(f"  Date range: {acled['event_date'].min().strftime('%Y-%m-%d')} to {acled['event_date'].max().strftime('%Y-%m-%d')}")
        
        # Create month column
        acled['period_start'] = acled['event_date'].dt.to_period('M').dt.to_timestamp()
        
        # Match admin names to codes
        acled['unit_id'] = acled['admin2'].apply(self._match_admin_name)
        
        # Check matching rate
        matched = acled['unit_id'].notna().sum()
        print(f"  Matched: {matched}/{len(acled)} ({100*matched/len(acled):.1f}%)")
        
        if matched < len(acled) * 0.5:
            unmatched = acled[acled['unit_id'].isna()]['admin2'].value_counts().head(5)
            print(f"  Top unmatched: {list(unmatched.index)}")
        
        # Aggregate by unit and month
        acled_agg = acled[acled['unit_id'].notna()].groupby(['unit_id', 'period_start']).agg({
            'event_id_cnty': 'count',
            'fatalities': 'sum'
        }).reset_index()
        acled_agg.columns = ['unit_id', 'period_start', 'conflict_events', 'conflict_fatalities']
        
        # Merge
        panel = panel.merge(acled_agg, on=['unit_id', 'period_start'], how='left')
        panel['conflict_events'] = panel['conflict_events'].fillna(0)
        panel['conflict_fatalities'] = panel['conflict_fatalities'].fillna(0)
        
        print(f"  ✓ Added conflict_events, conflict_fatalities")
        
        return panel
    
    def _merge_wfp(self, panel: pd.DataFrame, wfp_path: Path) -> pd.DataFrame:
        """Merge WFP food price data with regional fallback for Gaza."""
        
        try:
            # Handle files with comment headers
            wfp = pd.read_csv(wfp_path, comment='#')
            
            # Clean column names
            wfp.columns = [c.lstrip('#').strip() for c in wfp.columns]
            
            if 'date' not in wfp.columns:
                print(f"  ⚠ No 'date' column found")
                return panel
            
            wfp['date'] = pd.to_datetime(wfp['date'], format='mixed', dayfirst=False)
            print(f"  Records: {len(wfp)}")
            print(f"  Commodities: {wfp['commodity'].unique().tolist()}")
            
            # Create month column
            wfp['period_start'] = wfp['date'].dt.to_period('M').dt.to_timestamp()
            
            # Match admin names
            wfp['unit_id'] = wfp['admin2'].apply(self._match_admin_name)
            
            # Debug: show matching results
            matched = wfp[wfp['unit_id'].notna()]['admin2'].unique()
            unmatched = wfp[wfp['unit_id'].isna()]['admin2'].unique()
            print(f"  Matched admin2: {list(matched)[:10]}")
            if len(unmatched) > 0:
                print(f"  Unmatched admin2: {list(unmatched)}")
            
            # Process multiple commodities - use exact matches where possible
            commodities_to_process = [
                ('Bread', 'price_bread'),
                ('Wheat flour', 'price_flour'),  # More specific - match flour first
                ('Rice', 'price_rice'),  # Will match all rice types
                ('Wheat', 'price_wheat'),  # Match wheat grain (not flour)
                ('Sugar', 'price_sugar'),
                ('Oil', 'price_oil'),
                ('Milk', 'price_milk'),
                ('chicken', 'price_chicken'),  # lowercase to match "Meat (chicken)"
            ]
            
            for commodity, price_col in commodities_to_process:
                # Special handling for wheat (exclude flour matches)
                if commodity == 'Wheat' and 'price_flour' in panel.columns:
                    wfp_commodity = wfp[
                        wfp['commodity'].str.contains(commodity, case=False, na=False) &
                        ~wfp['commodity'].str.contains('flour', case=False, na=False)
                    ]
                else:
                    wfp_commodity = wfp[wfp['commodity'].str.contains(commodity, case=False, na=False)]
                
                if len(wfp_commodity) == 0:
                    print(f"  ⚠ No {commodity} prices found")
                    continue
                
                # Method 1: Direct district-level merge
                wfp_matched = wfp_commodity[wfp_commodity['unit_id'].notna()]
                
                if len(wfp_matched) > 0:
                    wfp_agg = wfp_matched.groupby(['unit_id', 'period_start']).agg({
                        'price': 'mean'
                    }).reset_index()
                    wfp_agg.columns = ['unit_id', 'period_start', price_col]
                    
                    panel = panel.merge(wfp_agg, on=['unit_id', 'period_start'], how='left')
                    matched_count = panel[price_col].notna().sum()
                    print(f"  ✓ Added {price_col} (district-level: {matched_count} rows)")
                else:
                    panel[price_col] = np.nan
                
                # Method 2: Regional fallback for Gaza (if district-level missing)
                gaza_districts = [uid for uid in panel['unit_id'].unique() 
                                if str(uid).startswith('PS02')]
                
                gaza_wfp_names = ['gaza', 'northgaza', 'rafah', 'kanyounis', 'khanyounis', 
                                'deiralbalah', 'deirelbalah', 'middlearea']
                wfp_gaza = wfp_commodity[wfp_commodity['admin2'].apply(
                    lambda x: clean_name(str(x)) in gaza_wfp_names if pd.notna(x) else False
                )]
                
                if len(wfp_gaza) > 0 and len(gaza_districts) > 0:
                    gaza_mask = panel['unit_id'].isin(gaza_districts)
                    missing_gaza = gaza_mask & panel[price_col].isna()
                    
                    if missing_gaza.sum() > 0:
                        gaza_regional = wfp_gaza.groupby('period_start').agg({
                            'price': 'mean'
                        }).reset_index()
                        gaza_regional.columns = ['period_start', 'gaza_regional_price']
                        
                        panel = panel.merge(gaza_regional, on='period_start', how='left')
                        
                        fill_mask = gaza_mask & panel[price_col].isna()
                        panel.loc[fill_mask, price_col] = panel.loc[fill_mask, 'gaza_regional_price']
                        panel = panel.drop(columns=['gaza_regional_price'])
                        
                        filled_count = fill_mask.sum()
                        print(f"  ✓ Filled Gaza {commodity} with regional average ({filled_count} rows)")
                
                final_count = panel[price_col].notna().sum()
                print(f"  ✓ Total {price_col} coverage: {final_count}/{len(panel)} rows")
                
        except Exception as e:
            print(f"  ⚠ WFP merge failed: {e}")
            import traceback
            traceback.print_exc()
        
        return panel
    
    def _merge_google_trends(self, panel: pd.DataFrame, gt_path: Path) -> pd.DataFrame:
        """Merge Google Trends data (country-level, applied to all districts)."""
        
        try:
            gt = pd.read_csv(gt_path)
            gt['date'] = pd.to_datetime(gt['date'])
            gt['period_start'] = gt['date'].dt.to_period('M').dt.to_timestamp()
            
            print(f"  Records: {len(gt)}")
            print(f"  Keywords: {gt['keyword'].nunique()}")
            
            # Pivot to wide format (one column per keyword)
            gt_pivot = gt.pivot_table(
                index='period_start', 
                columns='keyword', 
                values='score', 
                aggfunc='mean'
            ).reset_index()
            
            # Rename columns with gt_ prefix
            gt_pivot.columns = ['period_start'] + [f'gt_{c}' for c in gt_pivot.columns[1:]]
            
            # Merge (Google Trends is country-level, applies to all districts)
            panel = panel.merge(gt_pivot, on='period_start', how='left')
            
            gt_cols = [c for c in panel.columns if c.startswith('gt_')]
            print(f"  ✓ Added {len(gt_cols)} Google Trends columns")
            
        except Exception as e:
            print(f"  ⚠ Google Trends merge failed: {e}")
        
        return panel
    
    def _create_derived_variables(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Create derived and analysis variables."""
        
        # SAR-derived indices
        if 'vv' in panel.columns and 'vh' in panel.columns:
            panel['rvi'] = 4 * panel['vh'] / (panel['vv'] + panel['vh'] + 1e-10)
            panel['structure_proxy_vh'] = panel['vh']
            panel['flood_proxy_vv'] = panel['vv']
        
        # Log transformations
        log_cols = ['nightlights', 'ndvi', 'rainfall', 'conflict_events', 'conflict_fatalities']
        for col in log_cols:
            if col in panel.columns:
                panel[f'log_{col}'] = np.log1p(panel[col].clip(lower=0))
        
        # Price log
        price_cols = [c for c in panel.columns if c.startswith('price_')]
        for col in price_cols:
            panel[f'log_{col}'] = np.log(panel[col].replace(0, np.nan))
        
        # Time indicators
        panel['post'] = (panel['period_start'] >= self.disaster_date).astype(int)
        panel['post_intervention'] = (panel['period_start'] >= self.intervention_date).astype(int)
        
        panel['months_since_disaster'] = (
            (panel['period_start'].dt.year - self.disaster_date.year) * 12 + 
            (panel['period_start'].dt.month - self.disaster_date.month)
        )
        panel['months_since_intervention'] = (
            (panel['period_start'].dt.year - self.intervention_date.year) * 12 + 
            (panel['period_start'].dt.month - self.intervention_date.month)
        )
        
        # Treatment placeholders (to be updated when treatment file provided)
        panel['treated'] = 0
        panel['treated_x_post'] = 0
        
        print("  ✓ Created log transforms, time indicators")
        
        return panel
    
    def _print_summary(self, panel: pd.DataFrame, output_path: Path):
        """Print summary of built panel."""
        
        print(f"\n{'='*60}")
        print(f"✓ PANEL SAVED: {output_path}")
        print(f"{'='*60}")
        print(f"  Rows: {len(panel)}")
        print(f"  Columns: {len(panel.columns)}")
        print(f"  Period: {panel['period_start'].min().strftime('%Y-%m-%d')} to {panel['period_start'].max().strftime('%Y-%m-%d')}")
        print(f"  Units: {panel['unit_id'].nunique()}")
        
        pre = panel[panel['post'] == 0]
        post = panel[(panel['post'] == 1) & (panel['post_intervention'] == 0)]
        post_int = panel[panel['post_intervention'] == 1]
        
        print(f"\n  Pre-disaster: {len(pre)} obs ({pre['period_start'].nunique()} months)")
        print(f"  Post-disaster (pre-intervention): {len(post)} obs ({post['period_start'].nunique()} months)")
        print(f"  Post-intervention: {len(post_int)} obs ({post_int['period_start'].nunique()} months)")


def main():
    parser = argparse.ArgumentParser(description='Build panel data for impact evaluation')
    parser.add_argument('--config', '-c', required=True, help='Path to country config YAML')
    args = parser.parse_args()
    
    builder = PanelDataBuilder(args.config)
    builder.build()


if __name__ == "__main__":
    main()
