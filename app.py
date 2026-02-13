#!/usr/bin/env python3
"""
app.py — UNDP Impact Evaluation Pipeline Interface
===================================================
Web interface for configuring and running the pipeline.

Usage:
    streamlit run app.py
"""

import streamlit as st
import yaml
import subprocess
import sys
import os
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv
import glob

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="UNDP Impact Evaluation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
    /* Clean, professional look */
    .main .block-container { max-width: 1100px; padding-top: 2rem; }
    
    /* Header */
    .pipeline-header {
        background: linear-gradient(135deg, #0468B1 0%, #00A5E5 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .pipeline-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .pipeline-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 1rem; }
    
    /* Status cards */
    .status-card {
        background: #f8f9fa;
        border-left: 4px solid #0468B1;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .status-card.success { border-left-color: #28a745; }
    .status-card.warning { border-left-color: #ffc107; }
    .status-card.error { border-left-color: #dc3545; }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #0468B1;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="pipeline-header">
    <h1>🌍 UNDP Impact Evaluation Pipeline</h1>
    <p>Satellite-based causal inference for post-disaster recovery</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Mode selection
    mode = st.radio("Mode", ["Use existing config", "Create new config"], 
                    label_visibility="collapsed")
    
    if mode == "Use existing config":
        # Find existing configs
        config_files = sorted(glob.glob("config/*.yaml"))
        config_names = [Path(f).stem for f in config_files]
        
        if config_names:
            selected = st.selectbox("Country", config_names, 
                                   format_func=lambda x: x.replace('_', ' ').title())
            config_path = f"config/{selected}.yaml"
            
            # Load and display config
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            st.warning("No configs found in config/")
            config = None
            config_path = None
    
    else:
        # Create new config
        st.markdown("---")
        st.markdown("**Country Details**")
        
        country_name = st.text_input("Country Name", "Myanmar")
        country_code = st.text_input("ISO Code (3-letter)", "MMR").upper()
        
        st.markdown("**Event**")
        disaster_type = st.selectbox("Disaster Type", 
                                     ["earthquake", "conflict", "flood", "cyclone"])
        disaster_date = st.date_input("Disaster Date", date(2025, 3, 28))
        intervention_start = st.date_input("Intervention Start", 
                                           date(2025, 4, 1))
        
        st.markdown("**Study Period**")
        col1, col2 = st.columns(2)
        with col1:
            study_start = st.date_input("Start", date(2023, 10, 1))
        with col2:
            study_end = st.date_input("End", date(2026, 1, 30))
        
        st.markdown("**Admin Boundaries**")
        admin_level = st.selectbox("Admin Level", [1, 2, 3], index=2,
                                   help="1=State, 2=District, 3=Township")
        
        st.markdown("**Data Sources**")
        acled_country = st.text_input("ACLED Country Name", country_name)
        wfp_commodity = st.text_input("WFP Commodity", "Rice")
        
        # Optional fields
        with st.expander("Advanced Options"):
            disaster_label = st.text_input("Disaster Label (for plots)", 
                                          disaster_type.title())
            
            if disaster_type == "earthquake":
                st.markdown("**Epicenter**")
                epi_lat = st.number_input("Latitude", value=0.0, format="%.4f")
                epi_lon = st.number_input("Longitude", value=0.0, format="%.4f")
                epi_mag = st.number_input("Magnitude", value=0.0, format="%.1f")
        
        # Build config dict
        config = {
            'country_name': country_name,
            'country_code': country_code,
            'study_period_start': study_start.isoformat(),
            'study_period_end': study_end.isoformat(),
            'events': {
                'disaster_type': disaster_type,
                'disaster_date': disaster_date.isoformat(),
                'intervention_start': intervention_start.isoformat(),
            },
            'boundaries': {
                'file_path': f"data/{country_code.lower()}/boundaries/{country_code.lower()}_admin{admin_level}.geojson",
                'id_column': 'shapeID',
                'name_column': 'shapeName',
                'admin_level': admin_level,
            },
            'acled_country_name': acled_country,
            'wfp_commodity': wfp_commodity,
        }
        
        # Add optional fields
        if disaster_label:
            config['disaster_label'] = disaster_label
        if disaster_type == "earthquake" and epi_lat != 0:
            config['epicenter'] = {'lat': epi_lat, 'lon': epi_lon, 'magnitude': epi_mag}
        
        config_path = f"config/{country_code.lower()}.yaml"
    
    # Credentials
    st.markdown("---")
    st.markdown("### 🔑 Credentials")
    
    acled_email = st.text_input("ACLED Email", value=os.getenv("ACLED_EMAIL", ""),
                                type="default")
    acled_key = st.text_input("ACLED Key", value=os.getenv("ACLED_KEY", ""),
                              type="password")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if config is None:
    st.info("Select or create a configuration to get started.")
    st.stop()

# --- Config Summary ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Country", config['country_name'])
with col2:
    st.metric("Disaster", config['events']['disaster_date'])
with col3:
    st.metric("Study Period", 
              f"{config.get('study_period_start', 'N/A')[:7]} → {config.get('study_period_end', 'N/A')[:7]}")

# --- Data Status ---
st.markdown('<div class="section-header">📊 Data Status</div>', unsafe_allow_html=True)

code = config['country_code'].lower()
data_dir = Path(f"data/{code}")
output_dir = Path(f"outputs/{code}")

def check_file(path, description):
    """Check if file exists and return status."""
    p = Path(path)
    if p.exists():
        try:
            import pandas as pd
            if path.endswith('.csv'):
                n = len(pd.read_csv(p))
                return "success", f"✅ {description}: {n:,} rows"
            elif path.endswith('.geojson'):
                import geopandas as gpd
                n = len(gpd.read_file(p))
                return "success", f"✅ {description}: {n} units"
        except:
            pass
        return "success", f"✅ {description}"
    return "warning", f"⬚ {description}: not downloaded"

checks = [
    (config['boundaries']['file_path'], "Admin Boundaries"),
    (f"data/{code}/gee_panel_{code}_centroid.csv", "GEE Satellite Data"),
    (f"data/{code}/external/acled_conflict.csv", "ACLED Conflict"),
    (f"data/{code}/external/undp_projects.csv", "UNDP Projects"),
    (f"data/{code}/external/google_trends.csv", "Google Trends"),
    (f"outputs/{code}/panel_data.csv", "Panel Data"),
]

cols = st.columns(3)
for i, (path, desc) in enumerate(checks):
    status, msg = check_file(path, desc)
    with cols[i % 3]:
        st.markdown(f'<div class="status-card {status}">{msg}</div>', 
                   unsafe_allow_html=True)

# --- Pipeline Controls ---
st.markdown('<div class="section-header">🚀 Run Pipeline</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    run_data = st.button("📥 Download Data", use_container_width=True,
                         help="Download boundaries, GEE, ACLED, UNDP, Google Trends")
with col2:
    run_preprocess = st.button("🔧 Preprocess", use_container_width=True,
                               help="Build panel data from downloaded sources")
with col3:
    run_visuals = st.button("📈 Visualizations", use_container_width=True,
                            help="Generate time series, choropleths, timelines")
with col4:
    run_all = st.button("▶️ Run All", use_container_width=True, type="primary",
                        help="Full pipeline: data → preprocess → visuals")

def run_command(cmd, description):
    """Run a make command and display output."""
    st.markdown(f"**Running: {description}...**")
    
    # Set environment
    env = os.environ.copy()
    if acled_email:
        env['ACLED_EMAIL'] = acled_email
    if acled_key:
        env['ACLED_KEY'] = acled_key
    
    with st.expander("Terminal Output", expanded=True):
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                env=env, timeout=600
            )
            if result.stdout:
                st.code(result.stdout, language="bash")
            if result.returncode != 0 and result.stderr:
                st.error(result.stderr[-500:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            st.error("⏱ Timed out (10 min). Run from terminal for long operations.")
            return False

# Save config if creating new
if mode == "Create new config":
    save_col1, save_col2 = st.columns([3, 1])
    with save_col2:
        if st.button("💾 Save Config"):
            Path("config").mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            st.success(f"Saved: {config_path}")
            st.rerun()

country = config_path.replace("config/", "").replace(".yaml", "")

if run_data:
    run_command(f"make data COUNTRY={country}", "Data Acquisition")

if run_preprocess:
    run_command(f"make preprocess COUNTRY={country}", "Preprocessing")

if run_visuals:
    run_command(f"make visuals COUNTRY={country} && make trends COUNTRY={country}", 
                "Visualizations")

if run_all:
    success = run_command(f"make run COUNTRY={country}", "Full Pipeline")
    if success:
        st.balloons()

# --- Output Gallery ---
figures_dir = output_dir / "figures"
if figures_dir.exists():
    pngs = sorted(figures_dir.glob("*.png"))
    if pngs:
        st.markdown('<div class="section-header">🖼️ Output Gallery</div>', 
                   unsafe_allow_html=True)
        
        # Group by type
        ts_files = [f for f in pngs if f.name.startswith('ts_')]
        map_files = [f for f in pngs if f.name.startswith('map_')]
        other_files = [f for f in pngs if f not in ts_files + map_files]
        
        tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🗺️ Maps", "📊 Other"])
        
        with tab1:
            for f in ts_files:
                name = f.stem.replace('ts_', '').replace('_', ' ').title()
                st.markdown(f"**{name}**")
                st.image(str(f), use_container_width=True)
        
        with tab2:
            for f in map_files:
                name = f.stem.replace('map_', '').replace('_', ' ').title()
                st.markdown(f"**{name}**")
                st.image(str(f), use_container_width=True)
        
        with tab3:
            for f in other_files:
                name = f.stem.replace('_', ' ').title()
                st.markdown(f"**{name}**")
                st.image(str(f), use_container_width=True)

# --- YAML Preview ---
with st.expander("📝 Config YAML"):
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")
