#!/usr/bin/env python3
"""
app_v8.py — UNDP Impact Evaluation Dashboard (v8)
==================================================
Usage: streamlit run app_v8.py
v8: Recovery Scorecard as first visual, before tabs
"""

import streamlit as st
import streamlit.components.v1 as components
import yaml, subprocess, os, glob
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

st.set_page_config(page_title="UNDP Impact Evaluation Dashboard", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    .main .block-container { max-width: 1200px; padding: 1.5rem 2rem 4rem 2rem; background: #1e293b; }
    html, body, .stApp { font-family: 'Source Sans 3', sans-serif !important; background: #1e293b; }
    section[data-testid="stSidebar"] { background: #1e293b; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .country-card { background: #2d3b4e; border: 1px solid #3d4f65; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .5rem; }
    section[data-testid="stSidebar"] .country-name { color: white !important; font-weight: 600; font-size: 1.05rem; }
    section[data-testid="stSidebar"] .country-detail { color: #94a3b8 !important; font-size: .83rem; margin-top: .15rem; }
    section[data-testid="stSidebar"] input { background: #2d3b4e !important; color: white !important; border-color: #3d4f65 !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div { background: #2d3b4e !important; }
    section[data-testid="stSidebar"] hr { border-color: #3d4f65 !important; }
    .hero { background: linear-gradient(135deg, #0468B1 0%, #00A5E5 50%, #0098D8 100%); padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.5rem; }
    .hero h1 { font-family: 'Playfair Display', serif !important; font-size: 1.9rem; margin: 0 0 .3rem 0; font-weight: 700; color: white !important; }
    .hero p { font-size: 1rem; margin: 0; color: rgba(255,255,255,.9) !important; }
    .hero-badge { display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25); padding: .2rem .7rem; border-radius: 20px; font-size: .8rem; margin-top: .7rem; color: white !important; }
    .metric-row { display: flex; gap: .8rem; margin: 1rem 0; }
    .metric-box { flex: 1; background: #2d3b4e; border-radius: 10px; padding: .9rem 1.1rem; border-left: 3px solid #0468B1; }
    .metric-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: #94a3b8; font-weight: 600; }
    .metric-value { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-top: .1rem; }
    .country-flag { font-size: 1.5rem; margin-right: .4rem; }
    .data-row { display: flex; align-items: center; padding: .55rem 0; border-bottom: 1px solid #2d3b4e; }
    .data-row:last-child { border-bottom: none; }
    .data-icon { font-size: 1rem; width: 1.8rem; text-align: center; color: #e2e8f0; }
    .data-label { flex: 1; font-size: .9rem; font-weight: 500; color: #e2e8f0; }
    .data-status { font-size: .8rem; padding: .15rem .55rem; border-radius: 12px; font-weight: 600; }
    .data-ready { background: #1b4332; color: #6ee7b7; }
    .data-missing { background: #78350f; color: #fbbf24; }
    .section-title { font-family: 'Playfair Display', serif !important; font-size: 1.25rem; font-weight: 600; color: #e2e8f0; margin: 1.8rem 0 .7rem 0; padding-bottom: .4rem; border-bottom: 2px solid #0468B1; }
    button[data-baseweb="tab"] > div > p { color: #e2e8f0 !important; }
    button[data-baseweb="tab"][aria-selected="true"] > div > p { color: #00A5E5 !important; font-weight: 600 !important; }
    .setup-box { background: white; border: 1px solid #cce0f5; border-radius: 10px; padding: 1.2rem; margin: .8rem 0; }
    .setup-box, .setup-box p, .setup-box li, .setup-box ol, .setup-box a, .setup-box span { color: #111 !important; }
    .setup-box h4 { color: #0468B1 !important; margin: 0 0 .5rem 0; font-size: 1rem; }
    .setup-box code { background: #dce4ed; padding: .1rem .4rem; border-radius: 4px; font-size: .85rem; color: #333; }
    .footer { text-align: center; padding: 1.5rem 0 .5rem 0; border-top: 1px solid #2d3b4e; margin-top: 2.5rem; color: #64748b; font-size: .78rem; }
    
    /* ── SCORECARD STYLES ── */
    .scorecard-container { background: #2d3b4e; border-radius: 14px; padding: 1.5rem 2rem; margin: 1rem 0 1.5rem 0; }
    .scorecard-title { font-family: 'Playfair Display', serif !important; font-size: 1.3rem; font-weight: 700; color: white; margin: 0 0 .3rem 0; }
    .scorecard-subtitle { font-size: .85rem; color: #94a3b8; margin: 0 0 1.2rem 0; }
    .scorecard-row { display: flex; align-items: center; padding: .7rem 0; border-bottom: 1px solid #3d4f65; }
    .scorecard-row:last-child { border-bottom: none; }
    .scorecard-icon { font-size: 1.3rem; width: 2rem; text-align: center; }
    .scorecard-name { width: 140px; font-weight: 600; color: #e2e8f0; font-size: .9rem; }
    .scorecard-desc { width: 180px; font-size: .75rem; color: #94a3b8; }
    .scorecard-bars { flex: 1; padding: 0 .5rem; }
    .scorecard-bar-track { background: #1e293b; border-radius: 6px; height: 14px; position: relative; margin: 3px 0; overflow: visible; }
    .scorecard-bar-fill-overall { background: linear-gradient(90deg, #0468B1, #0580d0); height: 14px; border-radius: 6px; position: relative; transition: width 0.5s; }
    .scorecard-bar-fill-undp { background: linear-gradient(90deg, #00A5E5, #38bdf8); height: 14px; border-radius: 6px; position: relative; transition: width 0.5s; }
    .scorecard-bar-label { font-size: .7rem; color: #94a3b8; margin-top: 1px; }
    .scorecard-pct { width: 70px; text-align: right; font-weight: 700; font-size: 1.05rem; }
    .scorecard-pct-overall { color: #60a5fa; }
    .scorecard-pct-undp { color: #38bdf8; }
    .scorecard-ref { position: absolute; left: 100%; top: -2px; bottom: -2px; width: 2px; background: #22c55e; opacity: 0.6; }
    .scorecard-legend { display: flex; gap: 1.5rem; margin-top: .8rem; justify-content: center; }
    .scorecard-legend-item { display: flex; align-items: center; gap: .4rem; font-size: .8rem; color: #94a3b8; }
    .scorecard-legend-dot { width: 10px; height: 10px; border-radius: 3px; }
    .scorecard-note { font-size: .75rem; color: #64748b; text-align: center; margin-top: .6rem; font-style: italic; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header[data-testid="stHeader"] { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
COUNTRY_REFERENCE = {
    'Afghanistan': {'iso': 'AFG', 'acled': 'Afghanistan'},
    'Bangladesh': {'iso': 'BGD', 'acled': 'Bangladesh'},
    'Colombia': {'iso': 'COL', 'acled': 'Colombia'},
    'DR Congo': {'iso': 'COD', 'acled': 'Democratic Republic of Congo'},
    'Ethiopia': {'iso': 'ETH', 'acled': 'Ethiopia'},
    'Gaza Strip': {'iso': 'PSE', 'acled': 'Palestine'},
    'Haiti': {'iso': 'HTI', 'acled': 'Haiti'},
    'Honduras': {'iso': 'HND', 'acled': 'Honduras'},
    'India': {'iso': 'IND', 'acled': 'India'},
    'Indonesia': {'iso': 'IDN', 'acled': 'Indonesia'},
    'Iraq': {'iso': 'IRQ', 'acled': 'Iraq'},
    'Lebanon': {'iso': 'LBN', 'acled': 'Lebanon'},
    'Libya': {'iso': 'LBY', 'acled': 'Libya'},
    'Mozambique': {'iso': 'MOZ', 'acled': 'Mozambique'},
    'Myanmar': {'iso': 'MMR', 'acled': 'Myanmar'},
    'Nepal': {'iso': 'NPL', 'acled': 'Nepal'},
    'Nigeria': {'iso': 'NGA', 'acled': 'Nigeria'},
    'Pakistan': {'iso': 'PAK', 'acled': 'Pakistan'},
    'Philippines': {'iso': 'PHL', 'acled': 'Philippines'},
    'Somalia': {'iso': 'SOM', 'acled': 'Somalia'},
    'South Sudan': {'iso': 'SSD', 'acled': 'South Sudan'},
    'Sudan': {'iso': 'SDN', 'acled': 'Sudan'},
    'Syria': {'iso': 'SYR', 'acled': 'Syria'},
    'Turkey / Türkiye': {'iso': 'TUR', 'acled': 'Turkey'},
    'Ukraine': {'iso': 'UKR', 'acled': 'Ukraine'},
    'Yemen': {'iso': 'YEM', 'acled': 'Yemen'},
}
DEFAULT_KW = ['earthquake','flood','reconstruction','food','water','medicine','shelter','aid','cement','construction']
COUNTRIES = {
    'myanmar': {'flag':'🇲🇲','name':'Myanmar','event':'M7.7 Earthquake','event_date':'Mar 28, 2025'},
    'gaza': {'flag':'🇵🇸','name':'Gaza Strip','event':'Armed Conflict','event_date':'Oct 7, 2023'},
    'afghanistan': {'flag':'🇦🇫','name':'Afghanistan','event':'M6.3 Earthquake','event_date':'Oct 7, 2023'},
    'ukraine': {'flag':'🇺🇦','name':'Ukraine','event':'Armed Conflict','event_date':'Feb 24, 2022'},
}
THEMES = {
    '💡 Economic Recovery': {'desc':'Nighttime lights as proxy for economic activity, energy access, and market prices','indicators':['nightlights','price_rice','price_oil']},
    '🌿 Environmental Impact': {'desc':'Vegetation health, land use change, and agricultural recovery','indicators':['ndvi','rvi','rainfall']},
    '🏗️ Infrastructure & Debris': {'desc':'Structural changes via radar backscatter (damage, debris, reconstruction)','indicators':['vh','vv']},
    '⚔️ Conflict & Security': {'desc':'Armed conflict events and fatalities from ACLED','indicators':['conflict_events','conflict_fatalities']},
}
NAMES = {'nightlights':'Nighttime Lights (VIIRS)','ndvi':'Vegetation (NDVI)','vh':'SAR VH Backscatter','vv':'SAR VV Backscatter','rvi':'Radar Vegetation Index','rainfall':'Rainfall (CHIRPS)','conflict_events':'Conflict Events','conflict_fatalities':'Conflict Fatalities','price_rice':'Rice Price','price_oil':'Oil Price'}
RASTER_IND = ['viirs','ndvi','ndbi','bsi','vh','vv','evi','savi','nbr','swir','mndwi']
RASTER_NAMES = {'viirs':'Nighttime Lights (VIIRS)','ndvi':'Vegetation (NDVI)','ndbi':'Built-Up Index (NDBI)','bsi':'Bare Soil / Debris (BSI)','evi':'Enhanced Vegetation (EVI)','savi':'Soil-Adjusted Vegetation (SAVI)','nbr':'Burn Ratio (NBR)','vh':'SAR VH','vv':'SAR VV','swir':'Short-Wave Infrared','mndwi':'Water Index'}

# =============================================================================
st.markdown('<div class="hero"><h1>UNDP Post-Disaster Recovery Dashboard</h1><p>Satellite-based impact evaluation of UNDP reconstruction interventions</p><div class="hero-badge">📡 Powered by remote sensing & causal inference</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### 🌍 Select Country")
    cfg_files = sorted(glob.glob("config/*.yaml"))
    cfg_names = [Path(f).stem for f in cfg_files]
    if not cfg_names: st.warning("No configs found"); st.stop()
    selected = st.selectbox("Country", cfg_names, format_func=lambda n: f"{COUNTRIES.get(n,{}).get('flag','🌐')}  {COUNTRIES.get(n,{}).get('name',n.replace('_',' ').title())}", label_visibility="collapsed")
    with open(f"config/{selected}.yaml") as f: config = yaml.safe_load(f)
    ci = COUNTRIES.get(selected, {})
    st.markdown(f'<div class="country-card"><span class="country-flag">{ci.get("flag","🌐")}</span><span class="country-name">{config["country_name"]}</span><div class="country-detail">{ci.get("event","")} · {ci.get("event_date",config["events"]["disaster_date"])}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("Mode", ["📊 View Results", "🔧 Run Pipeline", "➕ Add New Country"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.75rem;">Built by AImpact Lab</div>', unsafe_allow_html=True)

code = config['country_code'].lower()
figures_dir = Path(f"outputs/{code}/figures")
output_dir = Path(f"outputs/{code}")
panel_path = output_dir / "panel_data.csv"
panel_info = ""
if panel_path.exists():
    try:
        p = pd.read_csv(panel_path, usecols=['unit_id','period_start'])
        panel_info = f"{p['unit_id'].nunique()} units × {p['period_start'].nunique()} mo"
    except: panel_info = "Available"
n_figs = len(list(figures_dir.rglob("*.png"))) if figures_dir.exists() else 0

st.markdown(f'<div class="metric-row"><div class="metric-box"><div class="metric-label">Event</div><div class="metric-value">{config["events"]["disaster_date"]}</div></div><div class="metric-box"><div class="metric-label">Study Period</div><div class="metric-value">{str(config.get("study_period_start",""))[:7]} → {str(config.get("study_period_end",""))[:7]}</div></div><div class="metric-box"><div class="metric-label">Panel</div><div class="metric-value">{panel_info or "Not built"}</div></div><div class="metric-box"><div class="metric-label">Figures</div><div class="metric-value">{n_figs}</div></div></div>', unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
def find_figs(indicator, pngs):
    out = []
    for f in pngs:
        s = f.stem.lower()
        if s == f"ts_{indicator}" or s == f"map_{indicator}": out.append(f)
        elif indicator in s and any(s.startswith(p) for p in ['ts_','map_']): out.append(f)
    return sorted(out)

def collect_rasters():
    candidates = []
    for sub in ['raster_maps','undp_maps']:
        d = figures_dir / sub
        if d.exists():
            for f in sorted(d.glob("*.png")):
                if 'three_panel' in f.stem.lower(): candidates.append(f)
    groups = {}
    for f in candidates:
        nm = f.stem.lower()
        ind = None
        for ri in RASTER_IND:
            if nm.startswith(ri): ind = ri; break
        if ind is None: ind = nm.split('_')[0]
        is_clean = 'clean' in nm
        is_geo = 'geocode' in nm
        if ind not in groups: groups[ind] = {'clean':None,'geocode':None,'other':None}
        if is_clean: groups[ind]['clean'] = f
        elif is_geo: groups[ind]['geocode'] = f
        else: groups[ind]['other'] = f
    result = []
    for ind in sorted(groups.keys()):
        g = groups[ind]
        picked = g['clean'] or g['geocode'] or g['other']
        if picked: result.append((ind, picked))
    return result

def render_scorecard():
    """Render the recovery scorecard from scorecard_data.csv or scorecard.png."""
    csv_path = figures_dir / "scorecard_data.csv"
    png_path = figures_dir / "scorecard.png"
    
    # Try interactive HTML scorecard first (from CSV)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        disaster_date = config['events']['disaster_date']
        intervention = config['events'].get('intervention_start', disaster_date)
        
        rows_html = ""
        for _, row in df.iterrows():
            overall = row["overall_recovery_pct"]
            if overall < 0: overall = 0
            undp = row["undp_recovery_pct"]
            if undp < 0: undp = 0
            
            # Color the percentage based on level
            if overall >= 75: overall_color = "#22c55e"  # green
            elif overall >= 50: overall_color = "#eab308"  # yellow
            else: overall_color = "#ef4444"  # red
            
            if undp >= 50: undp_color = "#38bdf8"
            elif undp >= 25: undp_color = "#60a5fa"
            else: undp_color = "#94a3b8"
            
            rows_html += f"""
            <div class="scorecard-row">
                <div class="scorecard-icon">{row.get('icon','📊')}</div>
                <div class="scorecard-name">{row.get('short_name', row.get('name',''))}</div>
                <div class="scorecard-desc">{row.get('description','')}</div>
                <div class="scorecard-bars">
                    <div class="scorecard-bar-track">
                        <div class="scorecard-bar-fill-overall" style="width: {overall}%;"></div>
                        <div class="scorecard-ref"></div>
                    </div>
                    <div class="scorecard-bar-label">Overall recovery</div>
                    <div class="scorecard-bar-track">
                        <div class="scorecard-bar-fill-undp" style="width: {min(undp, 100)}%;"></div>
                    </div>
                    <div class="scorecard-bar-label">Since UNDP arrival</div>
                </div>
                <div style="width: 80px; text-align: right;">
                    <div class="scorecard-pct scorecard-pct-overall" style="color: {overall_color}">{('—' if row['overall_recovery_pct'] < 0 else f"{row['overall_recovery_pct']:.0f}%")}</div>
                    <div class="scorecard-pct scorecard-pct-undp" style="color: {undp_color}">{('—' if row['undp_recovery_pct'] < 0 else f"{row['undp_recovery_pct']:.0f}%")}</div>
                </div>
            </div>
            """
        
        scorecard_html = f"""
        <style>
            body {{ background: transparent; margin: 0; padding: 0; font-family: 'Source Sans 3', -apple-system, sans-serif; }}
            .scorecard-container {{ background: #2d3b4e; border-radius: 14px; padding: 1.5rem 2rem; }}
            .scorecard-title {{ font-size: 1.3rem; font-weight: 700; color: white; margin: 0 0 .3rem 0; }}
            .scorecard-subtitle {{ font-size: .85rem; color: #94a3b8; margin: 0 0 1.2rem 0; }}
            .scorecard-row {{ display: flex; align-items: center; padding: .7rem 0; border-bottom: 1px solid #3d4f65; }}
            .scorecard-row:last-of-type {{ border-bottom: none; }}
            .scorecard-icon {{ font-size: 1.3rem; width: 2rem; text-align: center; }}
            .scorecard-name {{ width: 140px; font-weight: 600; color: #e2e8f0; font-size: .9rem; }}
            .scorecard-desc {{ width: 180px; font-size: .75rem; color: #94a3b8; }}
            .scorecard-bars {{ flex: 1; padding: 0 .5rem; }}
            .scorecard-bar-track {{ background: #1e293b; border-radius: 6px; height: 14px; position: relative; margin: 3px 0; }}
            .scorecard-bar-fill-overall {{ background: linear-gradient(90deg, #0468B1, #0580d0); height: 14px; border-radius: 6px; }}
            .scorecard-bar-fill-undp {{ background: linear-gradient(90deg, #00A5E5, #38bdf8); height: 14px; border-radius: 6px; }}
            .scorecard-bar-label {{ font-size: .7rem; color: #94a3b8; margin-top: 1px; }}
            .scorecard-pct {{ font-weight: 700; font-size: 1.05rem; }}
            .scorecard-ref {{ position: absolute; right: 0; top: -2px; bottom: -2px; width: 2px; background: #22c55e; opacity: 0.6; }}
            .scorecard-legend {{ display: flex; gap: 1.5rem; margin-top: .8rem; justify-content: center; flex-wrap: wrap; }}
            .scorecard-legend-item {{ display: flex; align-items: center; gap: .4rem; font-size: .8rem; color: #94a3b8; }}
            .scorecard-legend-dot {{ width: 10px; height: 10px; border-radius: 3px; }}
            .scorecard-note {{ font-size: .75rem; color: #64748b; text-align: center; margin-top: .6rem; font-style: italic; }}
        </style>
        <div class="scorecard-container">
            <div class="scorecard-title">Recovery Scorecard</div>
            <div class="scorecard-subtitle">
                How much has recovered since the disaster, and what changed after UNDP arrived?
                <br>Disaster: {disaster_date} &nbsp;|&nbsp; UNDP Intervention: {intervention}
            </div>
            {rows_html}
            <div class="scorecard-legend">
                <div class="scorecard-legend-item"><div class="scorecard-legend-dot" style="background: #0468B1;"></div>Overall Recovery — % of disaster drop recovered</div>
                <div class="scorecard-legend-item"><div class="scorecard-legend-dot" style="background: #00A5E5;"></div>Since UNDP — % of remaining gap closed</div>
                <div class="scorecard-legend-item"><div class="scorecard-legend-dot" style="background: #22c55e;"></div>Pre-disaster level (100%)</div>
            </div>
            <div class="scorecard-note">Recovery metrics compare national monthly averages. "Since UNDP" shows temporal correlation, not necessarily causation.</div>
        </div>
        """
        components.html(scorecard_html, height=180 + len(df) * 110, scrolling=False)
        return True
    
    # Fallback: show the PNG
    elif png_path.exists():
        st.image(str(png_path))
        return True
    
    return False


# =============================================================================
if mode == "📊 View Results":
    
    # ── SCORECARD FIRST (hero position)
    has_scorecard = render_scorecard()
    if not has_scorecard:
        st.info("💡 Run the scorecard to see recovery metrics: "
               "`python pipeline/recovery_scorecard.py --config config/{}.yaml`".format(selected))
    
    # ── DATA STATUS
    st.markdown('<div class="section-title">Data Availability</div>', unsafe_allow_html=True)
    srcs = [("🗺️","Boundaries",config['boundaries']['file_path']),("🛰️","Satellite",f"data/{code}/gee_panel_{code}_centroid.csv"),("⚔️","Conflict",f"data/{code}/external/acled_conflict.csv"),("🏗️","UNDP Projects",f"data/{code}/external/undp_projects.csv"),("📈","Google Trends",f"data/{code}/external/google_trends.csv"),("📊","Panel",f"outputs/{code}/panel_data.csv")]
    cols = st.columns(3)
    for i,(ic,lb,pt) in enumerate(srcs):
        pp=Path(pt); ok=pp.exists()
        try: nn=f" ({len(pd.read_csv(pp)):,})" if ok and pt.endswith('.csv') else ""
        except: nn=""
        s=f'<span class="data-status data-ready">✓{nn}</span>' if ok else '<span class="data-status data-missing">○</span>'
        with cols[i%3]: st.markdown(f'<div class="data-row"><div class="data-icon">{ic}</div><div class="data-label">{lb}</div>{s}</div>', unsafe_allow_html=True)

    summary_data = pd.read_csv(figures_dir/"summary_statistics.csv") if (figures_dir/"summary_statistics.csv").exists() else None
    if summary_data is not None:
        st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(summary_data, hide_index=True)

    # ── THEMATIC TABS
    if figures_dir.exists():
        all_pngs = sorted(figures_dir.glob("*.png"))
        if all_pngs:
            st.markdown('<div class="section-title">Detailed Indicators</div>', unsafe_allow_html=True)
            st.markdown("Explore each thematic area for detailed time series and maps:")
            tab_names = list(THEMES.keys())
            trend_files = sorted([f for f in all_pngs if f.name.startswith('trends_')])
            undp_files = sorted([f for f in all_pngs if 'timeline' in f.name.lower() or 'gantt' in f.name.lower()])
            impact_files = sorted([f for f in all_pngs if any(k in f.name.lower() for k in ['coef_plot','did_','impact_'])])
            raster_pairs = collect_rasters()
            if trend_files: tab_names.append("🔍 Google Trends")
            if undp_files: tab_names.append("🏗️ UNDP Programme")
            if impact_files: tab_names.append("📐 Impact Analysis")
            if raster_pairs: tab_names.append("🛰️ High-Resolution Maps")
            tabs = st.tabs(tab_names); ti=0

            for tn,th in THEMES.items():
                with tabs[ti]:
                    st.caption(th['desc']); found=False
                    for ind in th['indicators']:
                        figs=find_figs(ind,all_pngs)
                        if figs:
                            found=True; st.markdown(f"**{NAMES.get(ind,ind.replace('_',' ').title())}**")
                            ts=[f for f in figs if f.name.startswith('ts_')]; mp=[f for f in figs if f.name.startswith('map_')]
                            if ts and mp:
                                c1,c2=st.columns([3,2])
                                with c1: st.image(str(ts[0]),caption="Time Series")
                                with c2: st.image(str(mp[0]),caption="Map")
                            elif ts: st.image(str(ts[0]))
                            elif mp: st.image(str(mp[0]))
                            st.markdown("---")
                    if not found: st.info("No data for this area yet.")
                ti+=1
            if trend_files:
                with tabs[ti]:
                    st.caption("Google search interest as proxy for public concern")
                    for f in trend_files:
                        st.markdown(f"**{f.stem.replace('trends_','').replace('_',' ').title()}**"); st.image(str(f))
                ti+=1
            if undp_files:
                with tabs[ti]:
                    st.caption("UNDP project timeline by sector")
                    for f in undp_files: st.image(str(f),caption=f.stem.replace('_',' ').title())
                ti+=1
            if impact_files:
                with tabs[ti]:
                    st.caption("Difference-in-Differences causal estimates by buffer zone")
                    for f in impact_files:
                        st.markdown(f"**{f.stem.replace('coef_plot_','').replace('_buffer','').replace('_',' ').title()}**"); st.image(str(f))
                    for rp in [output_dir/"results"/"did_buffer_results.csv", figures_dir/"did_results.csv"]:
                        if rp.exists(): st.markdown("**Regression Results**"); st.dataframe(pd.read_csv(rp),hide_index=True); break
                ti+=1
            if raster_pairs:
                with tabs[ti]:
                    st.caption("Pixel-level satellite imagery: pre-disaster, post-disaster, post-UNDP")
                    for ind,f in raster_pairs:
                        st.markdown(f"**{RASTER_NAMES.get(ind, ind.upper())}**"); st.image(str(f))

elif mode == "🔧 Run Pipeline":
    st.markdown('<div class="section-title">Pipeline Setup & Execution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Credentials</div>', unsafe_allow_html=True)
    st.markdown('<div class="setup-box"><h4>🛰️ Google Earth Engine</h4><p>One-time terminal setup:</p><ol><li>Account: <a href="https://earthengine.google.com/">earthengine.google.com</a></li><li>Run: <code>earthengine authenticate</code></li><li>Follow browser prompts — token saved permanently</li></ol></div>', unsafe_allow_html=True)
    st.markdown('<div class="setup-box"><h4>⚔️ ACLED</h4><p>Register at <a href="https://acleddata.com/">acleddata.com</a>, enter below.</p></div>', unsafe_allow_html=True)
    ca,cb=st.columns(2)
    with ca: acled_email=st.text_input("ACLED Email",value=os.getenv("ACLED_EMAIL",""))
    with cb: acled_key=st.text_input("ACLED Password",value=os.getenv("ACLED_KEY",""),type="password")
    st.caption("💡 Save in `.env` to persist: `ACLED_EMAIL=...` and `ACLED_KEY=...`")
    st.markdown('<div class="section-title">2. Google Trends Keywords</div>', unsafe_allow_html=True)
    st.caption(f"**Defaults:** {', '.join(DEFAULT_KW)}")
    extra_kw=st.text_input("Additional keywords (comma-separated, no quotes needed)",placeholder="e.g., tsunami, passport, refugee")
    st.markdown('<div class="setup-box"><h4>💡 How to add keywords</h4><p>Type extra words separated by <b>commas</b>. No quotation marks needed.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">3. Run Pipeline</div>', unsafe_allow_html=True)
    steps=[("🗺️ Boundaries",f"make boundaries COUNTRY={selected}","Admin boundaries"),("🛰️ Satellite",f"make gee COUNTRY={selected}","VIIRS, NDVI, SAR, rainfall from GEE"),("⚔️ Conflict",f"make acled COUNTRY={selected}","ACLED events & fatalities"),("🏗️ UNDP",f"make undp COUNTRY={selected}","UNDP API + ML classification"),("📈 Trends",f"make gt COUNTRY={selected}","Google search interest"),("🔧 Panel",f"make preprocess COUNTRY={selected}","Build unit×month panel"),("📊 Visuals",f"make visuals COUNTRY={selected}","Time series, maps, timeline"),("🎯 Scorecard",f"python pipeline/recovery_scorecard.py --config config/{selected}.yaml","Recovery scorecard")]
    def run_cmd(cmd,desc):
        env=os.environ.copy()
        if acled_email: env['ACLED_EMAIL']=acled_email
        if acled_key: env['ACLED_KEY']=acled_key
        with st.status(f"{desc}...",expanded=True) as s:
            try:
                r=subprocess.run(cmd,shell=True,capture_output=True,text=True,env=env,timeout=600)
                st.code(r.stdout[-3000:] or "(none)",language="bash")
                if r.returncode==0: s.update(label=f"✅ {desc}",state="complete")
                else: st.error(r.stderr[-500:] if r.stderr else "Error"); s.update(label=f"⚠️ {desc}",state="error")
            except subprocess.TimeoutExpired: s.update(label="⏱️ Timeout",state="error")
    for lb,cmd,desc in steps:
        c1,c2=st.columns([1,3])
        with c1: cl=st.button(lb,use_container_width=True)
        with c2: st.caption(desc)
        if cl: run_cmd(cmd,lb)
    st.markdown("---")
    if st.button("▶️ Run Full Pipeline",use_container_width=True,type="primary"): run_cmd(f"make run COUNTRY={selected} && python pipeline/recovery_scorecard.py --config config/{selected}.yaml","Full Pipeline"); st.balloons()

elif mode == "➕ Add New Country":
    st.markdown('<div class="section-title">Add a New Country</div>', unsafe_allow_html=True)
    country_names = sorted(COUNTRY_REFERENCE.keys())
    country_choice = st.selectbox("Select Country", country_names, index=None, placeholder="Choose a country...")
    if country_choice and country_choice in COUNTRY_REFERENCE:
        ref = COUNTRY_REFERENCE[country_choice]
        default_name, default_iso, default_acled = country_choice, ref['iso'], ref['acled']
    else:
        default_name, default_iso, default_acled = "", "", ""
    with st.form("new_country"):
        st.markdown("**Country Details**")
        c1,c2,c3=st.columns(3)
        with c1: country_name=st.text_input("Country Name",value=default_name,placeholder="Nepal")
        with c2: country_code=st.text_input("ISO Code",value=default_iso,placeholder="NPL",max_chars=3)
        with c3: acled_country=st.text_input("ACLED Name",value=default_acled,placeholder="Nepal")
        st.markdown("**Disaster Event**")
        c1,c2,c3,c4=st.columns(4)
        with c1: dtype=st.selectbox("Type",["earthquake","conflict","flood","cyclone","drought"])
        with c2: ddate=st.date_input("Disaster Date",date(2025,1,1))
        with c3: idate=st.date_input("UNDP Start",date(2025,2,1))
        with c4: alevel=st.selectbox("Admin Level",[1,2,3],index=2)
        st.markdown("**Study Period**")
        c1,c2=st.columns(2)
        with c1: sstart=st.date_input("Start",date(2023,1,1))
        with c2: send=st.date_input("End",date(2026,1,31))
        if st.form_submit_button("💾 Save Configuration",type="primary"):
            cn=(country_name or "").strip(); cc=(country_code or "").strip().upper(); ac=(acled_country or cn).strip()
            if not cn or not cc or len(cc)!=3: st.error("Need country name + valid 3-letter ISO code.")
            else:
                nc={'country_name':cn,'country_code':cc,'study_period_start':sstart.isoformat(),'study_period_end':send.isoformat(),'events':{'disaster_type':dtype,'disaster_date':ddate.isoformat(),'intervention_start':idate.isoformat()},'boundaries':{'file_path':f"data/{cc.lower()}/boundaries/{cc.lower()}_admin{alevel}.geojson",'id_column':'shapeID','name_column':'shapeName','admin_level':alevel},'acled_country_name':ac,'google_trends_keywords':DEFAULT_KW}
                Path("config").mkdir(exist_ok=True)
                with open(f"config/{cc.lower()}.yaml",'w') as f: yaml.dump(nc,f,default_flow_style=False,sort_keys=False)
                st.success(f"✅ Saved! Select **{cn}** from the sidebar."); st.rerun()

with st.expander("📝 Configuration YAML"):
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")
st.markdown('<div class="footer">UNDP Impact Evaluation Pipeline · Built by AImpact Lab · VIIRS · MODIS · Sentinel-1 · CHIRPS · ACLED</div>', unsafe_allow_html=True)
