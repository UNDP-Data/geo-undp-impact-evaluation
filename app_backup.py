#!/usr/bin/env python3
"""
app.py — UNDP Early Recovery Impact (v15)
==========================================
Usage: streamlit run app.py
v15: Clean rewrite with all fixes:
  - Scorecard: green overall, blue intervention, no trend column, no pre-event
  - UNDP programme picker in View Results (below scorecard)
  - Simplified Setup & Run (no Add Country / Admin)
  - Sidebar hidden in Setup mode
  - Renamed indicators and tabs
"""

import streamlit as st
import streamlit.components.v1 as components
import yaml, subprocess, os, glob
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

st.set_page_config(page_title="UNDP Early Recovery Impact", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', -apple-system, sans-serif !important; background: #0d1117; color: #c9d1d9; }
    .main .block-container { max-width: 1300px; padding: 1.5rem 2rem 4rem 2rem; background: #0d1117; }
    section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    section[data-testid="stSidebar"] input { background: #161b22 !important; color: #c9d1d9 !important; border-color: #30363d !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div { background: #161b22 !important; }
    section[data-testid="stSidebar"] hr { border-color: #21262d !important; }
    .country-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: .5rem; }
    .country-card .country-name { color: #e6edf3 !important; font-weight: 600; font-size: 1.05rem; }
    .country-card .country-detail { color: #8b949e !important; font-size: .83rem; margin-top: .15rem; }
    .hero { background: linear-gradient(135deg, #0468B1 0%, #0d6ebd 50%, #1a7fd4 100%); padding: 2rem 2.5rem; border-radius: 8px; margin-bottom: 1.2rem; border: 1px solid #1a7fd4; }
    .hero h1 { font-family: 'Inter', sans-serif !important; font-size: 1.8rem; margin: 0 0 .25rem 0; font-weight: 700; color: white !important; letter-spacing: -0.02em; }
    .hero p { font-size: .95rem; margin: 0; color: rgba(255,255,255,.85) !important; font-weight: 400; }
    .hero-badge { display: inline-block; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2); padding: .2rem .7rem; border-radius: 2em; font-size: .78rem; margin-top: .6rem; color: rgba(255,255,255,.8) !important; }
    .metric-row { display: flex; gap: .6rem; margin: .8rem 0 1.2rem 0; }
    .metric-box { flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: .8rem 1rem; }
    .metric-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: #8b949e; font-weight: 500; }
    .metric-value { font-size: 1.05rem; font-weight: 600; color: #e6edf3; margin-top: .15rem; }
    .data-row { display: flex; align-items: center; padding: .5rem 0; border-bottom: 1px solid #21262d; }
    .data-row:last-child { border-bottom: none; }
    .data-icon { font-size: .9rem; width: 1.6rem; text-align: center; color: #8b949e; }
    .data-label { flex: 1; font-size: .85rem; font-weight: 500; color: #c9d1d9; }
    .data-status { border-radius: 2em; padding: 2px 10px; font-size: .75rem; font-weight: 500; }
    .data-ready { background: rgba(35, 134, 54, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
    .data-missing { background: rgba(187, 128, 9, 0.15); color: #d29922; border: 1px solid rgba(187, 128, 9, 0.3); }
    .section-title { font-family: 'Inter', sans-serif !important; font-size: 1.15rem; font-weight: 600; color: #e6edf3; margin: 1.5rem 0 .6rem 0; padding-bottom: .35rem; border-bottom: 1px solid #30363d; }
    button[data-baseweb="tab"] > div > p { color: #8b949e !important; }
    button[data-baseweb="tab"][aria-selected="true"] > div > p { color: #58a6ff !important; font-weight: 600 !important; }
    .footer { text-align: center; padding: 1.5rem 0 .5rem 0; border-top: 1px solid #21262d; margin-top: 2.5rem; color: #484f58; font-size: .75rem; }
    div[data-baseweb="select"] > div { border-color: #30363d !important; }
    div[data-baseweb="select"]:focus-within > div { border-color: #58a6ff !important; box-shadow: none !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
    header[data-testid="stHeader"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
def iso_to_flag(code):
    try:
        import pycountry
        c2 = pycountry.countries.get(alpha_3=code.upper())
        if c2: return ''.join(chr(0x1F1E6 + ord(ch) - ord('A')) for ch in c2.alpha_2)
    except Exception: pass
    return '🌐'

COUNTRIES = {
    'mmr': {'flag':'🇲🇲','name':'Myanmar','event':'M7.7 Earthquake','event_date':'Mar 28, 2025'},
    'myanmar': {'flag':'🇲🇲','name':'Myanmar','event':'M7.7 Earthquake','event_date':'Mar 28, 2025'},
    'pse': {'flag':'🇵🇸','name':'Gaza Strip','event':'Armed Conflict','event_date':'Oct 7, 2023'},
    'afg': {'flag':'🇦🇫','name':'Afghanistan','event':'M6.3 Earthquake','event_date':'Oct 7, 2023'},
    'ukr': {'flag':'🇺🇦','name':'Ukraine','event':'Armed Conflict','event_date':'Feb 24, 2022'},
    'sdn': {'flag':'🇸🇩','name':'Sudan','event':'Civil War','event_date':'Apr 15, 2023'},
    'syr': {'flag':'🇸🇾','name':'Syria','event':'M7.8 Earthquake','event_date':'Feb 6, 2023'},
    'cod': {'flag':'🇨🇩','name':'DR Congo','event':'M23 Conflict','event_date':'Mar 1, 2022'},
    'som': {'flag':'🇸🇴','name':'Somalia','event':'Drought','event_date':'Jan 1, 2023'},
    'yem': {'flag':'🇾🇪','name':'Yemen','event':'Armed Conflict','event_date':'Jan 1, 2022'},
    'hti': {'flag':'🇭🇹','name':'Haiti','event':'Armed Conflict','event_date':'Mar 1, 2024'},
    'lbn': {'flag':'🇱🇧','name':'Lebanon','event':'Armed Conflict','event_date':'Oct 1, 2024'},
    'irq': {'flag':'🇮🇶','name':'Iraq','event':'Armed Conflict','event_date':'Jan 1, 2022'},
    'eth': {'flag':'🇪🇹','name':'Ethiopia','event':'Tigray Conflict','event_date':'Nov 4, 2020'},
}

THEMES = {
    '💡 Economic Recovery': {'desc':'Nighttime lights as proxy for economic activity and energy access','indicators':['nightlights','price_rice','price_oil','price_bread','price_flour','price_chicken','price_milk','price_sugar']},
    '🌿 Vegetation': {'desc':'Vegetation and ecosystem recovery across affected regions','indicators':['ndvi','rvi','rainfall']},
    '🏚️ Debris & Surface': {'desc':'Surface roughness changes via radar — debris, rubble, and land clearing','indicators':['vv']},
    '⚔️ Conflict': {'desc':'Armed conflict events and fatalities from ACLED','indicators':['conflict_events','conflict_fatalities']},
}
NAMES = {'nightlights':'Energy & Economic Activity','ndvi':'Vegetation Health','vv':'Debris & Surface (Radar VV)','vh':'Infrastructure (Radar VH)','rvi':'Radar Vegetation','rainfall':'Rainfall','conflict_events':'Conflict Events','conflict_fatalities':'Conflict Fatalities','price_rice':'Rice Price','price_oil':'Oil Price','price_bread':'Bread Price','price_flour':'Flour Price','price_chicken':'Chicken Price','price_milk':'Milk Price','price_sugar':'Sugar Price'}
RASTER_IND = ['viirs','ndvi','ndbi','bsi','vv','evi','savi','nbr','swir','mndwi']
RASTER_NAMES = {'viirs':'Energy & Nightlights','ndvi':'Vegetation Health','ndbi':'Built-Up Areas','bsi':'Bare Soil & Debris','evi':'Enhanced Vegetation','savi':'Soil-Adjusted Vegetation','nbr':'Burn Ratio','vv':'Surface Roughness (Radar)','swir':'Short-Wave Infrared','mndwi':'Water Coverage'}

METHODOLOGY_NOTE = """<b>Methodological note.</b> Scores are calculated only for administrative units that experienced a significant shock, defined as a decline greater than 10% relative to the pre-event baseline. Indicators below this threshold are classified as <em>Stable</em> and are excluded from recovery scoring. To ensure comparability over time, Vegetation and Nightlights indicators are adjusted on a year-over-year basis to remove natural seasonal variation (e.g., harvest cycles, snow cover, or holiday effects). <b>Overall Recovery</b> measures the share of the initial decline that has been regained since the lowest observed point of the crisis (the trough), capped at 100%. <b>Since Intervention</b> captures the recovery achieved after the intervention start date, expressed as a percentage of the total disaster impact and likewise capped at 100%. These metrics describe observed temporal dynamics to support rapid prioritisation and resource allocation; they reflect recovery trends but do not imply direct causal attribution to any single organisation or intervention.<br><br><b>Disclaimer.</b> The accuracy of these results depends on the quality and resolution of the underlying satellite data. In some contexts, indicators such as nighttime lights may be affected by seasonal patterns, cloud cover, snow reflectance, or sensor calibration changes that are not fully captured by the year-over-year adjustment. This dashboard presents <em>correlations</em> between the timing of an event, the start of early recovery interventions, and changes in satellite-derived indicators; it does not constitute evidence of causal impact. In most contexts, UNDP is not the only actor on the ground, and observed recovery trends reflect the combined contributions of multiple organisations and local actors. Isolating UNDP's specific contribution requires econometric approaches (e.g., Difference-in-Differences, Synthetic Control) available in the Impact Measurement tab, which support attribution to the extent possible given data constraints."""

QUICK_REF = {
    'Afghanistan':  {'iso':'AFG','acled':'Afghanistan','etype':'earthquake','edate':'2023-10-07','commodity':'Wheat flour','admin':2},
    'Bangladesh':   {'iso':'BGD','acled':'Bangladesh','etype':'cyclone','edate':'2024-05-01','commodity':'Rice','admin':2},
    'DR Congo':     {'iso':'COD','acled':'Democratic Republic of Congo','etype':'conflict','edate':'2022-03-01','commodity':'Cassava flour','admin':2},
    'Ethiopia':     {'iso':'ETH','acled':'Ethiopia','etype':'conflict','edate':'2020-11-04','commodity':'Wheat','admin':2},
    'Gaza Strip':   {'iso':'PSE','acled':'Palestine','etype':'conflict','edate':'2023-10-07','commodity':'Bread','admin':2},
    'Haiti':        {'iso':'HTI','acled':'Haiti','etype':'conflict','edate':'2024-03-01','commodity':'Rice','admin':2},
    'Iraq':         {'iso':'IRQ','acled':'Iraq','etype':'conflict','edate':'2022-01-01','commodity':'Wheat flour','admin':2},
    'Lebanon':      {'iso':'LBN','acled':'Lebanon','etype':'conflict','edate':'2024-10-01','commodity':'Bread','admin':2},
    'Myanmar':      {'iso':'MMR','acled':'Myanmar','etype':'earthquake','edate':'2025-03-28','commodity':'Rice','admin':3},
    'Somalia':      {'iso':'SOM','acled':'Somalia','etype':'drought','edate':'2023-01-01','commodity':'Sorghum','admin':2},
    'Sudan':        {'iso':'SDN','acled':'Sudan','etype':'conflict','edate':'2023-04-15','commodity':'Sorghum','admin':2},
    'Syria':        {'iso':'SYR','acled':'Syria','etype':'earthquake','edate':'2023-02-06','commodity':'Bread','admin':2},
    'Ukraine':      {'iso':'UKR','acled':'Ukraine','etype':'conflict','edate':'2022-02-24','commodity':'Wheat flour','admin':2},
    'Yemen':        {'iso':'YEM','acled':'Yemen','etype':'conflict','edate':'2022-01-01','commodity':'Wheat flour','admin':2},
}


# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>UNDP Early Recovery Impact</h1>
    <p>Satellite data indicators to monitor early recovery interventions</p>
    <div class="hero-badge">📡 Powered by AImpact Lab — geospatial pipelines for satellite-based impact measurement</div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.radio("Mode", ["📊 View Results", "⚙️ Setup & Run"],
                    label_visibility="collapsed", key="main_mode")
    st.markdown("---")

    cfg_files = sorted(glob.glob("config/*.yaml"))
    cfg_names = [Path(f).stem for f in cfg_files]

    if mode == "📊 View Results":
        st.markdown("#### 🌍 Select Country")
        if not cfg_names:
            st.warning("No configs found. Go to Setup & Run.")
            st.stop()
        _auto = st.session_state.pop("auto_select_country", None)
        _def_idx = cfg_names.index(_auto) if _auto and _auto in cfg_names else 0
        selected = st.selectbox("Country", cfg_names, index=_def_idx,
                                format_func=lambda n: f"{COUNTRIES.get(n,{}).get('flag', iso_to_flag(n))}  {COUNTRIES.get(n,{}).get('name', n.replace('_',' ').title())}",
                                label_visibility="collapsed")
        with open(f"config/{selected}.yaml") as f:
            config = yaml.safe_load(f)
        ci = COUNTRIES.get(selected, {})
        st.markdown(f'<div class="country-card"><span style="font-size:1.5rem;margin-right:.4rem;">{ci.get("flag", iso_to_flag(selected))}</span><span class="country-name">{config["country_name"]}</span><div class="country-detail">{ci.get("event","")} · {ci.get("event_date", config["events"]["disaster_date"])}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("#### ⚙️ Setup & Run")
        st.caption("Configure a country and run the data pipeline.")
        # Load fallback config for shared variables
        if cfg_names:
            _auto = st.session_state.pop("auto_select_country", None)
            selected = _auto if _auto and _auto in cfg_names else cfg_names[0]
            with open(f"config/{selected}.yaml") as f:
                config = yaml.safe_load(f)
        else:
            selected = "none"
            config = {'country_name':'(none)','country_code':'XXX','events':{'disaster_date':'2025-01-01','intervention_start':'2025-01-01'},'study_period_start':'2024-01-01','study_period_end':'2025-12-31','boundaries':{'admin_level':2}}

    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.72rem;color:#484f58;">Built by AImpact Lab</div>', unsafe_allow_html=True)


# ─── SHARED VARIABLES ────────────────────────────────────────────────────────
code = config['country_code'].lower()
figures_dir = Path(f"outputs/{code}/figures")
output_dir = Path(f"outputs/{code}")
results_dir = output_dir / "results"
panel_path = output_dir / "panel_data.csv"
panel_info = ""
if panel_path.exists():
    try:
        p = pd.read_csv(panel_path, usecols=['unit_id','period_start'])
        panel_info = f"{p['unit_id'].nunique()} units × {p['period_start'].nunique()} mo"
    except Exception:
        panel_info = "Available"
n_figs = len(list(figures_dir.rglob("*.png"))) if figures_dir.exists() else 0
event_date = config['events']['disaster_date']
intervention = config['events'].get('intervention_start', event_date)

st.markdown(f'<div class="metric-row"><div class="metric-box"><div class="metric-label">Event Date</div><div class="metric-value">{event_date}</div></div><div class="metric-box"><div class="metric-label">Study Period</div><div class="metric-value">{str(config.get("study_period_start",""))[:7]} → {str(config.get("study_period_end",""))[:7]}</div></div><div class="metric-box"><div class="metric-label">Panel</div><div class="metric-value">{panel_info or "Not built"}</div></div><div class="metric-box"><div class="metric-label">Figures</div><div class="metric-value">{n_figs}</div></div></div>', unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def find_figs(indicator, pngs):
    out = []
    for f in pngs:
        s = f.stem.lower()
        if s == f"ts_{indicator}" or s == f"map_{indicator}":
            out.append(f)
        elif indicator in s and any(s.startswith(p) for p in ['ts_','map_']):
            out.append(f)
    return sorted(out)

def collect_rasters():
    candidates = []
    for sub in ['raster_maps','undp_maps']:
        d = figures_dir / sub
        if d.exists():
            for f in sorted(d.glob("*.png")):
                if 'three_panel' in f.stem.lower():
                    candidates.append(f)
    groups = {}
    for f in candidates:
        nm = f.stem.lower(); ind = None
        for ri in RASTER_IND:
            if nm.startswith(ri): ind = ri; break
        if ind is None: ind = nm.split('_')[0]
        k = 'clean' if 'clean' in nm else ('geocode' if 'geocode' in nm else 'other')
        if ind not in groups: groups[ind] = {'clean':None,'geocode':None,'other':None}
        groups[ind][k] = f
    return [(ind, g['clean'] or g['geocode'] or g['other']) for ind, g in sorted(groups.items()) if g['clean'] or g['geocode'] or g['other']]

def collect_showcase_files():
    showcase_dir = output_dir / "showcase"
    if not showcase_dir.exists():
        return {'plots':[],'tables':[],'pretrends':[],'coef_plots':[],'maps':[]}
    files = {'plots':[],'tables':[],'pretrends':[],'coef_plots':[],'maps':[]}
    for f in sorted(showcase_dir.iterdir()):
        if f.suffix == '.csv': files['tables'].append(f)
        elif f.suffix == '.png':
            nm = f.stem.lower()
            if 'pretrend' in nm: files['pretrends'].append(f)
            elif 'coef_plot' in nm: files['coef_plots'].append(f)
            elif 'three_panel' in nm or 'map' in nm or 'debris' in nm: files['maps'].append(f)
            else: files['plots'].append(f)
    return files


def render_scorecard():
    csv_path = figures_dir / "scorecard_data.csv"
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    df = df[df['indicator'] != 'vh']

    name_map = {'Nightlights':'Economic Activity', 'Vegetation':'Vegetation', 'Radar VV':'Debris & Surface', 'Radar VH':'Debris & Surface'}
    desc_map = {'Economic Activity':'Nighttime lights (VIIRS)', 'Vegetation':'Vegetation index (NDVI)', 'Debris & Surface':'Radar backscatter (Sentinel-1 VV)'}

    rows_html = ""
    for _, row in df.iterrows():
        overall = row['overall_recovery_pct']
        undp = row['undp_recovery_pct']
        n_aff = int(row.get('n_affected_units', 0))
        n_tot = int(row.get('n_total_units', 0))
        short = row.get('short_name', '')
        display_name = name_map.get(short, short)
        display_desc = desc_map.get(display_name, row.get('description', ''))

        if overall < 0:
            rows_html += f"""
            <div class="sc-row">
                <div class="sc-name">{display_name}</div>
                <div class="sc-desc">{display_desc}</div>
                <div class="sc-bars"><div class="sc-stable">Stable — no significant decline detected</div></div>
                <div class="sc-vals"><div class="sc-stable-badge">Stable</div></div>
            </div>"""
            continue

        oc = max(0, min(overall, 100))
        uc = max(0, min(undp, 100)) if undp >= 0 else 0
        if uc > oc and oc > 0:
            uc = oc

        oc_c = "#3fb950" if overall >= 75 else ("#d29922" if overall >= 50 else "#f85149")
        if undp < 0:
            uc_c = "#484f58"; ud = "N/A"
        elif undp >= 50:
            uc_c = "#58a6ff"; ud = f"{undp:.0f}%"
        elif undp >= 25:
            uc_c = "#79c0ff"; ud = f"{undp:.0f}%"
        else:
            uc_c = "#8b949e"; ud = f"{undp:.0f}%"

        units_label = f"{n_aff} of {n_tot} units" if n_tot > 0 else ""
        rows_html += f"""
            <div class="sc-row">
                <div class="sc-name">{display_name}<br><span class="sc-units">{units_label}</span></div>
                <div class="sc-desc">{display_desc}</div>
                <div class="sc-bars">
                    <div class="sc-track"><div class="sc-fill-a" style="width:{oc}%;"></div></div>
                    <div class="sc-track"><div class="sc-fill-b" style="width:{uc}%;"></div></div>
                </div>
                <div class="sc-vals">
                    <div class="sc-val-line"><span class="sc-pct" style="color:{oc_c};">{overall:.0f}%</span><span class="sc-lbl">overall</span></div>
                    <div class="sc-val-line"><span class="sc-pct" style="color:{uc_c};">{ud}</span><span class="sc-lbl">since programme</span></div>
                </div>
            </div>"""

    html = f"""
    <style>
        body {{ background: transparent; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; color: #c9d1d9; }}
        .sc {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem 1.8rem; }}
        .sc-title {{ font-size: 1.2rem; font-weight: 700; color: #e6edf3; margin: 0 0 .2rem 0; }}
        .sc-sub {{ font-size: .82rem; color: #8b949e; margin: 0 0 .8rem 0; }}
        .sc-legend {{ display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .sc-legend-item {{ display: flex; align-items: center; gap: .35rem; font-size: .75rem; color: #8b949e; }}
        .sc-legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}
        .sc-row {{ display: flex; align-items: center; padding: .65rem 0; border-bottom: 1px solid #21262d; }}
        .sc-row:last-of-type {{ border-bottom: none; }}
        .sc-name {{ width: 140px; font-weight: 600; color: #e6edf3; font-size: .85rem; line-height: 1.3; }}
        .sc-units {{ font-size: .68rem; color: #484f58; font-weight: 400; }}
        .sc-desc {{ width: 180px; font-size: .72rem; color: #8b949e; padding-right: .5rem; }}
        .sc-bars {{ flex: 1; padding: 0 .5rem; }}
        .sc-track {{ background: #0d1117; border-radius: 4px; height: 10px; position: relative; margin: 4px 0; }}
        .sc-fill-a {{ background: linear-gradient(90deg, #238636, #3fb950); height: 10px; border-radius: 4px; }}
        .sc-fill-b {{ background: linear-gradient(90deg, #1f6feb, #79c0ff); height: 10px; border-radius: 4px; }}
        .sc-vals {{ width: 160px; text-align: left; padding-left: .8rem; }}
        .sc-val-line {{ display: flex; align-items: baseline; gap: .3rem; }}
        .sc-pct {{ font-weight: 700; font-size: 1.05rem; min-width: 45px; text-align: right; }}
        .sc-lbl {{ color: #484f58; font-size: .72rem; }}
        .sc-stable {{ color: #3fb950; font-size: .82rem; padding: 6px 0; font-weight: 500; }}
        .sc-stable-badge {{ display: inline-block; background: rgba(35,134,54,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 2em; padding: 1px 10px; font-size: .75rem; font-weight: 500; }}
        .sc-note {{ font-size: .68rem; color: #8b949e; margin-top: .8rem; line-height: 1.6; }}
        .sc-note b {{ color: #c9d1d9; }}
        .sc-note em {{ color: #3fb950; font-style: normal; font-weight: 600; }}
    </style>
    <div class="sc">
        <div class="sc-title">Recovery Scorecard</div>
        <div class="sc-sub">Event: {event_date} &nbsp;·&nbsp; Intervention: {intervention}</div>
        <div class="sc-legend">
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#3fb950;"></div>Overall Recovery</div>
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#1f6feb;"></div>Since Selected Programme</div>
        </div>
        {rows_html}
        <div class="sc-note">{METHODOLOGY_NOTE}</div>
    </div>
    """
    n_rows = len(df)
    components.html(html, height=80 + n_rows * 70 + 40 + 240, scrolling=False)
    return True


# ═════════════════════════════════════════════════════════════════════════════
# VIEW RESULTS
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📊 View Results":
    # SCORECARD
    has_scorecard = render_scorecard()
    if not has_scorecard:
        st.info(f"Run the pipeline first: `make run COUNTRY={selected}`")

    # UNDP PROGRAMME PICKER (below scorecard)
    if has_scorecard:
        undp_csv = Path(f"data/{code}/external/undp_projects.csv")
        if undp_csv.exists():
            undp_df = pd.read_csv(undp_csv)
            if len(undp_df) > 0 and 'start_date' in undp_df.columns:
                undp_df['start_date'] = pd.to_datetime(undp_df['start_date'], errors='coerce')
                undp_valid = undp_df.dropna(subset=['start_date']).sort_values('start_date')
                event_dt = pd.Timestamp(event_date)
                post = undp_valid[undp_valid['start_date'] >= event_dt].copy()
                if len(post) > 0:
                    options = []
                    for _, pr in post.iterrows():
                        d = pr['start_date'].strftime('%Y-%m-%d')
                        t = str(pr.get('title',''))[:65]
                        b = f"${pr['total_budget']:,.0f}" if pd.notna(pr.get('total_budget')) else ""
                        options.append({'label': f"{d} — {t} {b}".strip(), 'date': d})
                    current_int = config['events'].get('intervention_start', event_date)
                    current_idx = 0
                    for i, o in enumerate(options):
                        if o['date'] == current_int:
                            current_idx = i
                            break
                    sel_prog = st.selectbox(
                        "UNDP Programme",
                        range(len(options)),
                        format_func=lambda i: options[i]['label'],
                        index=current_idx,
                        key="undp_prog_pick"
                    )
                    new_int = options[sel_prog]['date']
                    if new_int != current_int:
                        config['events']['intervention_start'] = new_int
                        with open(f"config/{selected}.yaml", 'w') as f_out:
                            yaml.dump(config, f_out, default_flow_style=False, sort_keys=False)
                        subprocess.run(
                            f"python pipeline/recovery_scorecard.py --config config/{selected}.yaml",
                            shell=True, capture_output=True, timeout=60
                        )
                        st.rerun()

    # DETAILED INDICATORS
    if figures_dir.exists():
        all_pngs = sorted(figures_dir.glob("*.png"))
        if all_pngs:
            st.markdown('<div class="section-title">Detailed Indicators</div>', unsafe_allow_html=True)
            tab_names = list(THEMES.keys())
            trend_files = sorted([f for f in all_pngs if f.name.startswith('trends_')])
            undp_files = sorted([f for f in all_pngs if 'timeline' in f.name.lower() or 'gantt' in f.name.lower()])
            raster_pairs = collect_rasters()
            impact_files = collect_showcase_files()
            has_impact = any(v for v in impact_files.values())
            if trend_files: tab_names.append("🔍 Google Trends")
            if undp_files: tab_names.append("🏗️ Programme Timeline")
            if has_impact: tab_names.append("📐 Impact Measurement")
            if raster_pairs: tab_names.append("🛰️ Satellite Maps")
            tabs = st.tabs(tab_names)
            ti = 0

            for tn, th in THEMES.items():
                with tabs[ti]:
                    st.caption(th['desc'])
                    found = False
                    for ind in th['indicators']:
                        figs = find_figs(ind, all_pngs)
                        if figs:
                            found = True
                            st.markdown(f"**{NAMES.get(ind, ind.replace('_',' ').title())}**")
                            ts = [f for f in figs if f.name.startswith('ts_')]
                            mp = [f for f in figs if f.name.startswith('map_')]
                            if ts and mp:
                                c1, c2 = st.columns(2)
                                with c1: st.image(str(ts[0]), width='stretch')
                                with c2: st.image(str(mp[0]), width='stretch')
                            elif ts: st.image(str(ts[0]), width='stretch')
                            elif mp: st.image(str(mp[0]), width='stretch')
                            st.markdown("---")
                    if not found: st.info("No data yet.")
                ti += 1

            if trend_files:
                with tabs[ti]:
                    st.caption("Google search interest as proxy for public concern")
                    for f in trend_files:
                        st.markdown(f"**{f.stem.replace('trends_','').replace('_',' ').title()}**")
                        st.image(str(f), width='stretch')
                ti += 1

            if undp_files:
                with tabs[ti]:
                    for f in undp_files: st.image(str(f), width='stretch')
                ti += 1

            if has_impact:
                with tabs[ti]:
                    st.caption("Econometric impact estimates using Difference-in-Differences and buffer zone analysis.")
                    if impact_files['maps']:
                        st.markdown("**Satellite Maps — UNDP Intervention Areas**")
                        for f in impact_files['maps']:
                            label = f.stem.replace('_clean','').replace('_three_panel','').replace('_',' ').title()
                            st.markdown(f"*{label}*")
                            st.image(str(f), width='stretch')
                        st.markdown("---")
                    if impact_files['coef_plots']:
                        st.markdown("**Buffer Zone Analysis — Treatment Effect by Distance**")
                        n_cols = min(len(impact_files['coef_plots']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['coef_plots']):
                            label = f.stem.replace('coef_plot_','').replace('_buffer','').replace('_',' ').title()
                            with cols[i % n_cols]: st.image(str(f), caption=label, width='stretch')
                        st.markdown("---")
                    if impact_files['plots']:
                        st.markdown("**Difference-in-Differences Estimates**")
                        for f in impact_files['plots']: st.image(str(f), width='stretch')
                        st.markdown("---")
                    if impact_files['pretrends']:
                        st.markdown("**Pre-Trends Validation**")
                        n_cols = min(len(impact_files['pretrends']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['pretrends']):
                            label = f.stem.replace('pretrends_','').replace('_baseline','').replace('_',' ').title()
                            with cols[i % n_cols]: st.image(str(f), caption=label, width='stretch')
                        st.markdown("---")
                    if impact_files['tables']:
                        st.markdown("**Regression Results**")
                        for f in impact_files['tables']:
                            df_table = pd.read_csv(f)
                            if 'comprehensive_unified' in f.name and 'control_group' in df_table.columns:
                                df_table = df_table[df_table['control_group'].str.contains('500m.*1.5km|500m-1.5km|0.5.*1.5', case=False, na=False)]
                            st.markdown(f"*{f.stem.replace('_',' ').title()}*")
                            st.dataframe(df_table, hide_index=True, use_container_width=True)
                ti += 1

            if raster_pairs:
                with tabs[ti]:
                    st.caption("Pre-Event, Post-Event, and Post-Intervention satellite imagery")
                    for ind, f in raster_pairs:
                        st.markdown(f"**{RASTER_NAMES.get(ind, ind.upper())}**")
                        st.image(str(f), width='stretch')

    # DATA & DOWNLOADS
    st.markdown('<div class="section-title">Data Availability</div>', unsafe_allow_html=True)
    srcs = [("🗺️","Boundaries",config['boundaries']['file_path']),("🛰️","Satellite",f"data/{code}/gee_panel_{code}_centroid.csv"),("⚔️","Conflict",f"data/{code}/external/acled_conflict.csv"),("🏗️","UNDP Projects",f"data/{code}/external/undp_projects.csv"),("📈","Google Trends",f"data/{code}/external/google_trends.csv"),("📊","Panel",f"outputs/{code}/panel_data.csv")]
    cols = st.columns(3)
    for i, (ic, lb, pt) in enumerate(srcs):
        pp = Path(pt); ok = pp.exists()
        try: nn = f" ({len(pd.read_csv(pp)):,})" if ok and pt.endswith('.csv') else ""
        except Exception: nn = ""
        s = f'<span class="data-status data-ready">✓{nn}</span>' if ok else '<span class="data-status data-missing">○</span>'
        with cols[i%3]: st.markdown(f'<div class="data-row"><div class="data-icon">{ic}</div><div class="data-label">{lb}</div>{s}</div>', unsafe_allow_html=True)

    if panel_path.exists():
        c1, c2 = st.columns(2)
        with c1:
            with open(panel_path, 'rb') as f:
                st.download_button("📥 Panel Dataset (CSV)", data=f, file_name=f"panel_data_{code}.csv", mime="text/csv", use_container_width=True)
        sc = figures_dir / "scorecard_data.csv"
        if sc.exists():
            with c2:
                with open(sc, 'rb') as f:
                    st.download_button("📥 Scorecard (CSV)", data=f, file_name=f"scorecard_{code}.csv", mime="text/csv", use_container_width=True)

    summary_data = pd.read_csv(figures_dir/"summary_statistics.csv") if (figures_dir/"summary_statistics.csv").exists() else None
    if summary_data is not None:
        with st.expander("📋 Summary Statistics"):
            st.dataframe(summary_data, hide_index=True)
    with st.expander("📝 Configuration"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")


# ═════════════════════════════════════════════════════════════════════════════
# SETUP & RUN
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "⚙️ Setup & Run":

    st.markdown('<div class="section-title">Configure Country</div>', unsafe_allow_html=True)

    pick = st.selectbox("Country", sorted(QUICK_REF.keys()), index=None,
                         placeholder="Choose a country...", key="setup_pick")

    setup_country_code = None

    if pick:
        ref = QUICK_REF[pick]
        cc = ref['iso'].lower()
        setup_country_code = cc
        existing = Path(f"config/{cc}.yaml").exists()

        if existing:
            with open(f"config/{cc}.yaml") as _f:
                ex = yaml.safe_load(_f)
            d_etype = ex.get('events',{}).get('disaster_type', ref['etype'])
            d_edate = ex.get('events',{}).get('disaster_date', ref['edate'])
            d_idate = ex.get('events',{}).get('intervention_start', ref['edate'])
            d_start = ex.get('study_period_start', '2022-01-01')
            d_end = ex.get('study_period_end', '2025-12-31')
            d_admin = ex.get('boundaries',{}).get('admin_level', ref['admin'])
            st.caption(f"Editing `config/{cc}.yaml`")
        else:
            d_etype, d_edate, d_idate = ref['etype'], ref['edate'], ref['edate']
            d_start = str(date(max(int(ref['edate'][:4])-1, 2000), 1, 1))
            d_end = '2025-12-31'
            d_admin = ref['admin']

        etypes = ["earthquake","conflict","flood","cyclone","drought"]
        c1, c2, c3 = st.columns(3)
        with c1: etype = st.selectbox("Event type", etypes, index=etypes.index(d_etype), key="cf_et")
        with c2: edate = st.date_input("Event date", date.fromisoformat(d_edate), key="cf_ed")
        with c3: idate = st.date_input("Intervention start", date.fromisoformat(d_idate), key="cf_id")

        c1, c2, c3 = st.columns(3)
        with c1: sstart = st.date_input("Study start", date.fromisoformat(d_start), key="cf_ss")
        with c2: send = st.date_input("Study end", date.fromisoformat(d_end), key="cf_se")
        with c3: admin_level = st.selectbox("Admin level", [1,2,3], index=[1,2,3].index(d_admin), key="cf_al")

        if st.button("💾 Save Configuration", type="primary", use_container_width=True, key="save_cfg"):
            nc = {
                'country_name': pick,
                'country_code': ref['iso'],
                'study_period_start': sstart.isoformat(),
                'study_period_end': send.isoformat(),
                'events': {
                    'disaster_type': etype,
                    'disaster_date': edate.isoformat(),
                    'intervention_start': idate.isoformat(),
                },
                'boundaries': {
                    'file_path': f"data/{cc}/boundaries/{cc}_admin{admin_level}.geojson",
                    'id_column': 'shapeID',
                    'name_column': 'shapeName',
                    'admin_level': admin_level,
                },
                'acled_country_name': ref['acled'],
                'wfp_commodity': ref['commodity'],
                'google_trends_keywords': {
                    'food': 'food', 'water': 'water', 'shelter': 'shelter',
                    'medicine': 'medicine', 'aid': 'aid', 'reconstruction': 'reconstruction',
                    'refugee': 'refugee', etype: etype,
                },
            }
            Path("config").mkdir(exist_ok=True)
            with open(f"config/{cc}.yaml", 'w') as f_out:
                yaml.dump(nc, f_out, default_flow_style=False, sort_keys=False)
            st.session_state.auto_select_country = cc
            st.success(f"✅ Saved `config/{cc}.yaml`")
            st.rerun()

    # CREDENTIALS
    st.markdown("---")
    with st.expander("🔑 Credentials (ACLED & GEE)", expanded=False):
        c1, c2 = st.columns(2)
        with c1: acled_email = st.text_input("ACLED Email", value=os.getenv("ACLED_EMAIL",""), key="acled_em")
        with c2: acled_key = st.text_input("ACLED Key", value=os.getenv("ACLED_KEY",""), type="password", key="acled_k")
        st.caption("GEE: run `earthengine authenticate` in terminal once.")

    # RUN PIPELINE
    run_country = setup_country_code or selected
    run_name = pick or config.get('country_name', selected)

    st.markdown("---")
    st.markdown(f'<div class="section-title">Run Pipeline — {run_name}</div>', unsafe_allow_html=True)

    def run_cmd(cmd, desc):
        env = os.environ.copy()
        if acled_email: env['ACLED_EMAIL'] = acled_email
        if acled_key:
            env['ACLED_KEY'] = acled_key
            env['ACLED_PASSWORD'] = acled_key
        with st.status(f"{desc}...", expanded=True) as s:
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=900)
                st.code(r.stdout[-3000:] or "(none)", language="bash")
                if r.returncode == 0: s.update(label=f"✅ {desc}", state="complete")
                else: st.error(r.stderr[-500:]); s.update(label=f"⚠️ {desc}", state="error")
            except subprocess.TimeoutExpired: s.update(label="⏱️ Timeout (15 min)", state="error")

    steps = [
        ("🗺️ Boundaries", f"make boundaries COUNTRY={run_country}"),
        ("🛰️ Satellite",  f"make gee COUNTRY={run_country}"),
        ("⚔️ Conflict",   f"make acled COUNTRY={run_country}"),
        ("🏗️ UNDP",       f"make undp COUNTRY={run_country}"),
        ("📈 Trends",     f"make gt COUNTRY={run_country}"),
        ("🔧 Panel",      f"make preprocess COUNTRY={run_country}"),
        ("📊 Visuals",    f"make visuals COUNTRY={run_country}"),
        ("🎯 Scorecard",  f"python pipeline/recovery_scorecard.py --config config/{run_country}.yaml"),
    ]
    cols = st.columns(4)
    for i, (lb, cmd) in enumerate(steps):
        with cols[i % 4]:
            if st.button(lb, use_container_width=True, key=f"run_{lb}"):
                run_cmd(cmd, lb)

    st.markdown("")
    if st.button("▶️ Run Full Pipeline", use_container_width=True, type="primary", key="run_full"):
        run_cmd(
            f"make run COUNTRY={run_country} && python pipeline/recovery_scorecard.py --config config/{run_country}.yaml",
            "Full Pipeline"
        )
        st.balloons()

    with st.expander("📝 Current Configuration"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">UNDP Early Recovery Impact · AImpact Lab · VIIRS · Sentinel-1 · MODIS · CHIRPS · ACLED</div>', unsafe_allow_html=True)
