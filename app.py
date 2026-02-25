#!/usr/bin/env python3
"""
app_v13.py — UNDP Early Recovery Impact (v13)
==============================================
Usage: streamlit run app.py
v13: Disclaimer, Impact Measurement tab with full file discovery,
     remove VH (keep VV only), equal map/ts columns, GitHub dark theme
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
    .setup-box { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1.2rem; margin: .8rem 0; }
    .setup-box, .setup-box p, .setup-box li, .setup-box ol, .setup-box a, .setup-box span { color: #c9d1d9 !important; }
    .setup-box h4 { color: #58a6ff !important; margin: 0 0 .5rem 0; font-size: .95rem; }
    .setup-box code { background: #21262d; padding: .15rem .4rem; border-radius: 4px; font-size: .82rem; color: #e6edf3; }
    .footer { text-align: center; padding: 1.5rem 0 .5rem 0; border-top: 1px solid #21262d; margin-top: 2.5rem; color: #484f58; font-size: .75rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}


    header[data-testid="stHeader"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
COUNTRY_REFERENCE = {
    'Afghanistan': {'iso': 'AFG', 'acled': 'Afghanistan'}, 'Bangladesh': {'iso': 'BGD', 'acled': 'Bangladesh'},
    'Colombia': {'iso': 'COL', 'acled': 'Colombia'}, 'DR Congo': {'iso': 'COD', 'acled': 'Democratic Republic of Congo'},
    'Ethiopia': {'iso': 'ETH', 'acled': 'Ethiopia'}, 'Gaza Strip': {'iso': 'PSE', 'acled': 'Palestine'},
    'Haiti': {'iso': 'HTI', 'acled': 'Haiti'}, 'Honduras': {'iso': 'HND', 'acled': 'Honduras'},
    'India': {'iso': 'IND', 'acled': 'India'}, 'Indonesia': {'iso': 'IDN', 'acled': 'Indonesia'},
    'Iraq': {'iso': 'IRQ', 'acled': 'Iraq'}, 'Lebanon': {'iso': 'LBN', 'acled': 'Lebanon'},
    'Libya': {'iso': 'LBY', 'acled': 'Libya'}, 'Mozambique': {'iso': 'MOZ', 'acled': 'Mozambique'},
    'Myanmar': {'iso': 'MMR', 'acled': 'Myanmar'}, 'Nepal': {'iso': 'NPL', 'acled': 'Nepal'},
    'Nigeria': {'iso': 'NGA', 'acled': 'Nigeria'}, 'Pakistan': {'iso': 'PAK', 'acled': 'Pakistan'},
    'Philippines': {'iso': 'PHL', 'acled': 'Philippines'}, 'Somalia': {'iso': 'SOM', 'acled': 'Somalia'},
    'South Sudan': {'iso': 'SSD', 'acled': 'South Sudan'}, 'Sudan': {'iso': 'SDN', 'acled': 'Sudan'},
    'Syria': {'iso': 'SYR', 'acled': 'Syria'}, 'Turkey / Türkiye': {'iso': 'TUR', 'acled': 'Turkey'},
    'Ukraine': {'iso': 'UKR', 'acled': 'Ukraine'}, 'Yemen': {'iso': 'YEM', 'acled': 'Yemen'},
}
DEFAULT_KW = ['earthquake','flood','reconstruction','food','water','medicine','shelter','aid','cement','construction']
COUNTRIES = {
    'myanmar': {'flag':'🇲🇲','name':'Myanmar','event':'M7.7 Earthquake','event_date':'Mar 28, 2025'},
    'gaza': {'flag':'🇵🇸','name':'Gaza Strip','event':'Armed Conflict','event_date':'Oct 7, 2023'},
    'afghanistan': {'flag':'🇦🇫','name':'Afghanistan','event':'M6.3 Earthquake','event_date':'Oct 7, 2023'},
    'ukraine': {'flag':'🇺🇦','name':'Ukraine','event':'Armed Conflict','event_date':'Feb 24, 2022'},
}
THEMES = {
    '💡 Economic Recovery': {'desc':'Nighttime lights as proxy for economic activity and energy access','indicators':['nightlights','price_rice','price_oil','price_bread','price_flour','price_chicken','price_milk','price_sugar']},
    '🌿 Agricultural Health': {'desc':'Vegetation and ecosystem recovery across affected regions','indicators':['ndvi','rvi','rainfall']},
    '🏚️ Debris & Surface': {'desc':'Surface roughness changes via radar — debris, rubble, and land clearing','indicators':['vv']},
    '⚔️ Conflict & Security': {'desc':'Armed conflict events and fatalities from ACLED','indicators':['conflict_events','conflict_fatalities']},
}
NAMES = {'nightlights':'Energy & Economic Activity','ndvi':'Agricultural Health','vv':'Debris & Surface (Radar VV)','vh':'Infrastructure (Radar VH)','rvi':'Radar Vegetation','rainfall':'Rainfall','conflict_events':'Conflict Events','conflict_fatalities':'Conflict Fatalities','price_rice':'Rice Price','price_oil':'Oil Price','price_bread':'Bread Price','price_flour':'Flour Price','price_chicken':'Chicken Price','price_milk':'Milk Price','price_sugar':'Sugar Price'}
RASTER_IND = ['viirs','ndvi','ndbi','bsi','vv','evi','savi','nbr','swir','mndwi']
RASTER_NAMES = {'viirs':'Energy & Nightlights','ndvi':'Agricultural Health','ndbi':'Built-Up Areas','bsi':'Bare Soil & Debris','evi':'Enhanced Vegetation','savi':'Soil-Adjusted Vegetation','nbr':'Burn Ratio','vv':'Surface Roughness (Radar)','swir':'Short-Wave Infrared','mndwi':'Water Coverage'}

METHODOLOGY_NOTE = """<b>Methodological note.</b> Scores are calculated only for administrative units that experienced a significant shock, defined as a decline greater than 10% relative to the pre-event baseline. Indicators below this threshold are classified as <em>Stable</em> and are excluded from recovery scoring. To ensure comparability over time, Vegetation and Nightlights indicators are adjusted on a year-over-year basis to remove natural seasonal variation (e.g., harvest cycles, snow cover, or holiday effects). <b>Overall Recovery</b> measures the share of the initial decline that has been regained since the lowest observed point of the crisis (the trough), capped at 100%. <b>Since Intervention</b> captures the recovery achieved after the intervention start date, expressed as a percentage of the total disaster impact and likewise capped at 100%. These metrics describe observed temporal dynamics to support rapid prioritisation and resource allocation; they reflect recovery trends but do not imply direct causal attribution to any single organisation or intervention.<br><br><b>Disclaimer.</b> The accuracy of these results depends on the quality and resolution of the underlying satellite data. In some contexts, indicators such as nighttime lights may be affected by seasonal patterns, cloud cover, snow reflectance, or sensor calibration changes that are not fully captured by the year-over-year adjustment. This dashboard presents <em>correlations</em> between the timing of an event, the start of early recovery interventions, and changes in satellite-derived indicators; it does not constitute evidence of causal impact. In most contexts, UNDP is not the only actor on the ground, and observed recovery trends reflect the combined contributions of multiple organisations and local actors. Isolating UNDP's specific contribution requires econometric approaches (e.g., Difference-in-Differences, Synthetic Control) available in the Impact Measurement tab, which support attribution to the extent possible given data constraints."""



# ─── HERO & SIDEBAR ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>UNDP Early Recovery Impact</h1>
    <p>Satellite data indicators to monitor early recovery interventions</p>
    <div class="hero-badge">📡 Powered by AImpact Lab — geospatial pipelines for satellite-based impact measurement</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### 🌍 Select Country")
    cfg_files = sorted(glob.glob("config/*.yaml"))
    cfg_names = [Path(f).stem for f in cfg_files]
    if not cfg_names: st.warning("No configs found"); st.stop()
    selected = st.selectbox("Country", cfg_names, format_func=lambda n: f"{COUNTRIES.get(n,{}).get('flag','🌐')}  {COUNTRIES.get(n,{}).get('name',n.replace('_',' ').title())}", label_visibility="collapsed")
    with open(f"config/{selected}.yaml") as f: config = yaml.safe_load(f)
    ci = COUNTRIES.get(selected, {})
    st.markdown(f'<div class="country-card"><span style="font-size:1.5rem;margin-right:.4rem;">{ci.get("flag","🌐")}</span><span class="country-name">{config["country_name"]}</span><div class="country-detail">{ci.get("event","")} · {ci.get("event_date",config["events"]["disaster_date"])}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("Mode", ["📊 View Results", "➕ Add Country", "🔧 Admin"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.72rem;color:#484f58;">Built by AImpact Lab</div>', unsafe_allow_html=True)

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
    except: panel_info = "Available"
n_figs = len(list(figures_dir.rglob("*.png"))) if figures_dir.exists() else 0
event_date = config['events']['disaster_date']
intervention = config['events'].get('intervention_start', event_date)

st.markdown(f'<div class="metric-row"><div class="metric-box"><div class="metric-label">Event Date</div><div class="metric-value">{event_date}</div></div><div class="metric-box"><div class="metric-label">Study Period</div><div class="metric-value">{str(config.get("study_period_start",""))[:7]} → {str(config.get("study_period_end",""))[:7]}</div></div><div class="metric-box"><div class="metric-label">Panel</div><div class="metric-value">{panel_info or "Not built"}</div></div><div class="metric-box"><div class="metric-label">Figures</div><div class="metric-value">{n_figs}</div></div></div>', unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
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
        nm = f.stem.lower(); ind = None
        for ri in RASTER_IND:
            if nm.startswith(ri): ind = ri; break
        if ind is None: ind = nm.split('_')[0]
        k = 'clean' if 'clean' in nm else ('geocode' if 'geocode' in nm else 'other')
        if ind not in groups: groups[ind] = {'clean':None,'geocode':None,'other':None}
        groups[ind][k] = f
    return [(ind, g['clean'] or g['geocode'] or g['other']) for ind, g in sorted(groups.items()) if g['clean'] or g['geocode'] or g['other']]

def collect_showcase_files():
    """Read curated impact measurement files from outputs/{code}/showcase/.
    To update what appears, simply add/remove files from that folder."""
    showcase_dir = output_dir / "showcase"
    if not showcase_dir.exists():
        return {'plots': [], 'tables': [], 'pretrends': [], 'coef_plots': [], 'maps': []}
    files = {'plots': [], 'tables': [], 'pretrends': [], 'coef_plots': [], 'maps': []}
    for f in sorted(showcase_dir.iterdir()):
        if f.suffix == '.csv':
            files['tables'].append(f)
        elif f.suffix == '.png':
            nm = f.stem.lower()
            if 'pretrend' in nm:
                files['pretrends'].append(f)
            elif 'coef_plot' in nm:
                files['coef_plots'].append(f)
            elif 'three_panel' in nm or 'map' in nm or 'debris' in nm:
                files['maps'].append(f)
            else:
                files['plots'].append(f)
    return files

def render_scorecard():
    csv_path = figures_dir / "scorecard_data.csv"
    if not csv_path.exists(): return False
    df = pd.read_csv(csv_path)
    # Filter out VH - keep only VV for debris
    df = df[df['indicator'] != 'vh']
    trend_icons = {'improving': '▲', 'declining': '▼', 'stagnant': '●', 'stable': '—'}
    trend_colors = {'improving': '#3fb950', 'declining': '#f85149', 'stagnant': '#d29922', 'stable': '#484f58'}
    rows_html = ""
    for _, row in df.iterrows():
        overall = row['overall_recovery_pct']
        undp = row['undp_recovery_pct']
        n_aff = int(row.get('n_affected_units', 0))
        n_tot = int(row.get('n_total_units', 0))
        trend = row.get('trend', 'stable')
        ti = trend_icons.get(trend, '●')
        tc = trend_colors.get(trend, '#8b949e')
        if overall < 0:
            rows_html += f"""
            <div class="sc-row">
                <div class="sc-name">{row.get('short_name','')}</div>
                <div class="sc-desc">{row.get('description','')}</div>
                <div class="sc-bars"><div class="sc-stable">Stable — no units with significant decline</div></div>
                <div class="sc-vals"><div class="sc-stable-badge">Stable</div></div>
                <div class="sc-trend" style="color: #484f58;">—</div>
            </div>"""
            continue
        oc = max(0, min(overall, 100)); uc = max(0, min(undp, 100)) if undp >= 0 else 0
        oc_c = "#3fb950" if overall >= 75 else ("#d29922" if overall >= 50 else "#f85149")
        if undp < 0: uc_c = "#484f58"; ud = "N/A"
        elif undp >= 50: uc_c = "#58a6ff"; ud = f"{undp:.0f}%"
        elif undp >= 25: uc_c = "#79c0ff"; ud = f"{undp:.0f}%"
        else: uc_c = "#8b949e"; ud = f"{undp:.0f}%"
        units_label = f"{n_aff} of {n_tot} units" if n_tot > 0 else ""
        rows_html += f"""
            <div class="sc-row">
                <div class="sc-name">{row.get('short_name','')}<br><span class="sc-units">{units_label}</span></div>
                <div class="sc-desc">{row.get('description','')}</div>
                <div class="sc-bars">
                    <div class="sc-track"><div class="sc-fill-a" style="width:{oc}%;"></div><div class="sc-ref"></div></div>
                    <div class="sc-track"><div class="sc-fill-b" style="width:{uc}%;"></div></div>
                </div>
                <div class="sc-vals">
                    <div><span style="color:{oc_c};font-weight:700;font-size:1.05rem;">{overall:.0f}%</span> <span style="color:#484f58;font-size:.75rem;">overall</span></div>
                    <div><span style="color:{uc_c};font-weight:700;font-size:1.05rem;">{ud}</span> <span style="color:#484f58;font-size:.75rem;">since intervention</span></div>
                </div>
                <div class="sc-trend" style="color:{tc};font-size:1.3rem;" title="{trend.title()}">{ti}</div>
            </div>"""

    html = f"""
    <style>
        body {{ background: transparent; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; color: #c9d1d9; }}
        .sc {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem 1.8rem; }}
        .sc-title {{ font-size: 1.2rem; font-weight: 700; color: #e6edf3; margin: 0 0 .2rem 0; }}
        .sc-sub {{ font-size: .82rem; color: #8b949e; margin: 0 0 .3rem 0; }}
        .sc-legend {{ display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .sc-legend-item {{ display: flex; align-items: center; gap: .35rem; font-size: .75rem; color: #8b949e; }}
        .sc-legend-dot {{ width: 8px; height: 8px; border-radius: 2px; }}
        .sc-row {{ display: flex; align-items: center; padding: .65rem 0; border-bottom: 1px solid #21262d; }}
        .sc-row:last-of-type {{ border-bottom: none; }}
        .sc-name {{ width: 150px; font-weight: 600; color: #e6edf3; font-size: .85rem; line-height: 1.3; }}
        .sc-units {{ font-size: .68rem; color: #484f58; font-weight: 400; }}
        .sc-desc {{ width: 200px; font-size: .72rem; color: #8b949e; padding-right: .5rem; }}
        .sc-bars {{ flex: 1; padding: 0 .5rem; }}
        .sc-track {{ background: #0d1117; border-radius: 4px; height: 10px; position: relative; margin: 4px 0; }}
        .sc-fill-a {{ background: linear-gradient(90deg, #0468B1, #58a6ff); height: 10px; border-radius: 4px; }}
        .sc-fill-b {{ background: linear-gradient(90deg, #1f6feb, #79c0ff); height: 10px; border-radius: 4px; }}
        .sc-ref {{ position: absolute; right: 0; top: -1px; bottom: -1px; width: 2px; background: #3fb950; opacity: 0.5; }}
        .sc-vals {{ width: 130px; text-align: right; padding-right: .5rem; }}
        .sc-trend {{ width: 30px; text-align: center; }}
        .sc-stable {{ color: #3fb950; font-size: .82rem; padding: 6px 0; font-weight: 500; }}
        .sc-stable-badge {{ display: inline-block; background: rgba(35,134,54,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 2em; padding: 1px 10px; font-size: .75rem; font-weight: 500; }}
        .sc-note {{ font-size: .68rem; color: #8b949e; margin-top: .8rem; line-height: 1.6; }}
        .sc-note b {{ color: #c9d1d9; }}
        .sc-note em {{ color: #3fb950; font-style: normal; font-weight: 600; }}
    </style>
    <div class="sc">
        <div class="sc-title">Recovery Scorecard</div>
        <div class="sc-sub">How much has been recovered since the event, and what changed after early recovery started?<br>Event: {event_date} &nbsp;·&nbsp; Intervention: {intervention}</div>
        <div class="sc-legend">
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#0468B1;"></div>Overall Recovery</div>
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#1f6feb;"></div>Since Intervention</div>
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#3fb950;"></div>Pre-event level</div>
            <div class="sc-legend-item">▲ Improving &nbsp; ● Stagnant &nbsp; ▼ Declining</div>
        </div>
        {rows_html}
        <div class="sc-note">{METHODOLOGY_NOTE}</div>

    </div>
    """
    n_rows = len(df)
    components.html(html, height=80 + n_rows * 80 + 40 + 240, scrolling=False)
    return True

# ─── VIEW RESULTS ────────────────────────────────────────────────────────────
if mode == "📊 View Results":
    # LEVEL 1: SCORECARD
    has_scorecard = render_scorecard()
    if not has_scorecard:
        st.info(f"Run: `python pipeline/recovery_scorecard.py --config config/{selected}.yaml`")

    # LEVEL 2: VISUAL INDICATORS & IMPACT
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
            tabs = st.tabs(tab_names); ti = 0

            # Thematic tabs
            for tn, th in THEMES.items():
                with tabs[ti]:
                    st.caption(th['desc']); found = False
                    for ind in th['indicators']:
                        figs = find_figs(ind, all_pngs)
                        if figs:
                            found = True
                            st.markdown(f"**{NAMES.get(ind, ind.replace('_',' ').title())}**")
                            ts = [f for f in figs if f.name.startswith('ts_')]
                            mp = [f for f in figs if f.name.startswith('map_')]
                            if ts and mp:
                                c1, c2 = st.columns([5, 6])
                                with c1: st.image(str(ts[0]), width='stretch')
                                with c2: st.image(str(mp[0]), width='stretch')
                            elif ts: st.image(str(ts[0]), width='stretch')
                            elif mp: st.image(str(mp[0]), width='stretch')
                            st.markdown("---")
                    if not found: st.info("No data yet.")
                ti += 1

            # Google Trends
            if trend_files:
                with tabs[ti]:
                    st.caption("Google search interest as proxy for public concern")
                    for f in trend_files:
                        st.markdown(f"**{f.stem.replace('trends_','').replace('_',' ').title()}**")
                        st.image(str(f), width='stretch')
                ti += 1

            # Programme Timeline
            if undp_files:
                with tabs[ti]:
                    for f in undp_files: st.image(str(f), width='stretch')
                ti += 1

            # Impact Measurement
            if has_impact:
                with tabs[ti]:
                    st.caption("Econometric impact estimates using Difference-in-Differences and buffer zone analysis. These approaches help isolate UNDP's contribution to recovery, though results should be interpreted alongside the caveats in the scorecard disclaimer.")

                    # Maps (three-panel, UNDP locations, debris)
                    if impact_files['maps']:
                        st.markdown("**Satellite Maps — UNDP Intervention Areas**")
                        for f in impact_files['maps']:
                            label = f.stem.replace('_clean','').replace('_three_panel','').replace('_',' ').title()
                            st.markdown(f"*{label}*")
                            st.image(str(f), width='stretch')
                        st.markdown("---")

                    # Coefficient plots (buffer zone analysis)
                    if impact_files['coef_plots']:
                        st.markdown("**Buffer Zone Analysis — Treatment Effect by Distance**")
                        n_cols = min(len(impact_files['coef_plots']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['coef_plots']):
                            label = f.stem.replace('coef_plot_','').replace('_buffer','').replace('_',' ').title()
                            with cols[i % n_cols]:
                                st.image(str(f), caption=label, width='stretch')
                        st.markdown("---")

                    # DiD summary and other plots
                    if impact_files['plots']:
                        st.markdown("**Difference-in-Differences Estimates**")
                        for f in impact_files['plots']:
                            st.image(str(f), width='stretch')
                        st.markdown("---")

                    # Pretrends
                    if impact_files['pretrends']:
                        st.markdown("**Pre-Trends Validation (Parallel Trends Assumption)**")
                        n_cols = min(len(impact_files['pretrends']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['pretrends']):
                            label = f.stem.replace('pretrends_','').replace('_baseline','').replace('_',' ').title()
                            with cols[i % n_cols]:
                                st.image(str(f), caption=label, width='stretch')
                        st.markdown("---")

                    # Regression tables
                    if impact_files['tables']:
                        st.markdown("**Regression Results**")
                        for f in impact_files['tables']:
                            df_table = pd.read_csv(f)
                            # Gaza: filter to 500m inner, 1500m outer + clean intervention names
                            if 'comprehensive_unified' in f.name and 'inner_m' in df_table.columns:
                                df_table = df_table[(df_table['inner_m'] == 500) & (df_table['outer_m'] == 1500)]
                                if 'intervention_type' in df_table.columns:
                                    df_table['intervention_type'] = df_table['intervention_type'].str.lstrip(')')
                            # Myanmar: filter to 0.5km-1.5km buffer
                            if 'buffer_results' in f.name and 'Buffer_Spec' in df_table.columns:
                                df_table = df_table[df_table['Buffer_Spec'] == '0.5km-1.5km']
                            if len(df_table) > 0:
                                st.markdown(f"*{f.stem.replace('_',' ').title()} — Buffer: 500m–1.5km*")
                                st.dataframe(df_table, hide_index=True, use_container_width=True)
                ti += 1

            # Satellite Maps
            if raster_pairs:
                with tabs[ti]:
                    st.caption("Pre-Event, Post-Event, and Post-Intervention satellite imagery")
                    for ind, f in raster_pairs:
                        st.markdown(f"**{RASTER_NAMES.get(ind, ind.upper())}**")
                        st.image(str(f), width='stretch')

    # LEVEL 3: DATA & DOWNLOADS
    st.markdown('<div class="section-title">Data Availability</div>', unsafe_allow_html=True)
    srcs = [("🗺️","Boundaries",config['boundaries']['file_path']),("🛰️","Satellite",f"data/{code}/gee_panel_{code}_centroid.csv"),("⚔️","Conflict",f"data/{code}/external/acled_conflict.csv"),("🏗️","UNDP Projects",f"data/{code}/external/undp_projects.csv"),("📈","Google Trends",f"data/{code}/external/google_trends.csv"),("📊","Panel",f"outputs/{code}/panel_data.csv")]
    cols = st.columns(3)
    for i, (ic, lb, pt) in enumerate(srcs):
        pp = Path(pt); ok = pp.exists()
        try: nn = f" ({len(pd.read_csv(pp)):,})" if ok and pt.endswith('.csv') else ""
        except: nn = ""
        s = f'<span class="data-status data-ready">✓{nn}</span>' if ok else '<span class="data-status data-missing">○</span>'
        with cols[i%3]: st.markdown(f'<div class="data-row"><div class="data-icon">{ic}</div><div class="data-label">{lb}</div>{s}</div>', unsafe_allow_html=True)

    if panel_path.exists():
        c1, c2 = st.columns(2)
        with c1:
            with open(panel_path, 'rb') as f:
                st.download_button("📥 Panel Dataset (CSV)", data=f, file_name=f"panel_data_{code}.csv", mime="text/csv", width='stretch')
        sc = figures_dir / "scorecard_data.csv"
        if sc.exists():
            with c2:
                with open(sc, 'rb') as f:
                    st.download_button("📥 Scorecard (CSV)", data=f, file_name=f"scorecard_{code}.csv", mime="text/csv", width='stretch')

    summary_data = pd.read_csv(figures_dir/"summary_statistics.csv") if (figures_dir/"summary_statistics.csv").exists() else None
    if summary_data is not None:
        with st.expander("📋 Summary Statistics"):
            st.dataframe(summary_data, hide_index=True)
    with st.expander("📝 Configuration"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

# ─── ADD COUNTRY ─────────────────────────────────────────────────────────────
elif mode == "➕ Add Country":
    st.markdown('<div class="section-title">Add a New Country</div>', unsafe_allow_html=True)
    country_names = sorted(COUNTRY_REFERENCE.keys())
    country_choice = st.selectbox("Select Country", country_names, index=None, placeholder="Choose a country...")
    if country_choice and country_choice in COUNTRY_REFERENCE:
        ref = COUNTRY_REFERENCE[country_choice]; dn, di, da = country_choice, ref['iso'], ref['acled']
    else: dn, di, da = "", "", ""
    with st.form("new_country"):
        st.markdown("**Country Details**")
        c1, c2, c3 = st.columns(3)
        with c1: country_name = st.text_input("Country Name", value=dn)
        with c2: country_code = st.text_input("ISO Code", value=di, max_chars=3)
        with c3: acled_country = st.text_input("ACLED Name", value=da)
        st.markdown("**Event & Intervention**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: dtype = st.selectbox("Event Type", ["earthquake","conflict","flood","cyclone","drought"])
        with c2: ddate = st.date_input("Event Date", date(2025,1,1))
        with c3: idate = st.date_input("Early Recovery Start", date(2025,2,1))
        with c4: alevel = st.selectbox("Admin Level", [1,2,3], index=2)
        st.markdown("**Study Period**")
        c1, c2 = st.columns(2)
        with c1: sstart = st.date_input("Start", date(2023,1,1))
        with c2: send = st.date_input("End", date(2026,1,31))
        if st.form_submit_button("💾 Save Configuration", type="primary"):
            cn = (country_name or "").strip(); cc = (country_code or "").strip().upper(); ac = (acled_country or cn).strip()
            if not cn or not cc or len(cc) != 3: st.error("Need country name + valid 3-letter ISO code.")
            else:
                nc = {'country_name':cn,'country_code':cc,'study_period_start':sstart.isoformat(),'study_period_end':send.isoformat(),'events':{'disaster_type':dtype,'disaster_date':ddate.isoformat(),'intervention_start':idate.isoformat()},'boundaries':{'file_path':f"data/{cc.lower()}/boundaries/{cc.lower()}_admin{alevel}.geojson",'id_column':'shapeID','name_column':'shapeName','admin_level':alevel},'acled_country_name':ac,'google_trends_keywords':DEFAULT_KW}
                Path("config").mkdir(exist_ok=True)
                with open(f"config/{cc.lower()}.yaml",'w') as f: yaml.dump(nc, f, default_flow_style=False, sort_keys=False)
                st.success(f"✅ Saved! Select **{cn}** from the sidebar."); st.rerun()

# ─── ADMIN ───────────────────────────────────────────────────────────────────
elif mode == "🔧 Admin":
    admin_pw = st.text_input("Admin Password", type="password", placeholder="Enter admin password")
    if admin_pw == os.getenv("ADMIN_PASSWORD", "undp2025"):
        st.markdown('<div class="section-title">Pipeline Execution</div>', unsafe_allow_html=True)
        st.markdown('<div class="setup-box"><h4>🛰️ Google Earth Engine</h4><p>One-time: <code>earthengine authenticate</code></p></div>', unsafe_allow_html=True)
        st.markdown('<div class="setup-box"><h4>⚔️ ACLED</h4><p>Register at <a href="https://acleddata.com/">acleddata.com</a></p></div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca: acled_email = st.text_input("ACLED Email", value=os.getenv("ACLED_EMAIL",""))
        with cb: acled_key = st.text_input("ACLED Password", value=os.getenv("ACLED_KEY",""), type="password")
        steps = [("🗺️ Boundaries",f"make boundaries COUNTRY={selected}"),("🛰️ Satellite",f"make gee COUNTRY={selected}"),("⚔️ Conflict",f"make acled COUNTRY={selected}"),("🏗️ UNDP",f"make undp COUNTRY={selected}"),("📈 Trends",f"make gt COUNTRY={selected}"),("🔧 Panel",f"make preprocess COUNTRY={selected}"),("📊 Visuals",f"make visuals COUNTRY={selected}"),("🎯 Scorecard",f"python pipeline/recovery_scorecard.py --config config/{selected}.yaml")]
        def run_cmd(cmd, desc):
            env = os.environ.copy()
            if acled_email: env['ACLED_EMAIL'] = acled_email
            if acled_key: env['ACLED_KEY'] = acled_key
            with st.status(f"{desc}...", expanded=True) as s:
                try:
                    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=600)
                    st.code(r.stdout[-3000:] or "(none)", language="bash")
                    if r.returncode == 0: s.update(label=f"✅ {desc}", state="complete")
                    else: st.error(r.stderr[-500:]); s.update(label=f"⚠️ {desc}", state="error")
                except subprocess.TimeoutExpired: s.update(label="⏱️ Timeout", state="error")
        for lb, cmd in steps:
            c1, c2 = st.columns([1,3])
            with c1:
                if st.button(lb, width='stretch'): run_cmd(cmd, lb)
            with c2: st.caption(cmd)
        st.markdown("---")
        if st.button("▶️ Run Full Pipeline", width='stretch', type="primary"):
            run_cmd(f"make run COUNTRY={selected} && python pipeline/recovery_scorecard.py --config config/{selected}.yaml", "Full Pipeline"); st.balloons()
    elif admin_pw:
        st.error("Incorrect password.")

st.markdown('<div class="footer">UNDP Early Recovery Impact · AImpact Lab · VIIRS · Sentinel-1 · MODIS · CHIRPS · ACLED</div>', unsafe_allow_html=True)
