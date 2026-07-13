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
import yaml, subprocess, os, glob, sys, shutil, json, time
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
import pandas as pd

USER_ENV_PATH = Path.home() / ".undp_dashboard" / ".env"
load_dotenv()
if USER_ENV_PATH.exists():
    load_dotenv(USER_ENV_PATH)


def safe_image(path, **kwargs):
    """Display an image only if it is a valid, non-empty image file."""
    try:
        from PIL import Image
        p = str(path)
        if (not os.path.exists(p)) or os.path.getsize(p) == 0:
            st.caption("(figure not available)")
            return False
        with Image.open(p) as _im:
            _im.verify()
        st.image(p, **kwargs)
        return True
    except Exception:
        st.caption("(figure could not be displayed)")
        return False


# ─── MULTI-USER WORKSPACES ─────────────────────────────────────────────────
# Each user gets an isolated, copy-on-write view of config/data/outputs.
# Shared (pre-built) countries stay read-only until a user edits or re-runs
# them, at which point that user's edit is cloned into their own workspace
# and every subsequent read for that country prefers the user's copy.
# Nothing about this is visible in the UI — countries are never renamed
# with a username suffix; the fork happens silently in the background.
REPO_ROOT = Path(__file__).resolve().parent
WORKSPACES_ROOT = Path.home() / ".undp_dashboard" / "workspaces"
USERS_REGISTRY = Path.home() / ".undp_dashboard" / "users.json"


def _load_users():
    try:
        return sorted(json.loads(USERS_REGISTRY.read_text()))
    except Exception:
        return []


def _register_user(name):
    users = set(_load_users())
    if name not in users:
        users.add(name)
        try:
            USERS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
            USERS_REGISTRY.write_text(json.dumps(sorted(users)))
        except Exception:
            pass


def user_workspace_root():
    """This user's private workspace. Also the cwd used for pipeline runs,
    so it carries symlinks to the shared, read-only pipeline/ code."""
    name = st.session_state.get("username", "_shared")
    r = WORKSPACES_ROOT / name
    (r / "config").mkdir(parents=True, exist_ok=True)
    (r / "data").mkdir(parents=True, exist_ok=True)
    (r / "outputs").mkdir(parents=True, exist_ok=True)
    for shared_name in ("pipeline", "run_pipeline.py"):
        link = r / shared_name
        if not link.exists():
            try:
                link.symlink_to(REPO_ROOT / shared_name)
            except Exception:
                pass
    return r


def resolve_config_path(name, for_write=False):
    """Path to a country's config. Reads prefer the user's own edited copy,
    falling back to the shared one. Writes always land in (and first
    copy-on-write clone into) the user's own copy."""
    shared = REPO_ROOT / "config" / (name + ".yaml")
    user_path = user_workspace_root() / "config" / (name + ".yaml")
    if for_write:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        if not user_path.exists() and shared.exists():
            shutil.copy2(shared, user_path)
        return user_path
    return user_path if user_path.exists() else shared


def resolve_dir(kind, code_, for_write=False):
    """Path to a country's data/ or outputs/ folder, same copy-on-write
    semantics as resolve_config_path. `kind` is 'data' or 'outputs'."""
    shared = REPO_ROOT / kind / code_
    user_path = user_workspace_root() / kind / code_
    if for_write:
        if not user_path.exists():
            if shared.exists():
                shutil.copytree(shared, user_path)
            else:
                user_path.mkdir(parents=True, exist_ok=True)
        return user_path
    return user_path if user_path.exists() else shared


def all_visible_config_names():
    """Shared countries plus this user's own additions, merged, no dupes,
    no username tags — exactly what the dropdown should show."""
    shared = {Path(p).stem for p in glob.glob(str(REPO_ROOT / "config" / "*.yaml"))}
    own = {Path(p).stem for p in glob.glob(str(user_workspace_root() / "config" / "*.yaml"))}
    return sorted(shared | own)


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

# ─── USERNAME GATE ───────────────────────────────────────────────────────────
# Lightweight, no-password workspace picker for the pilot. Every distinct
# name gets its own isolated copy-on-write workspace (see resolve_config_path
# / resolve_dir above); nothing here is a real login and it should not be
# treated as one.
if "username" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <h1>UNDP Early Recovery Impact</h1>
        <p>Enter your name to continue. This keeps your changes separate from everyone else's.</p>
    </div>
    """, unsafe_allow_html=True)
    _known_users = _load_users()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Continue as an existing user**")
        if _known_users:
            _pick_user = st.selectbox("Existing users", _known_users,
                                       key="pick_existing_user", label_visibility="collapsed")
            if st.button("Continue", key="continue_existing", width='stretch'):
                st.session_state.username = _pick_user
                st.rerun()
        else:
            st.caption("No users yet — create the first one on the right.")
    with c2:
        st.markdown("**Or create a new user**")
        _new_user = st.text_input("New username", key="new_username_input",
                                   label_visibility="collapsed", placeholder="e.g. carlo, joelle...")
        if st.button("Create & continue", key="create_user", width='stretch', type="primary"):
            _clean = "".join(ch for ch in _new_user.strip().lower().replace(" ", "_")
                              if ch.isalnum() or ch in "_-")
            if not _clean:
                st.error("Please enter a valid name (letters, numbers, - or _).")
            else:
                _register_user(_clean)
                st.session_state.username = _clean
                st.rerun()
    st.stop()

USERNAME = st.session_state.username

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
# ISO3 -> ISO2 for common country codes (extend as needed)
_ISO3_TO_ISO2 = {
    'VEN':'VE','MMR':'MM','PSE':'PS','AFG':'AF','UKR':'UA','SDN':'SD','SYR':'SY',
    'COD':'CD','SOM':'SO','NPL':'NP','JAM':'JM','MWI':'MW','COL':'CO','IDN':'ID',
    'BOL':'BO','NGA':'NG','KEN':'KE','ETH':'ET','TCD':'TD','HTI':'HT','YEM':'YE',
    'LBN':'LB','IRQ':'IQ','MLI':'ML','NER':'NE','BFA':'BF','MOZ':'MZ','ZWE':'ZW',
}
def iso_to_flag(code):
    if not code: return '🌐'
    c = code.upper()
    iso2 = _ISO3_TO_ISO2.get(c, c[:2] if len(c) == 2 else None)
    if iso2 and len(iso2) == 2 and iso2.isalpha():
        try:
            return ''.join(chr(0x1F1E6 + ord(ch) - ord('A')) for ch in iso2)
        except Exception:
            pass
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
    'jam': {'flag':'🇯🇲','name':'Jamaica','event':'Hurricane Melissa','event_date':'Oct 28, 2025'},
    'mwi': {'flag':'🇲🇼','name':'Malawi','event':'Cyclone Freddy','event_date':'Mar 12, 2023'},
    'npl': {'flag':'🇳🇵','name':'Nepal','event':'M6.4 Earthquake','event_date':'Nov 3, 2023'},
    'jamaica': {'flag':'🇯🇲','name':'Jamaica','event':'Hurricane Melissa','event_date':'Oct 28, 2025'},
    'cub': {'flag':'🇨🇺','name':'Cuba','event':'Hurricane Melissa','event_date':'Oct 29, 2025'},
    'bhs': {'flag':'🇧🇸','name':'Bahamas','event':'Hurricane Melissa','event_date':'Oct 30, 2025'},
    'hti': {'flag':'🇭🇹','name':'Haiti','event':'Armed Conflict','event_date':'Mar 1, 2024'},
    'lbn': {'flag':'🇱🇧','name':'Lebanon','event':'Armed Conflict','event_date':'Oct 1, 2024'},
    'eth': {'flag':'🇪🇹','name':'Ethiopia','event':'Armed Conflict','event_date':'Nov 4, 2020'},
    'irq': {'flag':'🇮🇶','name':'Iraq','event':'Conflict','event_date':'Jan 1, 2022'},
    'yem': {'flag':'🇾🇪','name':'Yemen','event':'Armed Conflict','event_date':'Jan 1, 2022'},
    'yem': {'flag':'🇾🇪','name':'Yemen','event':'Armed Conflict','event_date':'Jan 1, 2022'},
    'hti': {'flag':'🇭🇹','name':'Haiti','event':'Armed Conflict','event_date':'Mar 1, 2024'},
    'lbn': {'flag':'🇱🇧','name':'Lebanon','event':'Armed Conflict','event_date':'Oct 1, 2024'},
    'irq': {'flag':'🇮🇶','name':'Iraq','event':'Armed Conflict','event_date':'Jan 1, 2022'},
    'eth': {'flag':'🇪🇹','name':'Ethiopia','event':'Tigray Conflict','event_date':'Nov 4, 2020'},
    'afghanistan': {'flag':'🇦🇫','name':'Afghanistan','event':'M6.3 Earthquake','event_date':'Oct 7, 2023'},
    'gaza': {'flag':'🇵🇸','name':'Gaza Strip','event':'Armed Conflict','event_date':'Oct 7, 2023'},
    'ukraine': {'flag':'🇺🇦','name':'Ukraine','event':'Armed Conflict','event_date':'Feb 24, 2022'},
    'sudan': {'flag':'🇸🇩','name':'Sudan','event':'Civil War','event_date':'Apr 15, 2023'},
    'syria': {'flag':'🇸🇾','name':'Syria','event':'M7.8 Earthquake','event_date':'Feb 6, 2023'},
    'dr_congo': {'flag':'🇨🇩','name':'DR Congo','event':'M23 Conflict','event_date':'Mar 1, 2022'},
    'myanmar': {'flag':'🇲🇲','name':'Myanmar','event':'M7.7 Earthquake','event_date':'Mar 28, 2025'},
    # Additional countries
    'Albania':  {'iso':'ALB','acled':'Albania','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Argentina':  {'iso':'ARG','acled':'Argentina','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Azerbaijan':  {'iso':'AZE','acled':'Azerbaijan','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Bosnia and Herzegovina':  {'iso':'BIH','acled':'Bosnia and Herzegovina','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Botswana':  {'iso':'BWA','acled':'Botswana','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Cape Verde':  {'iso':'CPV','acled':'Cape Verde','etype':'drought','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Congo':  {'iso':'COG','acled':'Republic of Congo','etype':'flood','edate':'2025-01-01','commodity':'Cassava flour','admin':2},
    'Equatorial Guinea':  {'iso':'GNQ','acled':'Equatorial Guinea','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Gabon':  {'iso':'GAB','acled':'Gabon','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Kazakhstan':  {'iso':'KAZ','acled':'Kazakhstan','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Kosovo':  {'iso':'XKX','acled':'Kosovo','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Malaysia':  {'iso':'MYS','acled':'Malaysia','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Mongolia':  {'iso':'MNG','acled':'Mongolia','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Montenegro':  {'iso':'MNE','acled':'Montenegro','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Morocco':  {'iso':'MAR','acled':'Morocco','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'North Macedonia':  {'iso':'MKD','acled':'North Macedonia','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Paraguay':  {'iso':'PRY','acled':'Paraguay','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Russia':  {'iso':'RUS','acled':'Russia','etype':'conflict','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Saint Kitts and Nevis':  {'iso':'KNA','acled':'Saint Kitts and Nevis','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Serbia':  {'iso':'SRB','acled':'Serbia','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Thailand':  {'iso':'THA','acled':'Thailand','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Turkmenistan':  {'iso':'TKM','acled':'Turkmenistan','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Uruguay':  {'iso':'URY','acled':'Uruguay','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
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

METHODOLOGY_NOTE = """<b>Methodological note.</b> Scores are calculated only for administrative units that experienced a significant shock, defined as a decline greater than 10% relative to the pre-event baseline. Indicators below this threshold are classified as <em>Stable</em> and are excluded from recovery scoring. To ensure comparability over time, Vegetation and Nightlights indicators are adjusted on a year-over-year basis to remove natural seasonal variation (e.g., harvest cycles, snow cover, or holiday effects). <b>Targeted Recovery</b> measures the share of the initial decline that has been regained since the lowest observed point of the crisis (the trough), capped at 100%. <b>Since Intervention</b> captures the recovery achieved after the intervention start date, expressed as a percentage of the total disaster impact and likewise capped at 100%. These metrics describe observed temporal dynamics to support rapid prioritisation and resource allocation; they reflect recovery trends but do not imply direct causal attribution to any single organisation or intervention.<br><br><b>Disclaimer.</b> The accuracy of these results depends on the quality and resolution of the underlying satellite data. In some contexts, indicators such as nighttime lights may be affected by seasonal patterns, cloud cover, snow reflectance, or sensor calibration changes that are not fully captured by the year-over-year adjustment. This dashboard presents <em>correlations</em> between the timing of an event, the start of early recovery interventions, and changes in satellite-derived indicators; it does not constitute evidence of causal impact. In most contexts, UNDP is not the only actor on the ground, and observed recovery trends reflect the combined contributions of multiple organisations and local actors. Isolating UNDP's specific contribution requires econometric approaches (e.g., Difference-in-Differences, Synthetic Control) available in the Impact Measurement tab, which support attribution to the extent possible given data constraints."""

QUICK_REF = {
    # Current UNDP priority countries
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
    # Africa
    'Algeria':      {'iso':'DZA','acled':'Algeria','etype':'earthquake','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Angola':       {'iso':'AGO','acled':'Angola','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Burkina Faso': {'iso':'BFA','acled':'Burkina Faso','etype':'conflict','edate':'2025-01-01','commodity':'Millet','admin':2},
    'Burundi':      {'iso':'BDI','acled':'Burundi','etype':'conflict','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Cameroon':     {'iso':'CMR','acled':'Cameroon','etype':'conflict','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Central African Republic': {'iso':'CAF','acled':'Central African Republic','etype':'conflict','edate':'2025-01-01','commodity':'Cassava flour','admin':2},
    'Chad':         {'iso':'TCD','acled':'Chad','etype':'conflict','edate':'2025-01-01','commodity':'Millet','admin':2},
    'Côte d\'Ivoire': {'iso':'CIV','acled':'Ivory Coast','etype':'conflict','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Egypt':        {'iso':'EGY','acled':'Egypt','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Eritrea':      {'iso':'ERI','acled':'Eritrea','etype':'drought','edate':'2025-01-01','commodity':'Sorghum','admin':2},
    'Ghana':        {'iso':'GHA','acled':'Ghana','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Guinea':       {'iso':'GIN','acled':'Guinea','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Kenya':        {'iso':'KEN','acled':'Kenya','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Libya':        {'iso':'LBY','acled':'Libya','etype':'conflict','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Madagascar':   {'iso':'MDG','acled':'Madagascar','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Malawi':       {'iso':'MWI','acled':'Malawi','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Mali':         {'iso':'MLI','acled':'Mali','etype':'conflict','edate':'2025-01-01','commodity':'Millet','admin':2},
    'Mauritania':   {'iso':'MRT','acled':'Mauritania','etype':'drought','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Mozambique':   {'iso':'MOZ','acled':'Mozambique','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Niger':        {'iso':'NER','acled':'Niger','etype':'conflict','edate':'2025-01-01','commodity':'Millet','admin':2},
    'Nigeria':      {'iso':'NGA','acled':'Nigeria','etype':'conflict','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Rwanda':       {'iso':'RWA','acled':'Rwanda','etype':'flood','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Senegal':      {'iso':'SEN','acled':'Senegal','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Sierra Leone': {'iso':'SLE','acled':'Sierra Leone','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'South Africa': {'iso':'ZAF','acled':'South Africa','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'South Sudan':  {'iso':'SSD','acled':'South Sudan','etype':'conflict','edate':'2025-01-01','commodity':'Sorghum','admin':2},
    'Tanzania':     {'iso':'TZA','acled':'Tanzania','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Togo':         {'iso':'TGO','acled':'Togo','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Tunisia':      {'iso':'TUN','acled':'Tunisia','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Uganda':       {'iso':'UGA','acled':'Uganda','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Zambia':       {'iso':'ZMB','acled':'Zambia','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Zimbabwe':     {'iso':'ZWE','acled':'Zimbabwe','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    # Asia & Pacific
    'Cambodia':     {'iso':'KHM','acled':'Cambodia','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'China':        {'iso':'CHN','acled':'China','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Fiji':         {'iso':'FJI','acled':'Fiji','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'India':        {'iso':'IND','acled':'India','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Indonesia':    {'iso':'IDN','acled':'Indonesia','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Japan':        {'iso':'JPN','acled':'Japan','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Laos':         {'iso':'LAO','acled':'Laos','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Nepal':        {'iso':'NPL','acled':'Nepal','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Pakistan':     {'iso':'PAK','acled':'Pakistan','etype':'flood','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Papua New Guinea': {'iso':'PNG','acled':'Papua New Guinea','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Philippines':  {'iso':'PHL','acled':'Philippines','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Sri Lanka':    {'iso':'LKA','acled':'Sri Lanka','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Vanuatu':      {'iso':'VUT','acled':'Vanuatu','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Vietnam':      {'iso':'VNM','acled':'Vietnam','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    # Middle East
    'Iran':         {'iso':'IRN','acled':'Iran','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Jordan':       {'iso':'JOR','acled':'Jordan','etype':'flood','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Turkey':       {'iso':'TUR','acled':'Turkey','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    # Americas
    'Bolivia':      {'iso':'BOL','acled':'Bolivia','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Brazil':       {'iso':'BRA','acled':'Brazil','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Chile':        {'iso':'CHL','acled':'Chile','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Colombia':     {'iso':'COL','acled':'Colombia','etype':'conflict','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Cuba':         {'iso':'CUB','acled':'Cuba','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Dominican Republic': {'iso':'DOM','acled':'Dominican Republic','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Ecuador':      {'iso':'ECU','acled':'Ecuador','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'El Salvador':  {'iso':'SLV','acled':'El Salvador','etype':'earthquake','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Guatemala':    {'iso':'GTM','acled':'Guatemala','etype':'earthquake','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Honduras':     {'iso':'HND','acled':'Honduras','etype':'cyclone','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Mexico':       {'iso':'MEX','acled':'Mexico','etype':'earthquake','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Nicaragua':    {'iso':'NIC','acled':'Nicaragua','etype':'cyclone','edate':'2025-01-01','commodity':'Beans','admin':2},
    'Peru':         {'iso':'PER','acled':'Peru','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Venezuela':    {'iso':'VEN','acled':'Venezuela','etype':'conflict','edate':'2025-01-01','commodity':'Rice','admin':2},
    # Caribbean
    'Jamaica':      {'iso':'JAM','acled':'Jamaica','etype':'cyclone','edate':'2025-10-28','commodity':'Rice','admin':2},
    'Bahamas':      {'iso':'BHS','acled':'Bahamas','etype':'cyclone','edate':'2025-10-30','commodity':'Rice','admin':2},
    'Trinidad and Tobago': {'iso':'TTO','acled':'Trinidad and Tobago','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Barbados':     {'iso':'BRB','acled':'Barbados','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Guyana':       {'iso':'GUY','acled':'Guyana','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Suriname':     {'iso':'SUR','acled':'Suriname','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Belize':       {'iso':'BLZ','acled':'Belize','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Costa Rica':   {'iso':'CRI','acled':'Costa Rica','etype':'earthquake','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Panama':       {'iso':'PAN','acled':'Panama','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Grenada':      {'iso':'GRD','acled':'Grenada','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Saint Lucia':  {'iso':'LCA','acled':'Saint Lucia','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Saint Vincent and the Grenadines': {'iso':'VCT','acled':'Saint Vincent and the Grenadines','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Antigua and Barbuda': {'iso':'ATG','acled':'Antigua and Barbuda','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Dominica':     {'iso':'DMA','acled':'Dominica','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    # Pacific Islands
    'Tonga':        {'iso':'TON','acled':'Tonga','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Samoa':        {'iso':'WSM','acled':'Samoa','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Solomon Islands': {'iso':'SLB','acled':'Solomon Islands','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Timor-Leste':  {'iso':'TLS','acled':'Timor-Leste','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Tuvalu':       {'iso':'TUV','acled':'Tuvalu','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Marshall Islands': {'iso':'MHL','acled':'Marshall Islands','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Micronesia':   {'iso':'FSM','acled':'Micronesia','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    # South/Central Asia
    'Bhutan':       {'iso':'BTN','acled':'Bhutan','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Maldives':     {'iso':'MDV','acled':'Maldives','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    # Africa (additional)
    'Benin':        {'iso':'BEN','acled':'Benin','etype':'flood','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Comoros':      {'iso':'COM','acled':'Comoros','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Djibouti':     {'iso':'DJI','acled':'Djibouti','etype':'drought','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Eswatini':     {'iso':'SWZ','acled':'Eswatini','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Gambia':       {'iso':'GMB','acled':'Gambia','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Guinea-Bissau': {'iso':'GNB','acled':'Guinea-Bissau','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Lesotho':      {'iso':'LSO','acled':'Lesotho','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Liberia':      {'iso':'LBR','acled':'Liberia','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Namibia':      {'iso':'NAM','acled':'Namibia','etype':'drought','edate':'2025-01-01','commodity':'Maize','admin':2},
    'Sao Tome and Principe': {'iso':'STP','acled':'Sao Tome and Principe','etype':'flood','edate':'2025-01-01','commodity':'Rice','admin':2},
    'Seychelles':   {'iso':'SYC','acled':'Seychelles','etype':'cyclone','edate':'2025-01-01','commodity':'Rice','admin':2},
        # Caribbean
        # Europe & Central Asia
    'Armenia':      {'iso':'ARM','acled':'Armenia','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Georgia':      {'iso':'GEO','acled':'Georgia','etype':'conflict','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Kyrgyzstan':   {'iso':'KGZ','acled':'Kyrgyzstan','etype':'earthquake','edate':'2025-01-01','commodity':'Bread','admin':2},
    'Moldova':      {'iso':'MDA','acled':'Moldova','etype':'flood','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Tajikistan':   {'iso':'TJK','acled':'Tajikistan','etype':'earthquake','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
    'Uzbekistan':   {'iso':'UZB','acled':'Uzbekistan','etype':'earthquake','edate':'2025-01-01','commodity':'Wheat flour','admin':2},
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
    _uc1, _uc2 = st.columns([3, 1])
    with _uc1:
        st.caption("Signed in as **" + USERNAME + "**")
    with _uc2:
        if st.button("Switch", key="switch_user_btn", help="Switch to a different user"):
            del st.session_state["username"]
            st.rerun()
    # Admin/User mode
    is_admin = st.toggle("Admin Mode", value=False, key="admin_toggle")
    if is_admin:
        _modes = ["\U0001f4ca View Results", "\u2699\ufe0f Setup & Run", "\u2139\ufe0f About"]
    else:
        _modes = ["\U0001f4ca View Results", "\u2139\ufe0f About"]
    mode = st.radio("Mode", _modes, label_visibility="collapsed", key="main_mode")
    st.markdown("---")

    cfg_names = all_visible_config_names()

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
        with open(resolve_config_path(selected)) as f:
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
            with open(resolve_config_path(selected)) as f:
                config = yaml.safe_load(f)
        else:
            selected = "none"
            config = {'country_name':'(none)','country_code':'XXX','events':{'disaster_date':'2025-01-01','intervention_start':'2025-01-01'},'study_period_start':'2024-01-01','study_period_end':'2025-12-31','boundaries':{'admin_level':2}}

    st.markdown("---")
    st.markdown('<div style="text-align:center;font-size:.72rem;color:#484f58;">Built by AImpact Lab</div>', unsafe_allow_html=True)


# ─── SHARED VARIABLES ────────────────────────────────────────────────────────
code = config['country_code'].lower()
output_dir = resolve_dir("outputs", code)
figures_dir = output_dir / "figures"
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
# GDP per capita (World Bank WDI 2023) and population (UN estimates, millions)
GDP_PC = {
    'AFG':380,'BGD':2688,'COD':654,'ETH':1027,'PSE':3664,'HTI':1748,'IRQ':5765,
    'LBN':4136,'MMR':1093,'SOM':446,'SDN':516,'SYR':530,'UKR':4534,'YEM':585,
    'DZA':3900,'AGO':2080,'BFA':831,'BDI':259,'CMR':1663,'CAF':461,'TCD':694,
    'CIV':2549,'EGY':3699,'GHA':2363,'KEN':2099,'LBY':6357,'MDG':507,'MWI':634,
    'MLI':879,'MOZ':539,'NER':554,'NGA':1621,'RWA':966,'SSD':393,'TZA':1192,
    'UGA':964,'ZMB':1138,'ZWE':1737,'KHM':1785,'CHN':12541,'IND':2485,'IDN':4788,
    'NPL':1337,'PAK':1505,'PHL':3905,'LKA':3814,'VNM':4164,'IRN':4091,'JOR':4703,
    'TUR':10616,'BOL':3345,'BRA':8917,'CHL':16265,'COL':6104,'ECU':6083,'SLV':4409,
    'GTM':5474,'HND':2831,'MEX':10948,'PER':7002,'ARM':7279,'GEO':7629,'KGZ':1506,
    'MDA':5563,'TJK':1058,'VEN':3500,'NIC':2150,'CUB':3800,'DOM':9700,'SEN':1690,'JAM':5750,'CUB':3800,'BHS':28600,'TTO':17400,'BRB':16000,'GUY':14700,
    'SUR':5800,'BLZ':4800,'CRI':12500,'PAN':14400,'GRD':10400,'LCA':10400,
    'VCT':8600,'ATG':16700,'DMA':7700,'TON':5000,'WSM':4300,'SLB':2300,
    'TLS':1900,'BEN':1400,'DJI':3400,'SWZ':4000,'GMB':840,'GNB':700,
    'LBR':650,'NAM':5000,'LSO':1100,
}
POP_M = {
    'AFG':41.1,'BGD':173,'COD':102.3,'ETH':126.5,'PSE':5.4,'HTI':11.7,'IRQ':44.5,
    'LBN':5.5,'MMR':54.4,'SOM':18.1,'SDN':48,'SYR':23.2,'UKR':37,'YEM':33.7,
    'DZA':45.6,'AGO':36.7,'BFA':23.3,'BDI':13.2,'CMR':28.6,'CAF':5.7,'TCD':18.3,
    'CIV':29.4,'EGY':109.3,'GHA':34,'KEN':55.1,'LBY':7,'MDG':30,'MWI':20.9,
    'MLI':23.3,'MOZ':33.9,'NER':27,'NGA':223.8,'RWA':14,'SSD':11,'TZA':67,
    'UGA':48.6,'ZMB':20.6,'ZWE':16.3,'KHM':17,'CHN':1412,'IND':1429,'IDN':278,
    'NPL':30.9,'PAK':240.5,'PHL':117.3,'LKA':22,'VNM':100,'IRN':88.6,'JOR':11.3,
    'TUR':85.3,'BOL':12.4,'BRA':216,'CHL':19.6,'COL':52.1,'ECU':18,'SLV':6.3,
    'GTM':17.6,'HND':10.4,'MEX':129,'PER':34,'ARM':2.8,'GEO':3.7,'KGZ':7,
    'MDA':2.6,'TJK':10.1,'VEN':28.4,'NIC':7,'CUB':11.1,'DOM':11.2,'SEN':17.7,'JAM':2.8,'CUB':11.1,'BHS':0.4,'TTO':1.4,'BRB':0.3,'GUY':0.8,
    'SUR':0.6,'BLZ':0.4,'CRI':5.2,'PAN':4.4,'GRD':0.1,'LCA':0.2,
    'VCT':0.1,'ATG':0.1,'DMA':0.07,'TON':0.1,'WSM':0.2,'SLB':0.7,
    'TLS':1.3,'BEN':13.4,'DJI':1.1,'SWZ':1.2,'GMB':2.7,'GNB':2.1,
    'LBR':5.3,'NAM':2.6,'LSO':2.3,
}


def load_wb_data(code, event_year):
    """Load World Bank GDP and population data. Returns dict with pre-shock values."""
    wb_path = resolve_dir("data", code) / "external" / "wb_indicators.csv"
    if not wb_path.exists():
        wb_path = resolve_dir("outputs", code) / "external" / "wb_indicators.csv"
    result = {"gdp": 0, "gdp_pc": 0, "pop": 0, "year": 0, "source": "N/A"}
    if not wb_path.exists():
        # Fallback to hardcoded
        cc = code.upper()
        result["gdp_pc"] = GDP_PC.get(cc, 1000)
        result["pop"] = POP_M.get(cc, 10) * 1000000
        result["gdp"] = result["gdp_pc"] * result["pop"]
        result["year"] = event_year - 1
        result["source"] = "Hardcoded estimates"
        return result
    df = pd.read_csv(wb_path)
    # Get the latest year BEFORE the event
    for ind, key in [("gdp_current_usd", "gdp"), ("gdp_per_capita_usd", "gdp_pc"), ("population", "pop")]:
        subset = df[(df["indicator"] == ind) & (df["year"] < event_year)].sort_values("year")
        if not subset.empty:
            result[key] = float(subset.iloc[-1]["value"])
            result["year"] = int(subset.iloc[-1]["year"])
    result["source"] = "World Bank WDI"
    if result["gdp"] == 0 and result["gdp_pc"] > 0 and result["pop"] > 0:
        result["gdp"] = result["gdp_pc"] * result["pop"]
    return result


def render_copilot_sidebar(config, figures_dir, code):
    """AI co-pilot in sidebar."""
    ctx = []
    ctx.append("Country: " + config.get("country_name", "Unknown"))
    ctx.append("Disaster: " + str(config["events"].get("disaster_date", "")))
    ctx.append("Intervention: " + str(config["events"].get("intervention_start", "N/A")))
    sc_path = figures_dir / "scorecard_data.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        ctx.append("SCORECARD SUMMARY:")
        for _, r in sc.iterrows():
            nm = r.get("short_name", r.get("indicator", ""))
            ctx.append("  " + str(nm) + ": overall " + str(round(r.get("overall_recovery_pct", 0))) + "%, "
                       + "since project " + str(round(r.get("undp_recovery_pct", 0))) + "%, "
                       + str(int(r.get("n_affected_units", 0))) + "/" + str(int(r.get("n_total_units", 0))) + " units")
    undp_csv = resolve_dir("data", code) / "external" / "undp_projects.csv"
    if not undp_csv.exists(): undp_csv = resolve_dir("outputs", code) / "external" / "undp_projects.csv"
    if undp_csv.exists():
        udf = pd.read_csv(undp_csv)
        ctx.append("UNDP projects: " + str(len(udf)))
        if "total_budget" in udf.columns:
            ctx.append("Total budget: $" + "{:,.0f}".format(udf["total_budget"].sum()))
    # Load monthly time series from panel data for richer analysis
    panel_paths = [
        resolve_dir("outputs", code) / "panel_data.csv",
        resolve_dir("data", code) / ("gee_panel_" + code + "_centroid.csv"),
    ]
    for pp in panel_paths:
        if pp.exists():
            try:
                pdf = pd.read_csv(pp)
                pdf["_dt"] = pd.to_datetime(pdf.get("period_start", pdf.get("date", "")), errors="coerce")
                pdf = pdf.dropna(subset=["_dt"])
                ctx.append(chr(10) + "MONTHLY TIME SERIES (national averages):")
                indicators = [c for c in ["nightlights", "ndvi", "vv", "vh"] if c in pdf.columns]
                monthly = pdf.groupby(pdf["_dt"].dt.to_period("M"))[indicators].mean()
                event_dt = str(config["events"].get("disaster_date", ""))[:7]
                intv_dt = str(config["events"].get("intervention_start", ""))[:7]
                for period, row in monthly.iterrows():
                    marker = ""
                    if str(period) == event_dt: marker = " [EVENT]"
                    elif str(period) == intv_dt: marker = " [INTERVENTION]"
                    vals = ", ".join(str(c) + "=" + "{:.2f}".format(row[c]) for c in indicators if pd.notna(row[c]))
                    ctx.append("  " + str(period) + ": " + vals + marker)
                ctx.append(chr(10) + "Use this time series to identify trends, breakpoints, and anomalies.")
                ctx.append("The [EVENT] marker shows the disaster date, [INTERVENTION] shows when the UNDP project started.")
            except Exception:
                pass
            break
    # Add cross-country comparison data
    ctx.append(chr(10) + "CROSS-COUNTRY COMPARISON (all available countries):")
    import glob as _glob
    for _sc_file in sorted(_glob.glob("outputs/*/figures/scorecard_data.csv")):
        try:
            _cc_code = _sc_file.split("/")[1]
            _cc_sc = pd.read_csv(_sc_file)
            _cc_cfg_path = "config/" + _cc_code + ".yaml"
            if not Path(_cc_cfg_path).exists():
                for _cf in _glob.glob("config/*.yaml"):
                    _cfg_tmp = yaml.safe_load(open(_cf))
                    if _cfg_tmp.get("country_code", "").lower() == _cc_code:
                        _cc_cfg_path = _cf
                        break
            _cc_name = _cc_code.upper()
            if Path(_cc_cfg_path).exists():
                _cc_name = yaml.safe_load(open(_cc_cfg_path)).get("country_name", _cc_code)
            _nl = _cc_sc[_cc_sc["indicator"].str.contains("nightlight", case=False, na=False)]
            if not _nl.empty:
                _ov = round(_nl.iloc[0].get("overall_recovery_pct", 0))
                _un = round(_nl.iloc[0].get("undp_recovery_pct", 0))
                ctx.append("  " + _cc_name + ": NL targeted=" + str(_ov) + "%, since project=" + str(_un) + "%")
        except Exception:
            pass
    ctx.append("Use this to compare recovery across countries when asked.")
    context = chr(10).join(ctx)
    country_name = config.get("country_name", "Unknown")

    sys_prompt = (
        "You are a senior development economist (PhD, MIT-trained) embedded in a UNDP Early Recovery dashboard. "
        "You help policy makers interpret satellite-based recovery monitoring data with precision and clarity. "
        "You give direct, actionable answers grounded in the data. You avoid jargon but maintain analytical rigour. "
        "When you cite numbers, you explain what they mean in practical terms for affected populations." + chr(10) + chr(10)
        + "DATA for " + country_name + ":" + chr(10) + context + chr(10) + chr(10)
        + "RULES:" + chr(10)
        + "1. All indicators show CORRELATIONS, not causal impact. Be explicit about this." + chr(10)
        + "2. UNDP is one of many actors - never imply sole attribution." + chr(10)
        + "3. Nightlight-GDP elasticity of 0.3 (Henderson et al. 2012) is a global average." + chr(10)
        + "4. 'Since Project' = what happened AFTER intervention date, not BECAUSE of it." + chr(10)
        + "5. Be concise and actionable. Recommend specific next steps when appropriate." + chr(10)
        + "6. Give complete, thorough answers. Never cut off mid-sentence." + chr(10)
        + "7. You cannot see images, but you HAVE the full monthly time series data. When asked about trends, maps, or charts, analyze the time series numbers directly. Identify breakpoints, trends, and anomalies." + chr(10)
        + "8. Use plain language a policy maker would understand. Avoid academic hedging."
    )

    if "copilot_msgs" not in st.session_state:
        st.session_state.copilot_msgs = []

    # Display last few messages
    for msg in st.session_state.copilot_msgs[-6:]:
        role_label = "You" if msg["role"] == "user" else "Co-pilot"
        st.markdown("**" + role_label + ":** " + msg["content"])

    # Use a form to prevent rerun on every keystroke
    with st.form("copilot_form", clear_on_submit=True):
        query = st.text_input("Question", key="copilot_input", label_visibility="collapsed",
                              placeholder="Ask about this country's recovery data...")
        submitted = st.form_submit_button("Send")

    if submitted and query:
        st.session_state.copilot_msgs.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    try:
                        api_key = st.secrets["OPENAI_API_KEY"]
                    except Exception:
                        pass
                if not api_key:
                    env_path = Path(".env")
                    if env_path.exists():
                        for ln in env_path.read_text().splitlines():
                            if ln.startswith("OPENAI_API_KEY="):
                                api_key = ln.split("=", 1)[1].strip().strip(chr(34)).strip(chr(39))
                if not api_key:
                    reply = "API key not found. Add it in Setup > Credentials."
                else:
                    client = OpenAI(api_key=api_key)
                    messages = [{"role": "system", "content": sys_prompt}]
                    messages += [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.copilot_msgs[-6:]]
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", messages=messages,
                        max_tokens=2000, temperature=0.3, timeout=60)
                    reply = resp.choices[0].message.content
            except Exception as e:
                reply = "Error: " + str(e)[:200]
        st.session_state.copilot_msgs.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.copilot_msgs:
        if st.button("Clear chat", key="clr_copilot"):
            st.session_state.copilot_msgs = []
            st.rerun()


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


def estimate_economic_reactivation(config, figures_dir, code):
    """Indicative economic reactivation using nightlights-GDP elasticity.
    Henderson et al. (2012): elasticity ~0.3.
    Uses ONLY nightlights for GDP conversion (other indicators don't have GDP elasticity).
    Ties estimates to the currently selected project."""
    scorecard_csv = figures_dir / "scorecard_data.csv"
    undp_csv = resolve_dir("data", code) / "external" / "undp_projects.csv"
    if not scorecard_csv.exists() or not undp_csv.exists():
        return None
    sc = pd.read_csv(scorecard_csv)

    # ONLY use nightlights — the elasticity only applies to nightlights-GDP
    nl_row = sc[sc['indicator'].str.contains('nightlight', case=False, na=False)]
    if not nl_row.empty:
        nl_overall = float(nl_row.iloc[0].get('overall_recovery_pct', 0))
        nl_undp = float(nl_row.iloc[0].get('undp_recovery_pct', 0))
        n_affected = int(nl_row.iloc[0].get('n_affected_units', 0))
        n_total = int(nl_row.iloc[0].get('n_total_units', 0))
    else:
        nl_overall, nl_undp = 0.0, 0.0
        any_aff = sc[sc['n_affected_units'] > 0]
        n_affected = int(any_aff.iloc[0]['n_affected_units']) if not any_aff.empty else 0
        n_total = int(any_aff.iloc[0]['n_total_units']) if not any_aff.empty else 0

    # Read UNDP projects
    undp_df = pd.read_csv(undp_csv)
    if 'total_budget' not in undp_df.columns:
        if 'budget' in undp_df.columns:
            undp_df['total_budget'] = undp_df['budget']
        else:
            return None
    undp_df['start_date'] = pd.to_datetime(undp_df.get('start_date'), errors='coerce')
    event_dt = pd.Timestamp(config['events']['disaster_date'])
    intervention_dt = pd.Timestamp(config['events'].get('intervention_start', config['events']['disaster_date']))
    post = undp_df[undp_df['start_date'] >= event_dt].dropna(subset=['total_budget']).copy()
    if post.empty:
        return None

    ELASTICITY = 0.3
    coverage = n_affected / max(n_total, 1)

    # Build per-programme estimates
    # The selected project's recovery is nl_undp (since project)
    # Other programmes: use targeted recovery as a rough proxy
    rows = []
    for _, pr in post.iterrows():
        budget = float(pr['total_budget'])
        if pd.isna(budget) or budget <= 0:
            continue
        title = str(pr.get('title', ''))[:55]
        pr_date = pr['start_date'].strftime('%Y-%m-%d') if pd.notna(pr['start_date']) else '-'

        # Is this the currently selected project?
        is_selected = (pr_date == str(intervention_dt.date()))

        # Use programme-specific recovery if selected, overall otherwise
        if is_selected and nl_undp > 0:
            prog_recovery = nl_undp
            label = "Since project"
        elif nl_overall > 0:
            prog_recovery = nl_overall
            label = "Targeted"
        else:
            prog_recovery = 0
            label = "No recovery"

        gdp_effect = prog_recovery * ELASTICITY
        multiplier = max(1.0, 1 + (prog_recovery / 100) * ELASTICITY * 10 * coverage)
        indicative_value = budget * multiplier
        roi = (indicative_value - budget) / budget if budget > 0 else 0

        rows.append({
            'Project': title,
            'Start': pr_date,
            'Budget (USD)': budget,
            'NL Recovery': f"{prog_recovery:.0f}%",
            'GDP Effect': f"{gdp_effect:.1f}%",
            'Est. Econ. Value': indicative_value,
            'ROI': roi,
            '_selected': is_selected,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values('_selected', ascending=False)

    # Summary for selected project (or best available)
    sel = [r for r in rows if r['_selected']]
    if sel:
        s = sel[0]
    else:
        s = max(rows, key=lambda r: r['Budget (USD)'])

    summary = {
        'programme_name': s['Programme'],
        'programme_budget': s['Budget (USD)'],
        'nl_overall': nl_overall,
        'nl_undp': nl_undp,
        'nl_recovery_used': float(s['NL Recovery'].replace('%','')),
        'gdp_effect': float(s['GDP Effect'].replace('%','')),
        'indicative_value': s['Ind. Value'],
        'roi': s['ROI'],
        'n_programmes': len(rows),
        'n_affected': n_affected,
        'n_total': n_total,
        'is_selected': s['_selected'],
    }
    return df, summary

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
        if oc > 0 and uc > oc:
            uc = oc
        if oc <= 0:
            uc = 0

        # Use CAPPED values for both display and bars
        oc_c = "#3fb950" if oc >= 75 else ("#d29922" if oc >= 50 else "#f85149")
        if uc < 0 or undp < 0:
            uc_c = "#484f58"; ud = "N/A"
        elif uc >= 50:
            uc_c = "#58a6ff"; ud = f"{uc:.0f}%"
        elif uc >= 25:
            uc_c = "#79c0ff"; ud = f"{uc:.0f}%"
        else:
            uc_c = "#8b949e"; ud = f"{uc:.0f}%"

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
                    <div class="sc-val-line"><span class="sc-pct" style="color:{oc_c};">{oc:.0f}%</span></div>
                    <div class="sc-val-line"><span class="sc-pct" style="color:{uc_c};">{ud}</span></div>
                </div>
            </div>"""

    # --- Scorecard BARS (without methodology note) ---
    sc_style = """
    <style>
        body { background: transparent; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; color: #c9d1d9; }
        .sc { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem 1.8rem; }
        .sc-title { font-size: 1.2rem; font-weight: 700; color: #e6edf3; margin: 0 0 .2rem 0; }
        .sc-sub { font-size: .82rem; color: #8b949e; margin: 0 0 .8rem 0; }
        .sc-legend { display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
        .sc-legend-item { display: flex; align-items: center; gap: .35rem; font-size: .75rem; color: #8b949e; }
        .sc-legend-dot { width: 8px; height: 8px; border-radius: 2px; }
        .sc-row { display: flex; align-items: center; padding: .65rem 0; border-bottom: 1px solid #21262d; }
        .sc-row:last-of-type { border-bottom: none; }
        .sc-name { width: 140px; font-weight: 600; color: #e6edf3; font-size: .85rem; line-height: 1.3; }
        .sc-units { font-size: .68rem; color: #484f58; font-weight: 400; }
        .sc-desc { width: 180px; font-size: .72rem; color: #8b949e; padding-right: .5rem; }
        .sc-bars { flex: 1; padding: 0 .5rem; }
        .sc-track { background: #0d1117; border-radius: 4px; height: 10px; position: relative; margin: 4px 0; }
        .sc-fill-a { background: linear-gradient(90deg, #238636, #3fb950); height: 10px; border-radius: 4px; }
        .sc-fill-b { background: linear-gradient(90deg, #1f6feb, #79c0ff); height: 10px; border-radius: 4px; }
        .sc-vals { width: 100px; text-align: left; padding-left: .8rem; }
        .sc-val-line { display: flex; align-items: baseline; gap: .3rem; }
        .sc-pct { font-weight: 700; font-size: 1.05rem; min-width: 45px; text-align: right; }
        .sc-stable { color: #3fb950; font-size: .82rem; padding: 6px 0; font-weight: 500; }
        .sc-stable-badge { display: inline-block; background: rgba(35,134,54,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 2em; padding: 1px 10px; font-size: .75rem; font-weight: 500; }
    </style>"""

    bars_html = f"""{sc_style}
    <div class="sc">
        <div class="sc-title">Recovery Scorecard</div>
        <div class="sc-sub">Event: {event_date} &nbsp;·&nbsp; Intervention: {intervention}</div>
        <div class="sc-legend">
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#3fb950;"></div>Targeted Recovery</div>
            <div class="sc-legend-item"><div class="sc-legend-dot" style="background:#1f6feb;"></div>Since Selected Project</div>
        </div>
        {rows_html}
    </div>
    """
    n_rows = len(df)
    components.html(bars_html, height=80 + n_rows * 70 + 20, scrolling=False)

    # Store methodology note for later rendering
    st.session_state['_scorecard_note'] = METHODOLOGY_NOTE
    return True


# ═════════════════════════════════════════════════════════════════════════════
# VIEW RESULTS
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📊 View Results":
        # SCORECARD (bars only — note rendered after UNDP picker)
    # UNDP PROJECT SELECTOR
    _undp_p = resolve_dir("data", code) / "external" / "undp_projects.csv"
    if not _undp_p.exists():
        _undp_p = resolve_dir("outputs", code) / "external" / "undp_projects.csv"
    if _undp_p.exists():
        try:
            _udf = pd.read_csv(_undp_p)
            if len(_udf) > 0 and 'start_date' in _udf.columns:
                _udf['start_date'] = pd.to_datetime(_udf['start_date'], errors='coerce')
                _post_u = _udf[_udf['start_date'] >= pd.Timestamp(event_date)].dropna(subset=['start_date']).sort_values('start_date')
                if len(_post_u) > 0:
                    _opts = []
                    for _, _r in _post_u.iterrows():
                        _d = _r['start_date'].strftime('%Y-%m-%d')
                        _t = str(_r.get('title',''))[:60]
                        _b = "${:,.0f}".format(_r['total_budget']) if pd.notna(_r.get('total_budget')) else ""
                        _opts.append({"label": _d + " - " + _t + " " + _b, "date": _d})
                    _cur = config['events'].get('intervention_start', event_date)
                    _cidx = 0
                    for _i, _o in enumerate(_opts):
                        if _o['date'] == _cur: _cidx = _i; break
                    st.markdown('<div class="section-title">UNDP Project</div>', unsafe_allow_html=True)
                    _pick = st.selectbox("Select project", range(len(_opts)),
                        format_func=lambda i: _opts[i]['label'], index=_cidx, key="undp_proj_top")
                    _nint = _opts[_pick]['date']
                    if _nint != _cur:
                        config['events']['intervention_start'] = _nint
                        _cfg_write_path = resolve_config_path(selected, for_write=True)
                        resolve_dir("data", code, for_write=True)      # ensure user has own copy to read/write
                        resolve_dir("outputs", code, for_write=True)   # (cloned from shared on first touch)
                        with open(_cfg_write_path, 'w') as _fw:
                            yaml.dump(config, _fw, default_flow_style=False, sort_keys=False)
                        subprocess.run(
                            [sys.executable, "pipeline/recovery_scorecard.py", "--config", str(_cfg_write_path)],
                            cwd=str(user_workspace_root()), capture_output=True, timeout=60)
                        st.rerun()
        except Exception:
            pass

    has_scorecard = render_scorecard()
    if not has_scorecard:
        st.info("Run the pipeline first using the buttons under Setup & Run.")

    # UNDP PROGRAMME PICKER (between bars and methodology note)
    if has_scorecard:
        undp_csv = resolve_dir("data", code) / "external" / "undp_projects.csv"
        # UNDP picker moved above scorecard

    # METHODOLOGY & DISCLAIMERS
    if has_scorecard:
        disc_html = """<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:1.2rem 1.6rem;margin:.5rem 0;">
            <div style="font-size:.88rem;font-weight:600;color:#e6edf3;margin-bottom:.6rem;">Methodology & Assumptions</div>
            <div style="font-size:.78rem;color:#c9d1d9;line-height:1.7;">
                <p style="margin:0 0 .5rem 0;"><b>What this shows:</b> Observed changes in satellite indicators before and after disaster events and UNDP interventions. These are correlations, not causal estimates. UNDP is one of many actors on the ground, and observed recovery reflects the combined contributions of multiple organisations, government efforts, and local communities.</p>
                <p style="margin:0 0 .5rem 0;"><b>How recovery is calculated:</b> &ldquo;Targeted Recovery&rdquo; measures how much of the initial decline (from the pre-event average to the single worst post-event month) has been regained. &ldquo;Since Selected Project&rdquo; measures recovery after the intervention start date. Both are capped at 100%, and &ldquo;Since Programme&rdquo; never exceeds &ldquo;Overall&rdquo;. Only units with &gt;10% decline are included; others are &ldquo;Stable&rdquo;. For nightlights, the baseline is the simple mean of all pre-event months. For vegetation (NDVI), a year-over-year seasonal adjustment is applied. For events occurring in the last week of a month, the first full post-event month is used as the starting point.</p>
                <p style="margin:0 0 .5rem 0;"><b>GDP conversion:</b> Nightlight recovery converted to GDP using Henderson, Storeygard &amp; Weil (2012) elasticity of 0.3 (global average). Economic values use World Bank GDP per capita (WDI 2023) and UN population estimates, applied proportionally to affected administrative units.</p>
                <p style="margin:0 0 .5rem 0;"><b>Limitations:</b> (1) Budget = allocations, not disbursements. (2) Satellite data affected by cloud cover, snow, sensors. (3) Causal attribution requires econometric methods (e.g., Synthetic Controls, Difference-in-Differences) in the Impact Measurement tab.</p>
                <p style="margin:0;"><b>Purpose:</b> Rapid monitoring and prioritisation tool. Not a substitute for rigorous impact evaluation.</p>
            </div>
        </div>"""
        st.markdown(disc_html, unsafe_allow_html=True)

    # INDICATIVE ECONOMIC REACTIVATION
    if has_scorecard:
        _undp_path = resolve_dir("data", code) / "external" / "undp_projects.csv"
        if not _undp_path.exists():
            _undp_path = resolve_dir("outputs", code) / "external" / "undp_projects.csv"
        if _undp_path.exists():
            try:
                _undp = pd.read_csv(_undp_path)
                _undp["start_date"] = pd.to_datetime(_undp.get("start_date"), errors="coerce")
                _evt = pd.Timestamp(config["events"]["disaster_date"])
                _intv = pd.Timestamp(config["events"].get("intervention_start", config["events"]["disaster_date"]))
                _post = _undp[_undp["start_date"] >= _evt].copy()
                if "total_budget" not in _post.columns and "budget" in _post.columns:
                    _post["total_budget"] = _post["budget"]
                _sc_p = figures_dir / "scorecard_data.csv"
                _nl_ov, _nl_un, _n_aff, _n_tot = 0.0, 0.0, 0, 0
                if _sc_p.exists():
                    _sc = pd.read_csv(_sc_p)
                    _nl = _sc[_sc["indicator"].str.contains("nightlight", case=False, na=False)]
                    if not _nl.empty:
                        _nl_ov = float(_nl.iloc[0].get("overall_recovery_pct", 0))
                        _nl_un = float(_nl.iloc[0].get("undp_recovery_pct", 0))
                        _n_aff = int(_nl.iloc[0].get("n_affected_units", 0))
                        _n_tot = int(_nl.iloc[0].get("n_total_units", 0))
                if _nl_ov > 0 and _nl_un > _nl_ov: _nl_un = _nl_ov
                if _nl_ov <= 0: _nl_un = 0
                _has_nl = _nl_ov > 0 or _nl_un > 0
                _EL = 0.3
                _evt_year = _evt.year
                _wb = load_wb_data(code, _evt_year)
                _gdp_pc = _wb["gdp_pc"]
                _nat_gdp = _wb["gdp"]
                _pop = _wb["pop"]
                _wb_year = _wb["year"]
                _wb_src = _wb["source"]
                _cov = _n_aff / max(_n_tot, 1)
                _aff_pop = _pop * _cov
                _aff_gdp = _nat_gdp * _cov
                _sel_b, _tot_b = 0.0, 0.0
                if "total_budget" in _post.columns:
                    _tot_b = _post["total_budget"].dropna().sum()
                    _sm = _post[_post["start_date"].dt.date.astype(str) == str(_intv.date())]
                    if not _sm.empty and pd.notna(_sm.iloc[0]["total_budget"]):
                        _sel_b = float(_sm.iloc[0]["total_budget"])
                    elif not _post.dropna(subset=["total_budget"]).empty:
                        _sel_b = float(_post.dropna(subset=["total_budget"]).iloc[0]["total_budget"])
                if _nl_un > 0:
                    _rec = _nl_un; _rlbl = "since selected project"
                elif _nl_ov > 0:
                    _rec = _nl_ov; _rlbl = "overall"
                else:
                    _rec = 0; _rlbl = "N/A"
                _gdp_eff = _rec * _EL
                _react = _aff_gdp * (_gdp_eff / 100)
                _bshare = _sel_b / _tot_b if _tot_b > 0 else 1.0
                _prog_react = _react * _bshare
                _roi = _prog_react / _sel_b if _sel_b > 0 else 0

                with st.expander("Indicative Economic Reactivation", expanded=True):
                    if not _has_nl:
                        st.info('Economic reactivation estimates require nightlight recovery data.')
                    elif _sel_b <= 0:
                        st.info('Economic reactivation estimates require a UNDP project with budget data. Select a project above.')
                    if _has_nl and _sel_b > 0:
                        def _fmt_big(v):
                            if v >= 1e9: return "${:.1f}B".format(v/1e9)
                            if v >= 1e6: return "${:.1f}M".format(v/1e6)
                            if v >= 1e3: return "${:.0f}K".format(v/1e3)
                            return "${:,.0f}".format(v)
                        _cards = [
                            ("Pre-shock GDP", _fmt_big(_nat_gdp), str(_wb_year)),
                            ("Project Budget", _fmt_big(_sel_b), ""),
                            ("NL Recovery", "{:.0f}%".format(_rec), _rlbl),
                            ("Est. Reactivation", _fmt_big(_prog_react), ""),
                            ("Indicative ROI", "{:,.1f}:1".format(_roi), ""),
                        ]
                        _cards_html = '<div style="display:flex;justify-content:center;gap:1.5rem;margin:.8rem 0;">'
                        for _lbl, _val, _sub in _cards:
                            _cards_html += (
                                '<div style="text-align:center;min-width:120px;">'
                                + '<div style="font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;">'
                                + _lbl + '</div>'
                                + '<div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">'
                                + _val + '</div>'
                                + ('<div style="font-size:.65rem;color:#8b949e;">' + _sub + '</div>' if _sub else '')
                                + '</div>'
                            )
                        _cards_html += '</div>'
                        st.markdown(_cards_html, unsafe_allow_html=True)

                        # Explanation as clean HTML to avoid markdown/LaTeX issues
                        _expl = (
                            '<div style="font-size:.85rem;color:#c9d1d9;line-height:1.7;margin:.8rem 0;">'
                            '<p style="margin:0 0 .6rem 0;">'
                            '<b>Step 1 - Nightlight recovery:</b> '
                            + '{} nightlight recovery of <b>{:.0f}%</b>'.format(_rlbl.capitalize(), _rec)
                            + ' was observed across <b>{:,}</b> affected administrative units.'.format(_n_aff)
                            + '</p>'
                            '<p style="margin:0 0 .6rem 0;">'
                            '<b>Step 2 - Pre-shock GDP:</b> '
                            + 'The national GDP before the shock was '
                            + '<b>${:,.0f}</b> ({}, {}).'.format(_nat_gdp, _wb_src, _wb_year)
                            + ' The affected area ({:,.0f} of {:,.0f} admin units, '.format(_n_aff, _n_tot)
                            + '<b>{:.0%}</b> of the country) '.format(_cov)
                            + 'has an estimated GDP of <b>${:,.0f}</b>.'.format(_aff_gdp)
                            + '</p>'
                            '<p style="margin:0 0 .6rem 0;">'
                            '<b>Step 3 - GDP reactivation:</b> '
                            + 'Applying the Henderson et al. (2012) elasticity of 0.3, '
                            + '{:.0f}% nightlight recovery implies '.format(_rec)
                            + '<b>{:.1f}% GDP reactivation</b>, '.format(_gdp_eff)
                            + 'equivalent to approximately <b>${:,.0f}</b> '.format(_react)
                            + 'in economic activity across all actors on the ground.'
                            + '</p>'
                            '<p style="margin:0 0 .6rem 0;">'
                            '<b>Step 4 - Programme attribution:</b> '
                            + 'This project represents '
                            + '<b>{:.1%}</b> of total post-event UNDP spending '.format(_bshare)
                            + '(${:,.0f} of ${:,.0f}). '.format(_sel_b, _tot_b)
                            + 'Attributing a proportional share yields '
                            + '<b>${:,.0f}</b> '.format(_prog_react)
                            + 'with a benefit-cost ratio of <b>{:,.1f}:1</b>.'.format(_roi)
                            + '</p>'
                            '</div>'
                        )
                        st.markdown(_expl, unsafe_allow_html=True)

                        st.caption(
                            "Correlational estimate for rapid screening, not causal impact. "
                            "Recovery reflects all actors on the ground, not just UNDP. "
                            "Project share assumes proportional contribution (a simplification). "
                            "GDP data: " + _wb_src + " (" + str(_wb_year) + "). "
                            "Affected area GDP estimated as proportional share of national GDP "
                            "based on fraction of admin units affected. "
                            "Henderson et al. (2012) elasticity of 0.3 is a global average that may vary locally."
                        )
                    elif _has_nl:
                        st.metric("NL Recovery", "{:.0f}%".format(_rec))
                        st.markdown("Recovery detected but no project budget available for ROI.")
                    else:
                        st.info("Nightlights show no measurable recovery yet. Estimates appear when recovery is detected.")
                    if "total_budget" in _post.columns and len(_post.dropna(subset=["total_budget"])) > 0:
                        _rows = []
                        for _, _pr in _post.dropna(subset=["total_budget"]).iterrows():
                            _b = float(_pr["total_budget"])
                            if pd.isna(_b) or _b <= 0: continue
                            _t = str(_pr.get("title", ""))[:55]
                            _d = _pr["start_date"].strftime("%Y-%m-%d") if pd.notna(_pr["start_date"]) else "-"
                            _bs = _b / _tot_b if _tot_b > 0 else 0
                            if _has_nl:
                                _pr_r = _react * _bs
                                _pr_roi = _pr_r / _b if _b > 0 else 0
                                _rows.append({"Programme": _t, "Start": _d, "Budget": _b, "Share": "{:.0%}".format(_bs),
                                              "Est. Reactivation": _pr_r, "ROI": _pr_roi})
                            else:
                                _rows.append({"Programme": _t, "Start": _d, "Budget": _b, "Share": "{:.0%}".format(_bs),
                                              "Est. Reactivation": 0, "ROI": 0})
                        if _rows:
                            _tdf = pd.DataFrame(_rows)
                            _tdf["Budget"] = _tdf["Budget"].apply(lambda x: "${:,.0f}".format(x))
                            _tdf["Est. Reactivation"] = _tdf["Est. Reactivation"].apply(lambda x: "${:,.0f}".format(x) if x > 0 else chr(8212))
                            _tdf["ROI"] = _tdf["ROI"].apply(lambda x: "{:,.1f}:1".format(x) if x > 0 else chr(8212))
                            st.dataframe(_tdf, hide_index=True, width="stretch")

            except Exception:
                pass

            # DATA SOURCES & AVAILABILITY
    if has_scorecard:
        import base64 as _b64
        _logos_dir = Path("logos")
        _row1 = [
            ("undp.png", "UNDP Projects", str(resolve_dir("data", code) / "external" / "undp_projects.csv")),
            ("nasa.png", "Nightlights (VIIRS)", str(resolve_dir("data", code) / f"gee_panel_{code}_centroid.csv")),
            ("esa.png", "Radar (Sentinel-1)", str(resolve_dir("data", code) / f"gee_panel_{code}_centroid.csv")),
            ("acled.png", "Conflict (ACLED)", str(resolve_dir("data", code) / "external" / "acled_conflict.csv")),
        ]
        _row2 = [
            ("google.png", "Google Trends", str(resolve_dir("data", code) / "external" / "google_trends.csv")),
            ("chc.png", "Rainfall (CHIRPS)", str(resolve_dir("data", code) / f"gee_panel_{code}_centroid.csv")),
            ("wfp.png", "Food Prices (WFP)", str(resolve_dir("data", code) / "external" / "wfp_prices.csv")),
        ]
        def _make_chip(logo, label, fpath):
            pp = Path(fpath); ok = pp.exists()
            try: nn = f" ({len(pd.read_csv(pp)):,})" if ok and fpath.endswith('.csv') else ""
            except Exception: nn = ""
            st_txt = f"\u2713{nn}" if ok else "\u25CB"
            clr = "#3fb950" if ok else "#d29922"
            lp = _logos_dir / logo
            if lp.exists():
                b = _b64.b64encode(lp.read_bytes()).decode()
                img = f'<img src="data:image/png;base64,{b}" style="height:22px;width:22px;object-fit:contain;vertical-align:middle;"/>'
            else:
                img = ""
            return f'<div style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;background:#0d1117;border-radius:4px;border:1px solid #21262d;margin:3px 4px;">{img}<span style="font-size:.73rem;color:#8b949e;">{label}</span><span style="font-size:.73rem;font-weight:600;color:{clr};">{st_txt}</span></div>'
        r1 = "".join(_make_chip(*s) for s in _row1)
        r2 = "".join(_make_chip(*s) for s in _row2)
        ds_html = f'''<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 14px;">
            <div style="font-size:.78rem;font-weight:600;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em;">Data</div>
            <div style="display:flex;flex-wrap:wrap;justify-content:center;">{r1}</div>
            <div style="display:flex;flex-wrap:wrap;justify-content:center;margin-top:2px;">{r2}</div>
        </div>'''
        components.html(ds_html, height=130, scrolling=False)

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
                                c1, c2 = st.columns([45, 55])
                                with c1: safe_image(str(ts[0]), width='stretch')
                                with c2: safe_image(str(mp[0]), width='stretch')
                            elif ts: safe_image(str(ts[0]), width='stretch')
                            elif mp: safe_image(str(mp[0]), width='stretch')
                            st.markdown("---")
                    if not found: st.info("No data yet.")
                ti += 1

            if trend_files:
                with tabs[ti]:
                    st.caption("Google search interest as proxy for public concern")
                    for f in trend_files:
                        st.markdown(f"**{f.stem.replace('trends_','').replace('_',' ').title()}**")
                        safe_image(str(f), width='stretch')
                ti += 1

            if undp_files:
                with tabs[ti]:
                    for f in undp_files: safe_image(str(f), width='stretch')
                ti += 1

            if has_impact:
                with tabs[ti]:
                    st.caption("Econometric impact estimates using Difference-in-Differences and buffer zone analysis.")
                    if impact_files['maps']:
                        st.markdown("**Satellite Maps — UNDP Intervention Areas**")
                        for f in impact_files['maps']:
                            label = f.stem.replace('_clean','').replace('_three_panel','').replace('_',' ').title()
                            st.markdown(f"*{label}*")
                            safe_image(str(f), width='stretch')
                        st.markdown("---")
                    if impact_files['coef_plots']:
                        st.markdown("**Buffer Zone Analysis — Treatment Effect by Distance**")
                        n_cols = min(len(impact_files['coef_plots']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['coef_plots']):
                            label = f.stem.replace('coef_plot_','').replace('_buffer','').replace('_',' ').title()
                            with cols[i % n_cols]: safe_image(str(f), caption=label, width='stretch')
                        st.markdown("---")
                    if impact_files['plots']:
                        st.markdown("**Difference-in-Differences Estimates**")
                        for f in impact_files['plots']: safe_image(str(f), width='stretch')
                        st.markdown("---")
                    if impact_files['pretrends']:
                        st.markdown("**Pre-Trends Validation**")
                        n_cols = min(len(impact_files['pretrends']), 3)
                        cols = st.columns(n_cols)
                        for i, f in enumerate(impact_files['pretrends']):
                            label = f.stem.replace('pretrends_','').replace('_baseline','').replace('_',' ').title()
                            with cols[i % n_cols]: safe_image(str(f), caption=label, width='stretch')
                        st.markdown("---")
                    if impact_files['tables']:
                        st.markdown("**Regression Results**")
                        for f in impact_files['tables']:
                            df_table = pd.read_csv(f)
                            if 'comprehensive_unified' in f.name and 'control_group' in df_table.columns:
                                df_table = df_table[df_table['control_group'].str.contains('500m.*1.5km|500m-1.5km|0.5.*1.5', case=False, na=False)]
                            st.markdown(f"*{f.stem.replace('_',' ').title()}*")
                            st.dataframe(df_table, hide_index=True, width='stretch')
                ti += 1

            if raster_pairs:
                with tabs[ti]:
                    st.caption("Pre-Event, Post-Event, and Post-Intervention satellite imagery")
                    for ind, f in raster_pairs:
                        st.markdown(f"**{RASTER_NAMES.get(ind, ind.upper())}**")
                        safe_image(str(f), width='stretch')

    # AI CO-PILOT IN SIDEBAR
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### \U0001f916 AI Co-Pilot")
        render_copilot_sidebar(config, figures_dir, code)

    # DOWNLOADS
    if panel_path.exists():
        c1, c2 = st.columns([45, 55])
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
        existing = resolve_config_path(cc).exists()

        if existing:
            with open(resolve_config_path(cc)) as _f:
                ex = yaml.safe_load(_f)
            d_etype = ex.get('events',{}).get('disaster_type', ref['etype'])
            d_edate = ex.get('events',{}).get('disaster_date', ref['edate'])
            # intervention_start auto-set to event date
            d_start = ex.get('study_period_start', '2022-01-01')
            d_end = ex.get('study_period_end', '2025-12-31')
            d_admin = ex.get('boundaries',{}).get('admin_level', ref['admin'])
            st.caption(f"Editing `config/{cc}.yaml`")
        else:
            d_etype, d_edate = ref['etype'], ref['edate']
            d_start = str(date(max(int(ref['edate'][:4])-1, 2000), 1, 1))
            d_end = '2025-12-31'
            d_admin = ref['admin']

        etypes = ["earthquake","conflict","flood","cyclone","drought","pandemic"]
        c1, c2 = st.columns(2)
        with c1: etype = st.selectbox("Event type", etypes, index=etypes.index(d_etype), key="cf_et")
        with c2: edate = st.date_input("Event date", date.fromisoformat(d_edate), key="cf_ed")

        c1, c2, c3 = st.columns(3)
        with c1: sstart = st.date_input("Study start", date.fromisoformat(d_start), key="cf_ss")
        with c2: send = st.date_input("Study end", date.fromisoformat(d_end), key="cf_se")
        with c3: admin_level = st.selectbox("Admin level", [1,2,3], index=[1,2,3].index(d_admin), key="cf_al")

        if st.button("💾 Save Configuration", type="primary", width='stretch', key="save_cfg"):
            nc = {
                'country_name': pick,
                'country_code': ref['iso'],
                'study_period_start': sstart.isoformat(),
                'study_period_end': send.isoformat(),
                'events': {
                    'disaster_type': etype,
                    'disaster_date': edate.isoformat(),
                    'intervention_start': edate.isoformat(),
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
            _save_path = resolve_config_path(cc, for_write=True)
            with open(_save_path, 'w') as f_out:
                yaml.dump(nc, f_out, default_flow_style=False, sort_keys=False)
            st.session_state.auto_select_country = cc
            st.success(f"✅ Saved configuration for {pick}")
            st.rerun()

    # CREDENTIALS
    st.markdown("---")
    with st.expander("🔑 Credentials", expanded=False):
        st.caption("API keys are saved locally on this computer (in your home folder) and never shared.")
        c1, c2 = st.columns(2)
        with c1: acled_email = st.text_input("ACLED Email", value=os.getenv("ACLED_EMAIL",""), key="acled_em")
        with c2: acled_key = st.text_input("ACLED Key", value=os.getenv("ACLED_KEY",""), type="password", key="acled_k")
        c3, c4 = st.columns(2)
        with c3: gee_project = st.text_input("GEE Project ID", value=os.getenv("GEE_PROJECT",""), key="gee_proj")
        with c4: openai_key = st.text_input("AI Co-pilot Key", value=os.getenv("OPENAI_API_KEY",""), type="password", key="openai_k")
        if st.button("Save Credentials", key="save_creds"):
            env_path = USER_ENV_PATH
            env_path.parent.mkdir(parents=True, exist_ok=True)
            lines = env_path.read_text().splitlines() if env_path.exists() else []
            updates = {}
            if acled_email: updates["ACLED_EMAIL"] = acled_email
            if acled_key: updates["ACLED_KEY"] = acled_key
            if gee_project: updates["GEE_PROJECT"] = gee_project
            if openai_key: updates["OPENAI_API_KEY"] = openai_key
            new_lines = []
            seen = set()
            for ln in lines:
                k = ln.split("=", 1)[0] if "=" in ln else ""
                if k in updates:
                    new_lines.append(k + "=" + updates[k])
                    seen.add(k)
                else:
                    new_lines.append(ln)
            for k, v in updates.items():
                if k not in seen:
                    new_lines.append(k + "=" + v)
            env_path.write_text(chr(10).join(new_lines) + chr(10))
            os.environ.update(updates)
            st.success("Credentials saved. They will be remembered next time you open the app.")

    # RUN PIPELINE
    run_country = setup_country_code or selected
    run_name = pick or config.get('country_name', selected)

    st.markdown("---")
    st.markdown(f'<div class="section-title">Run Pipeline — {run_name}</div>', unsafe_allow_html=True)

    LOCK_PATH = Path("/tmp/undp_pipeline.lock")
    LOCK_STALE_SECONDS = 60 * 60  # a lock older than 1h is considered stale

    def _lock_active():
        import time as _t
        if not LOCK_PATH.exists():
            return False
        try:
            age = _t.time() - LOCK_PATH.stat().st_mtime
            if age > LOCK_STALE_SECONDS:
                LOCK_PATH.unlink(missing_ok=True)  # clear stale lock
                return False
            return True
        except Exception:
            return False

    def run_cmd(cmd, desc):
        import time as _t
        # --- single-job queue: only one pipeline runs at a time across all users ---
        if _lock_active():
            try:
                who = LOCK_PATH.read_text().strip()
            except Exception:
                who = "another user"
            st.warning("⏳ A pipeline is currently running (" + who + "). "
                       "Only one analysis can run at a time. Please wait a few minutes "
                       "and try again — your data and others' are kept separate and safe.")
            return
        env = os.environ.copy()
        if acled_email: env['ACLED_EMAIL'] = acled_email
        if acled_key:
            env['ACLED_KEY'] = acled_key
            env['ACLED_PASSWORD'] = acled_key
        _gee = os.getenv('GEE_PROJECT', '')
        if _gee: env['GEE_PROJECT'] = _gee
        try:
            LOCK_PATH.write_text(f"{desc} since {_t.strftime('%H:%M')}")
        except Exception:
            pass
        with st.status(desc + "...", expanded=True) as s:
            try:
                # cmd may be a list (cross-platform, preferred) or a string (legacy)
                _shell = isinstance(cmd, str)
                proc = subprocess.Popen(
                    cmd, shell=_shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env, bufsize=1, cwd=str(user_workspace_root()))
                output_box = st.empty()
                log = []
                for line in iter(proc.stdout.readline, ""):
                    log.append(line.rstrip())
                    if len(log) > 200: log = log[-200:]
                    output_box.code(chr(10).join(log[-40:]), language="bash")
                proc.wait()
                output_box.code(chr(10).join(log[-60:]) or "(no output)", language="bash")
                if proc.returncode == 0:
                    s.update(label="Done: " + desc, state="complete")
                else:
                    # surface the real reason, not just a generic label
                    tail = chr(10).join(log[-15:])
                    st.error("Step failed: " + desc + ". Last output:" + chr(10) + tail)
                    s.update(label="Error: " + desc, state="error")
            except Exception as e:
                st.error(str(e)[:300])
                s.update(label="Error: " + desc, state="error")
            finally:
                LOCK_PATH.unlink(missing_ok=True)   # release the queue lock

    _PY = sys.executable          # the venv's python — works on Windows and Mac
    # run_code is the actual country code (e.g. "ven"); prefer the code tied to
    # run_country specifically, since it can differ from the page's fallback `code`.
    run_code = setup_country_code or code
    _cfg_obj = resolve_config_path(run_country, for_write=True)   # user's own writable config
    resolve_dir("data", run_code, for_write=True)                 # cloned from shared on first touch
    resolve_dir("outputs", run_code, for_write=True)
    _cfg = str(_cfg_obj)
    def _stage(stage):            # call the cross-platform runner (no `make` needed)
        return [_PY, "run_pipeline.py", "--config", _cfg, "--stage", stage]
    steps = [
        ("🗺️ Boundaries", _stage("boundaries")),
        ("🛰️ Satellite",  _stage("gee")),
        ("⚔️ Conflict",   _stage("acled")),
        ("🏗️ UNDP",       _stage("undp")),
        ("📈 Trends",     _stage("gt")),
        ("🔧 Panel",      _stage("preprocess")),
        ("📊 Visuals",    _stage("visuals")),
        ("🎯 Scorecard",  _stage("scorecard")),
    ]
    cols = st.columns(4)
    for i, (lb, cmd) in enumerate(steps):
        with cols[i % 4]:
            if st.button(lb, width='stretch', key=f"run_{lb}"):
                run_cmd(cmd, lb)

    st.markdown("")
    if st.button("▶️ Run Full Pipeline", width='stretch', type="primary", key="run_full"):
        run_cmd(_stage("all"), "Full Pipeline")
        st.balloons()

    with st.expander("📝 Current Configuration"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")



# ════════════════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif mode.startswith("\u2139"):
    st.markdown('<div class="section-title">About This Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
### UNDP Early Recovery Impact Dashboard

A satellite-based monitoring and rapid screening tool developed by **AImpact Lab** for UNDP Crisis Bureau / C3RT. 
This dashboard tracks post-crisis recovery across multiple indicators to support evidence-based prioritisation 
and resource allocation for early recovery interventions.

**GitHub Repository:** [github.com/hgrueso/undp-impact-evaluation](https://github.com/hgrueso/undp-impact-evaluation)

---

### Data Sources

| Source | Indicator | Coverage | Use |
|--------|-----------|----------|-----|
| **NASA VIIRS** | Nighttime Lights | Monthly, 500m, 2012-present | Economic activity proxy |
| **ESA Sentinel-1** | Radar Backscatter (VV) | 12-day, 10m, 2014-present | Surface change, debris |
| **NASA MODIS** | Vegetation Index (NDVI) | 16-day, 250m, 2000-present | Agricultural health |
| **ACLED** | Conflict Events & Fatalities | Daily, geocoded | Security context |
| **World Bank WDI** | GDP, GDP/capita, Population | Annual | Economic baseline |
| **UNDP API** | Project budgets & timelines | Per project | Intervention tracking |
| **Google Trends** | Search interest indices | Weekly/monthly | Population sentiment |
| **WFP** | Food Prices | Monthly, market-level | Food security monitoring |
| **CHIRPS** | Rainfall | Monthly, 5km | Environmental control |

---

### Pipeline Architecture

The system follows a modular pipeline orchestrated via a Makefile:

1. **Configuration** — Each country is defined by a YAML file specifying event dates, boundaries, and parameters
2. **Data Download** — Satellite data via Google Earth Engine, conflict via ACLED API, economic via World Bank API
3. **Panel Construction** — All sources merged into an admin-unit x month panel dataset
4. **Recovery Scorecard** — Unit-level recovery calculation with configurable thresholds
5. **Visualization** — Time series, choropleth maps, and spatial comparisons
6. **Economic Estimation** — Nightlight-GDP conversion using Henderson et al. (2012) elasticity

Adding a new country takes approximately 30-60 minutes end-to-end.

---

### Recovery Scorecard Methodology

For each indicator and administrative unit:

- **Baseline:** Mean value across all available pre-event months. For vegetation (NDVI), a year-over-year seasonal adjustment is applied.
- **Trough:** The single worst post-event month (captures rapid-onset shocks).
- **Affected units:** Only units with >10% decline from baseline are scored. Others are classified as "Stable."
- **Targeted Recovery:** (Current - Trough) / (Baseline - Trough) x 100, capped at 100%.
- **Since Selected Project:** Recovery measured from the selected UNDP project start date.
- **Late-month events:** If the disaster occurs after the 21st of the month, the event month is treated as pre-event (monthly composites dilute within-month shocks).

The "Since Project" metric can never exceed "Targeted" recovery (logical consistency constraint).

---

### Economic Reactivation Model

Converts nightlight recovery to estimated USD values in 4 steps:

1. **Nightlight recovery** — Observed from the scorecard
2. **Pre-shock GDP** — National GDP from World Bank WDI for the year before the event
3. **Affected area GDP** — National GDP x share of affected admin units
4. **GDP reactivation** — Affected GDP x NL recovery x Henderson elasticity (0.3)

The Henderson, Storeygard & Weil (2012) elasticity of 0.3 means 10% nightlight recovery corresponds to approximately 3% GDP reactivation. This is a global average; local values may differ.

Project-specific ROI is calculated by attributing a proportional share of total reactivation based on the project's budget share of total post-event UNDP spending.

**Key limitations:**
- Correlational estimate, not causal impact
- UNDP is one of many actors on the ground
- Budget figures are allocations, not disbursements
- GDP proportionally distributed across admin units (simplification)

---

### AI Co-pilot

The integrated AI assistant (powered by OpenAI GPT-4o-mini) helps policy makers interpret recovery data. 
It has full context of the selected country's scorecard, project data, and recovery metrics. 
The co-pilot is pre-prompted to always emphasise correlation over causation and to recommend 
econometric validation where appropriate.

---

### Causal Impact Analysis

For specific cases where geocoded UNDP project locations are available (currently Myanmar and Gaza), 
the pipeline includes causal impact estimates using formal econometric methods:

- **Difference-in-Differences (DiD)** — Compares UNDP-treated vs control areas before and after intervention
- **Two-Way Fixed Effects** — Controls for time-invariant unit characteristics and common time shocks
- **Event Study Design** — Tests parallel pre-trends and visualises dynamic treatment effects
- **Synthetic Control** — Constructs counterfactual for areas without suitable within-country controls
- **Placebo Tests** — Validates results by assigning artificial treatment dates
- **Dose-Response Analysis** — Examines whether effects vary with proximity to intervention sites

These methods provide the econometric rigour needed for causal attribution, complementing the rapid 
monitoring provided by the scorecard. Results for Myanmar and Gaza are available in the Impact Measurement tab.

---

### Deployment

The dashboard can be deployed on any server with Python 3.10+:
```
git clone https://github.com/hgrueso/undp-impact-evaluation.git
cd undp-impact-evaluation
pip install -r requirements.txt
# Set credentials in .env
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For UNDP Azure deployment, the app can be containerised with Docker or deployed directly on an Azure App Service.

---

### Contact

**AImpact Lab** — Tech-enabled impact evaluation consultancy specialising in econometrics, 
machine learning, causal inference, and satellite remote sensing.

- Website: [www.aimpactlab.com](https://www.aimpactlab.com)
- Email: [info@aimpactlab.com](mailto:info@aimpactlab.com)

**Hernando Grueso, PhD** — Lead Data Scientist & Founder, AImpact Lab 
([hernando@aimpactlab.com](mailto:hernando@aimpactlab.com))

For questions about this dashboard or the underlying methodology, contact AImpact Lab or the C3RT team at UNDP Crisis Bureau.
""")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">UNDP Early Recovery Impact · AImpact Lab · VIIRS · Sentinel-1 · MODIS · CHIRPS · ACLED</div>', unsafe_allow_html=True)
