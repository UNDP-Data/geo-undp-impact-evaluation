#!/usr/bin/env python3
"""Round 32. Run from the repo root.
1. Country list: one entry per country_code (legacy 'afghanistan.yaml' and
   'afg.yaml' no longer show as two Afghanistans); prefers the file named
   after the code, which is the one Setup & Run writes.
2. Data logos: loaded from the app's own folder, not the working directory,
   so they appear on any server (including the UNDP deployment).
3. A scorecard error no longer blanks the rest of the page; the Data panel
   shows whenever the country has data."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

rep('''def all_visible_config_names():
    """Shared countries plus this user's own additions, merged, no dupes,
    no username tags — exactly what the dropdown should show."""
    shared = {Path(p).stem for p in glob.glob(str(REPO_ROOT / "config" / "*.yaml"))}
    own = {Path(p).stem for p in glob.glob(str(user_workspace_root() / "config" / "*.yaml"))}
    return sorted(shared | own)''',
'''def all_visible_config_names():
    """Shared countries plus this user's own additions, one entry per country.
    Several config files can describe the same country (e.g. a legacy
    'afghanistan.yaml' and 'afg.yaml'); keep one per country_code, preferring
    the file named after the code, which is the one Setup & Run writes."""
    shared = {Path(p).stem for p in glob.glob(str(REPO_ROOT / "config" / "*.yaml"))}
    own = {Path(p).stem for p in glob.glob(str(user_workspace_root() / "config" / "*.yaml"))}
    by_code = {}
    for name in sorted(shared | own):
        try:
            with open(resolve_config_path(name)) as _f:
                _cc = (yaml.safe_load(_f) or {}).get("country_code", name)
        except Exception:
            _cc = name
        _cc = str(_cc).lower()
        if _cc not in by_code or name == _cc:
            by_code[_cc] = name
    return sorted(by_code.values())''')

rep('''        _logos_dir = Path("logos")''', '''        _logos_dir = REPO_ROOT / "logos"''')

rep('''    has_scorecard = render_scorecard()''',
'''    try:
        has_scorecard = render_scorecard()
    except Exception as _sc_err:
        st.warning("The recovery scorecard for this country could not be displayed (" + str(_sc_err)[:120]
                   + "). Re-running the 🎯 Scorecard stage in Setup & Run usually fixes this.")
        has_scorecard = False''')

rep('''    if has_scorecard:
        import base64 as _b64''',
'''    if resolve_dir("data", code).exists():
        import base64 as _b64''')

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
