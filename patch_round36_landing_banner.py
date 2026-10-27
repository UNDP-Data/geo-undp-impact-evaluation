#!/usr/bin/env python3
"""Round 36: on the landing page, the blue banner sits in the same centred
column as the user box, so both have the same width. Run from the repo root."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
old = '''if "username" not in st.session_state:
    st.markdown("""
    <div class="hero">
        <h1>UNDP Early Recovery Impact</h1>
        <p>Enter your name to continue. This keeps your changes separate from everyone else's.</p>
    </div>
    """, unsafe_allow_html=True)
    _known_users = _load_users()
    _mid = st.columns([1, 1.3, 1])[1]
    with _mid:
        if _known_users:'''
new = '''if "username" not in st.session_state:
    _known_users = _load_users()
    _mid = st.columns([1, 1.3, 1])[1]
    with _mid:
        st.markdown("""
    <div class="hero" style="padding:1.4rem 1.6rem;">
        <h1 style="font-size:1.5rem;">UNDP Early Recovery Impact</h1>
        <p>Enter your name to continue. This keeps your changes separate from everyone else's.</p>
    </div>
    """, unsafe_allow_html=True)
        if _known_users:'''
if s.count(old) != 1:
    sys.exit(f"ABORT: landing block found {s.count(old)}x (expected 1)")
s = s.replace(old, new)
compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
