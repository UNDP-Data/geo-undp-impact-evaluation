#!/usr/bin/env python3
"""Round 34. Run from the repo root.
1. Landing page: centred 'Continue as an existing user', with a collapsed
   'Don't have a user yet? Create one here' below (open when no users exist).
2. Detailed Indicators: each chart/map pair is drawn in two frames of
   identical height, so the pair always lines up."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
def rep(old, new):
    global s
    c = s.count(old)
    if c != 1: sys.exit(f"ABORT: anchor found {c}x (expected 1): {old[:80]!r}")
    s = s.replace(old, new)

# ── 1. landing page ──
rep('''    _known_users = _load_users()
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
        if st.button("Create & continue", key="create_user", width='stretch', type="primary"):''',
'''    _known_users = _load_users()
    _mid = st.columns([1, 1.3, 1])[1]
    with _mid:
        if _known_users:
            st.markdown("**Continue as an existing user**")
            _pick_user = st.selectbox("Existing users", _known_users,
                                       key="pick_existing_user", label_visibility="collapsed")
            if st.button("Continue", key="continue_existing", width='stretch', type="primary"):
                st.session_state.username = _pick_user
                st.rerun()
            st.markdown("")
        with st.expander("Don't have a user yet? Create one here", expanded=not _known_users):
            _new_user = st.text_input("New username", key="new_username_input",
                                       label_visibility="collapsed", placeholder="Your name, e.g. carlo")
            if st.button("Create & continue", key="create_user", width='stretch',
                         type="secondary" if _known_users else "primary"):''')

rep('''            if not _clean:
                st.error("Please enter a valid name (letters, numbers, - or _).")
            else:
                _register_user(_clean)
                st.session_state.username = _clean
                st.rerun()
    st.stop()''',
'''                if not _clean:
                    st.error("Please enter a valid name (letters, numbers, - or _).")
                else:
                    _register_user(_clean)
                    st.session_state.username = _clean
                    st.rerun()
    st.stop()''')
rep('''            _clean = "".join(ch for ch in _new_user.strip().lower().replace(" ", "_")
                              if ch.isalnum() or ch in "_-")''',
'''                _clean = "".join(ch for ch in _new_user.strip().lower().replace(" ", "_")
                                  if ch.isalnum() or ch in "_-")''')

# ── 2. equal-height chart/map pairs ──
rep('''# ─── MULTI-USER WORKSPACES ─────────────────────────────────────────────────''',
'''def two_up(left, right, height=330):
    """A chart and its map side by side in two frames of identical height.
    Figures are produced at slightly different proportions, so equal columns
    alone give unequal heights; object-fit keeps each image's aspect ratio."""
    import base64 as _b64
    def _tag(path):
        try:
            from PIL import Image
            with Image.open(path) as _im:
                _im.verify()
            b = _b64.b64encode(Path(path).read_bytes()).decode()
            return ('<img src="data:image/png;base64,' + b + '" style="width:100%;height:'
                    + str(height) + 'px;object-fit:contain;display:block;"/>')
        except Exception:
            return ('<div style="height:' + str(height) + 'px;display:flex;align-items:center;'
                    'justify-content:center;color:#8b949e;font-size:.8rem;">(figure not available)</div>')
    frame = ('flex:1;min-width:0;background:#161b22;border:1px solid #30363d;'
             'border-radius:6px;padding:6px;')
    components.html(
        '<body style="margin:0;background:transparent;">'
        '<div style="display:flex;gap:12px;align-items:stretch;">'
        '<div style="' + frame + '">' + _tag(left) + '</div>'
        '<div style="' + frame + '">' + _tag(right) + '</div>'
        '</div></body>', height=height + 20, scrolling=False)


# ─── MULTI-USER WORKSPACES ─────────────────────────────────────────────────''')

rep('''                            if ts and mp:
                                c1, c2 = st.columns(2)
                                with c1: safe_image(str(ts[0]), width='stretch')
                                with c2: safe_image(str(mp[0]), width='stretch')''',
'''                            if ts and mp:
                                two_up(str(ts[0]), str(mp[0]))''')

compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
