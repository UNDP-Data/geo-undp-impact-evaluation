#!/usr/bin/env python3
"""Round 35 (app). Run from the repo root.
Chart/map pairs: pad both images to a common proportion (each with its own
background colour) and show them with st.image in equal columns, so they
line up AND keep the enlarge button and full resolution."""
import sys
p = "app.py"; s = open(p, encoding="utf-8").read(); orig = s
a = s.find("def two_up(left, right, height=330):")
b = s.find("# ─── MULTI-USER WORKSPACES", a)
if not (0 < a < b): sys.exit("ABORT: two_up block not found")
new = '''@st.cache_data(show_spinner=False)
def _padded_pair(left, right, left_mtime, right_mtime):
    """PNG bytes of two images padded (with each image's own background colour)
    to a common width:height ratio, so equal columns give equal heights.
    The mtime arguments only invalidate the cache when a figure is regenerated."""
    import io
    from PIL import Image
    ims = [Image.open(pth).convert("RGB") for pth in (left, right)]
    target = max(im.width / im.height for im in ims)
    out = []
    for im in ims:
        w, h = im.size
        if w / h < target - 1e-6:
            new_w = int(round(h * target))
            canvas = Image.new("RGB", (new_w, h), im.getpixel((0, 0)))
            canvas.paste(im, ((new_w - w) // 2, 0))
            im = canvas
        buf = io.BytesIO(); im.save(buf, format="PNG"); out.append(buf.getvalue())
    return out


def two_up(left, right):
    """A chart and its map side by side at the same height, with the usual
    enlarge button and full resolution preserved."""
    try:
        l_b, r_b = _padded_pair(str(left), str(right), os.path.getmtime(left), os.path.getmtime(right))
    except Exception:
        l_b, r_b = None, None
    c1, c2 = st.columns(2)
    with c1:
        if l_b: st.image(l_b, width='stretch')
        else: safe_image(str(left), width='stretch')
    with c2:
        if r_b: st.image(r_b, width='stretch')
        else: safe_image(str(right), width='stretch')


'''
s = s[:a] + new + s[b:]
if "components.html(\n        '<body style=\"margin:0;background:transparent;\">'" in s:
    sys.exit("ABORT: old frame code still present")
compile(s, p, "exec")
open(p, "w", encoding="utf-8").write(s)
print(f"patched app.py ({len(orig):,} -> {len(s):,} bytes); compiles OK")
