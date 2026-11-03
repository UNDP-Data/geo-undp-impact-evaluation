"""
Country brief (PDF) for the Early Recovery Impact dashboard.

One short document built from what already exists: header, recovery
scorecard, a few key charts, the selected programme's prediction if it was
shown, then data sources with latest dates and essential limitations.
Nothing is recomputed; only matplotlib and pandas are used.

    from pipeline.report_pdf import build_report
    pdf_bytes = build_report(code, config, outputs_dir, data_dir, roi=None, username="")
"""
import io
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

PAGE, M = (8.27, 11.69), 0.07          # A4 portrait, margin (fraction of width)
KEY_CHARTS = [                          # shown if the file exists, in this order
    ("ts_nightlights.png", "Nighttime lights (VIIRS monthly)"),
    ("map_nightlights.png", "Nighttime lights, map"),
    ("ts_bm_nightlights.png", "Nighttime lights (Black Marble daily)"),
    ("ts_ndvi.png", "Vegetation (NDVI)"),
    ("ts_conflict_events.png", "Conflict events (ACLED)"),
    ("undp_timeline.png", "UNDP programme timeline"),
]
SOURCES = [   # label, file (relative to data dir), note
    ("Satellite indicators (VIIRS, MODIS, Sentinel-1, CHIRPS)", "gee_panel_{code}_centroid.csv", "monthly VIIRS lags 3 to 4 months"),
    ("Black Marble daily nightlights (NASA VNP46A2)", "external/blackmarble_ntl.csv", "about one week lag"),
    ("Conflict events (ACLED)", "external/acled_conflict.csv", ""),
    ("Food prices (WFP)", "external/wfp_prices.csv", ""),
    ("Google Trends", "external/google_trends.csv", ""),
    ("UNDP project register", "external/undp_projects.csv", "budgets are allocations"),
]
LIMITATIONS = [
    "All figures are correlational: they show what was observed before and after the event and programme dates, not what the programmes caused.",
    "Recovery reflects every actor on the ground, not UNDP alone.",
    "The predicted reactivation is an upper bound that credits UNDP projects with all observed recovery in proportion to their budgets; the actual return is likely to be much lower.",
    "Seasonal indicators (vegetation, rainfall, radar) are compared with the same calendar month before the event; monthly nightlights are published with a 3 to 4 month lag.",
]


def _get(config, key, default=""):
    v = config.get(key)
    if v in (None, "") and isinstance(config.get("events"), dict):
        v = config["events"].get(key)
    return default if v in (None, "") else v


def _latest(path):
    """(rows, latest date) of a CSV; ('', '') if unreadable."""
    try:
        df = pd.read_csv(path)
        col = next((c for c in df.columns if c.lower() in ("period_start", "date", "event_date", "week")), None) \
            or next((c for c in df.columns if "date" in c.lower()), None)
        last = pd.to_datetime(df[col], errors="coerce").max() if col else pd.NaT
        return f"{len(df):,}", ("" if pd.isna(last) else str(last.date()))
    except Exception:
        return "", ""


class _Doc:
    def __init__(self, pdf, header):
        import matplotlib.pyplot as plt
        self.plt, self.pdf, self.header, self.n, self.fig = plt, pdf, header, 0, None
        self.y = 0.0

    def page(self):
        self.close()
        self.n += 1
        self.fig = self.plt.figure(figsize=PAGE)
        self.fig.text(M, 0.965, self.header, fontsize=8, color="#555555")
        self.fig.text(1 - M, 0.965, f"Page {self.n}", fontsize=8, color="#555555", ha="right")
        self.y = 0.93

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig); self.plt.close(self.fig); self.fig = None

    def need(self, h):
        if self.fig is None or self.y - h < 0.06:
            self.page()

    def text(self, s, size=9.5, bold=False, color="#222222", gap=0.01, width=110):
        for para in str(s).split("\n"):
            for ln in (textwrap.wrap(para, width=width) or [""]):
                self.need(0.02)
                self.fig.text(M, self.y, ln, fontsize=size, va="top", color=color,
                              fontweight="bold" if bold else "normal")
                self.y -= 0.017 * size / 9.5
        self.y -= gap

    def heading(self, s):
        self.need(0.12)
        self.y -= 0.008
        self.text(s, size=12, bold=True, gap=0.004)

    def table(self, df, widths=None, size=8):
        if df is None or len(df) == 0:
            return
        rows = min(len(df), 25)
        h = 0.021 * (rows + 1)
        self.need(h + 0.02)
        ax = self.fig.add_axes([M, self.y - h, 1 - 2 * M, h]); ax.axis("off")
        t = ax.table(cellText=df.head(rows).astype(str).values, colLabels=list(df.columns),
                     loc="upper left", cellLoc="left", colWidths=widths)
        t.auto_set_font_size(False); t.set_fontsize(size); t.scale(1, 1.2)
        for (r, _), cell in t.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if r == 0:
                cell.set_text_props(fontweight="bold"); cell.set_facecolor("#eef2f7")
        self.y -= h + 0.018

    def images(self, items, row_h=0.23):
        import matplotlib.image as mpimg
        w = (1 - 2 * M - 0.03) / 2
        for i in range(0, len(items), 2):
            self.need(row_h + 0.01)
            for j, (pth, cap) in enumerate(items[i:i + 2]):
                x = M + j * (w + 0.03)
                ax = self.fig.add_axes([x, self.y - row_h + 0.02, w, row_h - 0.02]); ax.axis("off")
                try:
                    ax.imshow(mpimg.imread(str(pth)))
                except Exception:
                    ax.text(0.5, 0.5, "(figure not available)", ha="center", va="center", fontsize=8)
                self.fig.text(x, self.y - row_h + 0.012, cap, fontsize=7.5, color="#444444")
            self.y -= row_h + 0.012


def build_report(code, config, outputs_dir, data_dir, roi=None, username=""):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    outputs_dir, data_dir = Path(outputs_dir), Path(data_dir)
    figs = outputs_dir / "figures"
    country = str(_get(config, "country_name", _get(config, "country", code.upper())))
    event = str(_get(config, "event_name", _get(config, "name", "Crisis event")))
    event_date = str(_get(config, "event_date"))[:10]
    start, end = str(config.get("study_period_start", ""))[:10], str(config.get("study_period_end", ""))[:10]
    today = date.today().isoformat()

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        doc = _Doc(pdf, f"UNDP Early Recovery Impact  |  {country}  |  exported {today}")
        doc.page()

        # header
        doc.text("UNDP Early Recovery Impact: country brief", size=18, bold=True, gap=0.0)
        doc.text(f"{country}: {event}", size=13, gap=0.006)
        doc.text(f"Event date {event_date or 'n/a'}   |   Analysis period {start or 'n/a'} to {end or 'n/a'}   |   "
                 f"Exported {today}" + (f" by {username}" if username else ""), size=9, color="#555555", gap=0.016)

        # scorecard
        doc.heading("Recovery scorecard")
        try:
            s = pd.read_csv(figs / "scorecard_data.csv")
            keep = [c for c in ["indicator", "overall_recovery_pct", "undp_recovery_pct", "status",
                                "n_affected_units", "n_total_units", "latest_date"] if c in s.columns]
            s = s[keep].rename(columns={"indicator": "Indicator", "overall_recovery_pct": "Overall recovery %",
                                        "undp_recovery_pct": "Since programme %", "status": "Status",
                                        "n_affected_units": "Affected units", "n_total_units": "Total units",
                                        "latest_date": "Latest data"})
            for c in ("Overall recovery %", "Since programme %"):
                if c in s.columns:
                    s[c] = pd.to_numeric(s[c], errors="coerce").map(
                        lambda v: "awaiting data" if v == -2 else "stable" if v == -1 else ("" if pd.isna(v) else f"{v:.0f}"))
            doc.table(s)
            doc.text("Overall: share of the post-event decline regained by the latest observation, in affected units. "
                     "Since programme: share regained after the programme start. Both are correlational.",
                     size=8, color="#555555")
        except Exception:
            doc.text("Scorecard not available for this country.", size=9, color="#777777")

        # key charts
        items = [(figs / f, cap) for f, cap in KEY_CHARTS if (figs / f).exists()]
        if items:
            doc.heading("Key indicators")
            doc.images(items)

        # prediction, only if the dashboard showed it this session
        if roi:
            doc.heading("Predicted economic reactivation (correlational)")
            doc.text("Upper-bound prediction, not a measured return. It credits UNDP projects with all the recovery "
                     "seen since they started, in proportion to their budgets. Many other actors contribute, so the "
                     "actual return is likely to be much lower.", size=8.5, color="#555555")
            try:
                doc.table(pd.DataFrame([{
                    "Programme": str(roi.get("title", ""))[:55], "Start": roi.get("start", ""),
                    "Budget": "${:,.0f}".format(roi.get("budget", 0)),
                    "Recovery since start": "{:.0f}%".format(roi.get("rec", 0)),
                    "Share of UNDP budget": "{:.0%}".format(roi.get("share", 0)),
                    "Predicted reactivation": "${:,.0f}".format(roi.get("react_project", 0)),
                    "Predicted ROI": "{:,.1f}:1".format(roi.get("roi", 0))}]),
                    widths=[0.30, 0.10, 0.12, 0.12, 0.11, 0.15, 0.10], size=7)
                doc.text(
                    "1. Recovery: nightlights in the {:,} affected areas regained {:.0f}% of their loss since this project started.\n"
                    "2. Economy at stake: these areas account for about ${:,.0f} of GDP ({:.0%} of admin units; national GDP ${:,.0f}, {}).\n"
                    "3. Economic reactivation: with the elasticity of 0.3 (Henderson et al., 2012), {:.0f}% more light corresponds to "
                    "{:.1f}% more GDP, about ${:,.0f} across all actors.\n"
                    "4. This project's share: {:.0%} of UNDP's post-event budget (${:,.0f} of ${:,.0f}), so it is credited with "
                    "${:,.0f}, a {:,.1f}:1 return on its budget.".format(
                        roi.get("n_aff", 0), roi.get("rec", 0), roi.get("aff_gdp", 0), roi.get("cov", 0),
                        roi.get("nat_gdp", 0), roi.get("gdp_year", ""), roi.get("rec", 0), roi.get("gdp_eff", 0),
                        roi.get("react_all", 0), roi.get("share", 0), roi.get("budget", 0), roi.get("tot_budget", 0),
                        roi.get("react_project", 0), roi.get("roi", 0)), size=8.5)
            except Exception:
                doc.text("Prediction details not available.", size=8, color="#777777")

        # sources and limitations
        doc.heading("Data sources and latest available dates")
        rows = []
        for label, rel, note in SOURCES:
            p = data_dir / rel.format(code=code)
            n, last = _latest(p) if p.exists() else ("not downloaded", "")
            rows.append({"Source": label, "Rows": n, "Latest date": last or "n/a", "Note": note})
        doc.table(pd.DataFrame(rows), widths=[0.46, 0.14, 0.14, 0.26], size=7.5)
        doc.heading("Limitations")
        for lim in LIMITATIONS:
            doc.text("\u2022 " + lim, size=8.5, gap=0.002)
        doc.close()
    return buf.getvalue()
