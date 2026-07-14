"""analysis/paper_style.py — shared style for all paper figures (both pilots)."""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

UN_BLUE = "#009EDB"
TREAT = "#1f4e79"
CONTROL = "#b45309"
ACCENT = "#c0392b"
GREY = "#8a8a8a"

# Classic nightlights look: deep black -> amber -> gold -> white (no brown mud)
NIGHTS = LinearSegmentedColormap.from_list(
    "nights", [(0.00, "#000000"), (0.35, "#0d0900"), (0.55, "#4a3200"),
               (0.72, "#c8871a"), (0.86, "#ffd166"), (1.00, "#ffffff")])

def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 9,
    })

def raster_triptych(raster_paths, labels, out_path, bbox=None, boundary_gdf=None,
                    sites_lonlat=None, epicenter=None, suptitle="",
                    site_label="UNDP sites", vmax_pct=99.0, clip_gdf=None,
                    boundary_color="#9aa0a6", boundary_lw=0.5):
    """Standardized 3-panel nightlights raster figure (same product, both pilots).
    raster_paths: list of 3 GeoTIFF paths. bbox: (minlon, maxlon, minlat, maxlat)."""
    import rasterio
    from rasterio.windows import from_bounds
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    clip_geoms = None
    if clip_gdf is not None:
        from shapely.ops import unary_union
        clip_geoms = [unary_union(clip_gdf.geometry)]
    arrays, extents = [], []
    for p in raster_paths:
        with rasterio.open(p) as src0:
            if clip_geoms is not None:
                from rasterio.mask import mask as rio_mask
                import rasterio.io
                arr, tr = rio_mask(src0, clip_geoms, crop=True, nodata=np.nan, filled=True)
                prof = src0.profile.copy(); prof.update(height=arr.shape[1], width=arr.shape[2],
                                                        transform=tr, dtype="float64", nodata=np.nan)
                mem = rasterio.io.MemoryFile()
                with mem.open(**prof) as mdst:
                    mdst.write(arr.astype("float64"))
                src = mem.open()
            else:
                src = src0
            if bbox is not None:
                win = from_bounds(bbox[0], bbox[2], bbox[1], bbox[3], src.transform)
                a = src.read(1, window=win)
                b = src.window_bounds(win)
            else:
                a = src.read(1)
                b = src.bounds
            a = np.where(np.isfinite(a) & (a >= 0), a, np.nan)
            arrays.append(np.log1p(a))
            extents.append((b[0], b[2], b[1], b[3]))
            if src is not src0:
                src.close()
    vmax = np.nanpercentile(np.concatenate([a[np.isfinite(a)].ravel() for a in arrays]), vmax_pct)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.04)
    for ax, a, ext, lab in zip(axes, arrays, extents, labels):
        im = ax.imshow(a, extent=ext, origin="upper", cmap=NIGHTS, vmin=0, vmax=vmax,
                       interpolation="bilinear", aspect="auto")
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_lw, alpha=0.9)
        if sites_lonlat is not None:
            ax.scatter(sites_lonlat[0], sites_lonlat[1], s=3.5, c=UN_BLUE,
                       alpha=0.85, linewidths=0, zorder=10)
        if epicenter is not None:
            ax.scatter([epicenter[0]], [epicenter[1]], marker="*", s=120, c=ACCENT,
                       linewidths=0, zorder=11)
        if bbox is not None:
            ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_title(lab)
        ax.grid(False)
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.015)
    cbar.set_label("VIIRS nighttime radiance, log(1 + nW/cm²/sr)")
    handles = []
    if sites_lonlat is not None:
        handles.append(mlines.Line2D([], [], marker="o", color=UN_BLUE, ms=5,
                                     ls="None", label=site_label))
    if epicenter is not None:
        handles.append(mlines.Line2D([], [], marker="*", color=ACCENT, ms=12,
                                     ls="None", label="Epicentre (M7.7)"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.44, -0.03))
    if suptitle:
        fig.suptitle(suptitle, y=1.03)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
