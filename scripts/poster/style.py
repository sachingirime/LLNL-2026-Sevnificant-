"""Poster styling for the mask-slice figures: palette, class colour ramps, chrome.

The colour scheme carries one idea. Every figure shows a signed distance field measured
off the CT mask, so a single ramp has to read on both sides of zero and the zero itself
has to be unmistakable, because that contour *is* the segmented surface -- the boundary
the whole detector is built on.

    outside metal   cold, dark, and slightly brighter close to the surface, so the void
                    around a strut shows how far it is from the nearest metal
    inside metal    the class colour, brightening toward the core

One consequence worth stating: the bands inside the metal are ~2 voxels' worth of range,
so they are a genuine but coarse local-thickness reading. They are drawn at a fixed
micron spacing across every figure, so band count is comparable between classes.
"""
from __future__ import annotations

import numpy as np

INK = "#080b14"
PANEL = "#05070f"
GRID = "#1b2337"
TEXT = "#e8edf7"
MUTED = "#7d8aa5"
DIM = "#4a5570"

CLASS_COLOR = {
    "missing": "#ff2d55",
    "broken": "#ff7a1a",
    "thin": "#ffd166",
    "thick": "#22d3ee",
    "dross": "#a78bfa",
    "node": "#39ff8b",
    "nominal": "#5b7099",
}

# Cold side of the ramp, far surface -> near surface.
VOID = ["#04060c", "#070b16", "#0b1122", "#111a30", "#18243f"]


def _rgb(c):
    from matplotlib.colors import to_rgb
    return np.array(to_rgb(c), float)


def _mix(a, b, t):
    return _rgb(a) * (1 - t) + _rgb(b) * t


def class_cmap(color, vmin, vmax, name="cls"):
    """Ramp with a hard identity change at zero, positioned for this figure's range."""
    from matplotlib.colors import LinearSegmentedColormap
    z = (0.0 - vmin) / (vmax - vmin)
    c = _rgb(color)
    stops = [(0.0, _rgb(VOID[0])),
             (z * 0.35, _rgb(VOID[1])),
             (z * 0.66, _rgb(VOID[2])),
             (z * 0.88, _rgb(VOID[3])),
             (max(z - 1e-4, 0.0), _rgb(VOID[4])),
             (z, c * 0.42),
             (z + (1 - z) * 0.35, c * 0.80),
             (z + (1 - z) * 0.72, c),
             (1.0, _mix(color, "#ffffff", 0.72))]
    stops = [(min(max(p, 0.0), 1.0), tuple(v)) for p, v in stops]
    return LinearSegmentedColormap.from_list(name, stops)


def rc():
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": INK,
        "savefig.facecolor": INK,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def frame(ax, color=GRID):
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(0.7)
    ax.tick_params(labelsize=8, length=2.5, pad=2)


def scalebar(ax, length_um, label=None, color=TEXT, pad=0.045, lw=2.4):
    """A bar in data units (microns), placed in the lower-right of the axes."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    xe = x1 - pad * w
    y = y0 + pad * h
    ax.plot([xe - length_um, xe], [y, y], color=color, lw=lw,
            solid_capstyle="butt", zorder=20)
    ax.text(xe - length_um / 2, y + 0.022 * h,
            label or f"{length_um:g} µm", color=color, fontsize=8.2,
            ha="center", va="bottom", zorder=20)


def callout(ax, xy, text, color, dxy=(0.0, 0.0), fontsize=9.4, ha="left"):
    """An arrow onto the feature plus its label, both in the class colour."""
    ax.annotate(text, xy=xy, xytext=(xy[0] + dxy[0], xy[1] + dxy[1]),
                color=color, fontsize=fontsize, ha=ha, va="center",
                zorder=25, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=0.16"))


def keyline(ax, x, y, entries, dy=0.052, fontsize=8.4):
    """A compact legend drawn as text, one styled sample per line."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=lw, ls=ls) for _, c, lw, ls in entries]
    labels = [e[0] for e in entries]
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y),
                    fontsize=fontsize, labelcolor=MUTED, handlelength=2.0,
                    handletextpad=0.6, borderpad=0.0, labelspacing=0.42,
                    borderaxespad=0.0)
    leg.set_zorder(30)
    return leg
