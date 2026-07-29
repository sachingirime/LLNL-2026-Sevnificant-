#!/usr/bin/env python
"""Spatial visualisations of the per-strut / per-node IoU tables from `src.lattice_iou`.

The distribution plots say *how many* struts are bad. These say *where*, which is the
question that decides whether a low-IoU population is real damage or a measurement
artefact. A genuine defect population scatters through the specimen; a residual
registration error, a bad slab of segmentation, or an edge effect all show up as
structure -- a gradient, a face, a plane -- and that is visible here and nowhere else.

Produces three figures:

  defect_map.png    every strut projected down each axis, healthy in recessive grey and
                    the flagged ones picked out, so clustering is obvious by eye
  quality_field.png median IoU binned over space, plus IoU against each axis. Flat means
                    the registration holds across the specimen; a ramp or a hot face
                    means it does not, and the defect count is not trustworthy yet
  node_map.png      node size and node IoU in space, same reasoning

    python scripts/visualize_lattice_iou.py --iou-cut 0.06 --station-cut 0.06
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lattice_iou import _BLUE, _GRID, _INK, _INK2, _ORANGE, _RED, _SURFACE, _style

# blue ramp, light -> dark, for magnitude
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GREY = "#d9d8d2"


def load(csv_path):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    return d


def fig_defect_map(d, out, iou_cut, station_cut, um):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    ok = (d["is_boundary"] == 0) & (d["embedded"] == 0)
    a = np.column_stack([d["z0"], d["y0"], d["x0"]])
    b = np.column_stack([d["z1"], d["y1"], d["x1"]])
    missing = ok & (d["iou"] < iou_cut)
    severed = ok & (d["iou"] >= iou_cut) & (d["profile_min"] < station_cut)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), facecolor=_SURFACE)
    pairs = [(2, 1, "x", "y"), (2, 0, "x", "z"), (1, 0, "y", "z")]
    for ax, (i, j, nx, ny) in zip(axes, pairs):
        _style(ax)
        ax.grid(False)
        seg = np.stack([a[:, [i, j]], b[:, [i, j]]], axis=1) * um / 1000.0
        rest = ok & ~missing & ~severed
        ax.add_collection(LineCollection(seg[rest], colors=GREY, linewidths=0.2,
                                         alpha=0.5, zorder=1))
        ax.add_collection(LineCollection(seg[severed], colors=_ORANGE, linewidths=1.6,
                                         zorder=2))
        ax.add_collection(LineCollection(seg[missing], colors=_RED, linewidths=2.2,
                                         zorder=3))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_xlabel(f"{nx} (mm)", color=_INK2, fontsize=8)
        ax.set_ylabel(f"{ny} (mm)", color=_INK2, fontsize=8)
        ax.set_title(f"projected along {'zyx'[3 - i - j]}", color=_INK, fontsize=10,
                     loc="left", pad=8)

    handles = [Line2D([], [], color=_RED, lw=2.2,
                      label=f"IoU < {iou_cut:g}  missing  (n={int(missing.sum())})"),
               Line2D([], [], color=_ORANGE, lw=1.6,
                      label=f"a station < {station_cut:g}  severed  (n={int(severed.sum())})"),
               Line2D([], [], color=GREY, lw=1.0,
                      label=f"rest  (n={int((ok & ~missing & ~severed).sum())})")]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=_INK2,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Where the flagged struts are   "
                 "(cuts are illustrative -- pass --iou-cut / --station-cut to change)",
                 color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def fig_quality_field(d, out, um, cell_vox, iou_cut):
    """Is the IoU level uniform across the specimen? If not, registration is still off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    ok = (d["is_boundary"] == 0) & (d["embedded"] == 0)
    P = np.column_stack([d["z"], d["y"], d["x"]])[ok]
    iou = d["iou"][ok]
    healthy = iou >= iou_cut          # exclude true defects from the *level* estimate

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), facecolor=_SURFACE)
    pairs = [(2, 1, "x", "y"), (2, 0, "x", "z"), (1, 0, "y", "z")]
    # Bin on the unit-cell period. Any other bin width beats against the lattice --
    # each cell holds several strut orientations at different mean IoU, so bins that
    # split a cell alternate between populations and paint a checkerboard that looks
    # like structure but is pure aliasing.
    def cell_edges(v):
        lo, hi = v.min(), v.max()
        n = max(int(round((hi - lo) / cell_vox)), 1)
        return np.linspace(lo, lo + n * cell_vox, n + 1)

    for ax, (i, j, nx, ny) in zip(axes[0], pairs):
        _style(ax)
        ax.grid(False)
        xe, ye = cell_edges(P[:, i]), cell_edges(P[:, j])
        H, xe, ye = np.histogram2d(P[healthy, i], P[healthy, j], bins=[xe, ye],
                                   weights=iou[healthy])
        N, _, _ = np.histogram2d(P[healthy, i], P[healthy, j], bins=[xe, ye])
        med = np.where(N >= 4, H / np.maximum(N, 1), np.nan)
        im = ax.imshow(med.T, origin="lower", cmap=cmap, aspect="equal",
                       extent=[xe[0] * um / 1000, xe[-1] * um / 1000,
                               ye[0] * um / 1000, ye[-1] * um / 1000],
                       vmin=np.nanpercentile(med, 2), vmax=np.nanpercentile(med, 98))
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("mean IoU", color=_INK2, fontsize=8)
        cb.ax.tick_params(colors=_INK2, labelsize=7)
        ax.set_xlabel(f"{nx} (mm)", color=_INK2, fontsize=8)
        ax.set_ylabel(f"{ny} (mm)", color=_INK2, fontsize=8)
        ax.set_title(f"mean IoU, projected along {'zyx'[3 - i - j]}",
                     color=_INK, fontsize=10, loc="left", pad=8)

    names = ["z", "y", "x"]
    for ax, axis in zip(axes[1], range(3)):
        _style(ax)
        # Rolling window over struts sorted by position, one unit cell wide. Same
        # anti-aliasing reasoning as the heatmaps, but it keeps a smooth trend line.
        order = np.argsort(P[:, axis])
        pos_s = P[order, axis] * um / 1000.0
        iou_s = iou[order]
        hlt_s = healthy[order]
        win = max(int(len(order) * cell_vox / (P[:, axis].max() - P[:, axis].min())), 50)
        step = max(win // 6, 1)
        cent, stats = [], []
        for k in range(0, len(order) - win, step):
            v = iou_s[k:k + win][hlt_s[k:k + win]]
            if v.size < 20:
                continue
            cent.append(pos_s[k:k + win].mean())
            stats.append(np.percentile(v, [10, 50, 90]))
        cent, stats = np.array(cent), np.array(stats)
        ax.fill_between(cent, stats[:, 0], stats[:, 2], color=_BLUE, alpha=0.18,
                        linewidth=0)
        ax.plot(cent, stats[:, 1], color=_BLUE, linewidth=2)

        edges = cell_edges(P[:, axis])
        idx = np.clip(np.digitize(P[:, axis], edges) - 1, 0, len(edges) - 2)
        centres = 0.5 * (edges[1:] + edges[:-1]) * um / 1000.0
        cnt = np.bincount(idx[~healthy], minlength=len(centres))[:len(centres)]
        twin = ax.twinx()
        twin.bar(centres, cnt, width=(centres[1] - centres[0]) * 0.8, color=_RED,
                 alpha=0.55, linewidth=0)
        twin.set_ylabel(f"flagged struts", color=_RED, fontsize=8)
        twin.tick_params(colors=_RED, labelsize=7)
        for s in twin.spines.values():
            s.set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{names[axis]} (mm)", color=_INK2, fontsize=8)
        ax.set_ylabel("IoU (median, 10-90%)", color=_INK2, fontsize=8)
        ax.set_title(f"IoU along {names[axis]}", color=_INK, fontsize=10,
                     loc="left", pad=8)

    fig.suptitle("Is the agreement uniform across the specimen?   "
                 "a flat blue level means registration holds everywhere; "
                 "a ramp or a hot face means it does not",
                 color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def fig_node_map(n, out, um):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    use = n["is_interior"] > 0
    P = np.column_stack([n["z"], n["y"], n["x"]])[use] * um / 1000.0
    dia = n["diameter_fill_um"][use]
    lo, hi = np.percentile(dia, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), facecolor=_SURFACE)
    pairs = [(2, 1, "x", "y"), (2, 0, "x", "z"), (1, 0, "y", "z")]
    for ax, (i, j, nx, ny) in zip(axes, pairs):
        _style(ax)
        ax.grid(False)
        sc = ax.scatter(P[:, i], P[:, j], c=dia, s=7, cmap=cmap, vmin=lo, vmax=hi,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("node diameter (um)", color=_INK2, fontsize=8)
        cb.ax.tick_params(colors=_INK2, labelsize=7)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{nx} (mm)", color=_INK2, fontsize=8)
        ax.set_ylabel(f"{ny} (mm)", color=_INK2, fontsize=8)
        ax.set_title(f"projected along {'zyx'[3 - i - j]}", color=_INK, fontsize=10,
                     loc="left", pad=8)

    fig.suptitle(f"Node size in space   n={int(use.sum())} interior nodes   "
                 f"median {np.median(dia):.0f} um",
                 color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="outputs/lattice_iou",
                   help="directory holding struts.csv / nodes.csv / summary.json")
    p.add_argument("--iou-cut", type=float, default=0.06)
    p.add_argument("--station-cut", type=float, default=0.06)
    a = p.parse_args()

    import json
    root = Path(a.dir)
    summary = json.loads((root / "summary.json").read_text())
    um = summary["geometry"]["um_per_voxel"]
    cell = summary["geometry"]["cell_edge_vox"]

    d = load(root / "struts.csv")
    n = load(root / "nodes.csv")

    fig_defect_map(d, root / "defect_map.png", a.iou_cut, a.station_cut, um)
    print(f"wrote {root / 'defect_map.png'}")
    fig_quality_field(d, root / "quality_field.png", um, cell, a.iou_cut)
    print(f"wrote {root / 'quality_field.png'}")
    fig_node_map(n, root / "node_map.png", um)
    print(f"wrote {root / 'node_map.png'}")


if __name__ == "__main__":
    main()
