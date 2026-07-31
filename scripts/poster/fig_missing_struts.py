"""Poster figure 1 -- missing struts.

Reads the finished `detect_lattice_defects` run and the STL ground truth. Nothing
here re-detects anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

AZIM, ELEV = 38, 22


def build_render(pos, pairs, sel, size=(1500, 1700)):
    seg = np.stack([pos[pairs[:, 0]][:, ::-1], pos[pairs[:, 1]][:, ::-1]], 1)
    slabs = C.depth_bins(seg[~sel], AZIM, ELEV, 4)
    bundles = [dict(seg=s, color=c, radius=1.0, sides=8)
               for s, c in zip(slabs, (C.STRUT, C.STRUT, C.STRUT, C.STRUT_HI))]
    bundles.append(dict(seg=seg[sel], color=C.CLASS_COLOR["missing"],
                        radius=3.8, glow=1.0))
    allpts = seg.reshape(-1, 3)
    bounds = (allpts[:, 0].min(), allpts[:, 0].max(), allpts[:, 1].min(),
              allpts[:, 1].max(), allpts[:, 2].min(), allpts[:, 2].max())
    layers = C.render_layers(bundles, size=size, azim=AZIM, elev=ELEV,
                             zoom=1.05, bounds=bounds)
    return C.composite(layers, (0.15, 0.22, 0.34, 0.55, 1.0))


def clustering(pos, pairs, sel, usable, n_draw=300, seed=0):
    """Fraction of flagged struts that share a junction with another flagged strut.

    Against a null of the same number of struts drawn uniformly from the measurable
    population -- otherwise 'they cluster' is unfalsifiable, since any 89 struts in a
    lattice this dense have neighbours.
    """
    import collections

    import lattice_iou
    _, node_of, _ = lattice_iou.dedupe_junctions(pos, pairs)

    def frac(idx):
        nn = node_of[pairs[idx]]
        c = collections.Counter(nn.ravel().tolist())
        return np.array([(c[a] >= 2) or (c[b] >= 2) for a, b in nn]), c

    obs, counter = frac(np.where(sel)[0])
    rng = np.random.default_rng(seed)
    pool = np.where(usable[:len(pairs)])[0]
    null = [frac(rng.choice(pool, int(sel.sum()), replace=False))[0].mean()
            for _ in range(n_draw)]
    return dict(observed=float(obs.mean()), null=float(np.mean(null)),
                null_sd=float(np.std(null)), worst_node=int(max(counter.values())))


def projection(mid_mm, sel, axes_pair, cell=4.56, grid=260):
    """Projected points and their smoothed areal density, per unit-cell footprint."""
    i, j = axes_pair
    pts = np.stack([mid_mm[:, i], mid_mm[:, j]], 1)
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    extent = (lo[0], hi[0], lo[1], hi[1])
    field = C.density_field(pts[sel], extent, 0.55 * cell, grid=grid)
    bin_mm2 = ((hi[0] - lo[0]) / grid) * ((hi[1] - lo[1]) / grid)
    return pts, field / bin_mm2 * cell ** 2, lo, hi, extent


def contour_panel(ax, pts, field, lo, hi, extent, sel, label, cmap, vmax):
    """Filled density contours of the flagged struts, projected down one axis."""
    levels = np.linspace(0, max(vmax, 1e-9), 11)
    cf = ax.contourf(field, levels=levels, cmap=cmap, extent=extent, extend="max")
    cf.set_edgecolor("face")          # kill the hairline seams between bands
    ax.contour(field, levels=levels[1::2], extent=extent,
               colors="#ffffff", linewidths=0.35, alpha=0.22)

    ax.scatter(pts[sel, 0], pts[sel, 1], s=7, c="#ffffff", alpha=0.85,
               linewidths=0, zorder=5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_facecolor(C.HEAT[0])
    for s in ax.spines.values():
        s.set_color(C.GRID)
    ax.tick_params(labelsize=8, length=2, pad=2)
    ax.set_title(label, color=C.TEXT, fontsize=10.5, pad=5, loc="left")
    return cf


def scorecard(ax, tp, fp, fn, n_pred, n_truth, prec, rec):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#0f1626",
                               edgecolor=C.GRID, lw=0.8))
    ax.text(0.055, 0.90, "scored against the CAD", color=C.TEXT, fontsize=11,
            weight="bold", va="center")
    ax.text(0.055, 0.795, "0.stl − 0.5.stl gives the struts the design "
                          "deleted", color=C.MUTED, fontsize=8.6, va="center")

    cells = [("true positive", tp, "#22c55e"), ("false positive", fp, "#ff2d55"),
             ("false negative", fn, "#ff7a1a")]
    for k, (name, val, col) in enumerate(cells):
        x = 0.055 + k * 0.315
        ax.text(x, 0.585, f"{val}", color=col, fontsize=30, weight="bold",
                va="center", ha="left")
        ax.text(x, 0.435, name, color=C.MUTED, fontsize=8.2, va="center", ha="left")

    ax.plot([0.055, 0.945], [0.355, 0.355], color=C.GRID, lw=0.8)
    for k, (name, val) in enumerate((("precision", f"{prec:.3f}"),
                                     ("recall", f"{rec:.3f}"),
                                     ("designed out", f"{n_truth}"))):
        x = 0.055 + k * 0.315
        ax.text(x, 0.225, val, color=C.TEXT, fontsize=19, weight="bold",
                va="center")
        ax.text(x, 0.088, name, color=C.MUTED, fontsize=8.2, va="center")


def main():
    import matplotlib
    matplotlib.use("Agg")
    C.rc()
    global plt
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    pos, pairs, geo = C.load_design()
    lab, meas = C.load_classes()
    truth, usable, info = C.load_stl_truth()
    sel = (lab == "missing") & usable[:len(lab)]

    mm = geo["um_per_voxel"] / 1000.0
    mid = C.strut_midpoints(pos, pairs) * mm          # (z, y, x) mm
    mid_xyz = mid[:, ::-1]                            # (x, y, z) mm

    img = build_render(pos, pairs, sel)
    clu = clustering(pos, pairs, sel, usable)

    fig = plt.figure(figsize=(13.2, 7.9), dpi=200)
    gs = gridspec.GridSpec(
        2, 3, figure=fig, width_ratios=[1.42, 1.0, 1.0], height_ratios=[1, 1],
        left=0.024, right=0.976, top=0.835, bottom=0.088, wspace=0.20, hspace=0.30)

    ax3d = fig.add_subplot(gs[:, 0])
    ax3d.imshow(img)
    ax3d.set_axis_off()
    ax3d.set_anchor("N")   # the cube is wider than tall; keep the slack at the
                           # bottom, where the callout sits
    # The clustering result sits on the render rather than under it: it is a claim
    # about this picture, and it is what stops "0.5%" being read as 0.5% everywhere.
    ax3d.text(0.035, 0.030,
              f"THE DEFECTS ARE NOT INDEPENDENT\n"
              f"{clu['observed']:.0%} share a junction with another missing strut\n"
              f"{clu['null']:.0%} ± {clu['null_sd']:.0%} if the same 89 were scattered "
              f"at random\none junction has {clu['worst_node']} of its struts gone",
              transform=ax3d.transAxes, ha="left", va="bottom",
              color=C.TEXT, fontsize=9.6, linespacing=1.75,
              bbox=dict(boxstyle="round,pad=0.6", facecolor=C.INK, alpha=0.78,
                        edgecolor=C.GRID, linewidth=0.7))

    cmap = C.heat_cmap()
    panels = [(gs[0, 1], (0, 1), "projected along z   (x–y, mm)"),
              (gs[0, 2], (0, 2), "projected along y   (x–z, mm)"),
              (gs[1, 1], (1, 2), "projected along x   (y–z, mm)")]

    # One shared colour scale across the three projections, so a hotter patch in
    # one view really is denser and not just differently normalised.
    projs = [projection(mid_xyz, sel, pair) for _, pair, _ in panels]
    vmax = max(float(p[1].max()) for p in projs)

    cf = None
    for (slot, _, title), p in zip(panels, projs):
        ax = fig.add_subplot(slot)
        cf = contour_panel(ax, p[0], p[1], p[2], p[3], p[4], sel, title, cmap, vmax)

    axs = fig.add_subplot(gs[1, 2])
    scorecard(axs, info["tp"], info["fp"], info["fn"], info["n_pred"],
              int(truth.sum()), info["precision"], info["recall"])

    cax = fig.add_axes([0.552, 0.048, 0.170, 0.0125])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("missing struts per unit-cell footprint, (4.56 mm)²",
                 color=C.MUTED, fontsize=8.0, labelpad=4)
    cb.outline.set_edgecolor(C.GRID)
    cb.ax.tick_params(labelsize=7.2, length=2, color=C.GRID)

    # ---- header
    fig.text(0.024, 0.953, "MISSING STRUTS", fontsize=28, weight="bold",
             color=C.TEXT, va="center")
    fig.text(0.024, 0.896,
             "no matched voxel on the nominal cylinder and every cross-section empty "
             "— a count of zero, so the call carries no threshold to tune",
             fontsize=10.8, color=C.MUTED, va="center")
    fig.text(0.976, 0.953, f"{100 * sel.sum() / usable[:len(lab)].sum():.3f}%",
             fontsize=28, weight="bold", color=C.CLASS_COLOR["missing"],
             va="center", ha="right")
    fig.text(0.976, 0.898,
             f"{int(sel.sum())} of {int(usable[:len(lab)].sum()):,} measurable struts",
             fontsize=10, color=C.MUTED, va="center", ha="right")
    fig.patches.append(plt.Rectangle((0.024, 0.862), 0.952, 0.0016,
                                     transform=fig.transFigure,
                                     facecolor=C.GRID, edgecolor="none"))

    fig.text(0.024, 0.052,
             "9×9×9 octet lattice · 58.18 µm/voxel · 350 µm nominal strut · "
             "registration correction applied",
             fontsize=8.4, color=C.MUTED, va="center")
    fig.text(0.024, 0.024,
             "972 boundary caps and 777 plate-embedded struts excluded, not counted "
             "as healthy",
             fontsize=8.4, color=C.MUTED, va="center")

    out = C.ensure_out() / "fig_missing_struts.png"
    fig.savefig(out, dpi=300, facecolor=C.INK)
    # PDF as well: poster software rescales the text without resampling it, and the
    # only raster in the page is the render itself.
    fig.savefig(out.with_suffix(".pdf"), facecolor=C.INK)
    print(f"wrote {out} (+ .pdf)")
    print(f"  missing {int(sel.sum())}  TP {info['tp']} FP {info['fp']} "
          f"FN {info['fn']}  P {info['precision']:.3f} R {info['recall']:.3f}")


if __name__ == "__main__":
    main()
