#!/usr/bin/env python
"""Visual audit of what `src.lattice_iou` actually measures, one strut at a time.

The IoU pipeline is a lot of arithmetic hidden behind a single number per strut, so
this renders the thing the number came from: the cropped Otsu mask resampled into the
strut's own frame, with the nominal 350 um cylinder drawn on top. If the cut, the trim
and the cylinder are where they should be, you can see it here; if the registration has
walked the design off the material, you see that here too.

Each panel is a longitudinal section through the strut axis (axis horizontal, one radial
direction vertical), sampled from the mask by trilinear interpolation. Solid lines are
the nominal cylinder wall at +/- r_nom, dashed lines the untrimmed junction-to-junction
extent, so the region between them is what the trim discards.

    python scripts/inspect_strut_crops.py --n 4
    python scripts/inspect_strut_crops.py --ids 12043 3310
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lattice_iou import (_BLUE, _GRID, _INK, _INK2, _ORANGE, _RED, _SURFACE,
                             dedupe_junctions, lattice_geometry, load_design)

DEF_MASK = "data/9x9x9_octet_lattice/segmentation/mask.tif"
DEF_JSON = ("data/missing_struts/registered_jsons/"
            "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
DEF_CORR = "outputs/registration/correction.json"
DEF_CSV = "outputs/lattice_iou/struts.csv"


def section(mask, p0, p1, r_nom, pad=2.5, margin=0.25, samples_r=81):
    """Sample the mask on a plane containing the segment axis.

    Returns (image, extent) with the axial coordinate running 0..1 over the *untrimmed*
    strut, extended by `margin` at both ends so the junctions are visible.
    """
    d = p1 - p0
    length = np.linalg.norm(d)
    u = d / length
    # any unit vector perpendicular to u
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v /= np.linalg.norm(v)

    reach = r_nom * pad
    t = np.linspace(-margin, 1 + margin, int(length * (1 + 2 * margin)) * 2)
    r = np.linspace(-reach, reach, samples_r)
    T, R = np.meshgrid(t, r, indexing="xy")
    pts = (p0[:, None, None] + T[None] * (d[:, None, None])
           + R[None] * v[:, None, None])
    img = map_coordinates(mask.astype(np.float32), pts.reshape(3, -1),
                          order=1, mode="constant", cval=0.0).reshape(T.shape)
    return img, (t[0], t[-1], r[0], r[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", default=DEF_MASK)
    ap.add_argument("--design", default=DEF_JSON)
    ap.add_argument("--correction", default=DEF_CORR)
    ap.add_argument("--csv", default=DEF_CSV)
    ap.add_argument("--out", default="outputs/lattice_iou/strut_crops.png")
    ap.add_argument("--trim-frac", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=3, help="examples per category")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="explicit strut ids; overrides the category sampling")
    ap.add_argument("--iou-cut", type=float, default=0.064)
    ap.add_argument("--station-cut", type=float, default=0.055)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = tifffile.imread(a.mask) > 0
    pos, pairs, _, corrected = load_design(a.design, a.correction)
    geom = lattice_geometry(pos, pairs)
    r_nom = geom["nominal_strut_radius_vox"]
    print(f"mask {mask.shape}, correction_applied={corrected}, r_nom={r_nom:.3f} vox")

    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    ok = (d["is_boundary"] == 0) & (d["embedded"] == 0)
    iou, pmin = d["iou"], d["profile_min"]

    if a.ids:
        picks = [(int(i), f"strut {int(i)}") for i in a.ids]
    else:
        rng = np.random.default_rng(a.seed)
        cats = [
            ("intact", ok & (iou >= a.iou_cut) & (pmin >= a.station_cut)),
            ("severed (a station is empty)", ok & (iou >= a.iou_cut) & (pmin < a.station_cut)),
            ("missing", ok & (iou < a.iou_cut)),
        ]
        picks = []
        for name, m in cats:
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            for i in rng.choice(idx, min(a.n, idx.size), replace=False):
                picks.append((int(i), name))

    ncol = a.n if not a.ids else min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.9 * nrow),
                             facecolor=_SURFACE, squeeze=False)

    for ax, (sid, name) in zip(axes.ravel(), picks):
        row = int(np.flatnonzero(d["strut_id"] == sid)[0])
        p0 = np.array([d["z0"][row], d["y0"][row], d["x0"][row]])
        p1 = np.array([d["z1"][row], d["y1"][row], d["x1"][row]])
        img, extent = section(mask, p0, p1, r_nom)

        ax.imshow(img, extent=extent, origin="lower", aspect="auto",
                  cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        for sign in (-1, 1):
            ax.axhline(sign * r_nom, color=_RED, linewidth=1.2)
        for t in (0.0, 1.0):
            ax.axvline(t, color=_ORANGE, linewidth=1.0, linestyle=(0, (4, 3)))
        for t in (a.trim_frac, 1 - a.trim_frac):
            ax.axvline(t, color=_BLUE, linewidth=1.0)
        ax.set_title(f"#{sid}  {name}\nIoU {d['iou'][row]:.3f}   fill {d['fill'][row]:.3f}"
                     f"   min station {d['profile_min'][row]:.3f}",
                     color=_INK, fontsize=8.5, loc="left", pad=6)
        ax.tick_params(colors=_INK2, labelsize=7, length=3, width=0.8)
        for s in ax.spines.values():
            s.set_color(_GRID)
        ax.set_xlabel("position along strut", color=_INK2, fontsize=7)
        ax.set_ylabel("radius (vox)", color=_INK2, fontsize=7)

    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")

    fig.suptitle("Cropped Otsu mask in the strut frame   "
                 "red = nominal 350 um cylinder wall,  blue = trimmed extent measured,  "
                 "orange dashed = junction centres",
                 color=_INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.06 / nrow * 3))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=_SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
