#!/usr/bin/env python
"""Cross-section every strut, then sort the results into defect classes.

Runs `src.strut_sections` over the whole lattice and joins it to the nominal-cylinder
IoU table, giving each strut two independent descriptions: how much of the nominal part
is filled, and what the real cross-section does along the edge. The classes fall out of
the second one, which is why the section pass is worth its runtime -- `missing` is the
only class a fill fraction alone can find.

    missing   nothing in the nominal cylinder along the whole edge
    broken    no path through material from one node to the other -- see
              `src.lattice_iou.measure_connectivity`
    thin      the radius profile sits low along its entire length (under-melt)
    thick     it sits high (over-melt, dross piled onto the surface)
    necked    normal median radius but a local pinch
    nominal   everything else

`broken` used to be "a run of empty cross-sections". That was the wrong instrument and
its output said so: among struts with an empty section the median centroid offset was
6.24 vox against the 6.01 vox acceptance radius, i.e. the sections were reading empty
because the material had walked out of the window, not because it was absent. Moving the
cut from 2 empty sections to 1 also swung the count 62 -> 99, which is a threshold
deciding a class rather than measuring one. Connectivity is topological and a stack of
independent planes cannot see it, so it is now measured directly and needs no threshold.

A `warped` class was removed for the same reason: it was built on that same contaminated
`offset_max`, where a blob accepted 6+ vox off-axis inside a 7.5 vox window may well be
the neighbouring strut. `offset_max` is still reported per strut, it just no longer
defines a class.

Order matters and is severity-first: a strut that is both broken and thin is reported
broken, because that is the one that matters.

`thin` and `thick` are a tolerance band on the *design* diameter (350 um), not on the
measured distribution: `--tol-frac 0.25` puts them at 262 and 438 um. This is deliberate
and it is the only defensible anchor at inspection time -- the as-built median is a
property of this print, so cutting at its own percentiles guarantees a fixed fraction of
struts in every class no matter how the part came out, and would call a uniformly
undersized lattice healthy. The cost is that a systematic process bias lands in the class
counts rather than being absorbed by the cut, so the run prints the as-built median
against nominal and the percentile each cut falls at. Read those two lines before
believing a thin/thick tally. `--thin`/`--thick` in um still override.

    python scripts/classify_strut_defects.py                  # measure + report
    python scripts/classify_strut_defects.py --tol-frac 0.20 --cached
    python scripts/classify_strut_defects.py --thin 210 --thick 459 --cached
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lattice_iou import (_BLUE, _GRID, _INK, _INK2, _ORANGE, _RED, _SURFACE,
                             _style, load_design, lattice_geometry,
                             measure_connectivity)
from src.strut_sections import longitudinal_section, measure_sections, section_stack

UM_PER_VOX_FALLBACK = 58.196

CLASS_COLORS = {
    "nominal": "#b8b7b2",
    "thin":    "#2a78d6",
    "thick":   "#eda100",
    "necked":  "#1baf7a",
    "broken":  "#eb6834",
    "missing": "#e34948",
}
ORDER = ["missing", "broken", "thin", "thick", "necked", "nominal"]


def _fmt(v, nd, unit=""):
    """Centreline metrics are NaN where too few clean sections survived to fit one."""
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}{unit}"


def measurable(d, sec, border_cut=0.25):
    """Struts whose geometry can be measured at all.

    Three exclusions, none of them defect classes -- a strut is left out because the
    measurement cannot be made, not because it is healthy or damaged:

      boundary-cap   the 972 struts on the outer surface; about half are never printed
      plate-embedded buried in the solid build plates at either z end, where an absent
                     strut leaves no signature to detect
      window-touching the cross-section runs to the edge of its frame, so the shape
                     describes this strut plus whatever bulk is beside it

    They are reported as an excluded count, never mixed into the class tallies.
    """
    return ((d["is_boundary"] == 0) & (d["embedded"] == 0)
            & ~sec["clipped"] & (sec["border_frac"] <= border_cut))


def classify(sec, n_matched, cuts, n_sections=16, speck_voxels=0, conn=None):
    """Assign one label per strut. Severity-first, so earlier rules win.

    No threshold is involved in `missing` or `broken`. Both rest on counts that are
    either zero or not: how many voxels of the nominal cylinder are material, and how
    many cross-sections along the edge contain none. An IoU cut was used here at first
    and it was the wrong instrument -- it invites a number like "0.06" that looks like a
    tolerance and is really just the middle of dead space, since 90 of these struts sit
    at exactly zero and the next one up is far away.

    `missing` means no strut at all: not one voxel inside the nominal cylinder, and every
    cross-section empty. Raising `speck_voxels` above 0 also admits struts holding only a
    speck -- on this specimen five struts have every section empty but 2 to 22 matched
    voxels out of ~1090 (0.2-2% fill), which is isolated segmentation noise rather than
    material. The next strut above them holds 75, so anything in 22..75 separates the
    two readings identically.
    """
    n = len(n_matched)
    lab = np.array(["nominal"] * n, dtype=object)
    r_med, r_min = sec["r_eq_med"], sec["r_eq_min"]

    necked = (r_min < cuts["neck"]) & (r_med >= cuts["thin"])
    lab[necked] = "necked"
    lab[r_med > cuts["thick"]] = "thick"
    lab[r_med < cuts["thin"]] = "thin"

    # Broken is now a topological fact, not a threshold: no path through material inside
    # the tube from one node to the other. Struts the test could not run on (clipped at
    # the volume edge, or both seeds landing on the same voxel) keep whatever the shape
    # rules gave them rather than being called broken by default.
    if conn is not None:
        lab[~conn["reachable"] & ~conn["clipped"] & ~conn["no_seed"]] = "broken"

    lab[(n_matched <= speck_voxels) & (sec["empty_sections"] >= n_sections)] = "missing"
    return lab


def report_distributions(sec, ok, um, out_path, nominal_um=None, band_um=None,
                         conn=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("r_eq_med", "median section radius (vox)", "thin / thick", 0),
        ("r_eq_min", "minimum section radius (vox)", "necked", 0),
        ("r_eq_cv", "radius variation along the edge (CV)", "roughness", 0),
        ("tilt_deg", "angle of the fitted centreline to the design axis (deg)", "reported only", 0),
        ("detour", "geodesic path length / straight span", "broken", 0),
        ("border_frac", "sections touching the window border", "diagnostic", 0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=_SURFACE)
    for ax, (key, xlabel, drives, _) in zip(axes.ravel(), panels):
        _style(ax)
        if key == "detour":
            # only the connected have a finite detour; the severed are the count in the
            # title, not a bar at infinity
            m = ok & conn["reachable"] & ~conn["clipped"] & ~conn["no_seed"]
            v = conn["detour"][m]
            drives = f"broken; {int((ok & ~conn['reachable'] & ~conn['clipped'] & ~conn['no_seed']).sum())} have no path at all"
        else:
            v = sec[key][ok]
        # tilt/bow are NaN where too few clean sections survived to fit a centreline
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.set_title(f"{key}: no finite values", color=_INK, fontsize=10, loc="left")
            continue
        hi = float(np.percentile(v, 99.5))
        ax.hist(v, bins=110, range=(float(v.min()), max(hi, float(v.min()) + 1e-6)),
                color=_BLUE, edgecolor="none")
        for q, col in ((1, _RED), (5, _ORANGE), (95, _ORANGE), (99, _RED)):
            x = float(np.percentile(v, q))
            if x <= hi:
                ax.axvline(x, color=col, linewidth=1.2)
        # On the radius panel the design value and the tolerance band matter more than
        # the percentiles: the gap between the nominal line and the histogram's peak is
        # the print's systematic shrinkage, and it is what makes a design-anchored thin
        # cut land where it does.
        if key == "r_eq_med" and nominal_um:
            ax.axvline(nominal_um / 2.0 / um, color=_INK, linewidth=1.4, linestyle="--")
            ax.annotate(f"nominal {nominal_um:.0f} um", xy=(nominal_um / 2.0 / um, 0.88),
                        xycoords=("data", "axes fraction"), color=_INK, fontsize=7.5,
                        rotation=90, ha="right", va="top")
            for x_um in (band_um or ()):
                ax.axvline(x_um / 2.0 / um, color=_ORANGE, linewidth=1.4,
                           linestyle=":")
        ax.set_title(f"{key}   (drives: {drives})", color=_INK, fontsize=10,
                     loc="left", pad=8)
        ax.set_xlabel(xlabel, color=_INK2, fontsize=8)
        ax.set_ylabel("struts", color=_INK2, fontsize=8)
        ax.annotate("lines: p1 / p5 / p95 / p99", xy=(0.99, 0.96),
                    xycoords="axes fraction", ha="right", va="top",
                    color=_INK2, fontsize=7.5)
    fig.suptitle(f"Cross-section metrics, {int(ok.sum())} measurable struts   "
                 f"1 vox = {um:.1f} um   read the cuts off these",
                 color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def plot_class_profiles(prof, lab, ok, out_path, um, nominal_um=None):
    """Median radius profile per class -- the shape is the class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=_SURFACE)
    ns = prof["r_eq"].shape[1]
    t = (np.arange(ns) + 0.5) / ns

    ax = axes[0]
    _style(ax)
    for name in ORDER:
        m = ok & (lab == name)
        if m.sum() < 3:
            continue
        live = np.where(prof["empty"][m], np.nan, prof["r_eq"][m])
        # a wholly missing strut has no live section anywhere, so nanmedian of an
        # all-NaN column is expected rather than a problem -- take it without the warning
        with np.errstate(all="ignore"):
            med = np.where(np.isfinite(live).any(0), np.nanmedian(
                np.where(np.isfinite(live).any(0), live, 0.0), axis=0), np.nan)
        ax.plot(t, med * 2 * um, color=CLASS_COLORS[name], linewidth=2,
                label=f"{name}  (n={int(m.sum())})")
    if nominal_um:
        ax.axhline(nominal_um, color=_INK, linewidth=1.2, linestyle="--")
        ax.annotate(f"nominal {nominal_um:.0f} um", xy=(0.01, nominal_um),
                    xycoords=("axes fraction", "data"), color=_INK, fontsize=7.5,
                    va="bottom")
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK2, ncol=2)
    ax.set_title("Median section diameter along the edge", color=_INK, fontsize=10,
                 loc="left", pad=8)
    ax.set_xlabel("position along strut (junctions trimmed off)", color=_INK2, fontsize=8)
    ax.set_ylabel("diameter (um)", color=_INK2, fontsize=8)

    ax = axes[1]
    _style(ax)
    counts = [(name, int((ok & (lab == name)).sum())) for name in ORDER]
    counts = [c for c in counts if c[1]]
    ypos = np.arange(len(counts))
    ax.barh(ypos, [c[1] for c in counts],
            color=[CLASS_COLORS[c[0]] for c in counts], height=0.68)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c[0] for c in counts], color=_INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    total = int(ok.sum())
    for y, (name, c) in zip(ypos, counts):
        ax.annotate(f"  {c}  ({100 * c / total:.2f}%)", xy=(c, y), va="center",
                    color=_INK2, fontsize=8)
    ax.set_title(f"Class counts of {total} measurable struts", color=_INK, fontsize=10,
                 loc="left", pad=8)
    ax.set_xlabel("struts (log scale)", color=_INK2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def plot_section_gallery(mask, pos, pairs, r_strut, lab, sec, ok, out_path, um,
                         n_total, trim_frac=0.20, window_factor=2.5, n_show=8, seed=0,
                         nominal_um=350.0, extra=None):
    """The actual sections, with the nominal circle on them -- the picture behind the number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    picks = []
    for name in ORDER:
        idx = np.flatnonzero(ok & (lab == name))
        if idx.size:
            picks.append((name, int(rng.choice(idx))))

    # Two views of the same strut on one row, because they fail in opposite directions.
    # The lateral view shows where material starts and stops -- a gap is a white column
    # and needs no statistic to see -- but it is one plane through the axis, so it misses
    # anything off that plane. The cross-sections see all round the strut but are
    # independent, so they cannot show continuity. Side by side, each covers the other.
    lat_w = 4.6
    fig = plt.figure(figsize=(1.5 * n_show + lat_w + 2.6, 1.75 * len(picks)),
                     facecolor=_SURFACE)
    # Explicit gridspec/subplots_adjust rather than tight_layout: the per-row caption is
    # wider than one panel, and tight_layout answers an overflowing text extent by
    # shrinking every axes, which collapses the sections to a few pixels each.
    gs = fig.add_gridspec(len(picks), n_show + 1,
                          width_ratios=[lat_w / 1.5] + [1] * n_show,
                          left=2.6 / (1.5 * n_show + lat_w + 2.6), right=0.995,
                          top=1 - 0.60 / (1.75 * len(picks)), bottom=0.03,
                          wspace=0.08, hspace=0.32)

    half = window_factor * r_strut
    for row, (name, sid) in enumerate(picks):
        p0, p1 = pos[pairs[sid, 0]], pos[pairs[sid, 1]]
        seg = p1 - p0

        ax = fig.add_subplot(gs[row, 0])
        img, extent = longitudinal_section(mask, p0, p1, r_strut)
        ax.imshow(img, extent=extent, origin="lower", aspect="auto", cmap="Greys",
                  vmin=0, vmax=1, interpolation="nearest")
        # The cylinder wall is drawn only across the trimmed span it is actually
        # measured over. Running it the full width of the panel implied the nominal
        # cylinder reaches into the junctions, which is exactly what the 20% trim exists
        # to prevent -- node material is ~3x the strut diameter and would swamp every
        # statistic if it were included.
        for sign in (-1, 1):
            ax.plot([trim_frac, 1 - trim_frac], [sign * r_strut] * 2,
                    color=_RED, linewidth=1.2, solid_capstyle="butt")
        for t in (0.0, 1.0):
            ax.axvline(t, color=_ORANGE, linewidth=0.9, linestyle=(0, (4, 3)))
        for t in (trim_frac, 1 - trim_frac):
            ax.axvline(t, color=_BLUE, linewidth=0.9)
        ax.tick_params(colors=_INK2, labelsize=6.5, length=2.5, width=0.7)
        ax.set_xlabel("position along strut", color=_INK2, fontsize=6.5, labelpad=1)
        ax.set_ylabel("radius (vox)", color=_INK2, fontsize=6.5, labelpad=1)
        for sp in ax.spines.values():
            sp.set_color(_GRID)
        first_ax = ax

        stack, g = section_stack(mask, p0 + trim_frac * seg, p1 - trim_frac * seg,
                                 half, 0.5, n_show)
        for col in range(n_show):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(stack[col], extent=(g[0], g[-1], g[0], g[-1]), origin="lower",
                      cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.add_patch(plt.Circle((0, 0), r_strut, fill=False, color=_RED, lw=1.1))
            ax.set_xlim(g[0], g[-1])
            ax.set_ylim(g[0], g[-1])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(_GRID)

        box = first_ax.get_position()
        fig.text(0.006, box.y0 + box.height * 0.70, f"{name}  #{sid}",
                 color=CLASS_COLORS[name], fontsize=10, fontweight="bold", va="center")
        conn = ""
        if extra is not None:
            conn = ("\nno path node-to-node" if not extra["reachable"][sid]
                    else f"\ndetour {extra['detour'][sid]:.3f}x")
        fig.text(0.006, box.y0 + box.height * 0.32,
                 f"median {sec['r_eq_med'][sid] * 2 * um:.0f} um   "
                 f"min {sec['r_eq_min'][sid] * 2 * um:.0f} um\n"
                 # all three centreline numbers, because each is blind to what the
                 # others catch: offset is how far off-axis it strays, tilt the net
                 # lean, bow the departure from a straight line. A strut that arcs out
                 # and back has a large offset and almost no tilt.
                 f"offset {sec['offset_max'][sid]:.1f} vox   "
                 f"tilt {_fmt(sec['tilt_deg'][sid], 1, 'deg')}\n"
                 f"bow {_fmt(sec['bow_vox'][sid], 2, ' vox')}   "
                 f"empty {int(sec['empty_sections'][sid])}/{n_total}{conn}",
                 color=_INK2, fontsize=7.5, va="center")

    fig.suptitle("One strut per class: lateral view through the axis (left) and "
                 f"cross-sections along it (right)   red = nominal {nominal_um:.0f} um   "
                 "blue = trimmed extent,  orange dashed = junction centres",
                 color=_INK, fontsize=11, x=0.006, ha="left", y=0.995)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", default="data/9x9x9_octet_lattice/segmentation/mask.tif")
    p.add_argument("--design", default="data/missing_struts/registered_jsons/"
                                       "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
    p.add_argument("--correction", default="outputs/registration/correction.json")
    p.add_argument("--dir", default="outputs/lattice_iou")
    p.add_argument("--sections", type=int, default=48,
                   help="cross-sections per strut across the trimmed span. 48 puts a "
                        "section every 0.69 vox; past ~64 (0.5 vox, two samples per "
                        "voxel) the extra sections are interpolation, not measurement")
    p.add_argument("--window-factor", type=float, default=2.5)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--cached", action="store_true",
                   help="reuse sections.npz instead of re-measuring")
    p.add_argument("--tol-frac", type=float, default=0.25,
                   help="thin/thick band as a fraction of the NOMINAL design diameter "
                        "(default 0.25 -> 262/438 um about 350). Anchored on the design "
                        "because the as-built median is not known at design time")
    p.add_argument("--nominal-um", type=float, default=None,
                   help="design strut diameter the band is taken about "
                        "(default: the value the IoU pass used, 350)")
    p.add_argument("--thin", type=float, default=None,
                   help="explicit thin cut in um, overrides --tol-frac")
    p.add_argument("--thick", type=float, default=None,
                   help="explicit thick cut in um, overrides --tol-frac")
    p.add_argument("--tube-factor", type=float, default=2.0,
                   help="radius of the connectivity search tube, in nominal strut radii")
    p.add_argument("--speck-voxels", type=int, default=0,
                   help="matched voxels still counted as 'no strut at all' (0 = strict)")
    p.add_argument("--neck", type=float, default=None, help="radius in vox (default p1)")
    p.add_argument("--border", type=float, default=0.25,
                   help="fraction of sections touching the window border above which a "
                        "strut is excluded (bulk metal is inside the frame)")
    p.add_argument("--max-struts", type=int, default=None)
    a = p.parse_args()

    import tifffile
    root = Path(a.dir)
    summary = json.loads((root / "summary.json").read_text())
    um = summary["geometry"]["um_per_voxel"]
    r_strut = summary["geometry"]["nominal_strut_radius_vox"]

    pos, pairs, edge_idx, corrected = load_design(a.design, a.correction)
    if a.max_struts:
        pairs = pairs[:a.max_struts]
    mask = tifffile.imread(a.mask) > 0
    print(f"mask {mask.shape}, correction_applied={corrected}, r_nom={r_strut:.3f} vox")

    cache = root / "sections.npz"
    if a.cached and cache.exists():
        z = np.load(cache, allow_pickle=False)
        sec = {k[4:]: z[k] for k in z.files if k.startswith("sec_")}
        prof = {k[5:]: z[k] for k in z.files if k.startswith("prof_")}
        # A cache measured at a different section count silently breaks both the break
        # criterion and `missing` (which asks for every section empty), so take the
        # cache's own count rather than the flag's.
        cached_sections = int(prof["r_eq"].shape[1])
        if cached_sections != a.sections:
            print(f"cache holds {cached_sections} sections, not {a.sections}; using "
                  f"{cached_sections}. Drop --cached to re-measure at {a.sections}")
            a.sections = cached_sections
        print(f"loaded {cache}")
    else:
        sec, prof = measure_sections(mask, pos, pairs, r_strut, window_factor=a.window_factor,
                                     n_sections=a.sections, step=a.step)
        np.savez_compressed(cache,
                            **{f"sec_{k}": v for k, v in sec.items()},
                            **{f"prof_{k}": v for k, v in prof.items()})
        print(f"wrote {cache}")

    d = np.genfromtxt(root / "struts.csv", delimiter=",", names=True)
    n = len(sec["r_eq_med"])
    d = {k: d[k][:n] for k in d.dtype.names}
    ok = measurable(d, sec, a.border)
    iou = d["iou"]

    pct = lambda k, q: float(np.percentile(sec[k][ok], q))
    span_vox = (1 - 2 * 0.20) * summary["geometry"]["strut_length_vox"]
    print(f"section spacing {span_vox / a.sections * um:.1f} um over a "
          f"{span_vox * um:.0f} um trimmed span")

    # Connectivity: the instrument for `broken`. Cached alongside the sections because it
    # is the same cost order (~3 min) and the same inputs.
    ccache = root / "connectivity.npz"
    if a.cached and ccache.exists():
        zc = np.load(ccache, allow_pickle=False)
        conn = {k: zc[k] for k in zc.files}
        print(f"loaded {ccache}")
    else:
        conn = measure_connectivity(mask, pos, pairs, r_strut, tube_factor=a.tube_factor)
        np.savez_compressed(ccache, **conn)
        print(f"wrote {ccache}")

    tested = ok & ~conn["clipped"] & ~conn["no_seed"]
    det = conn["detour"][tested & conn["reachable"]]
    print(f"\nconnectivity on {int(tested.sum())} measurable struts, tube "
          f"{a.tube_factor}x r_nom:  no node-to-node path {int((tested & ~conn['reachable']).sum())}")
    print(f"  detour ratio of the connected: median {np.median(det):.3f}  "
          f"p95 {np.percentile(det, 95):.3f}  p99.9 {np.percentile(det, 99.9):.3f}  "
          f"max {det.max():.3f}")

    nominal_um = a.nominal_um if a.nominal_um is not None else \
        summary["geometry"]["nominal_strut_diameter_um"]
    thin_um = a.thin if a.thin is not None else (1 - a.tol_frac) * nominal_um
    thick_um = a.thick if a.thick is not None else (1 + a.tol_frac) * nominal_um

    cuts = {
        "thin": thin_um / 2.0 / um,
        "thick": thick_um / 2.0 / um,
        "neck": a.neck if a.neck is not None else pct("r_eq_min", 1),
    }

    # The band is anchored on the design, so where it lands in the measured population
    # is a result, not a setting. Printed here because a shrunken print puts a large and
    # entirely real fraction of struts under the thin cut, and that has to be read as
    # process bias rather than mistaken for hundreds of individual defects.
    built_um = float(np.median(sec["r_eq_med"][ok])) * 2 * um
    frac_at = lambda v: 100.0 * float((sec["r_eq_med"][ok] * 2 * um < v).mean())
    print(f"\nnominal design diameter {nominal_um:.0f} um vs as-built median "
          f"{built_um:.0f} um  ({100 * built_um / nominal_um:.0f}% of nominal)")
    print(f"thin/thick band: {'explicit' if a.thin is not None else f'+/-{100 * a.tol_frac:.0f}% of nominal'}"
          f"  ->  {thin_um:.0f} / {thick_um:.0f} um"
          f"   (p{frac_at(thin_um):.1f} and p{frac_at(thick_um):.1f} of the measured struts)")
    print("\ncuts in use (override with --tol-frac/--thin/--thick/--neck). "
          "missing and broken use none:")
    for k, v in cuts.items():
        extra = f"  = {v * 2 * um:.0f} um dia" if k in ("thin", "thick", "neck") else ""
        print(f"  {k:16s} {v:.4f}{extra}")

    excl_cap = int((d["is_boundary"] > 0).sum())
    excl_emb = int(((d["is_boundary"] == 0) & (d["embedded"] > 0)).sum())
    excl_win = int(((d["is_boundary"] == 0) & (d["embedded"] == 0)
                    & (sec["border_frac"] > a.border)).sum())
    print(f"\nmeasurable struts: {int(ok.sum())} of {n}   excluded {n - int(ok.sum())}"
          f"  ({excl_cap} boundary-cap, {excl_emb} plate-embedded, "
          f"{excl_win} window-touching)")

    lab = classify(sec, d["n_matched"], cuts, a.sections, a.speck_voxels, conn)
    print(f"\n{'class':10s} {'count':>7s} {'pct':>8s}   median diameter")
    for name in ORDER:
        m = ok & (lab == name)
        if not m.sum():
            continue
        dia = np.median(sec["r_eq_med"][m]) * 2 * um if name != "missing" else float("nan")
        print(f"{name:10s} {int(m.sum()):7d} {100 * m.sum() / ok.sum():7.3f}%   "
              f"{dia:6.0f} um" if dia == dia else
              f"{name:10s} {int(m.sum()):7d} {100 * m.sum() / ok.sum():7.3f}%        --")

    hdr = ["strut_id", "label", "r_eq_med_um", "r_eq_min_um", "r_eq_cv",
           "offset_max_vox", "tilt_deg", "bow_vox", "aspect_med", "empty_sections",
           "gap_len", "gap_interior", "border_frac", "iou", "reachable", "detour",
           "geo_len_vox", "span_vox"]
    rows = np.column_stack([
        np.arange(n), sec["r_eq_med"] * 2 * um, sec["r_eq_min"] * 2 * um, sec["r_eq_cv"],
        sec["offset_max"], sec["tilt_deg"], sec["bow_vox"],
        sec["aspect_med"], sec["empty_sections"], sec["gap_len"],
        sec["gap_interior"].astype(int), sec["border_frac"], iou,
        conn["reachable"].astype(int),
        np.where(conn["reachable"], conn["detour"], -1.0),
        np.where(conn["reachable"], conn["geo_len_vox"], -1.0), conn["span_vox"]])
    with open(root / "strut_classes.csv", "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for k in range(n):
            fh.write(f"{int(rows[k,0])},{lab[k]}," +
                     ",".join(f"{v:.5f}" for v in rows[k, 1:]) + "\n")
    print(f"\nwrote {root / 'strut_classes.csv'}")

    report_distributions(sec, ok, um, root / "section_metrics.png",
                         nominal_um=nominal_um, band_um=(thin_um, thick_um), conn=conn)
    plot_class_profiles(prof, lab, ok, root / "class_profiles.png", um,
                        nominal_um=nominal_um)
    plot_section_gallery(mask, pos, pairs, r_strut, lab, sec, ok,
                         root / "section_gallery.png", um, a.sections,
                         window_factor=a.window_factor, nominal_um=nominal_um,
                         extra=conn)
    for f in ("section_metrics.png", "class_profiles.png", "section_gallery.png"):
        print(f"wrote {root / f}")


if __name__ == "__main__":
    main()
