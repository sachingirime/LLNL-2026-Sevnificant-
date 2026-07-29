#!/usr/bin/env python
"""Per-node health: sphere fill, cross-checked against how many struts arrive.

A missing junction leaves nothing behind, so a sphere about the node is simply empty.
Sorted sphere fill over the 2,456 interior nodes is 0.000 for two of them and 1.000 for
every other one -- the largest gap in the distribution is 0.90 wide, and the cut is read
off it rather than chosen. Sweeping the radius from the nominal 257 um out to 525 um
flags the same two nodes every time with a gap never narrower than 0.55, so nothing here
turns on the parameter.

The reading is that clean because a node is present whole or absent whole. In the design
JSON a junction is a bare `position` with no radius and no thickness -- only struts carry
`thickness` -- so it is not a separately printed part but the region where twelve struts
overlap, plus the fillet the melt pool leaves where their scan vectors converge. Fill
accordingly stays at or above 0.90 through the loss of one, two, three, four struts and
hits 0.000 only where all twelve are gone, with nothing in between. So a missing node
here is a localised build failure that took a whole junction with it, which is a more
useful thing to report than twelve unrelated missing struts -- and it is worth measuring
from the voxels even though the strut labels imply it, because the two are independent
evidence and they agree.

The independent check is degree. Every physical node knows which struts meet it, and
`measure_connectivity` already says for each strut whether metal joins its two ends, so
"of the struts that should meet here, how many arrive?" is a count needing no threshold
at all. It comes from the strut labels; the sphere fill comes from the voxels. The two
agree on exactly the same two nodes, which is worth more than either alone.

The same table also answers whether defects cluster. If missing and broken struts were
scattered independently the count meeting at a node would be binomial; a heavier tail
means damage concentrates at particular junctions, which is what a node-level process
problem looks like as opposed to independent strut failures. The comparison is printed
and plotted rather than tested to a p-value: with 2,456 nodes the null is easy to reject
on a technicality, and the shape of the discrepancy is more informative.

    python scripts/node_health.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lattice_iou import (_BLUE, _GRID, _INK, _INK2, _ORANGE, _RED, _SURFACE,
                             _style, dedupe_junctions, detrend_by_height,
                             lattice_geometry, load_design, measure_node_sphere_fill)

DEF_JSON = ("data/missing_struts/registered_jsons/"
            "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
DEF_MASK = "data/9x9x9_octet_lattice/segmentation/mask.tif"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", default=DEF_JSON)
    p.add_argument("--mask", default=DEF_MASK)
    p.add_argument("--correction", default="outputs/registration/correction.json")
    p.add_argument("--dir", default="outputs/lattice_iou")
    p.add_argument("--radius-factor", type=float, default=1.467,
                   help="sphere radius in nominal strut radii (default = the design "
                        "node). The answer is the same anywhere in 1.5-3.0.")
    a = p.parse_args()

    root = Path(a.dir)
    pos, pairs, _, corrected = load_design(a.design, a.correction)
    node_pos, node_of, degree = dedupe_junctions(pos, pairs)
    geom = lattice_geometry(pos, pairs)
    print(f"{len(pos)} junction entries -> {len(node_pos)} physical nodes, "
          f"correction_applied={corrected}")

    conn = dict(np.load(root / "connectivity.npz", allow_pickle=False))
    cls = np.genfromtxt(root / "strut_classes.csv", delimiter=",", names=True,
                        dtype=None, encoding=None)
    lab = cls["label"]
    st = np.genfromtxt(root / "struts.csv", delimiter=",", names=True)
    nd = np.genfromtxt(root / "nodes.csv", delimiter=",", names=True)

    n = len(lab)
    pairs = pairs[:n]
    st = {k: st[k][:n] for k in st.dtype.names}
    ends = node_of[pairs]                       # (n_struts, 2) physical node ids

    # A strut is only evidence about its nodes if it could be measured at all.
    usable = (st["is_boundary"] == 0) & (st["embedded"] == 0) & ~conn["clipped"] \
        & ~conn["no_seed"]
    bad = usable & ((lab == "missing") | (lab == "broken"))
    print(f"{int(usable.sum())} usable struts, {int(bad.sum())} of them missing or broken")

    n_nodes = len(node_pos)
    inc_usable = np.bincount(ends[usable].ravel(), minlength=n_nodes)
    inc_bad = np.bincount(ends[bad].ravel(), minlength=n_nodes)
    arrive = inc_usable - inc_bad

    interior = degree == 12
    # A node needs most of its struts measurable before "how many arrive" means anything;
    # on the surface and in the build plates the count is dominated by what was excluded.
    solid = interior & (inc_usable >= 8)
    print(f"{int(interior.sum())} interior (degree 12), of which {int(solid.sum())} have "
          f">=8 usable struts")

    # ---------------------------------------------------------------- sphere fill
    r_strut = geom["nominal_strut_radius_vox"]
    r_node = a.radius_factor * r_strut
    um = geom["um_per_voxel"]
    print(f"\nsphere fill: radius {a.radius_factor:.3f}r = {r_node:.2f} vox "
          f"= {r_node * um:.0f} um")
    mask = tifffile.imread(a.mask).astype(bool)
    fill = measure_node_sphere_fill(mask, node_pos, r_node)

    sv = np.sort(fill[solid])
    print(f"  min {sv[0]:.3f}   p1 {np.percentile(sv, 1):.3f}   median {np.median(sv):.3f}")

    # The cut is read off the gap, not chosen.
    g = int(np.argmax(np.diff(sv)))
    cut = 0.5 * (sv[g] + sv[g + 1])
    print(f"  largest gap {sv[g]:.3f} -> {sv[g + 1]:.3f} ({sv[g + 1] - sv[g]:.3f} wide), "
          f"cut at {cut:.3f}, {g + 1} node(s) below it")
    empty = solid & (fill <= cut)

    # Nothing turns on the radius: show that instead of asserting it.
    print(f"\n  {'radius':>8s} {'um':>5s} {'gap':>7s} {'below':>6s}")
    for k in (1.467, 1.8, 2.0, 2.4, 2.8, 3.0):
        f2 = measure_node_sphere_fill(mask, node_pos[solid], k * r_strut,
                                      progress_every=0)
        s2 = np.sort(f2)
        j = int(np.argmax(np.diff(s2)))
        print(f"  {k:8.3f} {k * r_strut * um:5.0f} {s2[j + 1] - s2[j]:7.3f} {j + 1:6d}")

    # ------------------------------------------------------------------- clustering
    print("\ndefective struts meeting at a node (interior, >=8 usable):")
    print(f"  {'bad':>4s} {'nodes':>7s} {'%':>7s}   {'expected if independent':>24s}")
    pbad = float(inc_bad[solid].sum()) / float(inc_usable[solid].sum())
    from math import comb
    tot = int(solid.sum())
    obs = np.bincount(inc_bad[solid], minlength=6)
    mean_k = float(inc_usable[solid].mean())
    for k in range(6):
        exp = tot * comb(int(round(mean_k)), k) * pbad ** k * (1 - pbad) ** (round(mean_k) - k)
        print(f"  {k:4d} {int(obs[k]):7d} {100 * obs[k] / tot:6.2f}%   {exp:24.1f}")
    if len(inc_bad[solid]) and inc_bad[solid].max() >= 6:
        print(f"  >=6  {int((inc_bad[solid] >= 6).sum()):6d}")

    # ------------------------------------------------------------------- candidates
    dead = solid & (arrive == 0)
    print("\nmissing-node candidates")
    print(f"  sphere fill (fill <= {cut:.3f}):  {int(empty.sum())}")
    print(f"  degree      (no strut arriving): {int(dead.sum())}")
    print(f"  both agree:                      {int((empty & dead).sum())}"
          f"   disagree: {int((empty ^ dead).sum())}")
    for k in np.flatnonzero(empty | dead):
        print(f"  node {k}  z={node_pos[k][0]:.0f} y={node_pos[k][1]:.0f} "
              f"x={node_pos[k][2]:.0f}  fill {fill[k]:.3f}  usable {inc_usable[k]}  "
              f"bad {inc_bad[k]}  {'sphere+degree' if empty[k] and dead[k] else 'ONE ONLY'}")

    worst = np.argsort(-inc_bad)[:15]
    print("\nmost-damaged nodes:")
    print(f"  {'node':>6s} {'deg':>4s} {'usable':>7s} {'bad':>4s} {'arrive':>7s}  "
          f"{'fill':>6s} {'dia_fill um':>12s}")
    lut = {int(v): i for i, v in enumerate(nd["node_id"].astype(int))}
    for k in worst:
        r = lut.get(int(k))
        dia = nd["diameter_fill_um"][r] if r is not None else np.nan
        print(f"  {int(k):6d} {int(degree[k]):4d} {int(inc_usable[k]):7d} "
              f"{int(inc_bad[k]):4d} {int(arrive[k]):7d}  {fill[k]:6.3f} {dia:12.0f}")

    # --------------------------------------------------------------- malformed nodes
    # Absence and malformation are different failures. Sphere fill only sees the first --
    # its 257 um probe sits inside a 937 um junction and saturates -- so a junction whose
    # struts arrived but never fused reads 1.000 there. Size does not saturate.
    dia = np.zeros(n_nodes)
    for k in range(n_nodes):
        r = lut.get(int(k))
        dia[k] = nd["diameter_fill_um"][r] if r is not None else np.nan
    live = solid & np.isfinite(dia) & (dia > 0)
    # Junction size falls with build height on this specimen, so the raw low tail is partly
    # a list of tall nodes. Rank on the ratio to what each node's height predicts, and print
    # both, because the trend is a real property of the print and not an error to hide.
    ratio, expect = detrend_by_height(dia, node_pos[:, 0], exclude=~live)
    corr = float(np.corrcoef(node_pos[live, 0], dia[live])[0, 1])
    corr_r = float(np.corrcoef(node_pos[live, 0], ratio[live])[0, 1])
    print(f"\nmalformed junctions -- printed but not formed (size does not saturate)")
    print(f"  diameter at 90% ball fill: median {np.median(dia[live]):.0f} um, "
          f"p1 {np.percentile(dia[live], 1):.0f}, p5 {np.percentile(dia[live], 5):.0f}")
    print(f"  vs build height: r = {corr:+.3f}  ->  after detrending {corr_r:+.3f}")
    order = np.flatnonzero(live)[np.argsort(dia[live])]
    sv_d = np.sort(dia[live])
    gp = int(np.argmax(np.diff(sv_d[:60])))
    print(f"  largest gap in the low tail: {sv_d[gp]:.1f} -> {sv_d[gp + 1]:.1f} um "
          f"({sv_d[gp + 1] - sv_d[gp]:.1f} wide), {gp + 1} node(s) below it")
    print(f"  {'rank':>4s} {'node':>6s} {'z':>5s} {'dia um':>8s} {'ratio':>6s} "
          f"{'bad':>4s} {'fill':>6s}")
    for rank, k in enumerate(order[:12]):
        print(f"  {rank:4d} {int(k):6d} {node_pos[k][0]:5.0f} {dia[k]:8.1f} "
              f"{ratio[k]:6.3f} {int(inc_bad[k]):4d} {fill[k]:6.3f}")
    malformed = order[0] if len(order) else None

    hdr = ["node_id", "z", "y", "x", "degree", "incident_usable", "incident_bad",
           "arriving", "arriving_frac", "sphere_fill", "missing_candidate",
           "diameter_fill_um", "size_ratio_to_height"]
    with open(root / "node_health.csv", "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for k in range(n_nodes):
            frac = arrive[k] / inc_usable[k] if inc_usable[k] else -1.0
            fh.write(f"{k},{node_pos[k][0]:.3f},{node_pos[k][1]:.3f},{node_pos[k][2]:.3f},"
                     f"{int(degree[k])},{int(inc_usable[k])},{int(inc_bad[k])},"
                     f"{int(arrive[k])},{frac:.4f},{fill[k]:.4f},"
                     f"{int(empty[k] or dead[k])},{dia[k]:.1f},{ratio[k]:.4f}\n")
    print(f"\nwrote {root / 'node_health.csv'}")

    # The headline is the UNFILTERED count -- every node the design declares, tested where
    # it declares it. The interior-only number is a second answer to a narrower question,
    # and on this specimen the two differ by 100x, so the figure has to show both and show
    # WHERE the hits are, because that is what decides which one to quote.
    sa = np.sort(fill)
    ga = int(np.argmax(np.diff(sa)))
    cut_all = 0.5 * (sa[ga] + sa[ga + 1])
    flag_all = fill <= cut_all
    print(f"\nunfiltered: {int(flag_all.sum())} of {n_nodes} nodes flagged "
          f"({100 * flag_all.sum() / n_nodes:.2f}%), cut {cut_all:.3f} off a "
          f"{sa[ga + 1] - sa[ga]:.3f}-wide gap")

    plot(root / "node_health.png", fill, solid, inc_bad, obs, tot, pbad, mean_k, cut,
         nd, lut, dia, ratio, live, node_pos[:, 0],
         flag_all=flag_all, interior=interior, pos=node_pos, cut_all=cut_all)
    print(f"wrote {root / 'node_health.png'}")

    # A representative of the flagged surface group, if they form a flat face. The whole
    # extent-vs-defect call rests on what those voxels look like, so put one in the figure
    # rather than asking the reader to trust the word "extent". Pick the one nearest the
    # middle of the face, so it is not also a plate or corner effect.
    face_pick = None
    fs = np.flatnonzero(flag_all & ~interior)
    if fs.size:
        pts = node_pos[fs]
        tol = 0.5 * geom["strut_length_vox"]
        for axis in range(3):
            # Count how many sit on the plane rather than measuring the full spread: a
            # single stray node elsewhere sends ptp to 709 vox and hides a real face.
            on = np.abs(pts[:, axis] - np.median(pts[:, axis])) < tol
            if on.mean() < 0.9:
                continue
            other = [k for k in range(3) if k != axis]
            mid = np.median(pts[on][:, other], axis=0)
            cand = fs[on]
            face_pick = int(cand[np.argmin(np.abs(pts[on][:, other] - mid).sum(1))])
            print(f"  face: {int(on.sum())} of {len(fs)} flagged surface nodes lie on "
                  f"{'zyx'[axis]} = {np.median(pts[:, axis]):.0f}; gallery uses node "
                  f"{face_pick} (z={node_pos[face_pick][0]:.0f} "
                  f"y={node_pos[face_pick][1]:.0f} x={node_pos[face_pick][2]:.0f})")
            break

    plot_node_gallery(root / "node_gallery.png", mask, node_pos, r_node, fill, solid,
                      inc_bad, empty | dead, um, malformed=malformed, dia=dia,
                      face_pick=face_pick, degree=degree)
    print(f"wrote {root / 'node_gallery.png'}")


def plot_node_gallery(out_path, mask, node_pos, r_node, fill, solid, inc_bad, flagged,
                      um, malformed=None, dia=None, face_pick=None, degree=None,
                      half=14):
    """The voxels behind the number, for one node per case.

    Three orthogonal centre planes each, raw on the left and with the sphere painted on
    the right -- blue where it is material, red where it is void. A missing junction is a
    blank window, which needs no statistic to see, and the point of the figure is that
    every other row is a solid disc.

    The `malformed` row is the one that cannot be read off the painted sphere at all: the
    sphere is full (fill 0.901, blue) because it is smaller than even a stunted junction.
    What is wrong there is visible only as the junction being visibly narrower than the
    typical row below it -- which is the whole argument for ranking on size, not fill.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.flatnonzero(solid & (inc_bad == 0))
    order = order[np.argsort(fill[order])]
    picks = [("missing node", int(k)) for k in np.flatnonzero(flagged)[:2]]
    # The face group, if there is one. It is empty exactly like the two above, so the row
    # that distinguishes them is not the sphere -- it is that the lattice simply stops
    # here, with no severed strut ends left behind and nothing beyond it.
    if face_pick is not None:
        tag = "surface node on the flagged face"
        if degree is not None:
            tag += f"\ndegree {int(degree[face_pick])}, not 12"
        picks.append((tag, int(face_pick)))
    # ...but not one of those two, which have all 12 struts gone and would just repeat a
    # row. The interesting case is a node that lost some struts and is still solid.
    # Malformed goes in BEFORE the strut rows. The smallest junction usually also has
    # defective struts, so picking by strut count first steals it and the size failure
    # never gets a row of its own -- which is exactly the distinction the figure is for.
    if malformed is not None:
        tag = "malformed: smallest junction"
        if dia is not None and np.isfinite(dia[int(malformed)]):
            tag += (f"\n{dia[int(malformed)]:.0f} um vs "
                    f"{np.nanmedian(dia[solid]):.0f} typical")
        picks.append((tag, int(malformed)))
    taken = {k for _, k in picks}
    d = np.flatnonzero(solid & (inc_bad >= 2) & ~flagged)
    d = np.array([k for k in d if int(k) not in taken], int)
    if d.size:
        picks.append(("2+ struts defective", int(d[np.argsort(fill[d])[0]])))
    picks += [("lowest fill, struts sound", int(order[0])),
              ("typical", int(order[len(order) // 2]))]

    ncol, cell, text_w = 6, 1.55, 3.0
    fig = plt.figure(figsize=(ncol * cell + text_w, cell * len(picks) + 0.85),
                     facecolor=_SURFACE)
    gs = fig.add_gridspec(len(picks), ncol,
                          left=text_w / (ncol * cell + text_w), right=0.995,
                          top=1 - 0.75 / (cell * len(picks) + 0.85), bottom=0.02,
                          wspace=0.06, hspace=0.10)
    planes = ("axial z", "coronal y", "sagittal x")

    o = np.arange(-half, half + 1, dtype=np.float32)
    dz, dy, dx = np.meshgrid(o, o, o, indexing="ij")

    for row, (name, k) in enumerate(picks):
        p = node_pos[k]
        base = np.round(p).astype(int)
        lo = base - half
        hi = lo + 2 * half + 1
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, mask.shape)
        box = np.zeros((2 * half + 1,) * 3, bool)
        box[lo_c[0] - lo[0]:hi_c[0] - lo[0],
            lo_c[1] - lo[1]:hi_c[1] - lo[1],
            lo_c[2] - lo[2]:hi_c[2] - lo[2]] = mask[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1],
                                                    lo_c[2]:hi_c[2]]
        f = p - base
        sphere = ((dz - f[0]) ** 2 + (dy - f[1]) ** 2 + (dx - f[2]) ** 2) <= r_node ** 2

        first_ax = None
        for axis in range(3):
            sl = [slice(None)] * 3
            sl[axis] = half
            m2, s2 = box[tuple(sl)], sphere[tuple(sl)]
            for col, painted in ((axis, False), (axis + 3, True)):
                ax = fig.add_subplot(gs[row, col])
                rgb = np.ones(m2.shape + (3,), float)
                rgb[m2] = (0.62, 0.62, 0.60)
                if painted:
                    rgb[s2 & m2] = (0.16, 0.47, 0.84)
                    rgb[s2 & ~m2] = (0.89, 0.29, 0.28)
                ax.imshow(rgb, extent=(-half - .5, half + .5, -half - .5, half + .5),
                          origin="lower", interpolation="nearest")
                ax.add_patch(plt.Circle((-f[2 if axis != 2 else 1],
                                         -f[0 if axis else 1]), r_node, fill=False,
                                        color=_INK, lw=0.9, alpha=0.75))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(_GRID)
                if row == 0:
                    ax.set_title(f"{planes[axis]}{' -- sphere' if painted else ''}",
                                 color=_INK2, fontsize=7.5, pad=3)
                if first_ax is None:
                    first_ax = ax

        b = first_ax.get_position()
        fig.text(0.006, b.y0 + b.height * 0.74, f"{name}   node {k}",
                 color=_RED if fill[k] < 0.5 else _INK, fontsize=9.5,
                 fontweight="bold", va="center")
        fig.text(0.006, b.y0 + b.height * 0.32,
                 f"sphere fill {fill[k]:.3f}\n"
                 f"{int(inc_bad[k])} defective struts of {12}\n"
                 f"z={p[0]:.0f} y={p[1]:.0f} x={p[2]:.0f}",
                 color=_INK2, fontsize=7.2, va="center")

    fig.suptitle(
        f"Node sphere fill.   Sphere of {r_node * um:.0f} um radius (black outline) "
        "about each junction.\nblue = material inside it,   red = void inside it.   "
        "A missing node is empty; every sound node is solid.",
        color=_INK, fontsize=10, x=0.006, ha="left", y=0.996, va="top")
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def plot(out_path, fill, solid, inc_bad, obs, tot, pbad, mean_k, cut, nd, lut,
         dia=None, ratio=None, live=None, zpos=None,
         flag_all=None, interior=None, pos=None, cut_all=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import comb

    nrow = 1 + (dia is not None) + (flag_all is not None)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 4.4 * nrow), facecolor=_SURFACE,
                             squeeze=False)
    rows = list(axes)
    axes = rows[0]
    extra = rows[1] if dia is not None else None
    space = rows[-1] if flag_all is not None else None
    sel = np.flatnonzero(solid)

    ax = axes[0]
    _style(ax)
    bins = np.linspace(0, 1, 61)
    if flag_all is not None:
        ax.hist(fill, bins=bins, color=_ORANGE, edgecolor="none",
                label=f"all {len(fill)} nodes")
        ax.hist(fill[sel], bins=bins, color=_BLUE, edgecolor="none",
                label=f"interior only ({len(sel)})")
        ax.axvline(cut_all, color=_RED, linewidth=1.4)
        ax.annotate(f"cut {cut_all:.2f}\n{int(flag_all.sum())} flagged", xy=(cut_all, 0.72),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=_RED, fontsize=7.5, va="top")
        leg = ax.legend(loc="upper center", fontsize=7.5, framealpha=0.9,
                        facecolor=_SURFACE)
        for t in leg.get_texts():
            t.set_color(_INK2)
        title = "Node sphere fill -- ALL nodes, unfiltered"
    else:
        ax.hist(fill[sel], bins=bins, color=_BLUE, edgecolor="none")
        ax.axvline(cut, color=_RED, linewidth=1.4)
        ax.annotate(f"cut {cut:.2f}\nread off the gap", xy=(cut, 0.72),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=_RED, fontsize=7.5, va="top")
        title = "Node sphere fill, interior nodes"
    ax.set_yscale("log")
    ax.set_title(title, color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("fraction of the sphere that is material", color=_INK2, fontsize=8)
    ax.set_ylabel("nodes (log)", color=_INK2, fontsize=8)

    # The whole case for the threshold in one line: sorted fill is flat at 1.0 and drops
    # off a cliff. Nothing sits in between, so no choice is being made.
    ax = axes[1]
    _style(ax)
    sv = np.sort(fill[sel])
    rank = np.arange(1, len(sv) + 1)
    ax.plot(rank, sv, color=_BLUE, linewidth=1.6)
    ax.scatter(rank[:2], sv[:2], s=30, color=_RED, zorder=3)
    ax.axhline(cut, color=_RED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xlim(0.8, len(sv) * 1.2)
    ax.set_ylim(-0.05, 1.08)
    ax.annotate(f"2 nodes at 0.000, then nothing\nuntil {sv[2]:.3f} -- a gap so wide the\n"
                "cut cannot be got wrong", xy=(0.30, 0.30), xycoords="axes fraction",
                color=_RED, fontsize=8, va="top")
    ax.set_title("Sorted fill, INTERIOR nodes: the gap does the thresholding",
                 color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("node, sorted by fill (log rank)", color=_INK2, fontsize=8)
    ax.set_ylabel("sphere fill", color=_INK2, fontsize=8)

    ax = axes[2]
    _style(ax)
    ks = np.arange(0, 6)
    exp = [tot * comb(int(round(mean_k)), int(k)) * pbad ** k
           * (1 - pbad) ** (round(mean_k) - k) for k in ks]
    ax.bar(ks - 0.19, obs[:6], width=0.38, color=_BLUE, label="observed")
    ax.bar(ks + 0.19, exp, width=0.38, color=_ORANGE,
           label="if defects were independent")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK2)
    ax.set_title("Defective struts meeting at one node", color=_INK, fontsize=10,
                 loc="left", pad=8)
    ax.set_xlabel("missing or broken struts at the node", color=_INK2, fontsize=8)
    ax.set_ylabel("interior nodes (log)", color=_INK2, fontsize=8)

    if extra is not None:
        s = np.flatnonzero(live)
        sd = np.sort(dia[s])

        # Where the fill panel above saturates, size does not -- this is the panel that
        # carries the malformed class.
        ax = extra[0]
        _style(ax)
        ax.hist(dia[s], bins=60, color=_BLUE, edgecolor="none")
        ax.axvline(sd[0], color=_RED, linewidth=1.4)
        ax.annotate(f"smallest junction\n{sd[0]:.0f} um", xy=(sd[0], 0.85),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=_RED, fontsize=7.5, va="top")
        ax.set_yscale("log")
        ax.set_title("Junction size -- does NOT saturate", color=_INK, fontsize=10,
                     loc="left", pad=8)
        ax.set_xlabel("diameter at 90% ball fill (um)", color=_INK2, fontsize=8)
        ax.set_ylabel("nodes (log)", color=_INK2, fontsize=8)

        # The confound, shown rather than corrected away in silence.
        ax = extra[1]
        _style(ax)
        ax.scatter(zpos[s], dia[s], s=3, color=_BLUE, alpha=0.25, edgecolors="none")
        k0 = s[np.argmin(dia[s])]
        ax.scatter([zpos[k0]], [dia[k0]], s=45, color=_RED, zorder=3)
        r = np.corrcoef(zpos[s], dia[s])[0, 1]
        ax.set_title(f"Size falls with build height (r = {r:+.2f})", color=_INK,
                     fontsize=10, loc="left", pad=8)
        ax.set_xlabel("z (voxels, build direction)", color=_INK2, fontsize=8)
        ax.set_ylabel("diameter at 90% ball fill (um)", color=_INK2, fontsize=8)
        ax.annotate("so the raw low tail is partly\na list of tall nodes",
                    xy=(0.04, 0.12), xycoords="axes fraction", color=_INK2, fontsize=8)

        # ...and the ranking that survives it.
        ax = extra[2]
        _style(ax)
        sr = np.sort(ratio[s])
        rank = np.arange(1, len(sr) + 1)
        ax.plot(rank, sr, color=_BLUE, linewidth=1.6)
        ax.scatter(rank[:1], sr[:1], s=35, color=_RED, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.8, len(sr) * 1.2)
        rr = np.corrcoef(zpos[s], ratio[s])[0, 1]
        ax.set_title(f"Detrended: size vs its own height (r = {rr:+.2f})", color=_INK,
                     fontsize=10, loc="left", pad=8)
        ax.set_xlabel("node, sorted by ratio (log rank)", color=_INK2, fontsize=8)
        ax.set_ylabel("size / what its height predicts", color=_INK2, fontsize=8)
        ax.annotate(f"lowest {sr[0]:.2f}, then {sr[1]:.2f}\n"
                    "no threshold applied -- read the gap",
                    xy=(0.06, 0.90), xycoords="axes fraction", color=_RED, fontsize=8,
                    va="top")

    if space is not None:
        f = np.flatnonzero(flag_all)
        fi = f[interior[f]]                       # interior hits -- the build failures
        fs = f[~interior[f]]                      # surface hits -- one face, on this part
        views = ((2, 1, "x", "y"), (2, 0, "x", "z"), (1, 0, "y", "z"))
        for ax, (a, b, na, nb) in zip(space, views):
            _style(ax)
            ax.scatter(pos[:, a], pos[:, b], s=1.5, color=_GRID, edgecolors="none")
            ax.scatter(pos[fs, a], pos[fs, b], s=16, color=_ORANGE, edgecolors="none",
                       label=f"surface ({len(fs)})")
            ax.scatter(pos[fi, a], pos[fi, b], s=70, facecolors="none", edgecolors=_RED,
                       linewidths=2.0, label=f"interior ({len(fi)})")
            ax.set_xlabel(f"{na} (voxels)", color=_INK2, fontsize=8)
            ax.set_ylabel(f"{nb} (voxels)", color=_INK2, fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(False)
        # The whole 184-vs-2 question in one picture. A flat face of the design with no
        # part behind it collapses to a LINE in the two views that contain its normal, and
        # fills the frame in the third; scattered build failures do neither. Label each
        # view by what it is actually showing rather than repeating one caption.
        space[0].set_title(
            f"WHERE the {int(flag_all.sum())} flagged nodes are  "
            f"(cut {cut_all:.3f}, all {len(fill)} nodes)",
            color=_INK, fontsize=10, loc="left", pad=8)
        space[1].set_title("face seen face-on: orange fills the frame",
                           color=_INK, fontsize=10, loc="left", pad=8)
        space[2].set_title("face edge-on: one line = extent mismatch",
                           color=_INK, fontsize=10, loc="left", pad=8)
        space[0].annotate("red = interior (fully surrounded by lattice), so not a "
                          "surface effect", xy=(0.02, -0.16), xycoords="axes fraction",
                          color=_RED, fontsize=7.5)
        leg = space[0].legend(loc="upper right", fontsize=8, framealpha=0.9,
                              facecolor=_SURFACE)
        for t in leg.get_texts():
            t.set_color(_INK2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
