#!/usr/bin/env python
"""Design-free missing-node detection on 2D node planes, with the raw-CT view to match.

Runs `src.node_planes_2d` over the whole volume and then does the two things the module
itself deliberately does not: it renders the slice a person actually looks at, with every
predicted lattice site drawn on it, and it measures each flagged site well enough to say
whether the node is ABSENT or merely UNDERSIZED. The 2D test alone answers "is there a
node-sized object here", which is both of those at once.

The point of the exercise is that this instrument shares no machinery with the
design-referenced one. It never opens the design JSON and never applies the registration,
so where it agrees with `measure_node_sphere_fill` the agreement means something. On the
0.5%-defect specimen it finds 3 empty interior sites in ~2,240 across 17 usable planes:
the same 2 nodes the 3D sphere fill flags, plus one the 3D test rates healthy at fill
0.901 -- which turns out to be the lowest non-zero fill in all 2,456 interior nodes, with
3 of 12 incident struts dead and a skeleton degree of 9. Measuring it settles what it is:
570 um inscribed against a 708 um population median, below the 1st percentile. Undersized,
not absent, and worth reporting as such.

The optional cross-check columns need the design and the earlier runs; without them the
script still produces the detection and the figures, which is the design-free result.

    python scripts/node_planes_2d.py
    python scripts/node_planes_2d.py --plane 188        # just the one slice, annotated
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import node_planes_2d as N
from src.lattice_iou import _BLUE, _GRID, _INK, _INK2, _ORANGE, _RED, _SURFACE, _style

_GREEN = "#2e9e63"

DEF_MASK = "data/9x9x9_octet_lattice/segmentation/mask.tif"
DEF_RAW = "data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif"
DEF_DESIGN = ("data/missing_struts/registered_jsons/"
              "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
DEF_CORR = "outputs/registration/correction.json"
DEF_OUT = "outputs/node_planes_2d"


# ------------------------------------------------------------------ characterise a site

def site_measurements(mask, raw, z, y, x, half=8):
    """Enough to separate 'absent' from 'undersized' at one flagged site."""
    from scipy import ndimage as ndi
    c = np.round([z, y, x]).astype(int)
    sl = tuple(slice(max(0, c[k] - half), min(mask.shape[k], c[k] + half + 1))
               for k in range(3))
    sub = mask[sl]
    frac = float(sub.mean()) if sub.size else 0.0
    # inscribed radius of the material at the site, in 3D, from a slightly wider box
    wide = tuple(slice(max(0, c[k] - 14), min(mask.shape[k], c[k] + 15)) for k in range(3))
    ed = ndi.distance_transform_edt(np.pad(mask[wide], 1))
    cc = [c[k] - wide[k].start + 1 for k in range(3)]
    gz, gy, gx = np.ogrid[:ed.shape[0], :ed.shape[1], :ed.shape[2]]
    near = ((gz - cc[0]) ** 2 + (gy - cc[1]) ** 2 + (gx - cc[2]) ** 2) <= 16
    r_ins = float(ed[near].max()) if near.any() else 0.0
    ct = float(np.asarray(raw[sl]).mean()) if raw is not None else float("nan")
    return {"material_frac": frac, "inscribed_radius_vox": r_ins, "raw_ct_mean": ct}


# ------------------------------------------------------------------------------ figures

def plot_plane(raw_slab, res, out_path, um_per_vox=None, title_extra=""):
    """The slice as seen, with every predicted lattice site drawn on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interior, edge = ~res["edge"], res["edge"]
    miss_int = interior & ~res["found"]
    miss_edge = edge & ~res["found"]

    fig, ax = plt.subplots(figsize=(11, 11), facecolor=_SURFACE)
    img = np.asarray(raw_slab, float)
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8))
    xy = res["sites_xy"]

    def scat(sel, **kw):
        if sel.any():
            ax.scatter(xy[sel, 1], xy[sel, 0], **kw)

    scat(interior & res["found"], s=34, facecolors="none", edgecolors=_GREEN,
         linewidths=0.9, label=f"interior node found ({int((interior & res['found']).sum())})")
    scat(edge & res["found"], s=34, facecolors="none", edgecolors=_INK2,
         linewidths=0.7, label=f"edge ring found ({int((edge & res['found']).sum())})")
    scat(miss_edge, s=90, marker="s", facecolors="none", edgecolors=_ORANGE,
         linewidths=1.6, label=f"edge ring EMPTY ({int(miss_edge.sum())}) - extent, not a defect")
    scat(miss_int, s=190, marker="o", facecolors="none", edgecolors=_RED,
         linewidths=2.4, label=f"INTERIOR SITE EMPTY ({int(miss_int.sum())})")

    pitch_um = f" = {res['pitch_vox'] * um_per_vox:.0f} um" if um_per_vox else ""
    ax.set_title(
        f"z = {res['z']}   {len(res['centroids'])} node blobs, "
        f"{len(res['sites_xy'])} lattice sites   "
        f"pitch {res['pitch_vox']:.1f} vox{pitch_um} at {res['basis_angle_deg']:.1f} deg"
        f"{title_extra}",
        color=_INK, fontsize=11, loc="left", pad=10)
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.92, facecolor=_SURFACE)
    for t in leg.get_texts():
        t.set_color(_INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)


def plot_explain(raw_slab, z, design, out_path, um_per_vox=None):
    """Label every bright spot in ONE arbitrary slice: node, or strut cross-section.

    This exists because a z slice away from a node plane is easy to misread. The octet's
    horizontal struts lie IN the node planes, so a node plane looks like a diamond grid,
    while a slice between planes shows only the inclined struts cut obliquely -- isolated
    dots, four of them clustered around each node position because that is where four
    inclined struts converge on the node above or below. A cluster missing a lobe is a
    missing STRUT, and a slice 20 vox off a node plane contains no nodes at all. Reading
    those clusters as nodes is what makes a 0.5%-defect part look riddled with holes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, pairs, node_pos, label = design
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=_SURFACE)
    img = np.asarray(raw_slab, float)
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8))

    # struts crossing this z
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    dz = b[:, 0] - a[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - a[:, 0]) / dz
    crosses = np.isfinite(t) & (t >= 0) & (t <= 1) & (np.abs(dz) > 1e-6)
    yx = a[:, 1:] + t[:, None] * (b[:, 1:] - a[:, 1:])
    styles = [("missing", _RED, 120, 2.2), ("broken", _ORANGE, 90, 1.7)]
    healthy = crosses.copy()
    for name, colour, size, lw in styles:
        sel = crosses & (label == name) if label is not None else np.zeros_like(crosses)
        healthy &= ~sel
        if sel.any():
            ax.scatter(yx[sel, 1], yx[sel, 0], s=size, marker="D", facecolors="none",
                       edgecolors=colour, linewidths=lw,
                       label=f"{name} strut crossing this slice ({int(sel.sum())})")
    ax.scatter(yx[healthy, 1], yx[healthy, 0], s=9, marker=".", color=_BLUE, alpha=0.55,
               label=f"sound strut crossing this slice ({int(healthy.sum())})")

    near = np.abs(node_pos[:, 0] - z) < 12
    if near.any():
        ax.scatter(node_pos[near, 2], node_pos[near, 1], s=150, facecolors="none",
                   edgecolors=_GREEN, linewidths=1.2,
                   label=f"node centre within 12 vox of this slice ({int(near.sum())})")

    ax.set_title(f"z = {z}   {int(near.sum())} nodes near this slice, "
                 f"{int(crosses.sum())} struts crossing it   "
                 f"-- every bright spot is one of the two",
                 color=_INK, fontsize=11, loc="left", pad=10)
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.92, facecolor=_SURFACE)
    for t_ in leg.get_texts():
        t_.set_color(_INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)


def plot_summary(result, sweep, out_path):
    """The two shapes the method is read off: the EDT valley and the plane periodicity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres, counts, valley = result["edt_hist"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), facecolor=_SURFACE)

    ax = axes[0]
    _style(ax)
    ax.bar(centres, counts, width=centres[1] - centres[0], color=_BLUE, edgecolor="none")
    ax.axvline(result["r_thr"], color=_RED, linewidth=1.6)
    ax.annotate(f"r_thr = {result['r_thr']:.2f} vox, read off the valley",
                xy=(result["r_thr"], 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -12), textcoords="offset points", color=_RED, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Pixel EDT over the node planes: struts on the left, nodes on the right",
                 color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("inscribed radius (vox)", color=_INK2, fontsize=8)
    ax.set_ylabel("pixels", color=_INK2, fontsize=8)

    ax = axes[1]
    _style(ax)
    n = result["n_blob_per_z"]
    ax.plot(np.arange(len(n)), n, color=_BLUE, linewidth=0.9)
    for res in result["planes"]:
        ax.axvline(res["z"], color=_GREEN, linewidth=0.7, alpha=0.7)
    for z, _ in result["rejected"]:
        ax.axvline(z, color=_ORANGE, linewidth=0.9, linestyle="--")
    ax.set_title(f"Node blobs per slice: the planes are the square wave "
                 f"({len(result['planes'])} kept, green; {len(result['rejected'])} "
                 f"rejected, orange)", color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("z (voxels)", color=_INK2, fontsize=8)
    ax.set_ylabel("blobs", color=_INK2, fontsize=8)

    ax = axes[2]
    _style(ax)
    r = [s[0] for s in sweep]
    ax.plot(r, [s[2] for s in sweep], color=_RED, marker="o", linewidth=1.4,
            label="empty interior sites")
    ax.plot(r, [s[1] for s in sweep], color=_INK2, marker=".", linewidth=1.0,
            linestyle="--", label="interior sites tested")
    ax.axvline(result["r_thr"], color=_RED, linewidth=1.0, alpha=0.5)
    ax.set_yscale("symlog")
    ax.set_title("Threshold sweep: how the answer moves with r_thr",
                 color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("r_thr (vox)", color=_INK2, fontsize=8)
    ax.set_ylabel("count", color=_INK2, fontsize=8)
    leg = ax.legend(fontsize=8, framealpha=0.9, facecolor=_SURFACE)
    for t in leg.get_texts():
        t.set_color(_INK2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", default=DEF_MASK)
    p.add_argument("--raw", default=DEF_RAW, help="grayscale CT, for the figures")
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--r-thr", type=float, default=None,
                   help="inscribed radius that defines a node blob; default reads the "
                        "EDT histogram valley")
    p.add_argument("--match-frac", type=float, default=0.25,
                   help="a site is answered by a blob within this fraction of the pitch")
    p.add_argument("--plane", type=int, default=None,
                   help="render only the plane nearest this z")
    p.add_argument("--design", default=DEF_DESIGN)
    p.add_argument("--correction", default=DEF_CORR)
    p.add_argument("--graph", default="outputs/skan_graph/graph.json")
    p.add_argument("--no-cross-check", action="store_true")
    p.add_argument("--explain-z", type=int, default=None,
                   help="also label every bright spot in this one slice as node or strut "
                        "cross-section; use it on a slice that looks full of holes")
    p.add_argument("--sweep", default="4.0,4.5,5.0,5.5,6.0,6.5")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"reading {args.mask}")
    mask = tifffile.imread(args.mask) > 0
    raw = tifffile.memmap(args.raw) if Path(args.raw).is_file() else None

    result = N.run(mask, r_thr=args.r_thr, match_frac=args.match_frac)
    planes = result["planes"]
    if not planes:
        print("no usable node planes"); return 1

    # -------------------------------------------------------------- detection summary
    rows, n_int_tot, n_miss_tot = [], 0, 0
    print("\n  z     blobs  sites  interior  edge   MISSING(int)  MISSING(edge)")
    for res in planes:
        interior = ~res["edge"]
        miss_int = interior & ~res["found"]
        n_int_tot += int(interior.sum())
        n_miss_tot += int(miss_int.sum())
        print(f"{res['z']:5d}  {len(res['centroids']):5d}  {len(res['sites_xy']):5d}  "
              f"{int(interior.sum()):8d}  {int(res['edge'].sum()):4d}  "
              f"{int(miss_int.sum()):12d}  {int((res['edge'] & ~res['found']).sum()):13d}")
        for s in np.flatnonzero(~res["found"]):
            rows.append({
                "z": res["z"], "y": float(res["sites_xy"][s, 0]),
                "x": float(res["sites_xy"][s, 1]),
                "ring": "edge" if res["edge"][s] else "interior",
                "nearest_blob_vox": float(res["dist"][s]),
            })
    print(f"\ninterior sites tested {n_int_tot}, empty {n_miss_tot} "
          f"({100 * n_miss_tot / max(n_int_tot, 1):.3f}%)")

    # ----------------------------------------------- absent vs undersized, per candidate
    cand = [r for r in rows if r["ring"] == "interior"]
    for r in cand:
        r.update(site_measurements(mask, raw, r["z"], r["y"], r["x"]))

    # -------------------------------------------------------------------- cross-checks
    if not args.no_cross_check and cand:
        from scipy.spatial import cKDTree
        try:
            from src import lattice_iou as L
            pos, pairs, _, corrected = L.load_design(
                args.design, args.correction if Path(args.correction).is_file() else None)
            npos, node_of, deg = L.dedupe_junctions(pos, pairs)
            geom = L.lattice_geometry(pos, pairs)
            fill = L.measure_node_sphere_fill(
                mask, npos, 1.467 * geom["nominal_strut_radius_vox"], log=lambda *_: None)
            tree = cKDTree(npos)
            for r in cand:
                d, i = tree.query([r["z"], r["y"], r["x"]])
                r.update(design_node=int(i), design_dist_vox=float(d),
                         design_degree=int(deg[i]), sphere_fill_3d=float(fill[i]))
            print(f"\ncross-check vs the design-referenced 3D test "
                  f"(registration applied: {corrected})")
        except (OSError, ValueError, KeyError) as err:
            print(f"\ndesign cross-check unavailable: {err}")
        try:
            g = json.loads(Path(args.graph).read_text())
            J = np.array([j["position"] for j in g["junctions"]], float)[:, ::-1]
            jd = np.array([j["degree"] for j in g["junctions"]])
            tj = cKDTree(J)
            for r in cand:
                d, i = tj.query([r["z"], r["y"], r["x"]])
                r.update(skan_dist_vox=float(d), skan_degree=int(jd[i]))
        except (OSError, ValueError, KeyError) as err:
            print(f"skeleton cross-check unavailable: {err}")

    umv = None
    try:
        from src import lattice_iou as L
        pos, pairs, _, _ = L.load_design(args.design, None)
        umv = L.lattice_geometry(pos, pairs)["um_per_voxel"]
    except (OSError, ValueError, KeyError):
        pass

    if cand:
        med_r = np.median([res["site_radius"][res["found"] & ~res["edge"]].mean()
                           for res in planes])
        print("\nEMPTY INTERIOR SITES -- absent or undersized?")
        for r in cand:
            um = f" = {r['inscribed_radius_vox'] * 2 * umv:.0f} um" if umv else ""
            verdict = ("ABSENT" if r["material_frac"] < 0.02 else
                       "UNDERSIZED (material present)")
            print(f"  z={r['z']:4d} y={r['y']:6.0f} x={r['x']:6.0f}  {verdict}")
            print(f"       mask material in 17^3 box {r['material_frac']:.3f}   "
                  f"3D inscribed radius {r['inscribed_radius_vox']:.2f} vox{um}")
            extra = []
            if "sphere_fill_3d" in r:
                extra.append(f"3D sphere fill {r['sphere_fill_3d']:.3f} "
                             f"(design node {r['design_node']}, "
                             f"{r['design_dist_vox']:.1f} vox away)")
            if "skan_degree" in r:
                extra.append(f"skeleton degree {r['skan_degree']} at "
                             f"{r['skan_dist_vox']:.1f} vox")
            if extra:
                print("       " + "   ".join(extra))
        print(f"  (mean blob radius of a found interior node: {med_r:.2f} vox)")

    # -------------------------------------------------------------------------- outputs
    keys = ["z", "y", "x", "ring", "nearest_blob_vox", "material_frac",
            "inscribed_radius_vox", "raw_ct_mean", "design_node", "design_dist_vox",
            "design_degree", "sphere_fill_3d", "skan_dist_vox", "skan_degree"]
    with (out / "empty_sites.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    summary = {
        "r_thr_vox": result["r_thr"],
        "match_frac": args.match_frac,
        "planes_used": [res["z"] for res in planes],
        "planes_rejected": [{"z": z, "reason": w} for z, w in result["rejected"]],
        "pitch_vox_median": float(np.median([res["pitch_vox"] for res in planes])),
        "basis_angle_deg_median": float(np.median([res["basis_angle_deg"]
                                                   for res in planes])),
        "interior_sites": n_int_tot,
        "interior_empty": n_miss_tot,
        "interior_empty_pct": 100 * n_miss_tot / max(n_int_tot, 1),
        "edge_empty": sum(1 for r in rows if r["ring"] == "edge"),
        "candidates": cand,
    }
    if umv:
        summary["um_per_voxel"] = umv
        summary["pitch_um_median"] = summary["pitch_vox_median"] * umv
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    sweep = N.threshold_sweep(mask, [res["z"] for res in planes],
                              [float(v) for v in args.sweep.split(",")],
                              args.match_frac)
    print("\nthreshold sweep (r_thr, interior sites, empty):")
    for r, ni, nm in sweep:
        print(f"  {r:4.1f}  {ni:6d}  {nm:4d}")
    summary["sweep"] = sweep
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    plot_summary(result, sweep, out / "method.png")
    todo = planes
    if args.plane is not None:
        todo = [min(planes, key=lambda r: abs(r["z"] - args.plane))]
    for res in todo:
        n_miss = int((~res["edge"] & ~res["found"]).sum())
        plot_plane(raw[res["z"]] if raw is not None else mask[res["z"]], res,
                   out / f"plane_z{res['z']:04d}.png", um_per_vox=umv,
                   title_extra=f"   {n_miss} interior site(s) empty" if n_miss else "")
    if args.explain_z is not None and raw is not None:
        try:
            from src import lattice_iou as L
            pos, pairs, _, _ = L.load_design(
                args.design, args.correction if Path(args.correction).is_file() else None)
            npos, _, _ = L.dedupe_junctions(pos, pairs)
            label = None
            cls_path = Path("outputs/lattice_iou/strut_classes.csv")
            if cls_path.is_file():
                cls = np.genfromtxt(cls_path, delimiter=",", names=True, dtype=None,
                                    encoding=None)
                label = np.full(len(pairs), "unknown", dtype=object)
                label[:len(cls["label"])] = cls["label"]
            plot_explain(raw[args.explain_z], args.explain_z, (pos, pairs, npos, label),
                         out / f"explain_z{args.explain_z:04d}.png", um_per_vox=umv)
            print(f"wrote {out}/explain_z{args.explain_z:04d}.png")
        except (OSError, ValueError, KeyError) as err:
            print(f"could not build the explain figure: {err}")

    print(f"\nwrote {out}/summary.json, empty_sites.csv, method.png and "
          f"{len(todo)} plane figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
