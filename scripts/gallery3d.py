#!/usr/bin/env python
"""Build the 3-D strut-class and missing-node galleries from a finished detector run.

Nothing here detects anything. The labels come from `strut_classes.csv`, the metrics from
`struts.csv` / `sections.npz` / `connectivity.npz` / `nodes.csv`, all written by
`detect_lattice_defects` with the registration correction applied; this script picks one
exemplar per class and draws the material behind it with `scripts.render3d`.

    python scripts/gallery3d.py --struts
    python scripts/gallery3d.py --nodes
    python scripts/gallery3d.py --struts --classes thin,thick --out outputs/g3d

Exemplars are **ranked on the quantity that defines the class** -- the thinnest `thin`,
the deepest pinch for `necked`, the most isolated `missing` -- and only then filtered for
legibility: clear of the build plates, cross-sections not running into their window, both
junctions printed. Ranking first and filtering second is the honest order. The gallery
this replaces drew a uniform random pick per class, which is defensible but shows the
median case for classes whose whole content is the extreme; picking by eye from the whole
population would be the dishonest version of the same move and is not what this does.

The renders are display objects. `render3d` surfaces the CT at the segmentation threshold
and smooths the result, so a surface here is not a measurement of thickness and the
caption on the figure says so. Every number printed beside a row is read from the cached
run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render3d as R  # noqa: E402
from scripts.classify_strut_defects import CLASS_COLORS, ORDER, measurable  # noqa: E402
from src.lattice_iou import (_GRID, _INK, _INK2, _SURFACE, dedupe_junctions,  # noqa: E402
                             lattice_geometry, load_design)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / ("data/missing_struts/registered_jsons/"
                 "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
CT = ROOT / ("data/missing_struts/tif_stacks/"
             "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif")
MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/mask.tif"

# Solid build plates cap both z ends and taper obliquely into the part. A strut inside one
# leaves no signature at all, so no exemplar is drawn from within this margin.
PLATE_MARGIN = 110


# ----------------------------------------------------------------------- run

class Run:
    """A finished detector run, loaded once. Read-only."""

    def __init__(self, directory, correction):
        self.dir = Path(directory)
        self.pos, self.pairs, _, self.corrected = load_design(
            str(DESIGN), str(correction) if correction else None)
        if not self.corrected:
            print("WARNING: no registration correction applied -- every window is cut "
                  "about the design axis and will sit off-centre. See AGENTS.md.")
        self.geo = lattice_geometry(self.pos, self.pairs, cell_mm=4.56,
                                    strut_diameter_um=350.0)
        summary = self.dir / "summary.json"
        if summary.is_file():
            self.geo.update(json.loads(summary.read_text()).get("geometry", {}))

        self.d = np.genfromtxt(self.dir / "struts.csv", delimiter=",", names=True)
        z = np.load(self.dir / "sections.npz", allow_pickle=False)
        self.sec = {k[4:]: z[k] for k in z.files if k.startswith("sec_")}
        self.n_sections = int(z["prof_r_eq"].shape[1])
        self.cls = np.genfromtxt(self.dir / "strut_classes.csv", delimiter=",",
                                 names=True, dtype=None, encoding=None)
        self.label = np.asarray(self.cls["label"], str)
        cpath = self.dir / "connectivity.npz"
        self.conn = dict(np.load(cpath, allow_pickle=False)) if cpath.is_file() else {}
        npath = self.dir / "nodes.csv"
        self.nodes = (np.genfromtxt(npath, delimiter=",", names=True)
                      if npath.is_file() else None)
        self.node_pos, self.node_of, self.degree = dedupe_junctions(self.pos, self.pairs)

    @property
    def um(self):
        return self.geo["um_per_voxel"]

    @property
    def r_nom(self):
        return self.geo["nominal_strut_radius_vox"]

    def ends(self, sid):
        return self.pos[self.pairs[sid, 0]], self.pos[self.pairs[sid, 1]]

    def col(self, name):
        """A per-strut column padded with NaN to the design's full strut count.

        The three tables disagree on length -- sections.npz covers every strut, the CSVs
        are written per measured strut -- so everything is indexed by strut_id throughout.
        """
        for src in (self.d.dtype.names or (), self.sec, self.conn,
                    self.cls.dtype.names or ()):
            if name in src:
                table = self.d if src is self.d.dtype.names else (
                    self.cls if src is (self.cls.dtype.names or ()) else src)
                v = np.asarray(table[name], float)
                break
        else:
            raise KeyError(name)
        n = len(self.pairs)
        if len(v) < n:
            v = np.concatenate([v, np.full(n - len(v), np.nan)])
        return v[:n]

    def labels_full(self):
        out = np.full(len(self.pairs), "", object)
        out[:len(self.label)] = self.label
        return out

    def node_fill(self):
        fill = np.full(len(self.node_pos), np.nan)
        if self.nodes is not None:
            fill[self.nodes["node_id"].astype(int)] = self.nodes["fill"]
        return fill

    def base(self):
        """Measurable, clear of the plates, and joining two junctions that printed."""
        n = len(self.label)
        ok = np.zeros(len(self.pairs), bool)
        ok[:n] = measurable(self.d, self.sec)[:n]
        mid = 0.5 * (self.pos[self.pairs[:, 0]] + self.pos[self.pairs[:, 1]])
        shape = np.array([761, 815, 837])
        clear = np.all((mid > PLATE_MARGIN) & (mid < shape - PLATE_MARGIN), axis=1)
        fill = self.node_fill()
        nn = self.node_of[self.pairs]
        printed = (np.nan_to_num(fill[nn[:, 0]], nan=1.0) > 0.5) & \
                  (np.nan_to_num(fill[nn[:, 1]], nan=1.0) > 0.5)
        return ok & clear & printed


def pick_exemplars(run, classes):
    """One strut per class, ranked on the quantity that defines it."""
    lab = run.labels_full()
    b = run.base()
    clean = b & (np.nan_to_num(run.col("border_frac"), nan=1.0) <= 0.0)
    r_med, r_min = run.col("r_eq_med_um"), run.col("r_eq_min_um")
    typical = np.nanmedian(r_med[b & (lab == "nominal")])

    score = {
        # Most isolated first: an absent strut reads as absent only when the space it
        # should occupy is otherwise empty, so rank on how little material the envelope
        # around the nominal cylinder holds.
        "missing": (b & (lab == "missing"), -run.col("env_material_frac")),
        # The break has to be inside the sectioned span or the figure shows an intact
        # strut with a caption claiming otherwise. `broken` itself is decided by
        # connectivity, which has no threshold; gap_len only orders the candidates.
        "broken": (b & (lab == "broken") & (run.col("gap_interior") > 0.5),
                   run.col("gap_len")),
        "thin": (clean & (lab == "thin"), -r_med),
        "thick": (clean & (lab == "thick"), r_med),
        # A neck is a local pinch on an otherwise normal strut, so rank on the depth of
        # the pinch rather than on either radius alone.
        "necked": (clean & (lab == "necked"), r_med - r_min),
        "nominal": (clean & (lab == "nominal"), -np.abs(r_med - typical)),
    }
    picks = []
    for name in ORDER:
        if name not in classes:
            continue
        sel, s = score[name]
        # `broken` with the break in frame may be empty on some runs; fall back to any.
        if not sel.any() and name == "broken":
            sel, s = b & (lab == "broken"), run.col("gap_len")
        idx = np.flatnonzero(sel)
        if not idx.size:
            print(f"  no exemplar available for {name}")
            continue
        picks.append((name, int(idx[np.argmax(np.nan_to_num(s[idx], nan=-np.inf))])))
    return picks


# --------------------------------------------------------------------- figures

def _imshow(ax, img, extent=None, **kw):
    ax.imshow(img, extent=extent, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(_GRID)
        sp.set_linewidth(0.7)


def _fmt(v, nd=1, unit=""):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}{unit}"


def strut_gallery(run, picks, out_path, ct, mask, width_px=1400, n_sections=8,
                  half_r=3.2, res=0.22, trim=0.20):
    """Two sub-rows per class: lateral and cutaway above, discs and flat sections below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for name, sid in picks:
        p0, p1 = run.ends(sid)
        print(f"  rendering {name} #{sid} ...", flush=True)
        frame = R.strut_frame(ct, mask, p0, p1, run.r_nom, half_r=half_r, res=res)
        mesh = R.isosurface(frame.field, frame.origin, frame.res)
        lat = R.render_lateral(frame, mesh, width=width_px, trim=trim)
        cut = R.render_cutaway(frame, mesh, width=width_px, trim=trim)
        dsk = R.render_disks(frame, width=width_px, n=n_sections, trim=trim)
        greys, gates, ext = R.section_images(frame, n=n_sections, trim=trim)
        panels.append((name, sid, lat, cut, dsk, greys, gates, ext, frame))
        del frame, mesh

    text_w, panel_w = 2.5, 7.6
    fig_w = text_w + 2 * panel_w
    heights = []
    for p in panels:
        heights += [panel_w * p[2].shape[0] / p[2].shape[1],
                    panel_w * p[4].shape[0] / p[4].shape[1]]
    fig_h = sum(heights) + 0.95
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_SURFACE)
    gs = fig.add_gridspec(2 * len(panels), 3, width_ratios=[text_w, panel_w, panel_w],
                          height_ratios=heights, left=0.001, right=0.998,
                          top=1 - 0.85 / fig_h, bottom=0.004, wspace=0.012, hspace=0.16)

    for row, (name, sid, lat, cut, dsk, greys, gates, ext, frame) in enumerate(panels):
        _imshow(fig.add_subplot(gs[2 * row, 1]), lat)
        _imshow(fig.add_subplot(gs[2 * row, 2]), cut)
        _imshow(fig.add_subplot(gs[2 * row + 1, 1]), dsk)

        # the flat sections, as one strip: CT greys with the mask boundary on top
        sub = gs[2 * row + 1, 2].subgridspec(1, len(greys), wspace=0.04)
        for k in range(len(greys)):
            ax = fig.add_subplot(sub[0, k])
            inside = gates[k]
            _imshow(ax, greys[k], extent=ext, origin="lower", cmap="Greys_r",
                    vmin=R.CT_WINDOW[0], vmax=R.CT_WINDOW[1], interpolation="bilinear")
            if inside.any():
                ax.contour(np.linspace(ext[0], ext[1], inside.shape[1]),
                           np.linspace(ext[2], ext[3], inside.shape[0]),
                           greys[k].astype(float), levels=[R.ISOLEVEL],
                           colors=["#39d1a0"], linewidths=0.8)
            ax.add_patch(plt.Circle((0, 0), run.r_nom, fill=False, color=R.GHOST, lw=1.0))
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            if row == 0 and k == len(greys) // 2:
                ax.set_title("cross-sections, head-on", color=_INK2, fontsize=7.5, pad=3)

        box = fig.add_subplot(gs[2 * row:2 * row + 2, 0])
        box.axis("off")
        um = run.um
        reach = run.conn.get("reachable")
        conn = ""
        if reach is not None and sid < len(reach):
            conn = ("no node-to-node path" if not reach[sid]
                    else f"detour {run.conn['detour'][sid]:.3f}x")
        box.text(0.02, 0.80, f"{name}  #{sid}", color=CLASS_COLORS[name], fontsize=13,
                 fontweight="bold", transform=box.transAxes, va="center")
        box.text(0.02, 0.55,
                 f"median {run.col('r_eq_med_um')[sid]:.0f} um\n"
                 f"min {run.col('r_eq_min_um')[sid]:.0f} um\n"
                 f"empty {int(run.col('empty_sections')[sid])}/{run.n_sections}\n"
                 f"gap {int(run.col('gap_len')[sid])}\n"
                 f"{conn}",
                 color=_INK2, fontsize=8.5, transform=box.transAxes, va="center",
                 linespacing=1.5)
        box.text(0.02, 0.17,
                 "top   side view  |  opened along its axis\n"
                 "below   sections as discs  |  head-on",
                 color=_INK2, fontsize=7.5, transform=box.transAxes, va="center",
                 linespacing=1.6)

    fig.suptitle(
        "One strut per class, surfaced from the CT at the segmentation threshold "
        f"({R.ISOLEVEL:.0f}), masked to the segmented part.\n"
        f"red = nominal {run.geo.get('nominal_strut_diameter_um', 350):.0f} um cylinder "
        f"over the measured span    green = the CT threshold contour    "
        "greys on the cut face and the sections are raw CT.    "
        "Surfaces are smoothed for display; thickness is read from sections.npz, "
        "never off a render.",
        color=_INK, fontsize=10, x=0.004, ha="left", y=0.997, va="top", linespacing=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def pick_nodes(run):
    """The cases a missing-node figure has to separate, one row each."""
    if run.nodes is None:
        raise SystemExit("nodes.csv not found in the run directory")
    fill = run.node_fill()
    interior = run.degree == 12
    rows = []

    empty_in = np.flatnonzero(interior & (fill < 0.01))
    for k in empty_in[:2]:
        rows.append(("missing, interior", int(k)))

    # A whole flat face of empties is the design reaching past the printed part, not a
    # build failure, and the two readings differ by ~65x on this specimen. The figure has
    # to show one so the reader can tell them apart by eye rather than by trusting a note.
    # Take the highest-degree one clear of the build plates: a degree-3 corner node is
    # empty for a third reason again and would muddle the comparison it is here to make.
    empty_out = np.flatnonzero(~interior & (fill < 0.01))
    if empty_out.size:
        pts = run.node_pos[empty_out]
        axis = int(np.argmax([np.sum(np.abs(pts[:, a] - np.median(pts[:, a])) < 20)
                              for a in range(3)]))
        on_face = empty_out[np.abs(pts[:, axis] - np.median(pts[:, axis])) < 20]
        clear = on_face[(run.node_pos[on_face][:, 0] > PLATE_MARGIN)
                        & (run.node_pos[on_face][:, 0] < 761 - PLATE_MARGIN)]
        pool = clear if clear.size else on_face
        if pool.size:
            rows.append((f"empty, but on the {'zyx'[axis]} face",
                         int(pool[np.argmax(run.degree[pool])])))

    live = interior & (fill >= 0.01)
    if run.nodes is not None and "diameter_um" in (run.nodes.dtype.names or ()):
        dia = np.full(len(run.node_pos), np.nan)
        dia[run.nodes["node_id"].astype(int)] = run.nodes["diameter_um"]
        cand = np.flatnonzero(live & np.isfinite(dia))
        if cand.size:
            rows.append(("smallest junction that printed", int(cand[np.argmin(dia[cand])])))
    ok = np.flatnonzero(live)
    if ok.size:
        rows.append(("typical", int(ok[np.argsort(fill[ok])[len(ok) // 2]])))
    return rows


def node_gallery(run, rows, out_path, ct, mask, radius_factor=1.467, res_wide=0.5,
                 res_close=0.25, close_half=20.0, width_px=760):
    """Per node: the cage around it, the junction close up, and the three centre planes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_sphere = radius_factor * run.r_nom
    wide_half = float(run.geo["strut_length_vox"]) * 1.05
    fill = run.node_fill()

    panels = []
    for name, nid in rows:
        p = run.node_pos[nid]
        print(f"  rendering node {nid} ({name}) ...", flush=True)
        gw, _, ow = R.node_box(ct, p, wide_half, res_wide)
        wide = R.render_node(gw, ow, r_sphere=r_sphere, width=width_px, azim=34, elev=22,
                             open_to_camera=True, inset_half=close_half)
        del gw
        gc, rawc, oc = R.node_box(ct, p, close_half, res_close, mask=mask)
        inc = np.where((run.node_of[run.pairs[:, 0]] == nid)
                       | (run.node_of[run.pairs[:, 1]] == nid))[0]
        far = np.where(run.node_of[run.pairs[inc, 0]] == nid,
                       run.pairs[inc, 1], run.pairs[inc, 0])
        close = R.render_node(gc, oc, r_sphere=r_sphere, spokes=run.pos[far] - p,
                              width=width_px, azim=34, elev=22)
        planes, ext = R.node_planes(rawc, oc)
        panels.append((name, nid, wide, close, planes, ext))
        del gc, rawc

    text_w, cell = 2.3, 2.3
    fig_w = text_w + 5 * cell
    fig_h = cell * len(panels) + 1.35
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_SURFACE)
    gs = fig.add_gridspec(len(panels), 6, width_ratios=[text_w] + [cell] * 5,
                          left=0.002, right=0.998, top=1 - 1.3 / fig_h, bottom=0.006,
                          wspace=0.03, hspace=0.05)
    # Every panel says its own half-extent and which volume axis runs which way. The three
    # flat cuts pair the axes differently from each other -- +y is up in the axial panel
    # and right in the sagittal one -- so without the labels a strut cannot be followed
    # from one panel to the next, and the row looks misaligned when it is not.
    titles = (f"+-1 strut length ({wide_half * run.um / 1000:.1f} mm)",
              f"the junction (+-{close_half * run.um / 1000:.2f} mm)",
              "axial   x right, y up", "coronal   x right, z up", "sagittal   y right, z up")

    for row, (name, nid, wide, close, planes, ext) in enumerate(panels):
        cells = [wide, close]
        for k, img in enumerate(cells):
            ax = fig.add_subplot(gs[row, 1 + k])
            _imshow(ax, img)
            if row == 0:
                ax.set_title(titles[k], color=_INK2, fontsize=7.5, pad=3)
        for k, pl in enumerate(planes):
            ax = fig.add_subplot(gs[row, 3 + k])
            _imshow(ax, pl, extent=ext, origin="lower", cmap="Greys_r",
                    vmin=R.CT_WINDOW[0], vmax=R.CT_WINDOW[1], interpolation="bilinear")
            ax.add_patch(plt.Circle((0, 0), r_sphere, fill=False, color=R.GHOST, lw=1.0))
            if row == 0:
                ax.set_title(titles[2 + k], color=_INK2, fontsize=7.5, pad=3)

        box = fig.add_subplot(gs[row, 0])
        box.axis("off")
        p = run.node_pos[nid]
        col = R.GHOST if fill[nid] < 0.5 else _INK
        box.text(0.03, 0.74, f"{name}", color=col, fontsize=10.5, fontweight="bold",
                 transform=box.transAxes, va="center")
        box.text(0.03, 0.40,
                 f"node {nid}   degree {int(run.degree[nid])}\n"
                 f"sphere fill {_fmt(fill[nid], 3)}\n"
                 f"z={p[0]:.0f}  y={p[1]:.0f}  x={p[2]:.0f}",
                 color=_INK2, fontsize=8, transform=box.transAxes, va="center",
                 linespacing=1.6)

    fig.suptitle(
        f"Junctions, surfaced from the CT at the segmentation threshold "
        f"({R.ISOLEVEL:.0f}).   red = the {r_sphere * run.um:.0f} um sphere the fill "
        "statistic is measured in,   blue = the design struts that should meet here.\n"
        "On a junction that printed the spokes disappear inside the metal; on one that "
        "did not they hang in the void. The wide view is cut open towards the camera, "
        "because a lattice is opaque at one strut length.\n"
        "Both 3-D panels are one oblique view of the volume axes (corner triad) at two "
        "zooms; the grey box in the wide view is the extent of the panel beside it.\n"
        f"The three flat cuts are that same +-{close_half:.0f} vox box, on one fixed CT "
        f"window ({R.CT_WINDOW[0]:.0f}-{R.CT_WINDOW[1]:.0f}), so an empty panel is empty "
        "rather than autoscaled.",
        color=_INK, fontsize=9, x=0.004, ha="left", y=0.997, va="top", linespacing=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="outputs/lattice_iou",
                   help="a finished detect_lattice_defects directory")
    p.add_argument("--correction", default="outputs/registration/correction.json")
    p.add_argument("--out", default="outputs/gallery3d")
    p.add_argument("--ct", default=str(CT))
    p.add_argument("--mask", default=str(MASK))
    p.add_argument("--struts", action="store_true")
    p.add_argument("--nodes", action="store_true")
    p.add_argument("--classes", default=",".join(ORDER))
    p.add_argument("--strut-ids", default="",
                   help="explicit strut ids to draw instead of the ranked exemplars")
    p.add_argument("--node-ids", default="")
    p.add_argument("--sections", type=int, default=8)
    p.add_argument("--res", type=float, default=0.22,
                   help="resample step in voxels for the strut frame")
    p.add_argument("--width", type=int, default=1400, help="render width in pixels")
    a = p.parse_args()
    if not (a.struts or a.nodes):
        a.struts = a.nodes = True

    import tifffile
    ct = tifffile.memmap(a.ct)
    mask = tifffile.memmap(a.mask)
    run = Run(a.dir, a.correction)
    out = Path(a.out)
    print(f"run {a.dir}   correction applied: {run.corrected}   "
          f"r_nom {run.r_nom:.2f} vox   {run.um:.2f} um/vox")

    if a.struts:
        if a.strut_ids.strip():
            lab = run.labels_full()
            picks = [(str(lab[int(s)]) or "unlabelled", int(s))
                     for s in a.strut_ids.replace(" ", "").split(",") if s]
        else:
            want = {c.strip() for c in a.classes.split(",") if c.strip()}
            unknown = want - set(ORDER)
            if unknown:
                raise SystemExit(f"unknown class(es) {sorted(unknown)}; "
                                 f"known: {', '.join(ORDER)}")
            picks = pick_exemplars(run, want)
        for name, sid in picks:
            print(f"  {name:8s} #{sid}")
        strut_gallery(run, picks, out / "strut_classes_3d.png", ct, mask,
                      width_px=a.width, n_sections=a.sections, res=a.res)

    if a.nodes:
        if a.node_ids.strip():
            rows = [("requested", int(s))
                    for s in a.node_ids.replace(" ", "").split(",") if s]
        else:
            rows = pick_nodes(run)
        for name, nid in rows:
            print(f"  node {nid}: {name}")
        node_gallery(run, rows, out / "missing_nodes_3d.png", ct, mask)


if __name__ == "__main__":
    main()
