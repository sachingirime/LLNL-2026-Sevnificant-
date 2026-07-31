#!/usr/bin/env python
"""As-built metal as a surface, with the classified struts inside it (one WebGL .html).

The line viewers show where the detector says the defects are. They cannot show whether a
strut called `missing` really has no metal at that location or whether the design line is
sitting a couple of voxels off it — and that distinction is defect versus registration
drift. This draws the segmentation as a translucent isosurface and the classified design
struts as coloured lines in the SAME voxel frame, so the two are checkable against each
other. Opacity and an x-clip slider are in the page; a lattice is opaque at one strut
length, so the clip is how you get inside it.

`scripts/overlay_webgl.py` already does this, but its `verdict` path carries a fixed
vocabulary — `disconnected`, `dross`, `bent` — with no `broken`, `thick` or `necked`.
Feeding it this detector's labels would mean renaming `thick` to `dross`, so its
isosurface extractor and viewer shell are reused here and only the class table is new.

Nothing is computed. Labels come from the run's `strut_classes.csv`.

    python scripts/dashboard/overlay3d.py --run outputs/lattice_iou -o out.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes3d import DESIGN, NODE_CROSS_VOX, load_labels, missing_nodes  # noqa: E402
from overlay_webgl import HTML, mask_surface  # noqa: E402
from scripts.classify_strut_defects import CLASS_COLORS, ORDER  # noqa: E402
from src.lattice_iou import load_design  # noqa: E402

SUPPLIED_MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/mask.tif"
NODE_INTERIOR = "#ffffff"
NODE_SURFACE = "#898781"
RULES = {
    "missing": "no matched voxel and every section empty",
    "broken":  "no path node-to-node through material",
    "thin":    "median section radius below the band",
    "thick":   "median section radius above the band",
    "necked":  "a local pinch under a normal median",
    "nominal": "everything the rules above did not claim",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="results directory with strut_classes.csv")
    p.add_argument("--mask", default="")
    p.add_argument("--design", default=str(DESIGN))
    p.add_argument("--correction", default="")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--max-dim", type=int, default=256,
                   help="longest mask axis after block-reduction; drives triangle count")
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--mask-alpha", type=float, default=0.16)
    p.add_argument("--exclude-caps", action="store_true",
                   help="drop boundary-cap struts; totals then stop matching the "
                        "counts reported elsewhere")
    p.add_argument("--fill-cut", type=float, default=0.194)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    mask = Path(args.mask) if args.mask else None
    if mask is None:
        for candidate in (run_dir / "mask.tif", run_dir.parent / "mask.tif", SUPPLIED_MASK):
            if candidate.is_file():
                mask = candidate
                break
    correction = Path(args.correction) if args.correction else None
    if correction is None:
        for candidate in (run_dir.parent / "registration" / "correction.json",
                          ROOT / "outputs/registration/correction.json"):
            if candidate.is_file():
                correction = candidate
                break

    print(f"mask       {mask}")
    print(f"correction {correction}")
    print("extracting isosurface (marching cubes on the block-reduced mask)…", flush=True)
    mverts, mfaces, factor = mask_surface(str(mask), args.max_dim, args.level)
    print(f"  {len(mfaces):,} triangles at reduction factor {factor}")

    pos, pairs, edge, corrected = load_design(str(args.design),
                                              str(correction) if correction else None)
    labels = load_labels(run_dir)
    label_of = np.asarray([labels.get(i, "") for i in range(len(pairs))], object)

    values, counts = np.unique(edge, return_counts=True)
    caps = set(values[counts < counts.max() * 0.5].tolist())
    is_cap = np.asarray([e in caps for e in edge])
    # Caps IN by default, so these totals match the counts reported everywhere else.
    keep = ~is_cap if args.exclude_caps else np.ones(len(pairs), bool)

    node_in, node_sf = missing_nodes(run_dir, args.fill_cut)

    # ONE shared centre for every geometry or nothing overlays. The mask verts are in
    # original voxel coordinates, so the design (also voxels, corrected) lines up.
    centre = (mverts.max(0) + mverts.min(0)) / 2.0
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]

    segments, meta_cls, cursor = [], [], 0
    n_keep = int(keep.sum())
    # Core and boundary cap as separate layers — see classes3d.py for why. It matters more
    # here: with the metal surface drawn, switching the cap layer on and off shows
    # directly that the cap shell tracks the outside of the part.
    totals = {}
    for name in ORDER:
        base = keep & (label_of == name)
        totals[name] = int(base.sum())
        hexcol = CLASS_COLORS.get(name, "#888888")
        recede = name == "nominal"
        for label, sel, alpha, extra in (
                (name, base & ~is_cap, 0.08 if recede else 0.95, ""),
                (f"{name} · cap", base & is_cap, 0.05 if recede else 0.55,
                 "  ·  outermost layer; endpoints are surface junctions (degree 3-8, not 12)")):
            if not sel.any():
                continue
            seg = np.stack([a[sel] - centre, b[sel] - centre], 1).astype(np.float32)
            segments.append(seg.reshape(-1, 3))
            meta_cls.append(dict(
                name=label, color=hexcol, rule=RULES.get(name, "") + extra,
                rgb=[int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)],
                start=cursor, count=int(sel.sum()),
                pct=round(100 * sel.sum() / max(1, n_keep), 2),
                on=True, alpha=alpha))
            cursor += int(sel.sum()) * 2

    for pts, colour, label, rule in (
            (node_in, NODE_INTERIOR, "missing node · interior",
             "degree 12, empty sphere — cannot be a surface effect"),
            (node_sf, NODE_SURFACE, "missing node · surface",
             "on the boundary — 181 sit on the single plane y = 759")):
        if not len(pts):
            continue
        arm = np.zeros((len(pts), 3, 2, 3), np.float32)
        centred = (pts - centre).astype(np.float32)
        for axis in range(3):
            delta = np.zeros(3, np.float32)
            delta[axis] = NODE_CROSS_VOX
            arm[:, axis, 0] = centred - delta
            arm[:, axis, 1] = centred + delta
        segments.append(arm.reshape(-1, 3))
        meta_cls.append(dict(
            name=label, color=colour, rule=rule,
            rgb=[int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)],
            start=cursor, count=len(pts) * 3, label_n=len(pts), pct=0.0,
            on=True, alpha=0.95))
        cursor += len(pts) * 3 * 2

    lines = np.concatenate(segments, 0)
    mvc = (mverts - centre).astype(np.float32)
    radius = float(max(np.linalg.norm(mvc, axis=1).max(),
                       np.linalg.norm(lines, axis=1).max()))

    meta = dict(
        lines=base64.b64encode(np.ascontiguousarray(lines, np.float32)).decode(),
        mverts=base64.b64encode(np.ascontiguousarray(mvc, np.float32)).decode(),
        mfaces=base64.b64encode(np.ascontiguousarray(mfaces, np.uint32)).decode(),
        mtris=int(len(mfaces)), malpha=args.mask_alpha, classes=meta_cls,
        radius=radius, zmin=float(mvc[:, 2].min()), zmax=float(mvc[:, 2].max()))

    flagged = sum(c["count"] for c in meta_cls
                  if not c["name"].startswith(("nominal", "missing node")))
    note = (f"surface = the segmentation at level {args.level}, block-reduced {factor}x "
            f"for display only &mdash; it is not a thickness measurement<br>"
            f"{flagged} of {n_keep} struts flagged &middot; labels from "
            f"{run_dir.name}/strut_classes.csv &middot; "
            f"correction {'applied' if corrected else 'NOT applied'}<br>"
            + "caps = the outermost strut layer, same 55.85 vox length as any other; "
        "their endpoints are surface junctions of degree 3-8 rather than 12<br>"
        "totals, core + cap: "
            + " &middot; ".join(f"{k} {totals[k]}" for k in ORDER if totals.get(k)))

    page = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", args.title or f"As-built overlay — {run_dir.name}")
                .replace("__NOTE__", note))
    # Same two fixes the line viewers need, applied to this copy only: the shell hardcodes
    # "material " before every rule (right for its own coverage classes, wrong for "no path
    # node-to-node through material"), and prints the segment count, which for a node cross
    # is three times the node count.
    # NB: this shell names the element `r`, not `rl` as graph_webgl does.
    # This shell sizes its HUD differently from graph_webgl (262px, and it already
    # sets max-height/overflow); without a max-width the long cap rules stretch the
    # panel across the model.
    page = page.replace("min-width:262px;", "min-width:262px; max-width:430px;")
    page = page.replace("r.textContent='material '+c.rule;", "r.textContent=c.rule;")
    page = page.replace(
        '<span class="ct">${c.count} · ${c.pct}%</span>',
        '<span class="ct">${c.label_n ?? c.count}${c.pct ? " · "+c.pct+"%" : ""}</span>')

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    for m in meta_cls:
        print(f"  {m['name']:26s} {m.get('label_n', m['count']):6d}")


if __name__ == "__main__":
    main()
