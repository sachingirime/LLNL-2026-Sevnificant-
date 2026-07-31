#!/usr/bin/env python
"""The detector's own classes as one interactive 3D page (self-contained .html).

The repo already has three WebGL viewers and none of them draws the labels the detector
actually assigned: `lattice_iou_webgl` colours by live IoU sliders, `graph_webgl` by
material-fraction cuts, `overlay_webgl` by a `measure_struts` table these runs do not
write. This reads `strut_classes.csv` and colours each strut by the class it was given —
`missing`, `broken`, `thin`, `thick`, `necked`, `nominal` — so what you rotate is the
classification, not a re-derivation of it.

Nothing is computed here. Labels come from the run, positions from the design JSON with
the run's registration correction applied, colours from
`scripts.classify_strut_defects.CLASS_COLORS`.

Missing junctions ride along as small 3-axis crosses. That is not a shortcut around the
line renderer — a cross IS three line segments, so it needs no second draw path, and a
cross reads unambiguously as "nothing is here" where a sphere would read as material.

    python scripts/dashboard/classes3d.py --run outputs/lattice_iou -o out.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graph_webgl import HTML  # noqa: E402  (reuse the viewer shell)
from scripts.classify_strut_defects import CLASS_COLORS, ORDER  # noqa: E402
from src.lattice_iou import load_design  # noqa: E402

DESIGN = ROOT / ("data/missing_struts/registered_jsons/"
                 "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")

NODE_INTERIOR = "#4a3aa7"       # violet: cannot be a surface effect
NODE_SURFACE  = "#898781"       # grey: on the boundary, likely the design overhanging
NODE_CROSS_VOX = 9.0            # arm half-length, ~1.5 nominal node radii
RULES = {
    "missing": "no matched voxel and every section empty",
    "broken":  "no path node-to-node through material",
    "thin":    "median section radius below the band",
    "thick":   "median section radius above the band",
    "necked":  "a local pinch under a normal median",
    "nominal": "everything the rules above did not claim",
}


def load_labels(run_dir: Path):
    with (run_dir / "strut_classes.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = "label" if rows and "label" in rows[0] else "class"
    labels = {}
    for row in rows:
        try:
            labels[int(row["strut_id"])] = row[key]
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def missing_nodes(run_dir: Path, fill_cut: float):
    """Junctions with no material, split interior vs surface. Returns (interior, surface).

    Both are returned on purpose. `detect_missing_nodes` reports 184 of 3430 as its
    HEADLINE — every junction the design declares — and 2 of 2456 restricted to interior
    (degree 12) sites, and it is explicit that the restricted number is the second one,
    because filtering by degree "can hide the larger count".

    They are not two estimates of one quantity. On this specimen 181 of the 184 lie on
    the single plane y = 759: a whole flat face, which reads as the design reaching past
    the printed part rather than as 181 build failures. The interior sites cannot be a
    surface effect at all. The tool warns the two readings differ by 61x, so drawing only
    one of them is the mistake — drawn as separate togglable classes, the face is visibly
    a face and you can judge it yourself.
    """
    path = run_dir / "nodes.csv"
    if not path.is_file():
        return np.zeros((0, 3)), np.zeros((0, 3))
    interior, surface = [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                if float(row["fill"]) > fill_cut:
                    continue
                point = (float(row["z"]), float(row["y"]), float(row["x"]))
            except (KeyError, TypeError, ValueError):
                continue
            is_interior = str(row.get("is_interior", "1")).strip() not in ("0", "0.00000", "False")
            (interior if is_interior else surface).append(point)
    return (np.asarray(interior, float).reshape(-1, 3),
            np.asarray(surface, float).reshape(-1, 3))


def build(run_dir: Path, design_path: Path, correction: Path | None, out_path: Path,
          include_caps: bool, fill_cut: float, title: str | None):
    pos, pairs, edge, corrected = load_design(str(design_path),
                                              str(correction) if correction else None)
    labels = load_labels(run_dir)

    # A boundary cap is the OUTERMOST layer of struts, not a half-strut: measured on this
    # design every strut is 55.85 vox, cap and core alike. What separates them is the
    # degree of their endpoints — a cap ends on a surface junction where 3, 5 or 8 struts
    # meet, a core strut ends on an interior junction where 12 do. graph_webgl drops them,
    # and inheriting that made this page disagree with every count reported elsewhere:
    # `missing` is 412 over all struts and 89 without caps. Caps are IN by default so the
    # totals match, split into their own layers so the distinction is not lost.
    values, counts = np.unique(edge, return_counts=True)
    caps = set(values[counts < counts.max() * 0.5].tolist())
    is_cap = np.asarray([e in caps for e in edge])
    keep = np.ones(len(pairs), bool) if include_caps else ~is_cap

    label_of = np.asarray([labels.get(i, "") for i in range(len(pairs))], object)
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]

    node_in, node_sf = missing_nodes(run_dir, fill_cut)
    everything = [pos] + [n for n in (node_in, node_sf) if len(n)]
    centre = np.concatenate(everything, 0)
    centre = (centre.max(0) + centre.min(0)) / 2.0

    segments, classes, cursor = [], [], 0
    n_keep = int(keep.sum())

    # Each class becomes up to two layers, core and boundary cap. Folding them together
    # gives the right total but hides that 323 of the 412 `missing` sit on the outer
    # shell — and on THIS specimen all 323 are on the single y-hi face, while the z-hi
    # and x-hi cap layers are fully present. That is the difference between a
    # build-failure rate and a one-face artefact. Splitting keeps the total (the rows
    # sum to it) while letting the cap shell be switched off, or viewed alone.
    totals = {}
    for name in ORDER:
        base = keep & (label_of == name)
        totals[name] = int(base.sum())
        hexcol = CLASS_COLORS.get(name, "#888888")
        recede = name == "nominal"
        for label, sel, alpha, extra in (
                (name, base & ~is_cap, 0.10 if recede else 0.95, ""),
                (f"{name} · cap", base & is_cap, 0.06 if recede else 0.55,
                 "  ·  outermost layer; endpoints are surface junctions (degree 3-8, not 12)")):
            if not sel.any():
                continue
            seg = np.stack([a[sel] - centre, b[sel] - centre], 1).astype(np.float32)
            segments.append(seg.reshape(-1, 3))
            classes.append(dict(
                name=label, color=hexcol,
                rgb=[int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)],
                rule=RULES.get(name, "") + extra,
                start=cursor, count=int(sel.sum()),
                pct=round(100 * sel.sum() / max(1, n_keep), 2),
                on=True, alpha=alpha))
            cursor += int(sel.sum()) * 2

    for pts, colour, label, rule in (
            (node_in, NODE_INTERIOR, "missing nodes / interior",
             "degree 12, empty sphere - cannot be a surface effect"),
            (node_sf, NODE_SURFACE, "missing nodes / surface",
             "on the boundary - mostly one flat face, i.e. design past the printed part")):
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
        classes.append(dict(
            name=label, color=colour,
            rgb=[int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)],
            rule=rule, start=cursor, count=len(pts) * 3,
            label_n=len(pts),  # nodes, not the 3 segments each cross costs
            pct=0.0, on=True, alpha=0.95))
        cursor += len(pts) * 3 * 2

    allseg = np.concatenate(segments, 0) if segments else np.zeros((0, 3), np.float32)
    radius = float(np.linalg.norm(allseg, axis=1).max()) if len(allseg) else 1.0
    meta = dict(pos=base64.b64encode(np.ascontiguousarray(allseg, np.float32)).decode(),
                classes=classes, radius=radius)

    flagged = sum(c["count"] for c in classes
                  if not c["name"].startswith(("nominal", "missing nodes")))
    # Per-class totals spelled out, because each class is now two rows and the headline
    # number people quote is the sum.
    per_class = " &middot; ".join(f"{k} {totals[k]}" for k in ORDER if totals.get(k))
    note = (f"{flagged} of {n_keep} struts flagged &middot; "
            f"{len(node_in)} interior + {len(node_sf)} surface junctions with no material<br>"
            f"caps = the outermost strut layer, same 55.85 vox length as any other; their "
            f"endpoints are surface junctions of degree 3-8 rather than 12<br>"
            f"totals, core + cap: {per_class}<br>"
            f"labels read from {run_dir.name}/strut_classes.csv &middot; "
            f"registration correction {'applied' if corrected else 'NOT APPLIED'}"
            + (" &middot; boundary caps included" if include_caps else " &middot; boundary caps HIDDEN"))

    page = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", title or f"Strut classes — {run_dir.name}")
                .replace("__NOTE__", note))

    # Two fixes to the shared viewer shell, applied to this copy only so graph_webgl.py
    # keeps behaving exactly as it does today:
    #   * it hardcodes "material " before every rule, which reads right for its own
    #     coverage classes and wrong for "no path node-to-node through material";
    #   * the legend prints `count`, which for the node classes is the SEGMENT count —
    #     a cross is three segments, so 2 nodes would advertise themselves as 6.
    page = page.replace("min-width:240px; }", "min-width:240px; max-width:430px; }")
    page = page.replace("rl.textContent='material '+c.rule;", "rl.textContent=c.rule;")
    page = page.replace(
        '<span class="ct">${c.count} · ${c.pct}%</span>',
        '<span class="ct">${c.label_n ?? c.count}${c.pct ? " · "+c.pct+"%" : ""}</span>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path, classes, len(node_in) + len(node_sf)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="results directory with strut_classes.csv")
    p.add_argument("--design", default=str(DESIGN))
    p.add_argument("--correction", default="")
    p.add_argument("-o", "--out", default="")
    p.add_argument("--exclude-caps", action="store_true",
                   help="drop boundary-cap struts; totals then stop matching the "
                        "counts reported elsewhere (missing 89 rather than 412)")
    p.add_argument("--fill-cut", type=float, default=0.194,
                   help="sphere fill at or below which a junction printed nothing; the "
                        "default is the cut detect_missing_nodes reads off the empty gap")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    correction = Path(args.correction) if args.correction else None
    if correction is None:
        for candidate in (run_dir.parent / "registration" / "correction.json",
                          ROOT / "outputs/registration/correction.json"):
            if candidate.is_file():
                correction = candidate
                break

    out = Path(args.out) if args.out else run_dir / "strut_classes_interactive.html"
    out, classes, n_nodes = build(run_dir, Path(args.design), correction, out,
                                  not args.exclude_caps, args.fill_cut, args.title)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    for c in classes:
        print(f"  {c['name']:28s} {c.get('label_n', c['count']):6d}")


if __name__ == "__main__":
    main()
