#!/usr/bin/env python
"""Two detector runs, one interactive 3D page: what agreed and what moved.

Counts cannot answer the question this draws. `missing 412 vs 412` is consistent with the
same 412 struts and with 400 shared plus 12 swapped, and those mean opposite things — the
identical-count-different-members trap that already cost this project one wrong ground
truth. So this compares by strut id, not by tally.

On the fresh-vs-stored pair the answer is: the 412 are bit-identical (intersection 412,
neither-only 0), 18,224 of 18,468 struts agree, and every one of the 244 that moved
crossed the `nominal` boundary — no strut went from one defect class to a different defect
class. That is why the changed struts get their own two colours by DIRECTION rather than
being coloured by their new label: the direction is the whole finding.

Nothing is computed here. Both label sets come from the runs' own `strut_classes.csv`.

    python scripts/dashboard/delta3d.py --fresh outputs/fresh_pass_.../defects \\
        --stored outputs/lattice_iou -o outputs/delta.html
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

from classes3d import (DESIGN, NODE_CROSS_VOX, load_labels,  # noqa: E402
                       missing_nodes)
from graph_webgl import HTML  # noqa: E402
from scripts.classify_strut_defects import CLASS_COLORS, ORDER  # noqa: E402
from src.lattice_iou import load_design  # noqa: E402

DEFECTS = [c for c in ORDER if c != "nominal"]
LOST = "#9085e9"      # violet: a defect label in the stored run, nominal in the fresh one
GAINED = "#e87ba4"    # magenta: the other direction
NODE_INTERIOR = "#ffffff"
NODE_SURFACE = "#898781"


def crosses(points, centre, half=NODE_CROSS_VOX):
    """Each point as three axis-aligned segments — the LINES renderer needs no new path."""
    arm = np.zeros((len(points), 3, 2, 3), np.float32)
    centred = (points - centre).astype(np.float32)
    for axis in range(3):
        delta = np.zeros(3, np.float32)
        delta[axis] = half
        arm[:, axis, 0] = centred - delta
        arm[:, axis, 1] = centred + delta
    return arm.reshape(-1, 3)


def build(fresh_dir, stored_dir, design_path, correction, out_path, include_caps, fill_cut):
    pos, pairs, edge, corrected = load_design(str(design_path),
                                              str(correction) if correction else None)
    fresh, stored = load_labels(fresh_dir), load_labels(stored_dir)

    values, counts = np.unique(edge, return_counts=True)
    caps = set(values[counts < counts.max() * 0.5].tolist())
    is_cap = np.asarray([e in caps for e in edge])
    keep = np.ones(len(pairs), bool) if include_caps else ~is_cap

    n = len(pairs)
    a_lab = np.asarray([fresh.get(i, "") for i in range(n)], object)
    b_lab = np.asarray([stored.get(i, "") for i in range(n)], object)
    both = np.asarray([i in fresh and i in stored for i in range(n)])
    agreed = both & (a_lab == b_lab)

    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    node_in, node_sf = missing_nodes(fresh_dir, fill_cut)
    stack = [pos] + [p for p in (node_in, node_sf) if len(p)]
    centre = np.concatenate(stack, 0)
    centre = (centre.max(0) + centre.min(0)) / 2.0

    groups = []
    # Agreed defect classes first: these are the findings both runs stand behind.
    for name in DEFECTS:
        sel = keep & agreed & (a_lab == name)
        groups.append((f"{name} · both runs", CLASS_COLORS.get(name, "#888"),
                       f"identical label in both runs", sel, 0.95))

    # Then the disagreement, split PER CLASS and per direction — one togglable layer for
    # every cell that moved. Splitting only by direction (which this did first) collapses
    # `thin -90` and `thick +24` into one bucket and makes the per-class deltas
    # unrecoverable, which is the whole table this figure exists to explain.
    #
    # Colour follows the CLASS, so a mover keeps the identity of the class it left or
    # joined; the two directions share that colour and are told apart by the layer name
    # and their own checkbox. Direction is never carried by hue alone.
    changed = keep & both & (a_lab != b_lab)
    accounted = np.zeros(len(pairs), bool)
    for name in DEFECTS:
        colour = CLASS_COLORS.get(name, "#888")
        lost = changed & (b_lab == name) & (a_lab == "nominal")
        gained = changed & (a_lab == name) & (b_lab == "nominal")
        if lost.any():
            groups.append((f"{name} → nominal  (lost)", colour,
                           f"{name} in stored, nominal in fresh", lost, 0.95))
        if gained.any():
            groups.append((f"nominal → {name}  (gained)", colour,
                           f"nominal in stored, {name} in fresh", gained, 0.95))
        accounted |= lost | gained

    # Anything that moved between two DEFECT classes would land here. On this pair the
    # set is empty, and that emptiness is itself the result — so the layer only appears
    # when it is non-empty rather than showing a permanent zero.
    other = changed & ~accounted
    if other.any():
        groups.append(("changed · defect → other defect", "#ffffff",
                       "moved between two defect classes, neither being nominal",
                       other, 0.95))

    groups.append(("nominal · both runs", "#b8b7b2",
                   "agreed nominal — context, not a finding",
                   keep & agreed & (a_lab == "nominal"), 0.06))

    segments, classes, cursor = [], [], 0
    n_keep = int(keep.sum())
    for label, colour, rule, sel, alpha in groups:
        seg = np.stack([a[sel] - centre, b[sel] - centre], 1).astype(np.float32)
        segments.append(seg.reshape(-1, 3))
        classes.append(dict(
            name=label, color=colour,
            rgb=[int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)],
            rule=rule, start=cursor, count=int(sel.sum()),
            pct=round(100 * sel.sum() / max(1, n_keep), 2), on=True, alpha=alpha))
        cursor += int(sel.sum()) * 2

    for pts, colour, label, rule in (
            (node_in, NODE_INTERIOR, "missing node · interior",
             "degree 12, empty sphere — cannot be a surface effect"),
            (node_sf, NODE_SURFACE, "missing node · surface",
             "on the boundary — 181 of these sit on the single plane y = 759")):
        if not len(pts):
            continue
        segments.append(crosses(pts, centre))
        classes.append(dict(
            name=label, color=colour,
            rgb=[int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)],
            rule=rule, start=cursor, count=len(pts) * 3, label_n=len(pts),
            pct=0.0, on=True, alpha=0.95))
        cursor += len(pts) * 3 * 2

    allseg = np.concatenate(segments, 0) if segments else np.zeros((0, 3), np.float32)
    radius = float(np.linalg.norm(allseg, axis=1).max()) if len(allseg) else 1.0
    meta = dict(pos=base64.b64encode(np.ascontiguousarray(allseg, np.float32)).decode(),
                classes=classes, radius=radius)

    # The headline claim, stated on the page so the figure travels with its evidence.
    # Quote `missing` over EVERY strut, caps included, because that is the population the
    # 412-vs-412 comparison was made on — and say so, since the legend below counts only
    # the core struts and would otherwise look like it contradicts this line.
    fresh_missing = {i for i in range(n) if a_lab[i] == "missing"}
    stored_missing = {i for i in range(n) if b_lab[i] == "missing"}
    n_changed = int((keep & both & (a_lab != b_lab)).sum())
    scope = "all struts" if include_caps else "core struts"
    caveat = ("" if include_caps else
              " &mdash; the layers below count core struts only, so their totals are smaller")
    note = (f"{int((keep & agreed).sum())} of {n_keep} {scope} agree &middot; "
            f"{n_changed} changed class<br>"
            f"missing over all {n} struts: {len(fresh_missing)} fresh, "
            f"{len(stored_missing)} stored, <b>{len(fresh_missing & stored_missing)} shared</b>, "
            f"{len(fresh_missing ^ stored_missing)} in only one{caveat}<br>"
            f"fresh {fresh_dir.name} vs stored {stored_dir.name} &middot; "
            f"correction {'applied' if corrected else 'NOT applied'}"
            + ("" if include_caps else " &middot; boundary caps hidden"))

    page = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", "Fresh vs stored — what moved")
                .replace("__NOTE__", note))
    page = page.replace("rl.textContent='material '+c.rule;", "rl.textContent=c.rule;")
    page = page.replace(
        '<span class="ct">${c.count} · ${c.pct}%</span>',
        '<span class="ct">${c.label_n ?? c.count}${c.pct ? " · "+c.pct+"%" : ""}</span>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path, classes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fresh", required=True)
    p.add_argument("--stored", required=True)
    p.add_argument("--design", default=str(DESIGN))
    p.add_argument("--correction", default="")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--include-caps", action="store_true")
    p.add_argument("--fill-cut", type=float, default=0.194)
    args = p.parse_args()

    fresh = Path(args.fresh) if Path(args.fresh).is_absolute() else ROOT / args.fresh
    stored = Path(args.stored) if Path(args.stored).is_absolute() else ROOT / args.stored
    correction = Path(args.correction) if args.correction else None
    if correction is None:
        for candidate in (fresh.parent / "registration" / "correction.json",
                          ROOT / "outputs/registration/correction.json"):
            if candidate.is_file():
                correction = candidate
                break

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out, classes = build(fresh, stored, Path(args.design), correction, out,
                         args.include_caps, args.fill_cut)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    for c in classes:
        print(f"  {c['name']:34s} {c.get('label_n', c['count']):6d}")


if __name__ == "__main__":
    main()
