#!/usr/bin/env python
"""The detector's classes against the CAD's designed-out set, as a confusion table.

Reads a finished validation (`validation.json` + `truth.npz`) and draws it. Measures
nothing: the orientation, the designed-out mask and every score come from
`src.stl_ground_truth`, which is the code `validate_against_stl` runs.

The header carries the two facts that decide whether the rest of the figure means
anything, because both are easy to omit and fatal to leave out:

  * the orientation's MARGIN over the runner-up. A complete lattice is invariant under all
    48 cube symmetries, so the alignment is fixed by the defect pattern alone. A thin
    margin means the identification is a guess and every number below it is a guess.
  * that the alignment used the detector's own missing calls as its reference, so
    precision and recall here are NOT a blind test. The designed-out COUNT is blind; the
    per-strut scores are not.

    python scripts/stl_confusion.py --run outputs/stl_validation_20260730
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.classify_strut_defects import CLASS_COLORS, ORDER  # noqa: E402
from src.lattice_iou import _GRID, _INK, _INK2, _SURFACE  # noqa: E402

OK_C, BAD_C, MUTE_C = "#0d8f57", "#d0342c", "#8f8d88"


def build(run_dir, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": _INK})
    meta = json.loads((run_dir / "validation.json").read_text())
    z = np.load(run_dir / "truth.npz", allow_pickle=False)
    truth, usable, lab = z["truth"], z["usable"], np.asarray(z["label"], str)
    n = len(lab)
    pad = np.zeros(len(truth), bool)

    W, H = 13.2, 8.6
    fig = plt.figure(figsize=(W, H), facecolor=_SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.45, H - 0.42, "Detector against the CAD: which struts were designed out",
            fontsize=15, va="top", fontweight="bold")
    margin_ok = meta["matched"] >= 3 * max(meta["runner_up"], 1)
    ax.text(0.45, H - 0.82,
            f"0.5.stl aligned by searching all 48 cube orientations.  Winner "
            f"perm {tuple(meta['perm'])} signs {tuple(meta['signs'])}: matches "
            f"{meta['matched']} of the detector's missing calls, "
            f"runner-up {meta['runner_up']}, chance {meta['chance']:.1f}.",
            fontsize=9, color=_INK2, va="top")
    ax.text(0.45, H - 1.10,
            ("Margin is wide, so the alignment is established.  " if margin_ok else
             "MARGIN IS NARROW -- the alignment is NOT established and everything "
             "below is unreliable.  ")
            + ("Independent check passed: the STL's plate axis maps to the design's "
               "build axis." if meta["plate_axis_ok"] else
               "Independent check FAILED: the plate axis is not the build axis."),
            fontsize=9, color=OK_C if (margin_ok and meta["plate_axis_ok"]) else BAD_C,
            va="top", fontweight="bold")

    # ------------------------------------------------- 2x2, the `missing` call
    pred = pad.copy()
    pred[:n] = lab == "missing"
    tp = int((pred & truth & usable).sum())
    fp = int((pred & ~truth & usable).sum())
    fn = int((~pred & truth & usable).sum())
    tn = int((~pred & ~truth & usable).sum())

    # Cell width is set so the 2x2 ends clear of the right-hand table; at 2*cw it ran
    # under the per-class rows and the TN figure landed on top of them.
    x0, y0, cw, ch = 1.75, H - 4.55, 2.30, 0.92
    ax.text(x0, y0 + 2 * ch + 0.55, "the `missing` call, over usable struts",
            fontsize=10.5, fontweight="bold")
    ax.text(x0 + cw / 2, y0 + 2 * ch + 0.20, "designed out", fontsize=9, ha="center",
            color=_INK2)
    ax.text(x0 + 1.5 * cw, y0 + 2 * ch + 0.20, "present in CAD", fontsize=9,
            ha="center", color=_INK2)
    for r, rowname in enumerate(("called missing", "not called missing")):
        ax.text(x0 - 0.12, y0 + (1 - r) * ch + ch / 2, rowname, fontsize=9, ha="right",
                va="center", color=_INK2)
    cells = [[(tp, OK_C), (fp, BAD_C)], [(fn, BAD_C), (tn, MUTE_C)]]
    for r in range(2):
        for c in range(2):
            v, colour = cells[r][c]
            ax.add_patch(plt.Rectangle((x0 + c * cw, y0 + (1 - r) * ch),
                                       cw - 0.06, ch - 0.06, facecolor=colour,
                                       alpha=0.16 if colour == MUTE_C else 0.24,
                                       edgecolor=colour, linewidth=1.2))
            ax.text(x0 + c * cw + cw / 2, y0 + (1 - r) * ch + ch / 2, f"{v:,}",
                    fontsize=17, ha="center", va="center", color=colour,
                    fontweight="bold")

    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    ax.text(x0, y0 - 0.34, f"precision {P:.3f}      recall {R:.3f}      F1 {F:.3f}",
            fontsize=11, fontweight="bold")
    ax.text(x0, y0 - 0.68,
            f"designed out: {meta['designed_out']} of 18,468 struts "
            f"({100 * meta['designed_out'] / 18468:.3f}%)  --  this COUNT owes nothing to "
            "the detector.",
            fontsize=8.6, color=_INK2)

    # ----------------------------------------- every class against the truth
    xr = 7.35
    ax.text(xr, H - 2.05, "every class, scored the same way", fontsize=10.5,
            fontweight="bold")
    ax.text(xr, H - 2.42, f"{'class':10s}{'called':>8s}{'TP':>7s}{'FP':>7s}{'FN':>7s}"
            f"{'prec':>8s}{'rec':>7s}", fontsize=8.6, color=_INK2, family="monospace")
    yy = H - 2.72
    for name in ORDER:
        p = pad.copy()
        p[:n] = lab == name
        t = int((p & truth & usable).sum())
        f = int((p & ~truth & usable).sum())
        m = int((~p & truth & usable).sum())
        pr = t / max(t + f, 1)
        rc = t / max(t + m, 1)
        ax.add_patch(plt.Rectangle((xr - 0.22, yy - 0.10), 0.10, 0.26,
                                   facecolor=CLASS_COLORS[name], edgecolor="none"))
        ax.text(xr, yy, f"{name:10s}{int((p & usable).sum()):>8d}{t:>7d}{f:>7d}{m:>7d}"
                f"{pr:>8.3f}{rc:>7.3f}", fontsize=8.6, family="monospace",
                color=_INK if t else _INK2)
        yy -= 0.34

    # ------------------------------- what the designed-out struts were called
    yy -= 0.30
    ax.text(xr, yy, "what the designed-out struts were actually labelled",
            fontsize=10.5, fontweight="bold")
    yy -= 0.36
    got = [(c, int((truth[:n] & usable[:n] & (lab == c)).sum())) for c in ORDER]
    total = sum(v for _, v in got)
    for name, v in got:
        if not v:
            continue
        ax.add_patch(plt.Rectangle((xr, yy - 0.09), 4.2 * v / max(total, 1), 0.24,
                                   facecolor=CLASS_COLORS[name], edgecolor="none"))
        ax.text(xr + 4.2 * v / max(total, 1) + 0.12, yy + 0.03,
                f"{name}  {v}  ({100 * v / max(total, 1):.0f}%)", fontsize=8.8,
                va="center", color=_INK)
        yy -= 0.36
    ax.text(xr, yy - 0.10,
            f"{int((truth & ~usable).sum())} more are boundary or plate-embedded and "
            "cannot be scored.", fontsize=8.4, color=_INK2, va="top")

    ax.text(0.45, 0.52,
            "The alignment is fixed using the detector's own missing calls, because a "
            "complete lattice is invariant under all 48 cube symmetries and carries no "
            "orientation information. So the designed-out COUNT is blind,\n"
            "and precision/recall are not. Produced by calling src.stl_ground_truth "
            "directly -- the MCP server was not connected, so this run has no MEP packet "
            "and no provenance record.",
            fontsize=8, color=_INK2, va="center", linespacing=1.7)

    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/stl_validation_20260730")
    p.add_argument("--out", default="")
    a = p.parse_args()
    run = Path(a.run)
    build(run, Path(a.out) if a.out else run / "stl_confusion.png")


if __name__ == "__main__":
    main()
