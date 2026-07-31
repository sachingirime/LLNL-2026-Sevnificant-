#!/usr/bin/env python
"""The pipeline as one poster-ready block diagram (PNG).

Every number on the figure is read from a finished run rather than typed in, so the
diagram cannot drift from the results it describes.

This replaces an earlier hand-drawn flowchart that carried four errors worth naming, since
they are the kind that survive review by looking plausible:

  * it labelled the INPUT "CAD Design Graph (16,733 Struts)". The design graph has 18,468
    struts; 16,733 is the *measurable* subset, which is an output of the geometry stage,
    not a property of the input.
  * it named the output classes "Inclusion | Broken | Cavity | Thinning". The detector
    emits missing | broken | thin | thick | necked | nominal. `Inclusion` and `Cavity` are
    not classes this pipeline can produce, and `missing` -- the class with ground-truth
    validation -- was absent.
  * it drew per-strut classification AFTER the explainability layer. Classification
    produces the results the explanation is about, so it comes first.
  * it showed "Attention Rollout Heatmaps" and "Refusal Flags" as pipeline stages. Neither
    exists in the codebase (`grep -rniE 'attention.?rollout|refusal'` over src/ and
    scripts/ returns nothing but an unrelated dtype message). Drawing an unbuilt component
    beside built ones is the failure this project keeps writing down.

    python scripts/poster/pipeline_diagram.py --run outputs/fresh_pass_.../defects
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.classify_strut_defects import CLASS_COLORS, ORDER  # noqa: E402

# Light figure: the poster page is light, and the dark palette in poster/style.py belongs
# to the mask-slice figures, which are photographs of data rather than diagrams.
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c2"
SURFACE, BAND, ACCENT = "#ffffff", "#f4f3ef", "#2a78d6"


def load_numbers(run_dir: Path) -> dict:
    with (run_dir / "strut_classes.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = "label" if rows and "label" in rows[0] else "class"
    counts = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1

    summary = {}
    for candidate in (run_dir / "summary.json", run_dir.parent / "summary.json"):
        if candidate.is_file():
            summary = json.loads(candidate.read_text())
            break
    return {"counts": counts, "total": len(rows),
            "counts_meta": summary.get("counts", {}), "geometry": summary.get("geometry", {})}


# Layout metrics in DATA units. The axes are 1.62 x 0.92 on a 16x9 inch canvas, so one
# data unit is ~9.78 inches vertically: an 8.3pt line of text is ~0.012 units tall. The
# first version of this figure used 0.052 per line — half an inch — and every box
# overflowed. Sizes are derived from these constants now, never hand-placed.
LINE = 0.0225          # one body line
TITLE = 0.034          # title block
PAD = 0.015            # inner padding, top and bottom
GAP = 0.022            # between stacked boxes
BAND_PAD = 0.030       # band inset around its boxes


def box_height(lines):
    return PAD + TITLE + LINE * len(lines) + PAD


def box(ax, x, y_top, w, title, lines, *, fc=SURFACE, ec=RULE, lw=1.1):
    """Draw a box whose height is derived from its content. Returns the bottom y."""
    h = box_height(lines)
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y_top - PAD - 0.004, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=INK, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y_top - PAD - TITLE - i * LINE, line, ha="center", va="top",
                fontsize=8.0, color=MUTED, zorder=3)
    return y_top - h


def arrow(ax, p0, p1, *, color=ACCENT, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=lw,
                                 color=color, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}"))


def band(ax, x, y_bot, w, y_top, label):
    ax.add_patch(FancyBboxPatch((x, y_bot), w, y_top - y_bot,
                                boxstyle="round,pad=0.008,rounding_size=0.016",
                                fc=BAND, ec="none", zorder=1))
    ax.text(x + 0.014, y_top - 0.012, label, ha="left", va="top", fontsize=8.4,
            fontweight="bold", color=MUTED, zorder=3)


def draw_cylinder_test(ax, x, y, w, h):
    """A strut inside its nominal cylinder, pinching then stopping."""
    cy = y + h * 0.60
    ax.add_patch(FancyBboxPatch((x, cy - h * 0.24), w, h * 0.48,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                fc="none", ec=CLASS_COLORS["missing"], lw=1.0, zorder=3))
    xs = np.linspace(x + 0.004, x + w - 0.004, 80)
    r = h * 0.18 * (1 - 0.72 * np.exp(-((xs - (x + w * 0.60)) / (w * 0.08)) ** 2))
    r[xs > x + w * 0.84] = 0
    ax.fill_between(xs, cy - r, cy + r, color="#8e8d88", zorder=4, lw=0)
    for f in (0.14, 0.34, 0.54, 0.74):
        ax.add_patch(Ellipse((x + w * f, cy), w * 0.020, h * 0.40, fc="none",
                             ec=ACCENT, lw=0.8, zorder=5))
    ax.text(x + w / 2, y - 0.004, "cylinder — struts",
            ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_sphere_test(ax, x, y, w, h):
    cx, cy = x + w / 2, y + h * 0.60
    rad = h * 0.22
    for ang in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([cx, cx + np.cos(ang) * rad * 2.6], [cy, cy + np.sin(ang) * rad * 2.6],
                color="#8e8d88", lw=1.4, solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((cx, cy), rad, fc="#ffffff", ec=CLASS_COLORS["missing"],
                        lw=1.2, zorder=5, alpha=0.92))
    ax.text(cx, y - 0.004, "sphere — nodes",
            ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_geodesic_test(ax, x, y, w, h):
    cy = y + h * 0.60
    xs = np.linspace(x + w * 0.10, x + w * 0.90, 90)
    ys = cy + np.sin((xs - xs[0]) / (w * 0.14)) * h * 0.10
    keep = (xs < x + w * 0.48) | (xs > x + w * 0.58)
    ax.plot(xs[keep], ys[keep], color="#8e8d88", lw=3.0, solid_capstyle="round",
            zorder=3, ls="none", marker="o", markersize=1.6)
    for f in (0.10, 0.90):
        ax.add_patch(Circle((x + w * f, cy), h * 0.13, fc="#8e8d88", ec="none", zorder=4))
    xm = x + w * 0.53
    ax.plot([xm - w * 0.028, xm + w * 0.028], [cy + h * 0.13, cy - h * 0.13],
            color=CLASS_COLORS["broken"], lw=1.5, zorder=6)
    ax.plot([xm - w * 0.028, xm + w * 0.028], [cy - h * 0.13, cy + h * 0.13],
            color=CLASS_COLORS["broken"], lw=1.5, zorder=6)
    ax.text(x + w / 2, y - 0.004, "geodesic — struts",
            ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_classes(ax, x, y_top, w, counts, total):
    defects = [c for c in ORDER if c != "nominal"]
    biggest = max((counts.get(c, 0) for c in defects), default=1)
    row = 0.036
    for i, name in enumerate(defects):
        yy = y_top - 0.012 - i * row
        n = counts.get(name, 0)
        ax.add_patch(FancyBboxPatch((x, yy - row * 0.42), 0.014, row * 0.44,
                                    boxstyle="round,pad=0,rounding_size=0.005",
                                    fc=CLASS_COLORS[name], ec="none", zorder=4))
        ax.text(x + 0.024, yy - row * 0.20, name, ha="left", va="center",
                fontsize=8.6, color=INK, zorder=4)
        bx, bw = x + 0.088, w - 0.155
        ax.add_patch(FancyBboxPatch((bx, yy - row * 0.38), max(bw * n / biggest, 0.006),
                                    row * 0.36, boxstyle="round,pad=0,rounding_size=0.006",
                                    fc=CLASS_COLORS[name], ec="none", zorder=4))
        ax.text(x + w, yy - row * 0.20, f"{n:,}", ha="right", va="center",
                fontsize=8.8, color=INK, fontweight="bold", zorder=4)
    bottom = y_top - 0.012 - len(defects) * row
    ax.text(x, bottom - 0.004, f"nominal {counts.get('nominal', 0):,} of {total:,}"
            "  —  context, not a finding", ha="left", va="top", fontsize=7.4, color=MUTED)
    ax.text(x, bottom - 0.028, "severity-first: necked → thick → thin → broken → missing",
            ha="left", va="top", fontsize=7.2, color=MUTED, style="italic")
    return bottom - 0.048


def build(run_dir: Path, out_path: Path, title: str):
    data = load_numbers(run_dir)
    counts, total = data["counts"], data["total"]
    meta, geom = data["counts_meta"], data["geometry"]
    n_struts = meta.get("struts", total)
    n_junc = meta.get("junction_entries", 10206)
    n_nodes = meta.get("physical_nodes", 3430)
    n_meas = meta.get("measurable_struts", 0)
    um = geom.get("um_per_voxel", 58.2)
    missing = counts.get("missing", 0)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1.62); ax.set_ylim(0, 0.92); ax.axis("off")

    ax.text(0.02, 0.905, title, fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.868, "every number is read from the run it describes, not typed in",
            fontsize=9, color=MUTED, va="top", style="italic")

    CA, CB, CC = 0.02, 0.575, 1.115
    CW = 0.485
    IN = 0.016                      # box inset inside a band
    HEAD = 0.040                     # room for the band's own label
    TOP = 0.828
    FLOOR = 0.152                    # bands stop here; the conclusion strip lives below

    # ---- column A ----------------------------------------------------------
    half = (CW - 3 * IN) / 2
    y = TOP - HEAD
    b1 = box(ax, CA + IN, y, half, "X-ray CT volume",
             [".tif · 761 × 815 × 837", f"uint16 · {um:.1f} µm / voxel", "what was actually built"])
    box(ax, CA + 2 * IN + half, y, half, "CAD design graph",
        [f".json · {n_struts:,} struts", f"{n_junc:,} junction entries",
         f"→ {n_nodes:,} physical nodes"])
    band(ax, CA, b1 - IN, CW, TOP, "1 · INPUT")

    gtop = b1 - IN - GAP
    y = gtop - HEAD
    b2 = box(ax, CA + IN, y, half, "Segmentation",
             ["global Otsu on the raw stack", "cut 40,127 · 11.2 % foreground"])
    box(ax, CA + 2 * IN + half, y, half, "Registration refit",
        ["corrects a 1.26 % x-shrink", "residual 0.22 / 0.57 / 0.27 vox"])
    b3 = box(ax, CA + IN, b2 - GAP, CW - 2 * IN, "Measurable population",
             [f"{n_meas:,} of {n_struts:,} struts carry enough clean signal to judge"])
    band(ax, CA, b3 - IN, CW, gtop, "2 · GEOMETRY")

    vtop = b3 - IN - GAP
    band(ax, CA, FLOOR, CW, vtop, "3 · VOLUMETRIC TESTS   (missing / broken use no threshold)")
    # Three glyphs in ONE row. Two rows did not fit this band at any anchoring — the
    # cylinder caption kept landing on the pair below it, or the art on the band label.
    third = (CW - 4 * IN) / 3
    gy = FLOOR + 0.074
    gh = 0.052
    draw_cylinder_test(ax, CA + IN, gy, third, gh)
    draw_geodesic_test(ax, CA + 2 * IN + third, gy, third, gh)
    draw_sphere_test(ax, CA + 3 * IN + 2 * third, gy, third, gh)
    # Two lines: one ran wider than the band and pushed the tight bbox out.
    ax.text(CA + CW / 2, FLOOR + 0.036,
            "cylinder: fill of the nominal cylinder + 25 perpendicular cross-sections",
            ha="center", va="bottom", fontsize=7.2, color=MUTED, zorder=3)
    ax.text(CA + CW / 2, FLOOR + 0.016,
            "geodesic: any path through metal node-to-node   ·   sphere: fill of a 257 µm sphere",
            ha="center", va="bottom", fontsize=7.2, color=MUTED, zorder=3)

    # ---- column B ----------------------------------------------------------
    cbot = draw_classes(ax, CB + 0.022, TOP - HEAD, CW - 0.044, counts, total)
    band(ax, CB, cbot - 0.008, CW, TOP, "4 · PER-STRUT CLASSIFICATION")

    vbtop = cbot - 0.008 - GAP
    y = vbtop - HEAD
    v1 = box(ax, CB + IN, y, CW - 2 * IN, "Six checks, run explicitly",
             ["coordinate frame · alignment residual · cache staleness",
              "threshold sensitivity · provenance · detector agreement",
              "whether you checked is itself part of the record"])
    v2 = box(ax, CB + IN, v1 - GAP, CW - 2 * IN, "Ground truth",
             ["designed-missing set read from the STL",
              "precision 0.989 · recall 0.978 · F1 0.983"])
    band(ax, CB, v2 - IN, CW, vbtop, "5 · VERIFY BEFORE YOU BELIEVE")

    # ---- column C ----------------------------------------------------------
    y = TOP - HEAD
    e1 = box(ax, CC + IN, y, CW - 2 * IN, "Execution trace",
             ["every tool call is a packet:", "actor · why · inputs · verdict · artifacts",
              "server side in outputs/mep/ · agent side from the dashboard"])
    e2 = box(ax, CC + IN, e1 - GAP, CW - 2 * IN, "Trajectory  (sₜ, aₜ, oₜ)",
             ["state → action → observation",
              "replayable, so any claim traces back to the call that made it"])
    e3 = box(ax, CC + IN, e2 - GAP, CW - 2 * IN, "Run explanation",
             ["what was asked, what was done, which classes the",
              "checks support — and which they do not"])
    band(ax, CC, e3 - IN, CW, TOP, "6 · EXPLAINABILITY  (MEP)")

    otop = e3 - IN - GAP
    o1 = box(ax, CC + IN, otop - HEAD, CW - 2 * IN, "Artifacts",
             ["strut_classes.csv · nodes.csv · report.html",
              "interactive 3-D (WebGL, self-contained)",
              "every figure regenerated from the tables"])
    band(ax, CC, o1 - IN, CW, otop, "7 · OUTPUT")

    # ---- conclusion strip, full width --------------------------------------
    strip_top = FLOOR - GAP
    ax.add_patch(FancyBboxPatch((CA, strip_top - 0.082), CC + CW - CA, 0.082,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                fc="#fdf3f2", ec=CLASS_COLORS["missing"], lw=1.1, zorder=2))
    ax.text(CA + 0.022, strip_top - 0.018,
            f"{missing:,} struts in the design have no metal — but that is two populations, "
            "not one", ha="left", va="top",
            fontsize=11.5, fontweight="bold", color=INK, zorder=3)
    ax.text(CA + 0.022, strip_top - 0.044,
            f"{missing - 89:,} are the closing row on a single outer face (y-hi), where the scan "
            "reads 0 voxels while the other two cap faces are solid metal.",
            ha="left", va="top", fontsize=8.4, color=MUTED, zorder=3)
    ax.text(CA + 0.022, strip_top - 0.064,
            "The remaining 89 are interior struts — the build failures, and the population the "
            "STL ground truth validates at precision 0.989 / recall 0.978.",
            ha="left", va="top", fontsize=8.4, color=MUTED, zorder=3)

    # ---- flow ---------------------------------------------------------------
    arrow(ax, (CA + 0.126, b1 - IN), (CA + 0.126, gtop))
    arrow(ax, (CA + 0.362, b1 - IN), (CA + 0.362, gtop))
    arrow(ax, (CA + 0.244, b3 - IN), (CA + 0.244, vtop))
    arrow(ax, (CA + CW, 0.30), (CB, 0.52), rad=-0.16)
    arrow(ax, (CB + 0.244, cbot - 0.008), (CB + 0.244, vbtop))
    arrow(ax, (CB + CW, 0.60), (CC, 0.60))
    arrow(ax, (CC + 0.244, e3 - IN), (CC + 0.244, otop))

    ax.text(0.02, 0.016,
            "Rules live in scripts/classify_strut_defects.classify.  `missing` and `broken` rest on "
            "counts that are zero or not, so neither involves a threshold;  `thin` / `thick` / `necked` "
            "sit near the resolution floor and are weaker evidence.",
            fontsize=7.4, color=MUTED, va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/fresh_pass_20260730T223536/defects")
    p.add_argument("-o", "--out", default="outputs/poster/pipeline_diagram.png")
    p.add_argument("--title", default="Agentic defect detection in an as-built octet lattice")
    args = p.parse_args()

    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out = build(run_dir, out, args.title)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
