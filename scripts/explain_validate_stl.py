#!/usr/bin/env python
"""How `validate_against_stl` scores the detector -- the method, drawn.

An explainer, not a result. It shows how a graph of strut centre-lines and a bare triangle
mesh are brought into one frame and compared, because that alignment is the part a reader
has to accept before any precision or recall number means anything.

The lattice sketches are real: struts are read from the nominal design
(`data/missing_struts/octet_truss_9x9x9.json`) and projected isometrically, so the
geometry in the figure is the geometry the tool works on. The numbers are measured, not
illustrative -- the bounding box and scale come from reading the STLs, printed in the
figure's own caption with the file they came from.

Nothing here measures the specimen. It draws the procedure defined in
`src/stl_ground_truth.py` and used by `validate_against_stl` in `src/mcp_server.py`.

    python scripts/explain_validate_stl.py
    python scripts/explain_validate_stl.py --out outputs/mep_figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lattice_iou import _GRID, _INK, _INK2, _SURFACE, load_design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOMINAL = ROOT / "data/missing_struts/octet_truss_9x9x9.json"

DESIGN_C = "#2a78d6"     # the graph: centre-lines, no thickness
MESH_C = "#6b6a67"       # the STL: solid material, no labels
GONE_C = "#d0342c"       # a strut the CAD deleted
OK_C = "#0d8f57"

# Measured, not assumed -- see the module docstring and the figure caption.
DESIGN_UNITS = 18.0
BBOX_MM = (41.4783, 51.0400, 41.4794)
SCALE_MM = 2.304381
TOL_MM = 0.7
MIDSPAN = (0.40, 0.45, 0.50, 0.55, 0.60)
N_ORIENT = 48


def iso(p, azim=35.0, elev=22.0):
    """Isometric projection of (n, 3) design coordinates to 2-D."""
    a, e = np.radians(azim), np.radians(elev)
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    up = np.array([-np.sin(a) * np.sin(e), -np.cos(a) * np.sin(e), np.cos(e)])
    p = np.asarray(p, float)
    return np.stack([p @ right, p @ up], axis=-1)


def unit_cell(radius=1.1):
    """One cell's worth of struts from the middle of the nominal design."""
    pos, pairs, _, _ = load_design(str(NOMINAL), None)
    mid = 0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])
    sel = np.all(np.abs(mid - pos.mean(0)) < radius, axis=1)
    seg = np.stack([pos[pairs[sel, 0]], pos[pairs[sel, 1]]], axis=1)
    return seg - seg.reshape(-1, 3).mean(0)


def draw_lattice(ax, seg, x, y, s, colour, lw=1.4, drop=None, drop_colour=GONE_C,
                 solid=False, alpha=1.0):
    """Project and draw a strut bundle at (x, y) scaled by s. `drop` indexes a gap."""
    for i, (a, b) in enumerate(seg):
        p = iso(np.stack([a, b])) * s + np.array([x, y])
        if drop is not None and i == drop:
            if solid:  # the mesh simply has nothing there
                continue
            ax.plot(p[:, 0], p[:, 1], color=drop_colour, lw=lw, ls=(0, (2.5, 2.0)),
                    alpha=alpha, solid_capstyle="round", zorder=3)
            continue
        ax.plot(p[:, 0], p[:, 1], color=colour, lw=lw * (2.6 if solid else 1.0),
                alpha=alpha, solid_capstyle="round", zorder=2)


def stage(ax, x, y, w, h, n, title, colour):
    ax.add_patch(plt_rect(x, y, w, h))
    ax.add_patch(plt_rect(x, y, 0.055, h, colour, edge="none"))
    ax.text(x + 0.22, y + h - 0.20, f"{n}", fontsize=10, color=_GRID, va="top",
            fontweight="bold")
    ax.text(x + 0.58, y + h - 0.19, title, fontsize=11.5, color=_INK, va="top",
            fontweight="bold")


def plt_rect(x, y, w, h, face="#ffffff", edge=_GRID):
    import matplotlib.pyplot as plt
    return plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                         linewidth=0.9, zorder=1)


def build(out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": _INK})
    seg = unit_cell()
    W, H = 13.6, 12.4
    fig = plt.figure(figsize=(W, H), facecolor=_SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.45, H - 0.42, "How the detector gets scored against the CAD",
            fontsize=15, color=_INK, va="top", fontweight="bold")
    ax.text(0.45, H - 0.80,
            "The design says where every strut should be. The CAD file shows what was "
            "actually asked for, with some struts deleted on purpose.\n"
            "To compare them, they first have to sit in the same space.",
            fontsize=9.5, color=_INK2, va="top", linespacing=1.6)

    # ---------------------------------------------------------------- stage 1
    y = H - 1.55
    h1 = 3.15
    stage(ax, 0.45, y - h1, 12.7, h1, 1, "Two inputs, two different spaces", _INK2)

    ax.text(1.5, y - 0.60, "the design  (JSON)", fontsize=10, color=DESIGN_C,
            fontweight="bold", ha="center")
    draw_lattice(ax, seg, 1.5, y - 1.85, 0.30, DESIGN_C, lw=1.3)
    ax.text(1.5, y - 2.78,
            "every strut, as a line\nfrom joint to joint\n0 to 18 design units\nno thickness",
            fontsize=8.2, color=_INK2, ha="center", va="top", linespacing=1.6)

    ax.text(5.1, y - 0.60, "the CAD  (0.5.stl)", fontsize=10, color=MESH_C,
            fontweight="bold", ha="center")
    draw_lattice(ax, seg, 5.1, y - 1.85, 0.30, MESH_C, lw=1.3, drop=7, solid=True)
    ax.text(5.1, y - 2.78,
            "3.5 million triangles\nsolid material, in mm\nno strut numbers, no\n"
            "record of which way up",
            fontsize=8.2, color=_INK2, ha="center", va="top", linespacing=1.6)
    ax.annotate("a strut is simply\nabsent here", xy=(5.45, y - 1.62),
                xytext=(6.35, y - 1.05), fontsize=8.2, color=GONE_C, va="top",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=GONE_C, linewidth=1.2,
                                connectionstyle="arc3,rad=-0.25"))

    ax.text(8.45, y - 0.60, "so the first job is to line them up", fontsize=9.5,
            color=_INK, va="top", fontweight="bold")
    ax.text(8.45, y - 0.95,
            "One is a stick figure in its own units.\n"
            "The other is a solid object in millimetres.\n"
            "Neither knows where the other is.\n\n"
            "Note this is NOT the scan registration.\n"
            "That one fits a design onto real printed\n"
            "metal and leaves an error behind. This is\n"
            "design against design -- same idealised\n"
            "shape twice -- so it comes out exact.",
            fontsize=8.4, color=_INK2, va="top", linespacing=1.65)

    # ---------------------------------------------------------------- stage 2
    y -= h1 + 0.42
    h2 = 3.05
    stage(ax, 0.45, y - h2, 12.7, h2, 2, "Line them up: two parts are free, one is not",
          DESIGN_C)

    bx = 1.05
    for k, (name, body, colour) in enumerate([
        ("SIZE", f"The lattice is {DESIGN_UNITS:.0f} units across.\n"
                 f"The CAD measures {BBOX_MM[0]:.3f} mm across.\n\n"
                 f"-> {SCALE_MM:.4f} mm per unit.\n\nRead straight off the bounding\n"
                 "box. Nothing is fitted.", OK_C),
        ("POSITION", "Shift the design so its centre\nis at the origin, which is where\n"
                     "the CAD already sits.\n\nAlso exact.", OK_C),
        (f"WHICH WAY UP  ({N_ORIENT} options)",
         "A cube can sit 48 ways and the\ntriangles carry no clue which.\n\n"
         "Try all 48. Keep the one that\nputs the CAD's holes where the\n"
         "detector says holes are.\n\nThis one is a judgement.", GONE_C),
    ]):
        ax.text(bx, y - 0.58, name, fontsize=9.5, color=colour, fontweight="bold")
        ax.text(bx, y - 0.90, body, fontsize=8.4, color=_INK2, va="top", linespacing=1.65)
        bx += 4.15

    ax.text(1.05, y - h2 + 0.30,
            "Because size and position are exact, every bit of doubt lives in the "
            "rotation -- and the rotation is settled using the detector's own answer, "
            "so the score is not quite blind. The tool says so, and reports how far "
            "ahead the winner was.",
            fontsize=8.4, color=_INK, va="center")

    # ---------------------------------------------------------------- stage 3
    y -= h2 + 0.42
    h3 = 2.55
    stage(ax, 0.45, y - h3, 12.7, h3, 3, "Now ask, for each strut: is anything here?",
          MESH_C)

    ax0, ax1 = 1.35, 5.55
    cy = y - 1.35
    for x0, x1, label, present, colour in (
            (ax0, ax0 + 3.0, "a strut that printed", True, OK_C),
            (ax1, ax1 + 3.0, "a strut the CAD deleted", False, GONE_C)):
        for jx in (x0, x1):  # the joints at both ends survive either way
            ax.add_patch(plt.Circle((jx, cy), 0.20, facecolor=MESH_C, edgecolor="none",
                                    zorder=3))
        if present:
            ax.plot([x0, x1], [cy, cy], color=MESH_C, lw=9, solid_capstyle="butt",
                    zorder=2)
        ax.plot([x0, x1], [cy, cy], color=DESIGN_C, lw=1.0, ls=(0, (3, 2.5)), zorder=4)
        for t in MIDSPAN:
            px = x0 + t * (x1 - x0)
            ax.plot([px], [cy], marker="o", markersize=5.5,
                    markerfacecolor=colour if not present else "#ffffff",
                    markeredgecolor=colour, markeredgewidth=1.5, zorder=5)
        ax.text((x0 + x1) / 2, cy + 0.42, label, fontsize=9, color=colour,
                ha="center", fontweight="bold")
        ax.text((x0 + x1) / 2, cy - 0.52,
                "material at every\nmiddle sample -> present" if present else
                f"nothing within {TOL_MM} mm of\nany middle sample -> deleted",
                fontsize=8.2, color=_INK2, ha="center", va="top", linespacing=1.55)

    ax.text(9.35, y - 0.58, "why the middle, never the ends", fontsize=9.5, color=_INK,
            fontweight="bold")
    ax.text(9.35, y - 0.92,
            "The joints at both ends belong to the\n"
            "other struts meeting there, so they are\n"
            "still solid even when the strut between\n"
            "them is gone.\n\n"
            "Sample near an end and every deleted\n"
            "strut looks present. So it samples only\n"
            f"{int(MIDSPAN[0] * 100)}-{int(MIDSPAN[-1] * 100)}% along.",
            fontsize=8.4, color=_INK2, va="top", linespacing=1.65)

    # ---------------------------------------------------------------- stage 4
    y -= h3 + 0.42
    h4 = 2.10
    stage(ax, 0.45, y - h4, 12.7, h4, 4, "Compare the two lists", OK_C)

    ax.text(1.05, y - 0.62, "the CAD's answer", fontsize=9, color=_INK,
            fontweight="bold")
    ax.text(1.05, y - 0.92, "struts actually deleted\non purpose", fontsize=8.4,
            color=_INK2, va="top", linespacing=1.6)
    ax.text(4.45, y - 0.62, "the detector's answer", fontsize=9, color=_INK,
            fontweight="bold")
    ax.text(4.45, y - 0.92, "struts it called `missing`\nfrom the CT scan", fontsize=8.4,
            color=_INK2, va="top", linespacing=1.6)
    ax.annotate("", xy=(4.25, y - 0.95), xytext=(3.35, y - 0.95),
                arrowprops=dict(arrowstyle="<|-|>", color=_GRID, linewidth=1.4))
    ax.text(7.85, y - 0.62, "what comes out", fontsize=9, color=_INK, fontweight="bold")
    ax.text(7.85, y - 0.92,
            "how many it got right, how many it missed, how many it invented\n"
            "(precision, recall, F1) -- plus the count of deleted struts, which\n"
            "owes nothing to the detector at all and is the fully independent part.",
            fontsize=8.4, color=_INK2, va="top", linespacing=1.65)

    ax.text(0.45, 0.62,
            f"Scale and bounding box measured by reading the STL files directly: "
            f"0.stl, 0.5.stl and 1.stl all span {BBOX_MM[0]:.4f} x {BBOX_MM[1]:.4f} x "
            f"{BBOX_MM[2]:.4f} mm -- identical to four decimals, so deleting struts never "
            "moves the outer edge and the size/position\nstep is safe. The long axis "
            "carries the build plates and is excluded from the scale. Procedure as "
            "implemented in src/stl_ground_truth.py; sampling positions and the "
            f"{TOL_MM} mm tolerance are that module's defaults.",
            fontsize=7.8, color=_INK2, va="center", linespacing=1.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/mep_figures")
    a = p.parse_args()
    build(Path(a.out) / "explain_validate_stl.png")


if __name__ == "__main__":
    main()
