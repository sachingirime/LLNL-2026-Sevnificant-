#!/usr/bin/env python
"""The pipeline as a plain block diagram (PNG) — blocks and arrows, nothing else.

A companion to pipeline_diagram.py, which carries the numbers and the caveats. This one
is for the poster's methods column, where the flow has to read in two seconds from a
metre away, so every block is a title and the detail lives in the caption you speak.

    python scripts/poster/block_diagram.py
    python scripts/poster/block_diagram.py --vertical -o outputs/poster/blocks_tall.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

INK, MUTED, RULE = "#0b0b0b", "#52514e", "#b9b8b2"
SURFACE, ACCENT = "#ffffff", "#2a78d6"
INPUT_FC, STAGE_FC, OUT_FC = "#eef4fc", "#f4f3ef", "#fdf3f2"
OUT_EC = "#e34948"

INPUTS = ["X-ray CT volume", "JSON graph"]
STAGES = [
    "Segmentation\n& correction",
    "Morphological\nanalysis",
    "Per-strut\nclassification",
    "Verification",
    "Explainable\npacket (MEP)",
]
OUTPUT = "Defect report"


def rbox(ax, x, y, w, h, label, *, fc, ec, fs, weight="bold", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.030",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=color, zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=17,
                                 lw=2.0, color=ACCENT, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def build_horizontal(out_path: Path, title: str | None):
    # Widths are solved so the last box lands inside the canvas: the chain is
    # 6 boxes + 5 gaps starting after the inputs, and the first attempt overflowed
    # the right edge by 0.6 in.
    W, H = 16.0, 3.0
    BW, BH, GAP = 1.75, 1.02, 0.34
    IW = 2.40                       # "CAD design graph" overran a 1.92 box at 11.5pt
    IX = 0.25
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    mid = 0.85                      # bottom of the chain row
    if title:
        ax.text(IX, H - 0.16, title, fontsize=15, fontweight="bold", color=INK, va="top")
        mid = 0.72

    cy = mid + BH / 2
    rbox(ax, IX, cy + 0.10, IW, 0.86, INPUTS[0], fc=INPUT_FC, ec=ACCENT, fs=11.5)
    rbox(ax, IX, cy - 0.96, IW, 0.86, INPUTS[1], fc=INPUT_FC, ec=ACCENT, fs=11.5)

    x = IX + IW + 0.46
    arrow(ax, (IX + IW, cy + 0.53), (x, cy), rad=-0.16)
    arrow(ax, (IX + IW, cy - 0.53), (x, cy), rad=0.16)

    for i, label in enumerate(STAGES):
        rbox(ax, x, mid, BW, BH, label, fc=STAGE_FC, ec=RULE, fs=11.5)
        ax.text(x + 0.12, mid + BH - 0.15, str(i + 1), fontsize=8.5,
                fontweight="bold", color=MUTED, ha="left", va="center", zorder=3)
        arrow(ax, (x + BW, cy), (x + BW + GAP, cy))
        x += BW + GAP

    rbox(ax, x, mid, BW, BH, OUTPUT, fc=OUT_FC, ec=OUT_EC, fs=11.5)

    fig.savefig(out_path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)
    return out_path


def build_vertical(out_path: Path, title: str | None):
    fig, ax = plt.subplots(figsize=(6.4, 12), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 6.4); ax.set_ylim(1.30, 12); ax.axis("off")

    if title:
        ax.text(3.2, 11.72, title, fontsize=15, fontweight="bold", color=INK,
                va="top", ha="center")

    bw, bh, gap = 2.55, 0.98, 0.60
    y = 10.55 if title else 10.95

    rbox(ax, 0.32, y, bw, bh, INPUTS[0], fc=INPUT_FC, ec=ACCENT, fs=12.5)
    rbox(ax, 3.52, y, bw, bh, INPUTS[1], fc=INPUT_FC, ec=ACCENT, fs=12.5)

    cx, cw = (6.4 - 3.30) / 2, 3.30
    y -= gap + bh
    arrow(ax, (0.32 + bw / 2, y + bh + gap), (cx + cw / 2, y + bh), rad=-0.18)
    arrow(ax, (3.52 + bw / 2, y + bh + gap), (cx + cw / 2, y + bh), rad=0.18)

    for i, label in enumerate(STAGES):
        rbox(ax, cx, y, cw, bh, label, fc=STAGE_FC, ec=RULE, fs=12.5)
        ax.text(cx + 0.16, y + bh - 0.15, str(i + 1), fontsize=9, fontweight="bold",
                color=MUTED, ha="left", va="center", zorder=3)
        if i < len(STAGES) - 1:
            arrow(ax, (cx + cw / 2, y), (cx + cw / 2, y - gap))
        y -= gap + bh

    rbox(ax, cx, y, cw, bh, OUTPUT, fc=OUT_FC, ec=OUT_EC, fs=12.5)
    arrow(ax, (cx + cw / 2, y + bh + gap), (cx + cw / 2, y + bh))

    fig.savefig(out_path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.26)
    plt.close(fig)
    return out_path


# The explainability half. Titles only — the detail belongs in what you say at the poster.
# NOTE: "Attention rollout" is on this diagram at the author's request; unlike the other
# five it has no implementation in this repo (no attention-rollout code in src/ or
# scripts/). Mark it as planned work if the poster is being reviewed on what was built.
EXPLAIN = [
    "Execution trace",
    "Trajectory  (s, a, o)",
    "Attention rollout",
    "Verification checks",
    "Run explanation",
]
PACKET_FIELDS = ("run_id · step · actor\nsession · tool · why\nargs · status\n"
                 "artifact · inputs\nverification")


def build_explain(out_path: Path, title: str | None):
    """Compact pipeline on the left, the explainability packet opened up on the right.

    The panel is sized by its own grid and top-aligned; tying its height to the left
    column made every cell 2.4 in tall for one line of text, and pushed the output box
    off the bottom of the canvas.
    """
    W, H = 12.0, 6.7
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    top = H - 0.22
    if title:
        ax.text(0.30, H - 0.12, title, fontsize=15, fontweight="bold", color=INK, va="top")
        top = H - 0.68

    # ---- left: the measurement pipeline ---------------------------------------
    lx, lw = 0.30, 2.70
    ih, sh = 0.72, 0.90
    y_in1 = top - ih
    y_in2 = y_in1 - ih - 0.14
    rbox(ax, lx, y_in1, lw, ih, INPUTS[0], fc=INPUT_FC, ec=ACCENT, fs=11)
    rbox(ax, lx, y_in2, lw, ih, INPUTS[1], fc=INPUT_FC, ec=ACCENT, fs=11)

    ys = y_in2 - 0.46 - sh
    for i, label in enumerate(STAGES[:3]):
        yy = ys - i * (sh + 0.34)
        rbox(ax, lx, yy, lw, sh, label, fc=STAGE_FC, ec=RULE, fs=11)
        ax.text(lx + 0.12, yy + sh - 0.14, str(i + 1), fontsize=8.5, fontweight="bold",
                color=MUTED, ha="left", va="center", zorder=3)
        if i:
            arrow(ax, (lx + lw / 2, yy + sh + 0.34), (lx + lw / 2, yy + sh))
    arrow(ax, (lx + lw, y_in1 + ih / 2), (lx + lw * 0.78, ys + sh), rad=-0.42)
    arrow(ax, (lx + lw / 2, y_in2), (lx + lw / 2, ys + sh))
    last_y = ys - 2 * (sh + 0.34)

    # ---- right: the packet, sized by its grid ---------------------------------
    px, pw = 3.55, W - 3.55 - 0.30
    cols, rows = 3, 2
    gw = (pw - 0.52 - (cols - 1) * 0.22) / cols
    gh = 1.50   # sized so panel + output ends level with the left column
    header, foot = 0.60, 0.26
    p_top = top
    p_bot = p_top - header - rows * gh - (rows - 1) * 0.22 - foot

    ax.add_patch(FancyBboxPatch((px, p_bot), pw, p_top - p_bot,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                fc="#f7f4fb", ec="#9085e9", lw=1.6, zorder=1))
    ax.text(px + 0.24, p_top - 0.20, "EXPLAINABLE PACKET  (MEP)", fontsize=11.5,
            fontweight="bold", color="#4a3aa7", va="top", zorder=3)

    gx = px + 0.26
    gy_top = p_top - header
    for i, name in enumerate(EXPLAIN):
        col, rowi = i % cols, i // cols
        bx = gx + col * (gw + 0.22)
        by = gy_top - gh - rowi * (gh + 0.22)
        rbox(ax, bx, by, gw, gh, name, fc=SURFACE, ec=RULE, fs=10.5)

    bx = gx + 2 * (gw + 0.22)
    by = gy_top - gh - (gh + 0.22)
    ax.add_patch(FancyBboxPatch((bx, by), gw, gh,
                                boxstyle="round,pad=0.010,rounding_size=0.035",
                                fc="#f2eefb", ec="#9085e9", lw=1.3, zorder=2))
    ax.text(bx + gw / 2, by + gh - 0.22, "Packet fields", ha="center", va="center",
            fontsize=9.4, fontweight="bold", color=INK, zorder=3)
    ax.text(bx + gw / 2, by + gh / 2 - 0.16, PACKET_FIELDS, ha="center", va="center",
            fontsize=6.6, color=MUTED, family="monospace", zorder=3, linespacing=1.5)

    arrow(ax, (lx + lw, last_y + sh / 2), (px, p_bot + 0.35), rad=-0.14)

    # ---- output ---------------------------------------------------------------
    ow, oh = 3.10, 0.76
    ox = px + pw / 2 - ow / 2
    oy = p_bot - 0.34 - oh
    rbox(ax, ox, oy, ow, oh, OUTPUT, fc=OUT_FC, ec=OUT_EC, fs=11.5)
    arrow(ax, (ox + ow / 2, p_bot), (ox + ow / 2, oy + oh))

    fig.savefig(out_path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="outputs/poster/block_diagram.png")
    p.add_argument("--vertical", action="store_true", help="tall layout for a poster column")
    p.add_argument("--explain", action="store_true",
                   help="compact pipeline + the explainability packet opened up")
    p.add_argument("--title", default=None, help="omit for a bare diagram")
    args = p.parse_args()

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    build = (build_explain if args.explain else
             build_vertical if args.vertical else build_horizontal)
    out = build(out, args.title)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
