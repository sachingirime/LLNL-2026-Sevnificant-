#!/usr/bin/env python
"""The Agentic MEP block diagram, instantiated on this project.

Redraws the three-stage Agentic MEP structure of Chaduvula et al., "From Features to
Actions" (arXiv:2602.06841) -- execution trace, trajectory, rubric flags, with the
multi-step loop closing back -- and fills each stage with what it actually is here, so the
paper's abstraction and this repository's code can be read against each other in one
picture.

The left column keeps the paper's wording. The right column names the real tools, files
and flags, all of them taken from the code rather than paraphrased:

    stage 1  the tools in src/mcp_server.py, each carrying the caller's `why=`
    stage 2  a packet per call in outputs/mep/<run>/trace.jsonl, the three MEP slots of
             mep.py: explanation artifact, linked evidence, verification signals
    stage 3  the five rubrics of mep_rubric.RUBRICS and the six check tools of
             check_tools.py

This is a diagram of the architecture, not of a run. For a particular run see
`scripts/mep_figures.py`.

    python scripts/mep_concept.py
    python scripts/mep_concept.py --out outputs/mep_figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lattice_iou import _INK, _INK2, _SURFACE  # noqa: E402

# The reference figure's palette: a green header, then warm / cool / green stages. These
# are structural fills behind black text, not a categorical data encoding -- every box is
# named in words, so nothing here depends on telling the hues apart.
HEADER = "#2f7d55"
STAGES = [("#fbe3c6", "#e0a15a"), ("#c9dcf5", "#7ba3d8"), ("#cfe8d4", "#7fb98e")]
LOOP = "#2f7d55"


def build(out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({"font.family": "DejaVu Serif", "text.color": _INK})

    W, H = 14.9, 9.9
    fig = plt.figure(figsize=(W, H), facecolor=_SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    x0, w = 0.75, 6.35          # the abstract column, as in the paper
    xg = x0 + w + 0.30          # a narrow gutter, for the loop back only
    xr = x0 + w + 1.45          # this project's instantiation, beside it

    def box(y, h, face, edge):
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                                    facecolor=face, edgecolor=edge, linewidth=1.6,
                                    zorder=2))

    # ------------------------------------------------------------------ header
    hh = 0.86
    y = H - 0.55 - hh
    ax.add_patch(FancyBboxPatch((x0, y), w, hh, boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=HEADER, edgecolor="none", zorder=2))
    ax.text(x0 + w / 2, y + hh / 2, "Agentic MEP", fontsize=21, color="#ffffff",
            ha="center", va="center", fontweight="bold", zorder=3)
    ax.text(xr, y + hh / 2, "as built in this repository", fontsize=12, color=_INK2,
            va="center", style="italic")

    stages = [
        ("Execution trace + reasoning", "(tool calls, decisions)",
         [("every tool call carries  why=  and  actor=", True),
          ("begin_analysis  records the user's prompt verbatim", False),
          ("segment -> refit registration -> detect defects -> detect nodes", False),
          ("-> validate -> visualize -> explain_run", False),
          ("settings the agent chose, not the user, are singled out", False)]),
        (r"Trajectory:  $(s_0, a_0, o_0, \ldots, s_T)$", "state, action, observation",
         [("one packet per call, appended to trace.jsonl", True),
          ("s  mask.tif, registered design, correction.json", False),
          ("     each fingerprinted by size + edge sha256", False),
          ("a  the tool and every argument it was given", False),
          ("o  what it reported, and the files it wrote", False)]),
        ("Rubric flags + replay checks", "(intent, tool correctness, state consistency)",
         [("five flags, scored over the finished run", True),
          ("State Tracking . Tool Correctness . Tool-Choice Accuracy", False),
          ("Error Awareness & Recovery . Intent Alignment", False),
          ("six checks the agent must call for itself:", True),
          ("coordinate frame . alignment residual . threshold sensitivity", False),
          ("cache staleness . provenance . detector agreement", False)]),
    ]

    tops, bots = [], []
    y -= 0.62
    for i, (title, sub, rows) in enumerate(stages):
        h = 1.62 + 0.005
        top = y
        bot = y - h
        box(bot, h, *STAGES[i])
        ax.text(x0 + w / 2, top - 0.42, title, fontsize=15.5, ha="center", va="center")
        ax.text(x0 + w / 2, top - 0.86, sub, fontsize=11.5, ha="center", va="center",
                color="#3a3a38")
        if i == 2:
            ax.text(x0 + w / 2, top - 1.24, "the agent's behaviour, not the server's",
                    fontsize=11.5, ha="center", va="center", color="#3a3a38")

        ry = top - 0.34
        for text, strong in rows:
            ax.text(xr, ry, text, fontsize=10.2 if strong else 9.6,
                    color=_INK if strong else _INK2, va="center",
                    fontweight="bold" if strong else "normal")
            ry -= 0.265
        tops.append(top)
        bots.append(bot)

        if i < len(stages) - 1:
            ax.annotate("", xy=(x0 + w / 2, bot - 0.50), xytext=(x0 + w / 2, bot - 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="#8b8a86", linewidth=2.2))
        y = bot - 0.62

    # ------------------------------------------------- the multi-step loop back
    # The loop lives in its own gutter. Curved across the right-hand column it ran
    # straight through the instantiation text and through its own label.
    ax.annotate("", xy=(xg, tops[0] - 0.55), xytext=(xg, bots[2] + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=LOOP, linewidth=2.0,
                                linestyle=(0, (5, 3)),
                                connectionstyle="arc3,rad=-0.16"))
    ax.text(xg + 0.62, (tops[0] + bots[2]) / 2, "multi-step", fontsize=11,
            color=LOOP, rotation=270, ha="center", va="center")

    ax.text(x0 + w / 2, bots[2] - 0.62,
            "Example:  9x9x9 octet-truss lattice, X-ray CT defect classification",
            fontsize=13, color=LOOP, ha="center", va="center", style="italic")

    ax.text(x0, 0.42,
            "Slot 3 is left empty when a packet is written and filled in afterwards. "
            "Verification here is a set of tools the agent has to choose to call, so "
            "whether a step was checked is a fact about the run;\nif the server verified "
            "everything automatically there would be nothing left to explain. "
            "Structure after Chaduvula et al., \"From Features to Actions\" "
            "(arXiv:2602.06841), sections 3.1 and 3.4.",
            fontsize=8.6, color=_INK2, va="center", linespacing=1.75)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/mep_figures")
    a = p.parse_args()
    build(Path(a.out) / "mep_concept.png")


if __name__ == "__main__":
    main()
