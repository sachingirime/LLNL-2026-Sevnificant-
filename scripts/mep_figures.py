#!/usr/bin/env python
"""Figures of an MEP run's own record: what was done, what it rested on, what it supports.

The story report already says all of this in prose. What prose cannot show is *shape* --
whether a check landed before or after the claim it verifies, whether two steps that read
"the same file" read the same bytes, whether the classes a run reports are the classes its
checks actually cover. Those are relations between steps, and a reader checks a relation by
looking at it.

Three figures, each answering one question:

    mep_trajectory.png   What happened, in order, by whom -- and for each check, which
                         steps it verifies and whether it ran BEFORE or AFTER them.
    mep_provenance.png   Which file every step read and wrote, and whether a file's
                         content changed between one step reading it and the next.
    mep_trust.png        Which defect classes the checks in this run actually cover, and
                         the five rubric dimensions.

Nothing here re-scores anything. Verdicts come from `mep_rubric.parse_check`, the rubric
flags from `mep_rubric.score`, the class sensitivity table from `mep_narrative`. This reads
`src/` and modifies none of it.

    python scripts/mep_figures.py --run real
    python scripts/mep_figures.py --run real --out outputs/mep_figures

**Packets are keyed by `packet_id` and ordered by the trace file, not by `step`.** One MCP
server process numbers its own steps, so a run assembled from several processes -- which is
what a sub-agent produces -- repeats step numbers. The `real` trace holds 11 packets under 8
distinct step numbers, and anything keying a dict on `step` silently keeps 8 of them. That
is a property of the trace, not of these figures; the figures report it rather than
inheriting it.

Palette: the actor hues and the status set are validated for CVD separation, chroma and
lightness against the #fcfcfb surface (`scripts/validate_palette.js` from the dataviz
skill). Both carry a sub-3:1 contrast warning, which is discharged by the direct labels on
every mark -- no state in these figures is carried by colour alone.
"""
from __future__ import annotations

import argparse
import os
import textwrap
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.mep_narrative as narrative  # noqa: E402
import src.mep_rubric as rubric  # noqa: E402
from src.lattice_iou import _GRID, _INK, _INK2, _SURFACE  # noqa: E402

# Validated categorical order. Assigned to actors in first-appearance order and never
# cycled: a sixth actor folds into NEUTRAL rather than repeating a hue, because a repeated
# hue in an attribution figure is worse than an unnamed one.
ACTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b6bbf", "#b8902b"]
NEUTRAL = "#8f8d88"

# Reserved status colours -- never used for an actor. Every use is accompanied by a word,
# so the figures survive greyscale printing and colour-vision deficiency.
STATUS = {"pass": "#0d8f57", "warn": "#e0a112", "fail": "#d0342c",
          "unverified": NEUTRAL, "check": _INK2}
STATUS_WORD = {"pass": "verified", "warn": "warned", "fail": "REFUTED",
               "unverified": "not checked", "check": "check"}
VERDICT_STATE = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}


# ------------------------------------------------------------------ the record

def load_trace(run):
    """Packets in the order the trace file recorded them.

    `mep_rubric.load` sorts by `step`, which interleaves the processes of a multi-agent run
    into an order that never happened. Timestamps are monotonic in file order, so file
    order is the trajectory.
    """
    import json
    path = rubric.mep.trace_path(run)
    if not os.path.isfile(path):
        raise SystemExit(f"no trace for run {run!r} at {path}")
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def checks_of(packets):
    return [c for c in (rubric.parse_check(p) for p in packets) if c]


def attach_by_id(packets):
    """{packet_id: [check, ...]} -- the same rule as `mep_rubric.link`, keyed safely.

    A check verifies any step that read or wrote its TARGET path. Keying on packet_id
    rather than step is the only difference, and it is the difference between describing
    11 packets and describing 8 of them.
    """
    attached = {p["packet_id"]: [] for p in packets}
    by_step = {}
    for p in packets:
        c = rubric.parse_check(p)
        if c:
            by_step[(c["check"], c["target"], c["step"])] = p["packet_id"]
    for check in checks_of(packets):
        target = check["target"]
        pid = by_step[(check["check"], target, check["step"])]
        for packet in packets:
            if packet.get("kind") == "check":
                continue
            touched = [r["path"] for r in packet.get("inputs", [])]
            touched += packet.get("artifact_files", [])
            if any(target == t or target.startswith(t + os.sep) or t.startswith(target + os.sep)
                   for t in touched):
                attached[packet["packet_id"]].append({**check, "_pid": pid})
    return attached


def live_checks(packets):
    """The set of check steps whose verdict is still current.

    A check re-run against the same target after its cause was addressed supersedes its own
    earlier result -- that is `mep_narrative.latest_verdicts`' rule, and it is what the
    story report tells the reader. `mep_rubric.state_of` does NOT supersede: it treats any
    attached FAIL as fatal forever. The two disagree on exactly the runs that recovered,
    which is the case worth drawing, so these figures use the superseding rule and mark the
    superseded arc rather than hiding it.
    """
    current = {}
    for p in packets:
        c = rubric.parse_check(p)
        if c:
            current[(c["check"], c["target"])] = p["packet_id"]
    return set(current.values())


def state_of(packet, attached, live=None):
    if packet.get("kind") == "check":
        return "check"
    live_attached = [c for c in attached
                     if live is None or c.get("_pid") in live]
    if not live_attached:
        return "unverified"
    if any(c["verdict"] == "FAIL" for c in live_attached):
        return "fail"
    if any(c["verdict"] == "WARN" for c in live_attached):
        return "warn"
    return "pass"


def voiceless_checks(packets):
    """Packets declared `kind="check"` that emit no parseable verdict.

    `validate_against_stl` is one. The consequence is not cosmetic: `link` skips every
    `kind="check"` packet when looking for steps to verify, so such a packet can neither
    verify anything nor be verified by anything. It is invisible to the rubric in both
    directions, and a reader counting checks would count it.
    """
    return [p for p in packets
            if p.get("kind") == "check" and rubric.parse_check(p) is None]


def actor_lut(packets):
    seen = []
    for p in packets:
        if p["actor"] not in seen:
            seen.append(p["actor"])
    return {a: (ACTOR_COLORS[i] if i < len(ACTOR_COLORS) else NEUTRAL)
            for i, a in enumerate(seen)}, seen


def short(path, keep=2, width=46):
    """Last `keep` path components, elided in the middle if still too long to fit."""
    parts = Path(path).parts
    name = os.sep.join(parts[-keep:]) if len(parts) > keep else path
    if len(name) <= width:
        return name
    head = (width - 3) // 2
    return name[:head] + "..." + name[-(width - 3 - head):]


# ------------------------------------------------------------------- plumbing

def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE,
        "font.size": 9, "text.color": _INK, "axes.labelcolor": _INK2,
        "xtick.color": _INK2, "ytick.color": _INK2,
    })
    return plt


def _bare(ax, keep=()):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(_GRID)
            ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0 if not keep else 3, colors=_INK2, labelsize=8)


def _caption(fig, text, y=0.008):
    fig.text(0.006, y, text, color=_INK2, fontsize=7.6, va="bottom", linespacing=1.55)


# ------------------------------------------------------------------ figure 1

def trajectory(packets, out_path):
    """The run in the order it happened, with each check joined to what it verifies.

    The arc direction is the point of the figure. A check drawn arcing *forwards* ran
    before the step it covers, so that step was measured knowing the check had passed. One
    arcing *backwards* ran after, so the claim was made first and audited later -- which is
    still worth something and is not the same thing, and no table of verdicts distinguishes
    them.
    """
    plt = _rc()
    lut, actors = actor_lut(packets)
    attached = attach_by_id(packets)
    n = len(packets)
    idx = {p["packet_id"]: i for i, p in enumerate(packets)}

    live = live_checks(packets)
    fig = plt.figure(figsize=(max(11.5, 1.34 * n), 4.4 + 0.46 * len(actors) + 2.1),
                     facecolor=_SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.15 + 0.38 * len(actors), 0.85],
                          left=0.095, right=0.995, top=0.905, bottom=0.155, hspace=0.06)
    ax_arc, ax_lane, ax_dur = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- arcs: check -> the steps whose files it verifies.
    # `rad` is scaled by 1/span so a long arc and a short one bulge by about the same
    # amount; at fixed rad the long ones balloon off the top of the panel and the short
    # ones flatten into the baseline, and the shape stops meaning anything.
    forward = backward = 0
    ax_arc.axhline(0, color=_GRID, linewidth=1.0, zorder=1)
    for pid, checks in attached.items():
        for c in checks:
            src, dst = idx.get(c["_pid"]), idx[pid]
            if src is None:
                continue
            state = VERDICT_STATE[c["verdict"]]
            ahead = dst > src
            forward += ahead
            backward += not ahead
            superseded = c["_pid"] not in live
            # ONE sign for both directions. arc3 offsets its control point along the
            # perpendicular of (dst - src), and that perpendicular already flips when the
            # arc runs backwards -- so a single sign puts forward arcs above the line and
            # backward ones below, automatically. Flipping the sign by direction as well
            # cancels that out and draws every arc on the same side, which is what the
            # first version did: it counted 3 backward arcs and drew all 13 above.
            rad = min(0.55, max(0.14, 1.9 / max(abs(dst - src), 1)))
            ax_arc.annotate(
                "", xy=(dst, 0), xytext=(src, 0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=STATUS[state], linewidth=1.5,
                                alpha=0.45 if superseded else 0.9, shrinkA=3, shrinkB=5,
                                linestyle=(0, (4, 2.5)) if superseded else "-",
                                connectionstyle=f"arc3,rad={-rad}"))
    ax_arc.set_xlim(-0.6, n - 0.4)
    ax_arc.set_ylim(-1.15, 1.15)
    _bare(ax_arc)
    ax_arc.set_xticks([])
    ax_arc.set_yticks([])
    ax_arc.text(0.002, 0.965, f"ABOVE the line: the check ran BEFORE the step it verifies "
                f"({forward})", transform=ax_arc.transAxes, fontsize=8, color=_INK2,
                va="top")
    ax_arc.text(0.002, 0.035, f"BELOW: it ran AFTER, so the claim was audited, not informed "
                f"({backward})    dashed = superseded by a later re-run of the same check",
                transform=ax_arc.transAxes, fontsize=8, color=_INK2, va="bottom")

    # ---- lanes: one row per actor
    ypos = {a: len(actors) - 1 - i for i, a in enumerate(actors)}
    for a in actors:
        ax_lane.axhline(ypos[a], color=_GRID, linewidth=0.8, zorder=0)
    voiceless = {p["packet_id"] for p in voiceless_checks(packets)}
    for i, p in enumerate(packets):
        y = ypos[p["actor"]]
        st = state_of(p, attached[p["packet_id"]], live)
        v = rubric.parse_check(p)
        is_check = v is not None
        mute = p["packet_id"] in voiceless
        if is_check:
            st = VERDICT_STATE[v["verdict"]]
        ax_lane.scatter([i], [y], s=210,
                        marker="D" if is_check else ("X" if mute else "o"),
                        facecolor=STATUS[st] if is_check else
                        (NEUTRAL if mute else lut[p["actor"]]),
                        edgecolor=_SURFACE, linewidth=1.6, zorder=3)
        ax_lane.annotate(p["tool"], xy=(i, y), xytext=(0, 13 if i % 2 == 0 else -20),
                         textcoords="offset points", ha="center", fontsize=7.4,
                         color=_INK, rotation=0)
        sub = ("declares kind=check,\nemits no verdict" if mute
               else None if is_check else STATUS_WORD[st])
        if sub:
            ax_lane.annotate(sub, xy=(i, y), xytext=(0, 24 if i % 2 == 0 else -31),
                             textcoords="offset points", ha="center", fontsize=6.8,
                             color=NEUTRAL if mute else STATUS[st], linespacing=1.35,
                             fontweight="bold" if st in ("fail", "unverified") and not mute
                             else "normal")
    ax_lane.set_xlim(-0.6, n - 0.4)
    ax_lane.set_ylim(-0.9, len(actors) - 0.1)
    ax_lane.set_yticks([ypos[a] for a in actors])
    ax_lane.set_yticklabels(actors, fontsize=8.5)
    ax_lane.set_xticks([])
    _bare(ax_lane, keep=("left",))

    # ---- durations, on their own axis: a second measure gets a second chart, not a
    # second y-scale on this one
    dur = [float(p.get("duration_s") or 0.0) for p in packets]
    ax_dur.bar(range(n), dur, width=0.42, color=_INK2, zorder=2)
    for i, d in enumerate(dur):
        if d >= 1.0:
            ax_dur.annotate(f"{d:.0f}s", xy=(i, d), xytext=(0, 3), ha="center",
                            textcoords="offset points", fontsize=7, color=_INK2)
    ax_dur.set_xlim(-0.6, n - 0.4)
    ax_dur.set_ylabel("seconds", fontsize=8)
    ax_dur.set_xticks(range(n))
    ax_dur.set_xticklabels([f"{i + 1}" for i in range(n)], fontsize=7.5)
    ax_dur.set_xlabel("packet, in the order the trace recorded it "
                      "(NOT the `step` field -- see the note below)", fontsize=8)
    _bare(ax_dur, keep=("left", "bottom"))
    ax_dur.grid(axis="y", color=_GRID, linewidth=0.6)
    ax_dur.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          markerfacecolor=lut[a], markeredgecolor=_SURFACE, label=a)
               for a in actors]
    handles += [plt.Line2D([], [], marker="D", linestyle="", markersize=8,
                           markerfacecolor=STATUS[s], markeredgecolor=_SURFACE,
                           label=f"check: {STATUS_WORD[s]}")
                for s in ("pass", "warn", "fail")]
    if voiceless:
        handles.append(plt.Line2D([], [], marker="X", linestyle="", markersize=9,
                                  markerfacecolor=NEUTRAL, markeredgecolor=_SURFACE,
                                  label="no verdict emitted"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.988),
               ncol=len(handles), frameon=False, fontsize=7.8, handletextpad=0.35,
               columnspacing=1.3)

    nstep = len({p["step"] for p in packets})
    fig.suptitle(f"Run {packets[0]['run_id']}: what was done, by whom, and what verified it",
                 color=_INK, fontsize=12, x=0.006, ha="left", y=0.985)
    _caption(fig,
             f"{len(packets)} packets under {nstep} distinct `step` numbers"
             + ("  --  step is per-process, so it is NOT a key; these figures index by "
                "packet_id and trace order." if nstep < len(packets) else ".")
             + "\nA step with no arc into it was never covered by a check: that is "
               "'not checked', which is a different claim from 'checked and fine'.")
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 2

def provenance(packets, out_path):
    """Every file each step touched, and whether its bytes changed underneath the run.

    The edges are content, not narration: each input carries an `edge_sha256`, so when the
    same path is read twice the figure can say whether it was the same file both times. A
    run that reads `correction.json` before and after something rewrote it is not a run
    whose steps agree with each other, and no ordering of tool names would reveal that.
    """
    plt = _rc()
    lut, actors = actor_lut(packets)

    touches = {}          # path -> [(packet index, "read"/"write", fingerprint)]
    for i, p in enumerate(packets):
        for r in p.get("inputs", []):
            touches.setdefault(r["path"], []).append((i, "read", r.get("edge_sha256")))
        for f in p.get("artifact_files", []):
            touches.setdefault(f, []).append((i, "write", None))
    if not touches:
        print("  no file touches recorded; skipping provenance figure")
        return

    order = sorted(touches, key=lambda k: (touches[k][0][0], k))
    n = len(packets)
    # Explicit vertical budget in inches. The x labels are tool names rotated 38 deg and
    # the caption sits under them; letting the axes take a fraction of the figure instead
    # puts the caption through the labels as soon as a run has more than a few steps.
    h_rows, h_top, h_labels, h_cap = 0.42 * len(order), 1.15, 1.45, 0.62
    height = h_rows + h_top + h_labels + h_cap
    fig = plt.figure(figsize=(max(11.0, 1.34 * n), height), facecolor=_SURFACE)
    ax = fig.add_axes([0.30, (h_labels + h_cap) / height, 0.685, h_rows / height])

    changed = []
    for row, path in enumerate(order):
        y = len(order) - 1 - row
        ax.axhline(y, color=_GRID, linewidth=0.8, zorder=0)
        events = touches[path]
        xs = [e[0] for e in events]
        if len(xs) > 1:
            ax.plot([min(xs), max(xs)], [y, y], color=_INK2, linewidth=1.0, alpha=0.5,
                    zorder=1)
        prints = [e[2] for e in events if e[1] == "read" and e[2]]
        drift = len(set(prints)) > 1
        if drift:
            changed.append(path)
        for i, kind, fp in events:
            ax.scatter([i], [y], s=132 if kind == "write" else 96,
                       marker="s" if kind == "write" else "o",
                       facecolor=lut[packets[i]["actor"]] if kind == "write" else _SURFACE,
                       edgecolor=lut[packets[i]["actor"]], linewidth=1.7, zorder=3)
        if drift:
            # Ringed AND labelled AND the row name turns red. One actor hue (#eb6834) sits
            # near the status red, so a lone coloured ring would be ambiguous against an
            # orange-actor marker; three channels make it unambiguous in greyscale too.
            ax.scatter(xs, [y] * len(xs), s=300, marker="o", facecolor="none",
                       edgecolor=STATUS["fail"], linewidth=2.2, zorder=4)
            ax.annotate("content changed between reads", xy=(max(xs), y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=STATUS["fail"], fontweight="bold")

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(p) for p in reversed(order)], fontsize=7.6)
    for tick, path in zip(ax.get_yticklabels(), reversed(order)):
        if path in changed:
            tick.set_color(STATUS["fail"])
            tick.set_fontweight("bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([p["tool"] for p in packets], rotation=38, ha="right", fontsize=7.4)
    _bare(ax, keep=("left", "bottom"))

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=_INK2,
                          label="wrote it"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          markerfacecolor=_SURFACE, markeredgecolor=_INK2,
                          label="read it"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                          markerfacecolor="none", markeredgecolor=STATUS["fail"],
                          label="edge_sha256 differs between reads")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
              frameon=False, fontsize=7.6, handletextpad=0.4, columnspacing=1.4)

    fig.suptitle(f"Run {packets[0]['run_id']}: what each step read and wrote",
                 color=_INK, fontsize=12, x=0.006, ha="left", y=1 - 0.30 / height)
    _caption(fig,
             "Marker colour is the actor, as in the trajectory figure. A row with only "
             "open circles is an input the run never produced -- its provenance is outside "
             "this trace.\n"
             + (f"{len(changed)} file(s) changed content between reads."
                if changed else "No file changed content between reads in this run."),
             y=0.10 / height)
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 3

def trust(packets, out_path):
    """Which classes this run's checks actually cover, and the five rubric dimensions.

    The cell that matters is "not run". A check that never ran leaves the classes it would
    have covered in exactly the state a failed check leaves them -- unsupported -- and a
    report that lists only the checks it did run makes that invisible.
    """
    plt = _rc()
    sens = narrative._CLASS_SENSITIVITY
    classes = [c for c in narrative.CLASSES if c != "nominal"]
    checks = list(sens)
    verdicts = {c["check"]: c["verdict"] for c in narrative.latest_verdicts(packets)}
    result = rubric.score(rubric.load(packets[0]["run_id"]))

    grid = np.empty((len(classes), len(checks)), object)
    for r, name in enumerate(classes):
        for c, check in enumerate(checks):
            if name in sens[check]["spares"]:
                grid[r, c] = "spared"
            elif check not in verdicts:
                grid[r, c] = "not run"
            else:
                grid[r, c] = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}[verdicts[check]]

    cell_face = {"pass": STATUS["pass"], "warn": STATUS["warn"], "fail": STATUS["fail"],
                 "not run": "#e6e4de", "spared": _SURFACE}
    cell_ink = {"pass": _SURFACE, "warn": _INK, "fail": _SURFACE,
                "not run": _INK2, "spared": _GRID}
    cell_word = {"pass": "covered", "warn": "warned", "fail": "REFUTED",
                 "not run": "not run", "spared": "n/a"}

    flags = result["flags"]
    height = 1.45 + 0.50 * len(classes) + 0.46 * len(flags) + 1.15
    fig = plt.figure(figsize=(13.4, height), facecolor=_SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.50 * len(classes), 0.46 * len(flags)],
                          left=0.115, right=0.99, top=1 - 1.12 / height,
                          bottom=0.98 / height, hspace=0.30)
    ax = fig.add_subplot(gs[0])

    for r in range(len(classes)):
        for c in range(len(checks)):
            s = grid[r, c]
            ax.add_patch(plt.Rectangle((c - 0.46, r - 0.42), 0.92, 0.84,
                                       facecolor=cell_face[s], edgecolor=_GRID,
                                       linewidth=0.8, zorder=2))
            ax.text(c, r, cell_word[s], ha="center", va="center", fontsize=7.8,
                    color=cell_ink[s], zorder=3,
                    fontweight="bold" if s == "fail" else "normal")
    ax.set_xlim(-0.55, len(checks) - 0.45)
    ax.set_ylim(-0.55, len(classes) - 0.45)
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([c.replace("check_", "") for c in checks], fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    _bare(ax)
    # No axes title here: the column headers sit on top of this axis, so a title has to
    # clear them and lands in the suptitle. The suptitle names the figure instead.
    fig.text(0.115, 1 - 0.62 / height, "Which checks cover which defect class",
             color=_INK, fontsize=10, va="bottom")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, len(flags) - 0.5)
    for i, (label, info) in enumerate(flags.items()):
        y = len(flags) - 1 - i
        bad = info["violated"]
        state = "fail" if bad else "pass"
        ax2.add_patch(plt.Rectangle((0.0, y - 0.34), 0.017, 0.68,
                                    facecolor=STATUS[state], edgecolor="none"))
        ax2.text(0.028, y, label, va="center", fontsize=8.6, color=_INK)
        ax2.text(0.40, y, "VIOLATED" if bad else "satisfied", va="center", fontsize=8.4,
                 color=STATUS[state], fontweight="bold" if bad else "normal")
        ev = " ".join((info["evidence"] or [""])[0].split())
        ax2.text(0.50, y, "\n".join(textwrap.wrap(ev, 78)[:2]), va="center",
                 fontsize=7.2, color=_INK2, linespacing=1.45)
    _bare(ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("The five rubric dimensions", color=_INK, fontsize=10, loc="left", pad=8)

    missing = [c for c in checks if c not in verdicts]
    # Checks the run DID emit that this table cannot place. A check whose name is not a key
    # of _CLASS_SENSITIVITY moves nothing here however it came out -- so a run can execute
    # a check, have it fail, and leave every cell above untouched. On the `real` trace that
    # is check_registration_drift, which no longer exists in check_tools.py: the trace was
    # recorded before it was renamed check_alignment_residual. The column therefore reads
    # "not run" for a concern the run did in fact test, under the older name.
    unmapped = sorted({c["check"] for c in narrative.latest_verdicts(packets)}
                      - set(checks))
    fig.suptitle(f"Run {packets[0]['run_id']}: what the run's own checks support",
                 color=_INK, fontsize=12, x=0.006, ha="left", y=1 - 0.22 / height)
    lines = ["'n/a' means the check cannot move that class, so not running it costs "
             "nothing. 'not run' means it can, and did not: those classes are "
             "unsupported, not sound.",
             (f"{len(missing)} of {len(checks)} sensitivity checks never ran: "
              + ", ".join(c.replace("check_", "") for c in missing)
              if missing else "Every sensitivity check ran.")]
    if unmapped:
        lines.append("Ran but absent from the sensitivity table, so moving no cell above: "
                     + ", ".join(unmapped)
                     + "  --  check the trace is not older than the tool set.")
    _caption(fig, "\n".join(lines), y=0.10 / height)
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------ figure 4

KIND_COLOR = {"goal": "#6b6a67", "check": "#2a78d6", "analysis": "#0d8f57",
              "view": "#9b6bbf"}
KIND_WORD = {"goal": "the request", "check": "check", "analysis": "measure",
             "view": "render"}


def flow(packets, out_path, wrap=74):
    """The run read top to bottom: the prompt, then each tool and why the agent chose it.

    The plainest possible view, and the one to reach for first. The other three figures
    answer questions a reader has *after* they know what happened; this one is what
    happened. Every line of it is quoted from the trace -- the tool name, the agent's own
    `why`, the verdict a check printed, the argument values it passed -- so it is a
    transcript with a shape, not a summary.

    Decisions sit in the right-hand gutter beside the step that made them. A parameter is
    listed there when it is a judgement call rather than a path (`mep_narrative._DISCRETIONARY`)
    and the user never named it, which is the difference between "the agent measured it this
    way" and "the agent was told to".
    """
    plt = _rc()
    goal = narrative.read_goal(packets)
    choices = narrative.discretionary_choices(packets)
    live = live_checks(packets)
    attached = attach_by_id(packets)
    lut, actors = actor_lut(packets)

    by_step = {}
    for c in choices:
        by_step.setdefault((c["step"], c["tool"]), []).append(c)

    # ---- measure first, then place: box height follows its own wrapped text
    W, x0, x1, xg = 13.6, 0.45, 8.5, 8.95
    line_h, pad, gap = 0.175, 0.16, 0.42
    rows = []
    for i, p in enumerate(packets):
        why = textwrap.wrap(" ".join((p.get("why") or "").split()), wrap) or ["(no reason recorded)"]
        note = by_step.get((p["step"], p["tool"]), [])
        # A call that raised is not a step that happened. The packet records status and
        # error and nothing was drawing them, so a crashed call sat in the flow looking
        # exactly like a completed one -- which is the failure mode this whole figure set
        # exists to catch, reproduced inside the figure set.
        err = (textwrap.wrap(" ".join((p.get("error") or "").split()), wrap)
               if p.get("status") not in (None, "ok") else [])
        h = pad * 2 + 0.24 + line_h * (len(why) + len(err)) + (0.10 if err else 0.0)
        rows.append(dict(p=p, why=why, err=err, note=note,
                         h=max(h, 0.62 + 0.20 * len(note))))

    ask = textwrap.wrap('"' + (goal.get("user_prompt") or "not recorded") + '"', 96)
    read_as = textwrap.wrap("read as: " + goal["goal"], 92) if goal.get("goal") else []
    head_h = 0.55 + line_h * (len(ask) + len(read_as)) + 0.30
    H = head_h + sum(r["h"] + gap for r in rows) + 0.95

    fig = plt.figure(figsize=(W, H), facecolor=_SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- the request
    y = H - 0.38
    ax.text(x0, y, "WHAT WAS ASKED", fontsize=8.5, color=_INK2, va="top",
            fontweight="bold")
    y -= 0.30
    for ln in ask:
        ax.text(x0, y, ln, fontsize=11.5, color=_INK, va="top")
        y -= line_h * 1.35
    for ln in read_as:
        ax.text(x0, y, ln, fontsize=8.6, color=_INK2, va="top")
        y -= line_h
    y -= 0.22

    # ---- the steps
    for i, r in enumerate(rows):
        p, why, note, err = r["p"], r["why"], r["note"], r["err"]
        kind = p.get("kind", "analysis")
        v = rubric.parse_check(p)
        colour = KIND_COLOR.get(kind, _INK2)
        if v:
            colour = STATUS[VERDICT_STATE[v["verdict"]]]
        if err:
            colour = STATUS["fail"]

        top, bot = y, y - r["h"]
        ax.add_patch(plt.Rectangle((x0, bot), x1 - x0, r["h"], facecolor="#ffffff",
                                   edgecolor=_GRID, linewidth=0.9, zorder=2))
        ax.add_patch(plt.Rectangle((x0, bot), 0.055, r["h"], facecolor=colour,
                                   edgecolor="none", zorder=3))

        ax.text(x0 + 0.20, top - pad - 0.02, f"{i + 1}", fontsize=9, color=_GRID,
                va="top", fontweight="bold")
        ax.text(x0 + 0.52, top - pad - 0.02, p["tool"], fontsize=10.5, color=_INK,
                va="top", fontweight="bold")
        tag = KIND_WORD.get(kind, kind)
        if err:
            tag = "RAISED - this step did not run"
        elif v:
            tag = f"{tag} -> {v['verdict']}"
            if p["packet_id"] not in live:
                tag += ", later superseded"
        elif kind == "check":
            tag = "check -> no verdict emitted"
        ax.text(x1 - 0.18, top - pad - 0.02, tag, fontsize=8.4, color=colour,
                va="top", ha="right",
                fontweight="bold" if err or (v and v["verdict"] == "FAIL") else "normal")
        if p["actor"] != packets[0]["actor"]:
            ax.text(x1 - 0.18, top - pad - 0.24, f"by {p['actor']}", fontsize=7.6,
                    color=lut[p["actor"]], va="top", ha="right")

        ty = top - pad - 0.34
        for ln in why:
            ax.text(x0 + 0.52, ty, ln, fontsize=8.6, color=_INK2, va="top")
            ty -= line_h
        if err:
            ty -= 0.06
            for ln in err:
                ax.text(x0 + 0.52, ty, ln, fontsize=8.2, color=STATUS["fail"], va="top")
                ty -= line_h

        # what came back, when it is a state a reader should carry forward
        st = state_of(p, attached[p["packet_id"]], live)
        if kind == "analysis" and st == "unverified" and not err:
            ax.text(x0 + 0.52, ty - 0.02, "no check covers this step", fontsize=7.8,
                    color=STATUS["fail"], va="top", fontweight="bold")

        if note:
            ax.text(xg, top - pad - 0.02, "chose on its own:", fontsize=7.8,
                    color=_INK2, va="top", fontweight="bold")
            ny = top - pad - 0.24
            for c in note:
                ax.text(xg + 0.06, ny, f"{c['label']} = {c['value']}", fontsize=8.2,
                        color=_INK, va="top")
                ny -= 0.20
            ax.plot([xg - 0.22, xg - 0.22], [ny + 0.10, top - pad + 0.02],
                    color=_GRID, linewidth=1.2, zorder=1)

        if i < len(rows) - 1:
            ax.annotate("", xy=(x0 + 0.9, bot - gap + 0.06), xytext=(x0 + 0.9, bot - 0.03),
                        arrowprops=dict(arrowstyle="-|>", color=_GRID, linewidth=1.4))
        y = bot - gap

    # Distinct settings, not settings-times-steps: the same five parameters passed to two
    # detect calls is five decisions taken twice, and reporting ten overstates the surface
    # the reader has to audit.
    n_distinct = len({c["param"] for c in choices})
    n_steps = len({c["step"] for c in choices})
    raised = [r["p"]["tool"] for r in rows if r["err"]]
    ax.text(x0, 0.42, f"{len(rows)} tool calls."
            f"  {n_distinct} distinct setting(s) the agent picked itself, across {n_steps} "
            "step(s), listed beside the step that picked them -- the user named none of them.",
            fontsize=8, color=_INK2, va="center")
    if raised:
        ax.text(x0, 0.20, f"{len(raised)} call(s) raised and produced nothing: "
                + ", ".join(raised)
                + ".  Whatever that step was for did not happen in this run.",
                fontsize=8, color=STATUS["fail"], va="center", fontweight="bold")
    fig.savefig(out_path, dpi=170, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ------------------------------------------------------------------------ cli

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="real", help="run id under outputs/mep/")
    p.add_argument("--out", default="outputs/mep_figures")
    p.add_argument("--flow-only", action="store_true",
                   help="just the plain top-to-bottom flow")
    a = p.parse_args()

    packets = load_trace(a.run)
    if not packets:
        raise SystemExit(f"run {a.run!r} has no packets")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {a.run}: {len(packets)} packets, actors "
          f"{sorted({p['actor'] for p in packets})}")

    flow(packets, out / "mep_flow.png")
    if not a.flow_only:
        trajectory(packets, out / "mep_trajectory.png")
        provenance(packets, out / "mep_provenance.png")
        trust(packets, out / "mep_trust.png")


if __name__ == "__main__":
    main()
