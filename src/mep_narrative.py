"""The human-facing account of a run: the question, the answer, and how much to believe it.

``mep_rubric`` produces an audit -- correct, complete, and written for whoever is
debugging the agent. This module answers the question the person who asked for
the analysis actually has: what did you find, how did you get there, and which
parts of the answer should I not rely on.

Three things it does that the audit view does not:

1. **Leads with the finding.** The defect classes and their counts, not the tool
   calls that produced them.

2. **Trusts each class separately.** This is the substantive part. The classes in
   this pipeline do not degrade together, so a report that marks a whole run
   "untrustworthy" throws away the part of it that was never in doubt. `thin`,
   `thick` and `necked` are radii compared to a tolerance band, so they inherit
   every error in the segmentation cut and the registration. `broken` looks
   threshold-free but is not: it asks whether a path exists *through material*,
   and a higher cut can sever a one-voxel bridge and manufacture a break. Only
   `missing` is genuinely robust to the cut, and only because the data says so --
   those struts hold zero matched voxels while the next one up holds 75, so every
   cut inside that gap gives the same answer. The map lives in
   ``_CLASS_SENSITIVITY`` and follows ``scripts/classify_strut_defects.classify``,
   which is where the rules actually are.

3. **Separates what the user asked for from what the agent decided.** Every
   parameter the agent chose without being told is a place the answer could have
   come out differently, and those are invisible in a prose summary.

The user's request enters through ``begin_analysis``. That is the goal slot g_t
of Selim et al. (2026) section 3.1 -- the one part of XAC that transfers to a
tool server, because a goal can be passed as an argument where a planning loop
cannot.
"""

import csv
import html
import os
import time

try:
    from . import mep_rubric
except ImportError:
    import mep_rubric


# Which checks bear on which defect classes, and why. The reason strings are shown
# to the reader, so they have to say what the dependency actually is rather than
# just naming it.
_CLASS_SENSITIVITY = {
    "check_threshold_sensitivity": {
        "affects": ("thin", "thick", "necked", "broken"),
        "spares": ("missing",),
        "because": "thin, thick and necked compare a radius read off the mask to a "
        "tolerance band, so moving the cut moves every radius. `broken` moves too, for a "
        "different reason: it asks whether a path exists through material from one node "
        "to the other, and a higher cut can sever a one-voxel bridge and manufacture a "
        "break. `missing` is the exception -- those struts hold zero matched voxels and "
        "the next strut up holds 75, so no cut inside that gap changes the answer.",
    },
    "check_alignment_residual": {
        "affects": ("missing", "broken", "thin", "thick", "necked"),
        "spares": (),
        "because": "every class is measured inside a window centred on where the graph "
        "says the strut is. The residual offset between a graph node and the material it "
        "names is what decides whether that window lands on the strut or beside it -- and "
        "measured on this specimen, leaving it uncorrected flips 1 strut in 11 at the far "
        "x face and inflates `broken` 2.6-fold.",
    },
    "check_coordinate_frame": {
        "affects": ("missing", "broken", "thin", "thick", "necked"),
        "spares": (),
        "because": "the voxel pitch is recovered from the design's own strut length, so a "
        "design in the wrong frame silently rescales every radius and every window in the run.",
    },
    "check_cache_staleness": {
        "affects": ("missing", "broken"),
        "spares": ("thin", "thick", "necked"),
        "because": "`missing` requires every cross-section to be empty, so it is defined "
        "against the section grid; `broken` is read from the cached connectivity arrays. "
        "A cache written at a different section count is honoured at its own count, which "
        "redefines both without raising an error.",
    },
}

# The classes classify_strut_defects.classify can return, severity-first. `dross` was
# removed from the pipeline -- excess_frac no longer drives a class -- so a report that
# expects it is reading an older run.
CLASSES = ("missing", "broken", "thin", "thick", "necked", "nominal")

# How the agent's tool calls group into the stages a reader thinks in.
_PHASES = [
    ("Prepare the volume", {"segment_ct_dataset", "filter_ct_volume"},
     "Turn raw CT intensities into a binary mask of where the metal is."),
    ("Align design to scan", {"refit_lattice_registration", "rasterize_lattice"},
     "Put the nominal lattice design into the scan's coordinates, so each strut can be "
     "found where the design says it should be."),
    ("Measure and classify", {"detect_lattice_defects", "measure_lattice_iou", "skeletonize",
                              "detect_missing_nodes", "detect_missing_nodes_2d"},
     "Measure every strut and node against its nominal geometry and assign a class."),
    ("Validate", {"validate_against_stl"},
     "Compare the calls against the designed-missing set from the STL."),
    ("Report", {"visualize_slice", "compare_slices", "visualize_lattice_3d",
                "export_lattice_html", "visualize_strut_classes"},
     "Render the result."),
]

# Parameters whose value is a judgement call rather than a path. These are the
# knobs the answer is sensitive to, so the reader is shown which ones the agent
# picked on its own.
_DISCRETIONARY = {
    "threshold": "segmentation cut",
    "tolerance_fraction": "thin/thick band width",
    "sections": "cross-sections per strut",
    "strut_diameter_um": "nominal strut diameter",
    "cell_mm": "unit-cell edge",
    "trim_fraction": "fraction trimmed off each strut end",
    "envelope_factor": "measurement window size",
    "stations": "fill-profile bands",
    "embedded_fraction": "buried-strut exclusion cut",
    "match_tolerance_vox": "detector agreement radius",
    "use_cache": "reuse cached arrays",
}


def read_goal(packets) -> dict:
    """The request this run was made to answer, if the agent recorded one."""
    for packet in packets:
        if packet["tool"] == "begin_analysis":
            args = packet.get("args", {})
            return {
                "user_prompt": args.get("user_prompt", "").strip(),
                "goal": args.get("goal", "").strip(),
                "step": packet["step"],
            }
    return {}


_TRUE = ("1", "1.0", "true", "True", "TRUE", "yes")


def read_classes(packets) -> dict | None:
    """The defect class tally, read from whichever results directory the run wrote.

    Counts over the MEASURABLE population only. detect_lattice_defects excludes struts it
    cannot measure -- outer boundary caps, struts buried in the solid build plates at both
    z ends, and struts whose cross-section window runs off the volume edge -- and its own
    documentation is explicit that these are "reported separately, never mixed into the
    tallies". Mixing them in is not a rounding error: on this specimen it turns 89 missing
    struts (0.53% of 16,733 measurable) into 412 (2.23% of all 18,468), because a strut
    fused into a build plate leaves no signature to distinguish absence from burial. It
    also inflates `thick` roughly fourfold, since plate-embedded struts read as fat.

    Prefers the most recent classification step, so a run that re-ran the detector with
    different settings reports the settings it finished on.
    """
    for packet in reversed(packets):
        if packet["tool"] != "detect_lattice_defects" or packet["status"] != "ok":
            continue
        for directory in packet.get("artifact_files", []):
            path = os.path.join(directory, "strut_classes.csv")
            if not os.path.isfile(path):
                continue
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue

            # Older result formats predate the column. Without it the excluded struts
            # cannot be separated, so the report says so rather than quoting a rate that
            # is silently inflated.
            has_flag = "measurable" in rows[0]
            counted = [r for r in rows if r.get("measurable") in _TRUE] if has_flag else rows

            counts = {}
            for row in counted:
                label = row.get("label", "")
                counts[label] = counts.get(label, 0) + 1
            if not counts:
                continue
            return {
                "counts": counts,
                "total": len(counted),
                "designed": len(rows),
                "excluded": len(rows) - len(counted),
                "population_known": has_flag,
                "source": path,
                "step": packet["step"],
            }
    return None


def figures(packets, budget_bytes=14 << 20) -> list:
    """Images the run produced, newest step last, inlined so the page travels alone.

    A report that references images by relative path stops working the moment it is
    emailed, moved, or opened from anywhere but its own directory -- which is exactly what
    happens to a deliverable. Each figure is base64'd into the page instead.

    ``budget_bytes`` caps the total embedded size; anything past it is listed by path
    rather than silently dropped, so the reader knows a figure exists and where it is.
    """
    found, seen, spent = [], set(), 0
    for packet in packets:
        if packet.get("kind") != "view" or packet["status"] != "ok":
            continue
        paths = []
        for target in packet.get("artifact_files", []):
            if os.path.isfile(target) and target.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(target)
            elif os.path.isdir(target):
                paths += sorted(
                    os.path.join(target, name)
                    for name in os.listdir(target)
                    if name.lower().endswith((".png", ".jpg", ".jpeg"))
                )
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            size = os.path.getsize(path)
            found.append({
                "path": path,
                "name": os.path.basename(path),
                "tool": packet["tool"],
                "actor": packet["actor"],
                "why": packet.get("why", ""),
                "size": size,
                "embed": spent + size <= budget_bytes,
            })
            if spent + size <= budget_bytes:
                spent += size
    return found


def _data_uri(path) -> str | None:
    """Read an image as a base64 data URI, or None if it cannot be read."""
    import base64

    suffix = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    try:
        with open(path, "rb") as handle:
            return f"data:{mime};base64," + base64.b64encode(handle.read()).decode("ascii")
    except OSError:
        return None


def latest_verdicts(packets) -> list:
    """The current verdict of each check, one per (check, target).

    A check re-run after its cause was addressed supersedes its own earlier result.
    Counting every verdict ever recorded would let a fixed problem go on condemning the
    classes it used to touch, so a run could never recover -- and the report would be
    telling the reader to fix something already fixed.
    """
    current = {}
    for packet in packets:
        parsed = mep_rubric.parse_check(packet)
        if parsed:
            current[(parsed["check"], parsed["target"])] = parsed
    return list(current.values())


def class_trust(packets) -> dict:
    """Per-class verdict: which failed checks undermine which classes, and why.

    Returns {class_name: {"trusted": bool, "undermined_by": [(check, reason), ...]}}.
    """
    failed = [c for c in latest_verdicts(packets) if c["verdict"] == "FAIL"]
    classes = tuple(c for c in CLASSES if c != "nominal")
    trust = {name: {"trusted": True, "undermined_by": []} for name in classes}

    for check in failed:
        rule = _CLASS_SENSITIVITY.get(check["check"])
        if not rule:
            continue
        for name in rule["affects"]:
            trust[name]["trusted"] = False
            trust[name]["undermined_by"].append((check["check"], rule["because"]))
    return trust


def recovery_path(packets) -> list:
    """Which failed check to fix first, ordered by how much of the answer it returns.

    When several checks fail their effects overlap, and "nothing is trustworthy" is a
    useless thing to tell someone. What they need is the shortest route back to a usable
    result: fixing the check that is the *sole* obstacle to the most classes recovers the
    most for one piece of work.
    """
    failed = [
        c["check"] for c in latest_verdicts(packets)
        if c["verdict"] == "FAIL" and c["check"] in _CLASS_SENSITIVITY
    ]
    if not failed:
        return []

    blockers = {}
    for name in tuple(c for c in CLASSES if c != "nominal"):
        blocking = [c for c in failed if name in _CLASS_SENSITIVITY[c]["affects"]]
        if blocking:
            blockers[name] = blocking

    ranked = []
    for check in dict.fromkeys(failed):
        # Classes this check alone is holding back, and classes still blocked after it.
        sole = [n for n, b in blockers.items() if b == [check]]
        remaining = sorted({n for n, b in blockers.items() if check in b and b != [check]})
        ranked.append({"check": check, "recovers": sorted(sole), "still_blocked": remaining})

    ranked.sort(key=lambda r: len(r["recovers"]), reverse=True)
    return ranked


def discretionary_choices(packets) -> list:
    """Parameters the agent set itself, which the person who asked did not specify.

    Each is a place the reported numbers could have come out differently. Only
    values actually passed are listed; defaults the agent never touched are not
    presented as decisions it made.
    """
    choices = []
    for packet in packets:
        if packet.get("kind") == "check":
            continue
        for name, label in _DISCRETIONARY.items():
            if name not in packet.get("args", {}):
                continue
            value = packet["args"][name]
            if value in ("", None):
                continue
            choices.append({
                "step": packet["step"], "actor": packet["actor"], "tool": packet["tool"],
                "param": name, "label": label, "value": value,
                "why": packet.get("why", ""),
            })
    return choices


def phases(packets) -> list:
    """Group the run into reader-facing stages, keeping checks beside what they test."""
    grouped = []
    for title, tools, blurb in _PHASES:
        steps = [p for p in packets if p["tool"] in tools]
        if steps:
            grouped.append({"title": title, "blurb": blurb, "steps": steps})

    checked = {p["step"] for g in grouped for p in g["steps"]}
    other = [
        p for p in packets
        if p["step"] not in checked and p["tool"] not in ("begin_analysis", "explain_run")
    ]
    if other:
        grouped.append({
            "title": "Verification",
            "blurb": "Checks the agent chose to run against its own inputs and results.",
            "steps": other,
        })
    return grouped


# ---------------------------------------------------------------------- render


def story(run_id=None, report_filepath=None) -> str:
    """The plain-language account of a run, as text."""
    packets = mep_rubric.load(run_id)
    result = mep_rubric.score(packets)
    goal = read_goal(packets)
    classes = read_classes(packets)
    trust = class_trust(packets)
    choices = discretionary_choices(packets)
    rid = packets[0]["run_id"] if packets else (run_id or "")

    out = ["=" * 74, f"WHAT WAS ASKED", "=" * 74]
    if goal.get("user_prompt"):
        out += ["", '  "' + goal["user_prompt"] + '"']
        if goal.get("goal"):
            out += ["", f"  Read as: {goal['goal']}"]
    else:
        out += ["", "  Not recorded. The agent did not call begin_analysis, so this report",
                "  cannot show what it was asked to do -- only what it did."]

    out += ["", "=" * 74, "WHAT WAS FOUND", "=" * 74, ""]
    if classes:
        total = classes["total"]
        if classes["population_known"]:
            out.append(f"  {total:,} measurable struts of {classes['designed']:,} designed")
            out.append(f"  ({classes['excluded']:,} excluded: boundary caps, struts buried in "
                       "the build plates, and windows clipped by the volume edge)")
        else:
            out.append(f"  {total:,} struts, WITHOUT the measurable/excluded split")
            out.append("  This results file predates the `measurable` column, so struts that "
                       "cannot be measured are")
            out.append("  counted as if they were. Rates below are inflated -- treat them as "
                       "upper bounds only.")
        out.append("")
        for label in CLASSES:
            count = classes["counts"].get(label, 0)
            if label == "nominal":
                out.append(f"  {label:<9} {count:>7,}  {count / total:>6.2%}")
                continue
            state = "trusted" if trust[label]["trusted"] else "NOT RELIABLE in this run"
            out.append(f"  {label:<9} {count:>7,}  {count / total:>6.2%}   {state}")
        out += ["", f"  from {classes['source']}"]
    else:
        out += ["  No strut classification in this run."]

    unreliable = [c for c in trust if not trust[c]["trusted"]]
    if classes and unreliable:
        out += ["", "-" * 74, "WHY SOME OF THAT IS NOT RELIABLE", "-" * 74]
        seen = set()
        for name in unreliable:
            for check, because in trust[name]["undermined_by"]:
                if check in seen:
                    continue
                seen.add(check)
                affected = ", ".join(_CLASS_SENSITIVITY[check]["affects"])
                spared = _CLASS_SENSITIVITY[check]["spares"]
                out += ["", f"  {check} failed.",
                        f"    Undermines: {affected}"]
                if spared:
                    out.append(f"    Leaves standing: {', '.join(spared)}")
                out.append(f"    Because {because}")

    out += ["", "=" * 74, "HOW IT GOT THERE", "=" * 74]
    for phase in phases(packets):
        out += ["", f"  {phase['title'].upper()}", f"    {phase['blurb']}"]
        for packet in phase["steps"]:
            state = result["states"][packet["step"]]
            mark = mep_rubric._glyph(packet, state)
            broke = packet["status"] != "ok"
            out.append(f"      {mark} {packet['tool']}  [{packet['actor']}]"
                       + ("   DID NOT RUN" if broke else ""))
            if packet.get("why"):
                out.append(f'           "{packet["why"]}"')
            if broke:
                out.append(f"           {(packet.get('error') or packet['status'])[:110]}")

    out += ["", "=" * 74, "DECISIONS THE AGENT MADE ON ITS OWN", "=" * 74, ""]
    if choices:
        out.append(f"  {len(choices)} setting(s) the request did not specify. Each is a place")
        out.append("  the numbers above could have come out differently:")
        out.append("")
        for choice in choices:
            out.append(
                f"    {choice['label']:<34} = {choice['value']}"
                f"   (step {choice['step']}, {choice['tool']})"
            )
    else:
        out.append("  None recorded.")

    out += ["", "=" * 74, "BOTTOM LINE", "=" * 74, ""]
    trusted = [c for c in trust if trust[c]["trusted"]]
    if classes and trusted and unreliable:
        out += [f"  Believe: {', '.join(trusted)}.",
                f"  Do not report: {', '.join(unreliable)} until the failed checks above are resolved."]
    elif classes and not unreliable:
        out.append("  Every class rests on checks that passed.")
    elif classes:
        out.append("  Nothing here is reportable yet. Shortest route back:")
        out.append("")
        for rank in recovery_path(packets):
            if rank["recovers"]:
                out.append(f"    Fix {rank['check']} -> recovers {', '.join(rank['recovers'])}")
            else:
                blocked = ", ".join(rank["still_blocked"])
                out.append(f"    Fix {rank['check']} -> recovers nothing on its own ({blocked} "
                           "are blocked by another check too)")
    else:
        out.append("  No classification was produced, so there is nothing to trust or doubt.")

    if result["unverified"]:
        out += ["", f"  {len(result['unverified'])} step(s) were never checked at all: "
                + ", ".join(sorted({p['tool'] for p in result['unverified']}))]

    out += ["", "  This report describes one run against one specimen. It says whether these",
            "  numbers are internally sound, not whether the method is right in general."]

    if report_filepath:
        write_story_html(packets, result, goal, classes, trust, choices, rid, report_filepath)
        out += ["", f"  Report written to {report_filepath}"]
    return "\n".join(out)


_STORY_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#e6e6e3;--card:#fff;
 --ok:#2f7d4f;--bad:#c0392b;--warn:#b07d00;--accent:#4c6ef5}
@media(prefers-color-scheme:dark){:root{--bg:#191918;--fg:#eeeeec;--dim:#9d9d98;
 --line:#34342f;--card:#212120;--ok:#6cc48d;--bad:#e8776a;--warn:#d9ab4a;--accent:#8ba4f9}}
:root[data-theme=dark]{--bg:#191918;--fg:#eeeeec;--dim:#9d9d98;--line:#34342f;--card:#212120;
 --ok:#6cc48d;--bad:#e8776a;--warn:#d9ab4a;--accent:#8ba4f9}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#e6e6e3;--card:#fff;
 --ok:#2f7d4f;--bad:#c0392b;--warn:#b07d00;--accent:#4c6ef5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:54rem;margin:0 auto;padding:3rem 1.25rem 6rem}
h1{font-size:1.7rem;margin:0 0 2.4rem;font-weight:650;letter-spacing:-.02em}
h2{font-size:.76rem;text-transform:uppercase;letter-spacing:.11em;color:var(--dim);
 margin:3rem 0 .9rem;font-weight:650}
.ask{border-left:3px solid var(--accent);padding:.15rem 0 .15rem 1.1rem;margin-bottom:.5rem}
.ask q{font-size:1.12rem;line-height:1.5;quotes:'"' '"'}
.read{color:var(--dim);font-size:.87rem;margin-top:.6rem}
table{width:100%;border-collapse:collapse;font-size:.93rem}
th{text-align:left;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;
 color:var(--dim);font-weight:600;padding:0 .6rem .5rem 0;border-bottom:1px solid var(--line)}
td{padding:.62rem .6rem .62rem 0;border-bottom:1px solid var(--line)}
td.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
.cls{font-weight:600}
tr.dim td{color:var(--dim)}
.tag{display:inline-block;font-size:.7rem;font-weight:650;padding:.13rem .5rem;border-radius:99px}
.t-ok{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.t-bad{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.t-warn{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--bad);
 border-radius:0 6px 6px 0;padding:.9rem 1.1rem;margin:.8rem 0;font-size:.9rem}
.note b{color:var(--bad)}
.note .aff{color:var(--dim);font-size:.84rem;margin-top:.45rem}
.phase{margin-bottom:1.7rem}
.ph-t{font-weight:650;font-size:1rem}
.ph-b{color:var(--dim);font-size:.87rem;margin:.15rem 0 .7rem}
.step{display:flex;gap:.7rem;align-items:baseline;padding:.4rem 0 .4rem 1rem;
 border-left:2px solid var(--line);margin-left:.2rem}
.dot{width:.5rem;height:.5rem;border-radius:99px;flex:0 0 auto;margin-top:.45rem}
.d-pass{background:var(--ok)}.d-fail{background:var(--bad)}.d-warn{background:var(--warn)}
.d-unverified{background:var(--dim);opacity:.45}
.st{font:.87rem ui-monospace,Menlo,monospace;font-weight:600}
.who{font-size:.72rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.wy{color:var(--dim);font-size:.86rem;font-style:italic}
.bottom{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:1.2rem 1.3rem;font-size:.95rem;margin-top:1rem}
.bottom p{margin:0 0 .6rem}.bottom p:last-child{margin:0}
.scope{color:var(--dim);font-size:.8rem;margin-top:2.5rem;line-height:1.6;
 border-top:1px solid var(--line);padding-top:1.2rem}
.scroll{overflow-x:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,20rem),1fr));gap:1rem}
.fig{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.fig img{display:block;width:100%;height:auto}
.fig figcaption{padding:.6rem .8rem;font-size:.8rem;border-top:1px solid var(--line)}
.fig .wy{margin-top:.2rem}
.nofig{padding:1.6rem .9rem;color:var(--dim);font-size:.8rem;text-align:center}
.st.struck{text-decoration:line-through;text-decoration-thickness:1px;opacity:.65}
.broke{color:var(--bad);font-size:.82rem;margin-top:.25rem}
"""


def write_story_html(packets, result, goal, classes, trust, choices, rid, path) -> None:
    """Write the plain-language report as a self-contained page."""
    esc = html.escape
    started = time.strftime("%d %b %Y, %H:%M", time.localtime(packets[0]["ts"])) if packets else ""

    if goal.get("user_prompt"):
        ask = (f'<div class="ask"><q>{esc(goal["user_prompt"])}</q></div>'
               + (f'<div class="read">Read as: {esc(goal["goal"])}</div>' if goal.get("goal") else ""))
    else:
        ask = ('<div class="read">The request was not recorded — the agent did not call '
               "<code>begin_analysis</code>, so this report can show what it did but not "
               "what it was asked for.</div>")

    if classes:
        total = classes["total"]
        rows = []
        for label in CLASSES:
            count = classes["counts"].get(label, 0)
            pct = count / total if total else 0
            if label == "nominal":
                tag = ""
                cls = ' class="dim"'
            else:
                ok = trust[label]["trusted"]
                tag = (f'<span class="tag {"t-ok" if ok else "t-bad"}">'
                       f'{"trusted" if ok else "not reliable"}</span>')
                cls = ""
            rows.append(
                f'<tr{cls}><td class="cls">{label}</td><td class="n">{count:,}</td>'
                f'<td class="n">{pct:.2%}</td><td>{tag}</td></tr>'
            )
        if classes["population_known"]:
            note = (f'{total:,} measurable struts of {classes["designed"]:,} designed. '
                    f'{classes["excluded"]:,} excluded — boundary caps, struts buried in the '
                    "build plates, and windows clipped by the volume edge. Shares are of the "
                    "measurable population; counting the excluded ones inflates "
                    "<code>missing</code> and <code>thick</code> severalfold.")
        else:
            note = ('<b style="color:var(--bad)">This results file predates the '
                    "<code>measurable</code> column</b>, so unmeasurable struts are counted as "
                    f"if they were measured. The {total:,} below is the full designed set and "
                    "every rate is inflated — treat them as upper bounds only.")
        found = (
            '<div class="scroll"><table><thead><tr><th>Class</th><th style="text-align:right">'
            'Struts</th><th style="text-align:right">Share</th><th>Verdict</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<div class="read">{note}</div>'
        )
    else:
        found = '<div class="read">No strut classification was produced in this run.</div>'

    notes, seen = [], set()
    for name, info in trust.items():
        for check, because in info["undermined_by"]:
            if check in seen:
                continue
            seen.add(check)
            rule = _CLASS_SENSITIVITY[check]
            spared = (f' &nbsp;·&nbsp; Leaves standing: <b style="color:var(--ok)">'
                      f'{", ".join(rule["spares"])}</b>' if rule["spares"] else "")
            notes.append(
                f'<div class="note"><b>{esc(check)} failed.</b> {esc(rule["because"])}'
                f'<div class="aff">Undermines: <b>{", ".join(rule["affects"])}</b>{spared}</div></div>'
            )

    gallery = ""
    figs = figures(packets)
    if figs:
        cards = []
        for fig in figs:
            caption = (f'<figcaption><b>{esc(fig["name"])}</b>'
                       f'<span class="who"> {esc(fig["tool"])}</span>'
                       + (f'<div class="wy">{esc(fig["why"])}</div>' if fig["why"] else "")
                       + "</figcaption>")
            uri = _data_uri(fig["path"]) if fig["embed"] else None
            if uri:
                cards.append(f'<figure class="fig"><img src="{uri}" alt="{esc(fig["name"])}">'
                             f"{caption}</figure>")
            else:
                cards.append(
                    f'<figure class="fig"><div class="nofig">Not embedded '
                    f'({fig["size"] / 1e6:.1f} MB, over the page budget)<br><code>'
                    f'{esc(fig["path"])}</code></div>{caption}</figure>'
                )
        gallery = f'<h2>What it looks like</h2><div class="gallery">{"".join(cards)}</div>'

    phase_html = []
    for phase in phases(packets):
        steps = []
        for packet in phase["steps"]:
            state = result["states"][packet["step"]]
            dot = state if state != "check" else (
                (mep_rubric.parse_check(packet) or {}).get("verdict", "warn").lower()
            )
            dot = {"pass": "pass", "fail": "fail", "warn": "warn"}.get(dot, "unverified")

            # A step that errored did not happen. Rendering it beside the steps that did,
            # with its stated intent and nothing else, reads as though it succeeded -- a
            # reader would conclude the ground truth was scored when the call crashed.
            broke = packet["status"] != "ok"
            note = ""
            if broke:
                dot = "fail"
                note = (f'<div class="broke"><b>did not run</b> — '
                        f'{esc((packet.get("error") or packet["status"])[:160])}</div>')

            steps.append(
                f'<div class="step"><div class="dot d-{dot}"></div><div>'
                f'<span class="st{" struck" if broke else ""}">{esc(packet["tool"])}</span> '
                f'<span class="who">{esc(packet["actor"])}</span>'
                + (f'<div class="wy">{esc(packet["why"])}</div>' if packet.get("why") else "")
                + note
                + "</div></div>"
            )
        phase_html.append(
            f'<div class="phase"><div class="ph-t">{esc(phase["title"])}</div>'
            f'<div class="ph-b">{esc(phase["blurb"])}</div>{"".join(steps)}</div>'
        )

    if choices:
        rows = "".join(
            f'<tr><td>{esc(c["label"])}</td><td class="n">{esc(str(c["value"]))}</td>'
            f'<td class="who">{esc(c["tool"])}</td></tr>'
            for c in choices
        )
        decisions = (
            f'<div class="read">{len(choices)} setting(s) your request did not specify. '
            "Each is a place the numbers above could have come out differently.</div>"
            '<div class="scroll"><table><thead><tr><th>Setting</th>'
            '<th style="text-align:right">Chosen</th><th>Set by</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        decisions = '<div class="read">No discretionary settings were recorded.</div>'

    trusted = [c for c in trust if trust[c]["trusted"]]
    unreliable = [c for c in trust if not trust[c]["trusted"]]
    bottom = []
    if classes and trusted and unreliable:
        bottom.append(f'<p><b>Believe:</b> {", ".join(trusted)}.</p>')
        bottom.append(
            f'<p><b>Do not report:</b> {", ".join(unreliable)} — until the failed checks above '
            "are resolved.</p>"
        )
    elif classes and not unreliable:
        bottom.append("<p>Every class rests on checks that passed.</p>")
    elif classes:
        bottom.append("<p><b>Nothing here is reportable yet.</b> Shortest route back:</p>")
        for rank in recovery_path(packets):
            if rank["recovers"]:
                bottom.append(
                    f'<p>Fix <code>{esc(rank["check"])}</code> &rarr; recovers '
                    f'<b style="color:var(--ok)">{", ".join(rank["recovers"])}</b>.</p>'
                )
            else:
                bottom.append(
                    f'<p class="read">Fix <code>{esc(rank["check"])}</code> &rarr; recovers '
                    f'nothing on its own; {", ".join(rank["still_blocked"])} are blocked by '
                    "another check too.</p>"
                )
    else:
        bottom.append("<p>No classification was produced, so there is nothing to trust or doubt.</p>")
    if result["unverified"]:
        tools = ", ".join(sorted({p["tool"] for p in result["unverified"]}))
        bottom.append(
            f'<p class="read">{len(result["unverified"])} step(s) were never checked at all: '
            f"{esc(tools)}.</p>"
        )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defect analysis &mdash; {esc(rid)}</title><style>{_STORY_CSS}</style></head>
<body><div class="wrap">
<h1>How this defect analysis was done</h1>
<h2>What you asked</h2>{ask}
<h2>What was found</h2>{found}
{"<h2>Why some of that is not reliable</h2>" + "".join(notes) if notes else ""}
{gallery}
<h2>How it got there</h2>{"".join(phase_html)}
<h2>Decisions the agent made on its own</h2>{decisions}
<h2>Bottom line</h2><div class="bottom">{"".join(bottom)}</div>
<div class="scope">One run, one specimen. This says whether these numbers are internally
sound &mdash; every input accounted for, every sensitive setting checked &mdash; not whether the
method is correct in general. The italic lines are the agent's own stated reasons, recorded
verbatim; they are what it said, which is not necessarily why it acted.
&nbsp;·&nbsp; {esc(rid)}, {esc(started)}</div>
</div></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
