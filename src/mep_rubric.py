"""Scores a recorded run and renders its explanation.

Reads the Minimal Explanation Packets written by ``mep.py``, resolves each
step's verification state against the check tools the agent chose to call, and
scores the trajectory on behavioural rubrics adapted from Chaduvula et al.
(arXiv:2602.06841, section 3.3).

Five of the paper's six categories are used. Plan Adherence is dropped: it
requires a plan object, and an MCP server never sees one. The remaining five are
evaluated deterministically from the trace rather than by an LLM judge, so a
flag here is a fact about what happened, not a judgement about it.

What this CANNOT see, and what the flags therefore do not cover: the agent's
reasoning. MCP's tools/call request carries a name and arguments and nothing
else, and a subagent's context is discarded when it returns, so the only
rationale available is the self-reported ``why`` string. Section 3.2 of the
source paper applies to it in full -- it records what the agent claimed its
reason was, which is not necessarily what caused the call.

The paper's prevalence and reliability statistics (its equations 1-4) are NOT
computed. Those need many runs against many labelled tasks; this project has one
CT volume and one ground truth, so the flags below are reported per run and
descriptively. Treating them as validated failure predictors would be reading a
rate off a sample of one.
"""

import html
import json
import os
import re
import time

try:
    from . import mep
except ImportError:
    import mep


_HEADER = re.compile(r"^CHECK (\w+) VERDICT=(PASS|WARN|FAIL) TARGET=(.*)$")

# Tools whose input should be a segmentation mask rather than raw CT intensities.
# Per-sub-volume Otsu is unstable on the raw volumes here (background sits around
# 32k of 65k), so measuring one of these against raw CT is a tool-choice error.
_MASK_CONSUMERS = {
    "measure_lattice_iou",
    "detect_lattice_defects",
    "detect_missing_nodes",
    "detect_missing_nodes_2d",
    "refit_lattice_registration",
    "skeletonize",
    "validate_against_stl",
}

# Tools that produce a mask.
_MASK_PRODUCERS = {"segment_ct_dataset", "filter_ct_volume"}


def load(run_id=None) -> list:
    """Return the run's packets in recorded order."""
    path = mep.trace_path(run_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no trace for run {run_id or mep.run_id()} at {path}")
    packets = []
    for line in open(path, encoding="utf-8"):
        try:
            packets.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sorted(packets, key=lambda p: p["step"])


def parse_check(packet) -> dict | None:
    """Pull the machine-readable verdict out of a check tool's artifact."""
    if packet.get("kind") != "check":
        return None
    artifact = packet.get("artifact") or ""
    match = _HEADER.match(artifact.splitlines()[0] if artifact else "")
    if not match:
        return None
    name, verdict, target = match.groups()
    return {"check": name, "verdict": verdict, "target": target.strip(), "step": packet["step"]}


def link(packets) -> dict:
    """Attach each check to the steps whose files it verifies.

    A check declares a TARGET path. It verifies any step that read that path as
    an input or declared it as an output. This is the retroactive fill of the
    MEP's third slot: at call time a step does not know whether it will be
    checked, and requiring it to would make verification automatic and therefore
    unobservable.

    Returns {step_number: [check_record, ...]}.
    """
    checks = [c for c in (parse_check(p) for p in packets) if c]
    attached = {p["step"]: [] for p in packets}

    for check in checks:
        target = check["target"]
        for packet in packets:
            if packet.get("kind") == "check":
                continue
            touched = [r["path"] for r in packet.get("inputs", [])]
            touched += packet.get("artifact_files", [])
            if any(target == t or target.startswith(t + os.sep) or t.startswith(target + os.sep)
                   for t in touched):
                attached[packet["step"]].append(check)
    return attached


def state_of(packet, attached_checks) -> str:
    """The verification state of one step: 'pass', 'fail', or 'unverified'."""
    if packet.get("kind") == "check":
        return "check"
    if not attached_checks:
        return "unverified"
    if any(c["verdict"] == "FAIL" for c in attached_checks):
        return "fail"
    if any(c["verdict"] == "WARN" for c in attached_checks):
        return "warn"
    return "pass"


# ------------------------------------------------------------------- rubrics
#
# Each returns (flag, [evidence lines]). flag is True when the constraint was
# VIOLATED, matching the source paper's convention where a raised flag marks a
# violation rather than a success.


def rubric_state_tracking(packets, attached) -> tuple:
    """Did every claim rest on a file whose origin the run can account for?

    The paper's dominant failure mode, 2.7x over-represented in failed runs. The
    concrete form here: a mask overwritten by a second actor between the step
    that produced it and the step that measured against it, or a correction file
    that one step applied and the next silently dropped.
    """
    violations = []

    writes = {}
    for packet in packets:
        for path in packet.get("artifact_files", []):
            writes.setdefault(path, []).append(packet)
    for path, producers in writes.items():
        if len(producers) > 1:
            actors = sorted({p["actor"] for p in producers})
            violations.append(
                f"{os.path.basename(path)} written {len(producers)}x "
                f"(steps {', '.join(str(p['step']) for p in producers)}"
                + (f"; actors {', '.join(actors)}" if len(actors) > 1 else "")
                + ") -- consumers may hold different versions"
            )

    # A correction file used earlier and then dropped by a later step that accepts one.
    used_correction = {
        p["args"]["correction_filepath"]
        for p in packets
        if p.get("args", {}).get("correction_filepath")
    }
    if used_correction:
        for packet in packets:
            args = packet.get("args", {})
            if "correction_filepath" in args and not args["correction_filepath"]:
                violations.append(
                    f"step {packet['step']} [{packet['actor']}] {packet['tool']} ran with no "
                    "correction_filepath while the run had one -- its defect population is "
                    "swamped by registration error"
                )

    return bool(violations), violations


def rubric_tool_correctness(packets, attached) -> tuple:
    """Were the tools invoked with arguments that mean what the caller thinks?

    Covers hard errors and the documented silent case: reusing a cache measured
    at a different section count, which is honoured at its own count and quietly
    redefines both `missing` and the break rule.
    """
    violations = []
    for packet in packets:
        if packet["status"] != "ok":
            violations.append(
                f"step {packet['step']} [{packet['actor']}] {packet['tool']} -> "
                f"{packet['status']}: {(packet.get('error') or '')[:120]}"
            )

    stale = {
        c["target"]
        for c in (parse_check(p) for p in packets)
        if c and c["check"] == "check_cache_staleness" and c["verdict"] == "FAIL"
    }
    for packet in packets:
        if packet.get("args", {}).get("use_cache") and any(
            out in stale for out in packet.get("artifact_files", [])
        ):
            violations.append(
                f"step {packet['step']} {packet['tool']} ran use_cache=True against a cache "
                "already reported stale -- the returned classes are not the ones requested"
            )
    return bool(violations), violations


def rubric_tool_choice(packets, attached) -> tuple:
    """Was each tool applied to the substrate it is valid on?

    The recurring error in this project is measuring per-strut geometry against
    raw CT rather than a segmentation mask.
    """
    violations = []
    produced_masks = {
        path
        for packet in packets
        if packet["tool"] in _MASK_PRODUCERS and packet["status"] == "ok"
        for path in packet.get("artifact_files", [])
    }
    for packet in packets:
        if packet["tool"] not in _MASK_CONSUMERS:
            continue
        mask_arg = packet.get("args", {}).get("mask_filepath") or packet.get("args", {}).get(
            "input_filepath"
        )
        if not mask_arg:
            continue
        path = os.path.abspath(mask_arg)
        if path in produced_masks:
            continue
        if not produced_masks:
            continue  # nothing was segmented in this run; the mask was supplied
        violations.append(
            f"step {packet['step']} [{packet['actor']}] {packet['tool']} measured against "
            f"{os.path.basename(path)}, which no segmentation step in this run produced"
        )
    return bool(violations), violations


def rubric_error_recovery(packets, attached) -> tuple:
    """After a check failed, did the run change course?

    A FAIL that is recorded and then ignored is worse than no check at all: it
    puts a refutation in the trace and a number in the report.
    """
    violations = []
    for packet in packets:
        check = parse_check(packet)
        if not check or check["verdict"] != "FAIL":
            continue
        later = [
            p
            for p in packets
            if p["step"] > packet["step"]
            and p.get("kind") == "analysis"
            and any(
                check["target"] == r["path"] or r["path"].startswith(check["target"] + os.sep)
                for r in p.get("inputs", [])
            )
        ]
        if later:
            steps = ", ".join(str(p["step"]) for p in later)
            violations.append(
                f"{check['check']} FAILED at step {check['step']} on "
                f"{os.path.basename(check['target'])}, then step(s) {steps} used it anyway"
            )
    return bool(violations), violations


def rubric_intent_alignment(packets, attached) -> tuple:
    """Did each step arrive with a stated reason?

    The weakest of the five, and deliberately so. This tests only that the
    rationale slot was filled, never whether the stated reason is the real one --
    the trace has no access to the agent's reasoning, and a self-report cannot be
    checked against something the server never sees.
    """
    violations = [
        f"step {p['step']} [{p['actor']}] {p['tool']} gave no reason"
        for p in packets
        if not (p.get("why") or "").strip()
    ]
    return bool(violations), violations


RUBRICS = [
    ("State Tracking Consistency", rubric_state_tracking),
    ("Tool Correctness", rubric_tool_correctness),
    ("Tool-Choice Accuracy", rubric_tool_choice),
    ("Error Awareness & Recovery", rubric_error_recovery),
    ("Intent Alignment", rubric_intent_alignment),
]


def score(packets) -> dict:
    """Run every rubric and resolve per-step verification states."""
    attached = link(packets)
    flags = {}
    for label, rubric in RUBRICS:
        violated, evidence = rubric(packets, attached)
        flags[label] = {"violated": violated, "evidence": evidence}

    analysis = [p for p in packets if p.get("kind") == "analysis"]
    states = {p["step"]: state_of(p, attached[p["step"]]) for p in packets}
    unverified = [p for p in analysis if states[p["step"]] == "unverified"]
    failed = [p for p in analysis if states[p["step"]] == "fail"]

    return {
        "attached": attached,
        "states": states,
        "flags": flags,
        "n_steps": len(packets),
        "n_analysis": len(analysis),
        "unverified": unverified,
        "failed": failed,
        "actors": sorted({p["actor"] for p in packets}),
        "trustworthy": not failed and not unverified and not flags["State Tracking Consistency"]["violated"],
    }


# -------------------------------------------------------------------- render


_GLYPH = {"pass": "[ok]", "fail": "[XX]", "warn": "[!!]", "unverified": "[--]", "check": "[??]"}

# A check step is shown by its own verdict, not by whether something checked it.
_CHECK_GLYPH = {"PASS": "[ok]", "WARN": "[!!]", "FAIL": "[XX]"}


def _glyph(packet, state) -> str:
    # A step that errored or threw did not produce a result, whatever else is true of
    # it. That has to win over the verification state, or the marker contradicts the
    # text beside it.
    if packet.get("status") != "ok":
        return _GLYPH["fail"]
    if state == "check":
        parsed = parse_check(packet)
        return _CHECK_GLYPH.get(parsed["verdict"], "[??]") if parsed else "[??]"
    return _GLYPH[state]


def explain(run_id=None, report_filepath=None) -> str:
    """Render the run's explanation as text, optionally also writing the HTML report."""
    packets = load(run_id)
    result = score(packets)
    rid = packets[0]["run_id"] if packets else (run_id or mep.run_id())

    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[0]["ts"])) if packets else "-"
    out = [
        f"RUN {rid}   started {started}   {result['n_steps']} steps   "
        f"actors: {', '.join(result['actors'])}",
        "",
    ]

    for packet in packets:
        state = result["states"][packet["step"]]
        checks = result["attached"][packet["step"]]
        note = ""
        if state == "check":
            parsed = parse_check(packet)
            note = f"VERDICT={parsed['verdict']} on {os.path.basename(parsed['target'])}" if parsed else ""
        elif checks:
            note = ", ".join(f"{c['check'].replace('check_', '')}={c['verdict']}" for c in checks)
        elif state == "unverified":
            note = "UNVERIFIED"

        out.append(
            f"{packet['step']:>3} {_glyph(packet, state)} [{packet['actor']}] {packet['tool']}"
            + (f"   {note}" if note else "")
        )
        if packet.get("why"):
            out.append(f'        why: "{packet["why"]}"')
        if packet["status"] != "ok":
            out.append(f"        {(packet.get('error') or packet['status'])[:110]}")

    out += ["", "RUBRIC"]
    for label, _ in RUBRICS:
        flag = result["flags"][label]
        out.append(f"  {'VIOLATED' if flag['violated'] else 'ok      '}  {label}")
        for line in flag["evidence"][:4]:
            out.append(f"              - {line}")
        if len(flag["evidence"]) > 4:
            out.append(f"              ... {len(flag['evidence']) - 4} more")

    out += ["", "VERDICT"]
    if result["trustworthy"]:
        out.append("  Every analysis step rests on a checked input and no rubric was violated.")
    else:
        if result["failed"]:
            out.append(f"  {len(result['failed'])} analysis step(s) rest on a FAILED check.")
        if result["unverified"]:
            names = ", ".join(sorted({p["tool"] for p in result["unverified"]}))
            out.append(f"  {len(result['unverified'])} analysis step(s) unverified: {names}")
        violated = [l for l, _ in RUBRICS if result["flags"][l]["violated"]]
        if violated:
            out.append(f"  Rubrics violated: {', '.join(violated)}")
        out.append("  Numbers from this run are not audit-ready without the above resolved.")

    out += [
        "",
        "Scope: flags are deterministic facts about the recorded trace, reported for this "
        "run only -- not validated failure predictors (one specimen, one ground truth). "
        "Agent reasoning is not visible to an MCP server; `why` is self-reported.",
    ]

    if report_filepath:
        write_html(packets, result, rid, report_filepath)
        out += ["", f"HTML report written to {report_filepath}"]
    return "\n".join(out)


_ACTOR_COLORS = ["#4c8dd8", "#c77c3c", "#5aa469", "#9b6bbf", "#c25b6d", "#3f9ca3"]

_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#e3e3e0;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#191918;--fg:#eeeeec;--dim:#9a9a96;--line:#33332f;--card:#212120}}
:root[data-theme=dark]{--bg:#191918;--fg:#eeeeec;--dim:#9a9a96;--line:#33332f;--card:#212120}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#e3e3e0;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1{font-size:1.45rem;margin:0 0 .2rem;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:.85rem;margin-bottom:2rem}
.step{display:grid;grid-template-columns:2.4rem .7rem 1fr;gap:.7rem;
 border-top:1px solid var(--line);padding:.7rem 0;align-items:start}
.step:last-of-type{border-bottom:1px solid var(--line)}
.num{color:var(--dim);font:.78rem ui-monospace,SFMono-Regular,Menlo,monospace;padding-top:.15rem}
.lane{width:3px;border-radius:2px;height:100%;min-height:1.3rem}
.tool{font:.9rem ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.actor{font-size:.72rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em;
 margin-left:.5rem;font-weight:500}
.why{color:var(--dim);font-size:.85rem;font-style:italic;margin-top:.2rem}
.badge{display:inline-block;font-size:.7rem;font-weight:600;padding:.1rem .4rem;
 border-radius:3px;margin-left:.4rem;vertical-align:1px}
.pass{background:#2f7d4f22;color:#2f7d4f}.fail{background:#c0392b22;color:#c0392b}
.warn{background:#b8860022;color:#b88600}.unverified{background:#8883;color:var(--dim)}
.check{background:#4c8dd822;color:#4c8dd8}
@media(prefers-color-scheme:dark){.pass{color:#6cc48d}.fail{color:#e8776a}.warn{color:#d9ab4a}.check{color:#7fb0e8}}
details{margin-top:.4rem}
summary{cursor:pointer;font-size:.78rem;color:var(--dim);user-select:none}
pre{background:var(--card);border:1px solid var(--line);border-radius:5px;
 padding:.7rem .8rem;overflow-x:auto;font-size:.76rem;line-height:1.45;margin:.4rem 0 0;
 white-space:pre-wrap;word-break:break-word}
h2{font-size:1rem;margin:2.5rem 0 .6rem;font-weight:650}
.rub{display:flex;gap:.6rem;padding:.5rem 0;border-top:1px solid var(--line);font-size:.88rem}
.rub ul{margin:.3rem 0 0;padding-left:1.1rem;color:var(--dim);font-size:.8rem}
.verdict{margin-top:1.6rem;padding:1rem 1.1rem;border-radius:6px;border:1px solid var(--line);
 background:var(--card);font-size:.88rem}
.verdict.no{border-color:#c0392b66}.verdict.yes{border-color:#2f7d4f66}
.scope{margin-top:2rem;color:var(--dim);font-size:.76rem;line-height:1.5}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.72rem;color:var(--dim);margin-bottom:1.4rem}
"""

_JS = """
document.querySelectorAll('.step').forEach(s=>{
  const d=s.querySelector('details'); if(!d) return;
  s.querySelector('.tool').style.cursor='pointer';
  s.querySelector('.tool').addEventListener('click',()=>d.open=!d.open);
});
"""


def write_html(packets, result, rid, path) -> None:
    """Write the run explanation as a self-contained HTML report."""
    colors = {a: _ACTOR_COLORS[i % len(_ACTOR_COLORS)] for i, a in enumerate(result["actors"])}
    esc = html.escape

    rows = []
    for packet in packets:
        step = packet["step"]
        state = result["states"][step]
        checks = result["attached"][step]

        if state == "check":
            parsed = parse_check(packet)
            label = parsed["verdict"] if parsed else "CHECK"
            cls = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(label, "check")
            badge = f'<span class="badge {cls}">{label}</span>'
        elif checks:
            badge = "".join(
                f'<span class="badge {c["verdict"].lower()}">'
                f'{esc(c["check"].replace("check_", ""))} {c["verdict"]}</span>'
                for c in checks
            )
        else:
            badge = '<span class="badge unverified">unverified</span>'

        detail = {
            "args": packet.get("args", {}),
            "inputs": [
                {"path": r["path"], "size": r["size"], "fingerprint": r["edge_sha256"]}
                for r in packet.get("inputs", [])
            ],
            "writes": packet.get("artifact_files", []),
            "status": packet["status"],
            "duration_s": packet["duration_s"],
            "artifact": packet.get("artifact"),
        }

        rows.append(
            f'<div class="step">'
            f'<div class="num">{step}</div>'
            f'<div class="lane" style="background:{colors.get(packet["actor"], "#888")}"></div>'
            f"<div>"
            f'<span class="tool">{esc(packet["tool"])}</span>'
            f'<span class="actor">{esc(packet["actor"])}</span>{badge}'
            + (f'<div class="why">{esc(packet["why"])}</div>' if packet.get("why") else "")
            + f"<details><summary>packet</summary><pre>"
            f"{esc(json.dumps(detail, indent=2)[:6000])}</pre></details>"
            f"</div></div>"
        )

    rubric_rows = []
    for label, _ in RUBRICS:
        flag = result["flags"][label]
        cls, text = ("fail", "VIOLATED") if flag["violated"] else ("pass", "ok")
        items = "".join(f"<li>{esc(e)}</li>" for e in flag["evidence"][:6])
        rubric_rows.append(
            f'<div class="rub"><span class="badge {cls}">{text}</span>'
            f"<div><b>{esc(label)}</b>{f'<ul>{items}</ul>' if items else ''}</div></div>"
        )

    if result["trustworthy"]:
        verdict = (
            '<div class="verdict yes"><b>Audit-ready.</b> Every analysis step rests on a '
            "checked input and no rubric was violated.</div>"
        )
    else:
        bits = []
        if result["failed"]:
            bits.append(f"{len(result['failed'])} step(s) rest on a FAILED check")
        if result["unverified"]:
            bits.append(f"{len(result['unverified'])} analysis step(s) unverified")
        violated = [l for l, _ in RUBRICS if result["flags"][l]["violated"]]
        if violated:
            bits.append("rubrics violated: " + ", ".join(violated))
        verdict = (
            '<div class="verdict no"><b>Not audit-ready.</b> '
            + esc("; ".join(bits))
            + ". Numbers from this run should not be reported without resolving these.</div>"
        )

    legend = " ".join(
        f'<span><span class="lane" style="display:inline-block;width:10px;height:10px;'
        f'background:{c};border-radius:2px"></span> {esc(a)}</span>'
        for a, c in colors.items()
    )
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime(packets[0]["ts"])) if packets else "-"

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run explanation {esc(rid)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>Run explanation</h1>
<div class="sub">{esc(rid)} &middot; started {started} &middot; {result['n_steps']} steps
&middot; {result['n_analysis']} analysis</div>
<div class="legend">{legend}</div>
{''.join(rows)}
<h2>Rubric</h2>{''.join(rubric_rows)}
{verdict}
<div class="scope">Flags are deterministic facts about the recorded trace, for this run only &mdash;
not validated failure predictors, since this project has one specimen and one ground truth.
Agent reasoning is not visible to an MCP server: the <i>why</i> line is self-reported and
records what the agent claimed, not what caused the call.</div>
</div><script>{_JS}</script></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)
