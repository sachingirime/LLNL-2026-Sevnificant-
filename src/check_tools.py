"""Deterministic verification checks, exposed to the agent as MCP tools.

These fill the third slot of the Minimal Explanation Packet (Chaduvula et al.,
arXiv:2602.06841, section 3.4). The paper fills that slot with LLM-judged
behavioural rubrics; this project has registered ground truth and known,
measurable failure modes, so the checks here are deterministic instead. Each
answers a question with a number, not an opinion.

They are TOOLS, not a decorator, on purpose. If every tool call were verified
automatically then verification would be a constant across the trace and there
would be no behaviour left to explain. As tools, "did the agent verify this
before believing it?" becomes an observable property of the trajectory -- which
is exactly the shape of the paper's rubric flags, with harder evidence behind
it.

Every check returns a first line that ``rubric.py`` parses:

    CHECK <name> VERDICT=<PASS|WARN|FAIL> TARGET=<path>

followed by the numbers a human needs to see why.
"""

import csv
import json
import os

import numpy as np

try:
    from . import lattice_iou
    from . import mep
except ImportError:
    import lattice_iou
    import mep


# Nominal scan geometry for this specimen family, from the dataset note --
# used only to detect a design JSON in the wrong units, never to rescale one.
EXPECTED_UM_PER_VOXEL = 58.1


def _verdict(name: str, verdict: str, target: str, lines: list) -> str:
    """Format a check result with the machine-readable header rubric.py reads."""
    head = f"CHECK {name} VERDICT={verdict} TARGET={os.path.abspath(target)}"
    return "\n".join([head, ""] + lines)


def _fail(name: str, target: str, message: str) -> str:
    return _verdict(name, "FAIL", target, [message])


# ---------------------------------------------------------------- 0. provenance


def check_provenance(target_filepath: str, run_id: str = "") -> str:
    """
    Traces a file back through the current run: which step wrote it, with what
    arguments, and whether it has changed since.

    This is the check for the failure mode the source paper found dominant --
    inconsistent state tracking, 2.7x over-represented in failed runs. Here it takes a
    concrete form: a mask measured against a design that a different step re-segmented,
    or a defect count taken from a directory whose cache another actor overwrote. MCP
    tools are stateless and each call looks locally valid, so a broken chain is invisible
    from inside any single step.

    Reports every step in the run that declared this path as an output or consumed it as
    an input, in order, plus whether the bytes on disk still match the fingerprint
    recorded when it was written.

    Args:
        target_filepath: The file whose history you want -- a mask, design JSON,
            correction.json or results directory.
        run_id: Which run's trace to read. Empty means the current run.

    Returns:
        PASS if the file has a single unambiguous producer inside this run, WARN if it
        came from outside it, FAIL if it was rewritten after being consumed.
    """
    name = "check_provenance"
    target = os.path.abspath(target_filepath)
    path = mep.trace_path(run_id or None)
    if not os.path.isfile(path):
        return _fail(name, target, f"No trace at {path}; nothing has been recorded yet.")

    produced, consumed = [], []
    for line in open(path, encoding="utf-8"):
        try:
            packet = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(target == out or target.startswith(out + os.sep)
               for out in packet.get("artifact_files", [])):
            produced.append(packet)
        for record in packet.get("inputs", []):
            if record["path"] == target:
                consumed.append((packet, record))

    if not produced and not consumed:
        return _verdict(name, "WARN", target, [
            "This file appears nowhere in the run trace -- neither written nor read by "
            "any traced tool. It originates outside this run, so nothing here can "
            "establish how it was made.",
        ])

    lines = []
    if produced:
        lines.append("Written by:")
        for packet in produced:
            lines.append(
                f"  step {packet['step']:>3}  [{packet['actor']}]  {packet['tool']}  "
                f"status={packet['status']}"
                + (f'  why="{packet["why"]}"' if packet["why"] else "")
            )
    else:
        lines.append("Written by: nothing in this run (pre-existing input).")

    if consumed:
        lines.append("")
        lines.append("Read by:")
        for packet, record in consumed:
            lines.append(
                f"  step {packet['step']:>3}  [{packet['actor']}]  {packet['tool']}  "
                f"fingerprint {record['edge_sha256']}"
            )

    verdict = "PASS"
    notes = []

    actors = {p["actor"] for p in produced}
    if len(produced) > 1:
        verdict = "FAIL"
        notes.append(
            f"Written {len(produced)} times in this run"
            + (f" by different actors ({', '.join(sorted(actors))})" if len(actors) > 1 else "")
            + ". Any step that read it holds whichever version existed at the time, and "
            "the trace cannot tell you which unless the fingerprints below differ."
        )

    # A file rewritten after a consumer read it invalidates that consumer's result.
    seen = {}
    for packet, record in consumed:
        seen.setdefault(record["edge_sha256"], []).append(packet["step"])
    if len(seen) > 1:
        verdict = "FAIL"
        notes.append(
            "Consumers saw DIFFERENT content: "
            + "; ".join(f"{h} at step(s) {s}" for h, s in seen.items())
            + ". Results downstream of the earlier fingerprint were computed against a "
            "file that no longer exists."
        )

    current = mep.fingerprint_file(target)
    if current and consumed:
        last = consumed[-1][1]["edge_sha256"]
        lines += ["", f"on disk now: {current['edge_sha256']}   last read as: {last}"]
        if current["edge_sha256"] != last:
            verdict = "FAIL"
            notes.append(
                "The file on disk no longer matches what the last consumer read; that "
                "step's output cannot be reproduced from the current inputs."
            )
    elif not produced and not consumed:
        verdict = "WARN"

    if not produced and consumed and verdict == "PASS":
        verdict = "WARN"
        notes.append(
            "Consumed but never produced inside this run, so its provenance is outside "
            "the trace. Fine for supplied data; a gap for anything generated."
        )

    return _verdict(name, verdict, target, lines + ([""] + notes if notes else []))


# --------------------------------------------------------------------- 1. drift


def check_alignment_residual(
    correction_filepath: str,
    design_filepath: str,
    correction_applied: bool = True,
    tolerance_vox: float = 1.0,
) -> str:
    """
    Reports how well the design lines up with the scan, and therefore whether a per-strut
    rate from it can be believed.

    The confounder being guarded against is about WHERE the struts are, not how thick they
    are. The design JSON is a graph -- junction positions and the pairs they join -- and
    the registration fits an affine to those junction positions only; the file's single
    constant `thickness` field plays no part, and the nominal diameter is a tool argument.
    `scale_correction_zyx` is therefore a scale on the node grid.

    On this specimen that scale is 0.9876 in x: the printed lattice came out ~1.2% smaller
    in x than the design places its nodes, which over a 715-voxel span accumulates to ~8.9
    voxels of position error at the far face. As-built struts are ~4 voxels across, so near
    one face the design sits on the strut and near the other it points at the gap beside
    it, and every strut over there reads thin, broken or missing. Defect calls then climb
    with x, which looks like a part that degrades to the right.

    WHAT IS JUDGED DEPENDS ON WHETHER THE CORRECTION IS APPLIED, and getting this backwards
    inverts the answer. `refit_lattice_registration` exists to solve the drift, and
    `correction.json` records `residual_rms_zyx` -- the error left over AFTER the fit. When
    the correction is applied, the residual is the live error and the size of the
    correction is merely history. Judging the correction's own magnitude instead would
    penalise a refit for correcting more, which is exactly backwards: a better fit moves
    the design further and would score worse.

    Args:
        correction_filepath: correction.json from refit_lattice_registration.
        design_filepath: The registered design .json the correction applies to.
        correction_applied: True if the run passes this correction to its measurement
            tools (i.e. `correction_filepath` is set on them). Set False to ask the other
            question -- how far off the raw registered design is on its own.
        tolerance_vox: FAIL above this residual. Default 1.0, about half an as-built strut
            radius at this resolution.

    Returns:
        PASS/WARN/FAIL, the post-fit residual, and the correction's own size as context.
    """
    name = "check_alignment_residual"
    if not os.path.isfile(correction_filepath):
        return _fail(name, correction_filepath, "Error: correction file not found.")
    if not os.path.isfile(design_filepath):
        return _fail(name, design_filepath, "Error: design JSON not found.")

    pos, pairs, _, _ = lattice_iou.load_design(design_filepath)
    correction = json.loads(open(correction_filepath).read())
    A = np.asarray(correction["A_zyx"], float)
    t = np.asarray(correction["t_zyx"], float)

    # How far the correction moves the design: the error it is removing, not the one left.
    disp = pos @ A.T + t
    magnitude = np.linalg.norm(disp, axis=1)
    x = pos[:, 2]
    slope_x = float(np.polyfit(x, disp[:, 2], 1)[0])
    span_x = float(slope_x * (x.max() - x.min()))

    residual = correction.get("residual_rms_zyx")
    lines = [f"{len(pos)} junctions.", ""]

    if correction_applied and residual is not None:
        residual = np.asarray(residual, float)
        worst = float(residual.max())
        verdict = "PASS" if worst <= tolerance_vox else (
            "WARN" if worst <= 2 * tolerance_vox else "FAIL"
        )
        lines += [
            "Correction IS applied, so the live error is what the fit left behind:",
            "  residual RMS (z, y, x) = "
            + ", ".join(f"{v:.3f}" for v in residual)
            + f"   worst {worst:.3f} vox",
            f"  tolerance {tolerance_vox} vox (about half an as-built strut radius)",
            "",
            "For context, the error the fit removed:",
            f"  displacement median {np.median(magnitude):.2f} vox, "
            f"{span_x:+.2f} vox accumulated across the {x.max() - x.min():.0f}-voxel x span",
            "  scale correction (z, y, x) = "
            + ", ".join(f"{v:.4f}" for v in correction.get("scale_correction_zyx", [])),
        ]
        if verdict == "PASS":
            lines += [
                "",
                "Sub-voxel after correction: per-strut measurements are placed on their "
                "struts. NOTE the residual is a single RMS over the whole part -- it cannot "
                "show whether what remains is uniform or still structured in x. Confirming "
                "that needs refit_lattice_registration to write per-junction residuals.",
            ]
        else:
            lines += [
                "",
                "The fit did not bring the design onto the struts. Measurement windows are "
                "still off by more than half a strut radius, so per-strut classes describe "
                "whatever sits next to the strut. Refit before quoting any rate.",
            ]
    else:
        p99 = float(np.percentile(magnitude, 99))
        verdict = "PASS" if p99 <= tolerance_vox else (
            "WARN" if p99 <= 2 * tolerance_vox else "FAIL"
        )
        if abs(span_x) > tolerance_vox:
            verdict = "FAIL"
        why = ("Correction is NOT applied" if not correction_applied
               else "correction.json has no residual_rms_zyx")
        lines += [
            f"{why}, so the raw design-to-scan offset is the live error:",
            "  displacement (z, y, x) medians "
            + ", ".join(f"{np.median(disp[:, i]):+.3f}" for i in range(3)),
            f"  magnitude median {np.median(magnitude):.3f}  p99 {p99:.3f}  "
            f"max {magnitude.max():.3f}",
            "",
            f"x-gradient {slope_x:+.5f} vox per vox = {span_x:+.2f} vox across the "
            f"{x.max() - x.min():.0f}-voxel span.",
        ]
        if verdict != "PASS":
            lines.append(
                "A graded offset produces a defect rate that varies with position. Pass "
                "correction_filepath to the measurement tools, or restrict to a low-drift "
                "subvolume."
            )
    return _verdict(name, verdict, design_filepath, lines)


# ---------------------------------------------------------------- 2. cache state


def check_cache_staleness(output_directory: str, sections: int = 25) -> str:
    """
    Checks whether cached intermediate arrays in a results directory were measured at
    the section count about to be requested.

    detect_lattice_defects caches sections.npz and connectivity.npz and reuses them when
    use_cache=True. A cache written at a different section count is honoured at its own
    count, which silently changes what `missing` and the break rule mean -- the counts
    come back with no error and no warning. This makes that mismatch visible before the
    run rather than after.

    Args:
        output_directory: A detect_lattice_defects results directory.
        sections: The section count you are about to request.

    Returns:
        PASS if no cache or a matching cache, FAIL on a mismatch.
    """
    name = "check_cache_staleness"
    sec_path = os.path.join(output_directory, "sections.npz")
    conn_path = os.path.join(output_directory, "connectivity.npz")

    if not os.path.isdir(output_directory):
        return _verdict(name, "PASS", output_directory,
                        ["Directory does not exist yet; nothing cached, nothing to go stale."])
    if not os.path.isfile(sec_path):
        return _verdict(name, "PASS", output_directory,
                        ["No sections.npz present; the run will measure from cold."])

    try:
        cached = np.load(sec_path, allow_pickle=False)
        shape = cached["prof_area"].shape
    except (OSError, KeyError, ValueError) as exc:
        return _fail(name, output_directory, f"sections.npz present but unreadable: {exc}")

    cached_sections = int(shape[1])
    lines = [
        f"sections.npz: {shape[0]} struts x {cached_sections} sections",
        f"requested:    {sections} sections",
        f"connectivity.npz present: {os.path.isfile(conn_path)}",
    ]
    if cached_sections == sections:
        return _verdict(name, "PASS", output_directory, lines + ["Cache matches the request."])

    lines.append(
        f"MISMATCH. use_cache=True will reuse the {cached_sections}-section arrays and "
        f"ignore sections={sections}. Both the `missing` count and the break rule are "
        "defined against the section grid, so the returned classes will not be the ones "
        "requested. Either pass sections=" + str(cached_sections) + " or delete the cache."
    )
    return _verdict(name, "FAIL", output_directory, lines)


# ------------------------------------------------------------- 3. coordinate frame


def check_coordinate_frame(
    design_filepath: str,
    cell_mm: float = 4.56,
    expected_um_per_voxel: float = EXPECTED_UM_PER_VOXEL,
    tolerance_fraction: float = 0.05,
) -> str:
    """
    Confirms a design JSON is in the coordinate frame the tools assume.

    These JSONs appear in two frames: nominal half-cell units (junction coordinates
    running 0..18) and CT voxels (running ~24..774). Every measurement downstream derives
    the voxel pitch from the median strut length, so a file in the wrong frame does not
    error -- it silently rescales every radius, IoU and diameter in the run.

    Recovers the voxel pitch from the design's own median strut length (an octet strut
    spans cell/sqrt(2)) and compares it to the scan's known pitch.

    Args:
        design_filepath: The design .json to check.
        cell_mm: Unit-cell edge in mm.
        expected_um_per_voxel: The scan's known pitch. 58.1 for this specimen family.
        tolerance_fraction: Allowed relative disagreement before FAIL.

    Returns:
        PASS/FAIL with the recovered pitch and the implied frame.
    """
    name = "check_coordinate_frame"
    if not os.path.isfile(design_filepath):
        return _fail(name, design_filepath, "Error: design JSON not found.")

    pos, pairs, _, _ = lattice_iou.load_design(design_filepath)
    geom = lattice_iou.lattice_geometry(pos, pairs, cell_mm=cell_mm)
    recovered = geom["um_per_voxel"]
    ratio = recovered / expected_um_per_voxel

    lines = [
        f"{len(pos)} junctions, {len(pairs)} struts",
        f"coordinate range (z, y, x): "
        + ", ".join(f"{pos[:, i].min():.1f}..{pos[:, i].max():.1f}" for i in range(3)),
        f"median strut length: {geom['strut_length_vox']:.3f} units",
        f"recovered pitch:     {recovered:.2f} um/unit",
        f"expected pitch:      {expected_um_per_voxel:.2f} um/voxel",
        f"ratio:               {ratio:.3f}",
    ]
    if abs(ratio - 1.0) <= tolerance_fraction:
        return _verdict(name, "PASS", design_filepath,
                        lines + ["Design is in CT voxel coordinates, as the tools assume."])

    lines.append(
        f"Design is NOT in CT voxels. At ratio {ratio:.3f} it is most likely in "
        + ("nominal half-cell units -- register it first."
           if ratio > 1.5 else "an unexpected frame -- do not measure against the mask.")
    )
    return _verdict(name, "FAIL", design_filepath, lines)


# ------------------------------------------------------- 4. threshold sensitivity


def check_threshold_sensitivity(
    input_filepath: str,
    threshold: float,
    delta_fraction: float = 0.10,
    stride: int = 4,
    tolerance_fraction: float = 0.15,
) -> str:
    """
    Measures how much the segmented foreground fraction moves when the threshold is
    perturbed, i.e. whether the chosen cut sits on a plateau or on a slope.

    A threshold on a plateau is a property of the specimen; one on a slope is a property
    of the operator. This distinguishes them by re-measuring at +/- delta_fraction. It
    does NOT recommend a threshold -- picking the cut from a statistic is how the
    self-fulfilling classifications in this project's history arose.

    Subsamples by `stride` on every axis; foreground fraction is a global statistic and
    converges quickly, and the volumes here reach several GB.

    Args:
        input_filepath: The raw CT .tif/.npy being segmented.
        threshold: The cut about to be used.
        delta_fraction: Relative perturbation applied either side. Default 0.10.
        stride: Subsampling step per axis. 4 keeps ~1/64 of the voxels.
        tolerance_fraction: FAIL if the foreground fraction changes by more than this
            fraction of itself across the perturbation band.

    Returns:
        PASS/WARN/FAIL with foreground fraction at the three thresholds.
    """
    name = "check_threshold_sensitivity"
    if not os.path.isfile(input_filepath):
        return _fail(name, input_filepath, "Error: input volume not found.")

    ext = os.path.splitext(input_filepath)[1].lower()
    try:
        if ext == ".npy":
            volume = np.load(input_filepath, mmap_mode="r")[::stride, ::stride, ::stride]
        else:
            import tifffile

            volume = tifffile.imread(input_filepath)[::stride, ::stride, ::stride]
    except (OSError, ValueError) as exc:
        return _fail(name, input_filepath, f"Could not read volume: {exc}")

    volume = np.asarray(volume)
    lo, hi = threshold * (1 - delta_fraction), threshold * (1 + delta_fraction)
    fractions = [float((volume > cut).mean()) for cut in (lo, threshold, hi)]
    f_lo, f_mid, f_hi = fractions

    swing = abs(f_lo - f_hi) / f_mid if f_mid > 0 else float("inf")
    verdict = "PASS" if swing <= tolerance_fraction else (
        "WARN" if swing <= 2 * tolerance_fraction else "FAIL"
    )

    lines = [
        f"volume {volume.shape} subsampled by {stride}, dtype {volume.dtype}",
        f"  threshold {lo:>12.1f} (-{delta_fraction:.0%})  foreground {f_lo:.4%}",
        f"  threshold {threshold:>12.1f}  (chosen)   foreground {f_mid:.4%}",
        f"  threshold {hi:>12.1f} (+{delta_fraction:.0%})  foreground {f_hi:.4%}",
        "",
        f"relative swing across the band: {swing:.1%}  (tolerance {tolerance_fraction:.0%})",
    ]
    if verdict != "PASS":
        lines.append(
            "The cut sits on a slope, so the foreground fraction -- and every count "
            "derived from it -- is sensitive to a threshold chosen by hand. Report the "
            "sensitivity alongside any rate taken from this mask."
        )
    return _verdict(name, verdict, input_filepath, lines)


# ------------------------------------------------------------ 5. detector agreement


def check_detector_agreement(
    csv_a: str,
    csv_b: str,
    key_columns: str = "z,y,x",
    flag_column_a: str = "",
    flag_column_b: str = "",
    match_tolerance_vox: float = 3.0,
) -> str:
    """
    Compares the sites flagged by two independent detectors and reports their overlap.

    Two detectors sharing an input can agree because both are right or because both
    inherit the same error -- agreement is necessary, not sufficient. Disagreement is
    the informative direction: it bounds how much of each detector's output is method
    rather than specimen.

    Matches rows across the two files by position within `match_tolerance_vox`, since
    independent detectors will not report bit-identical coordinates.

    Args:
        csv_a: First detector's CSV, e.g. outputs/lattice_iou/node_health.csv.
        csv_b: Second detector's CSV, e.g. outputs/node_planes_2d/empty_sites.csv.
        key_columns: Comma-separated position columns present in both. Default "z,y,x".
        flag_column_a: Column in csv_a selecting flagged rows (truthy). Empty = all rows.
        flag_column_b: As above for csv_b.
        match_tolerance_vox: Euclidean distance within which two rows are the same site.

    Returns:
        PASS/WARN with counts, Jaccard overlap and the unmatched sites on each side.
    """
    name = "check_detector_agreement"
    keys = [k.strip() for k in key_columns.split(",") if k.strip()]

    def load(path, flag_column):
        if not os.path.isfile(path):
            return None, f"Error: {path} not found."
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return np.empty((0, len(keys))), None
        missing = [k for k in keys + ([flag_column] if flag_column else []) if k not in rows[0]]
        if missing:
            return None, f"Error: {path} has no column(s) {missing}. Present: {list(rows[0])}"
        if flag_column:
            rows = [r for r in rows if r[flag_column] not in ("", "0", "False", "false", "nan")]
        try:
            return np.array([[float(r[k]) for k in keys] for r in rows], float), None
        except ValueError as exc:
            return None, f"Error: non-numeric key column in {path}: {exc}"

    pts_a, err = load(csv_a, flag_column_a)
    if err:
        return _fail(name, csv_a, err)
    pts_b, err = load(csv_b, flag_column_b)
    if err:
        return _fail(name, csv_b, err)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return _verdict(name, "WARN", csv_a, [
            f"A: {len(pts_a)} flagged sites from {csv_a}",
            f"B: {len(pts_b)} flagged sites from {csv_b}",
            "One side is empty; there is nothing to compare.",
        ])

    distance = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
    a_matched = (distance <= match_tolerance_vox).any(axis=1)
    b_matched = (distance <= match_tolerance_vox).any(axis=0)
    shared = int(a_matched.sum())
    union = len(pts_a) + len(pts_b) - shared
    jaccard = shared / union if union else 0.0

    verdict = "PASS" if jaccard >= 0.5 else "WARN"
    lines = [
        f"A: {len(pts_a)} flagged sites  ({os.path.basename(csv_a)}"
        + (f", flag={flag_column_a}" if flag_column_a else "") + ")",
        f"B: {len(pts_b)} flagged sites  ({os.path.basename(csv_b)}"
        + (f", flag={flag_column_b}" if flag_column_b else "") + ")",
        "",
        f"matched within {match_tolerance_vox} vox: {shared}",
        f"A only: {len(pts_a) - shared}    B only: {len(pts_b) - shared}",
        f"Jaccard overlap: {jaccard:.3f}",
        "",
        "Agreement alone is not confirmation if both detectors read the same mask and "
        "the same registration; the unmatched sites are the informative set.",
    ]
    return _verdict(name, verdict, csv_a, lines)
