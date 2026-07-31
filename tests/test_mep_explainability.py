"""Tests for the MEP explanation layer: tracing, checks, and rubric scoring."""

import json
import math

import numpy as np
import pytest

from src import check_tools, mep, mep_narrative, mep_rubric
from src.mep import mep_tool


# The pitch check_coordinate_frame expects, and the strut length that produces it:
# an octet strut spans cell/sqrt(2), so cell_vox = 4560 um / 58.1 um and the strut
# is that over sqrt(2).
_GOOD_STRUT_VOX = (4560.0 / check_tools.EXPECTED_UM_PER_VOXEL) / math.sqrt(2.0)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Isolate the trace for one test into tmp_path, and reset the step counter."""
    monkeypatch.setattr(mep, "MEP_ROOT", str(tmp_path / "mep"))
    monkeypatch.setattr(mep, "_RUN_ID", "test")
    monkeypatch.setattr(mep, "_STEP", 0)
    return "test"


def write_design(path, strut_length_vox=_GOOD_STRUT_VOX, n=6):
    """A minimal design JSON: n junctions on the x axis, each joined to the next."""
    junctions = [{"position": [i * strut_length_vox, 0.0, 0.0]} for i in range(n)]
    struts = [{"junction0": i, "junction1": i + 1} for i in range(n - 1)]
    path.write_text(json.dumps({"junctions": junctions, "struts": struts}))
    return str(path)


# ------------------------------------------------------------------ tracing


def test_packet_records_actor_rationale_and_inputs(run, tmp_path):
    source = tmp_path / "volume.npy"
    np.save(source, np.zeros((2, 2, 2), dtype=np.uint8))

    @mep_tool(kind="analysis")
    def fake_tool(input_filepath: str, output_filepath: str) -> str:
        return "done"

    fake_tool(
        input_filepath=str(source),
        output_filepath=str(tmp_path / "out.npy"),
        actor="threshold-optimizer",
        why="probing the cut",
    )

    packet = json.loads(open(mep.trace_path()).read())
    assert packet["actor"] == "threshold-optimizer"
    assert packet["why"] == "probing the cut"
    assert packet["kind"] == "analysis"
    assert packet["status"] == "ok"
    assert packet["artifact"] == "done"
    # The verification slot is filled retroactively by the rubric, never at call time.
    assert packet["verification"] == []
    # An input is fingerprinted; a declared output is an artifact, not evidence.
    assert [r["path"] for r in packet["inputs"]] == [str(source)]
    assert packet["artifact_files"] == [str(tmp_path / "out.npy")]


def test_packet_records_a_returned_error_as_a_failed_step(run, tmp_path):
    @mep_tool(kind="analysis")
    def fake_tool(input_filepath: str) -> str:
        return "Error: mask file not found"

    fake_tool(input_filepath=str(tmp_path / "absent.npy"))

    packet = json.loads(open(mep.trace_path()).read())
    assert packet["status"] == "error"
    assert packet["actor"] == "main"
    assert packet["inputs"] == []


def test_packet_is_written_even_when_the_tool_raises(run, tmp_path):
    @mep_tool(kind="analysis")
    def fake_tool(input_filepath: str) -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fake_tool(input_filepath=str(tmp_path / "x.npy"))

    packet = json.loads(open(mep.trace_path()).read())
    assert packet["status"] == "exception"
    assert "boom" in packet["error"]


# ------------------------------------------------------------------- checks


def test_coordinate_frame_passes_on_a_design_in_ct_voxels(tmp_path):
    design = write_design(tmp_path / "design.json")
    assert "VERDICT=PASS" in check_tools.check_coordinate_frame(design)


def test_coordinate_frame_fails_on_a_design_in_nominal_units(tmp_path):
    # Nominal half-cell units are far smaller than voxels, so the recovered pitch
    # comes out large and the ratio well above 1.
    design = write_design(tmp_path / "nominal.json", strut_length_vox=1.41)
    result = check_tools.check_coordinate_frame(design)
    assert "VERDICT=FAIL" in result
    assert "not in ct voxels" in result.lower()


def test_cache_staleness_fails_on_a_section_count_mismatch(tmp_path):
    np.savez(tmp_path / "sections.npz", prof_area=np.zeros((10, 25), dtype=np.float32))
    result = check_tools.check_cache_staleness(str(tmp_path), sections=33)
    assert "VERDICT=FAIL" in result
    assert "MISMATCH" in result


def test_cache_staleness_passes_when_the_cache_matches(tmp_path):
    np.savez(tmp_path / "sections.npz", prof_area=np.zeros((10, 25), dtype=np.float32))
    assert "VERDICT=PASS" in check_tools.check_cache_staleness(str(tmp_path), sections=25)


def test_cache_staleness_passes_when_nothing_is_cached(tmp_path):
    assert "VERDICT=PASS" in check_tools.check_cache_staleness(str(tmp_path), sections=25)


def _correction(tmp_path, A_xx=0.0, t=(0.0, 0.0, 0.0), residual=None):
    doc = {
        "A_zyx": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, A_xx]],
        "t_zyx": list(t),
    }
    if residual is not None:
        doc["residual_rms_zyx"] = list(residual)
    path = tmp_path / "correction.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_a_large_correction_with_a_small_residual_passes(tmp_path):
    # The whole point of the refit is to remove a big misalignment. Judging the
    # correction's own size would penalise it for working: a better fit moves the
    # design further and would score worse. What matters is what it left behind.
    design = write_design(tmp_path / "design.json", n=40)
    result = check_tools.check_alignment_residual(
        _correction(tmp_path, A_xx=0.01, residual=(0.27, 0.55, 0.32)), design,
        correction_applied=True,
    )
    assert "VERDICT=PASS" in result
    assert "0.548" in result or "0.550" in result


def test_a_large_residual_fails_however_small_the_correction(tmp_path):
    design = write_design(tmp_path / "design.json", n=40)
    result = check_tools.check_alignment_residual(
        _correction(tmp_path, A_xx=0.0, residual=(0.3, 3.1, 0.4)), design,
        correction_applied=True,
    )
    assert "VERDICT=FAIL" in result
    assert "did not bring the design onto the struts" in result


def test_an_unapplied_correction_is_judged_on_the_raw_offset_instead(tmp_path):
    # Same file, opposite question: if nothing applies the correction, the offset it
    # would have removed is the live error.
    design = write_design(tmp_path / "design.json", n=40)
    correction = _correction(tmp_path, A_xx=0.01, residual=(0.27, 0.55, 0.32))

    assert "VERDICT=PASS" in check_tools.check_alignment_residual(
        correction, design, correction_applied=True)
    unapplied = check_tools.check_alignment_residual(
        correction, design, correction_applied=False)
    assert "VERDICT=FAIL" in unapplied
    assert "x-gradient" in unapplied


def test_drift_falls_back_to_the_raw_offset_when_no_residual_was_recorded(tmp_path):
    design = write_design(tmp_path / "design.json", n=40)
    result = check_tools.check_alignment_residual(
        _correction(tmp_path, t=(0.2, 0.1, 0.15)), design, correction_applied=True)
    assert "VERDICT=PASS" in result
    assert "no residual_rms_zyx" in result


def test_provenance_fails_when_a_file_is_written_twice(run, tmp_path):
    target = tmp_path / "mask.tif"
    target.write_bytes(b"mask")

    @mep_tool(kind="analysis")
    def segment(output_filepath: str) -> str:
        return "ok"

    segment(output_filepath=str(target), actor="main")
    segment(output_filepath=str(target), actor="threshold-optimizer")

    result = check_tools.check_provenance(str(target), run_id="test")
    assert "VERDICT=FAIL" in result
    assert "threshold-optimizer" in result


def test_provenance_warns_for_a_file_from_outside_the_run(run, tmp_path):
    target = tmp_path / "supplied.tif"
    target.write_bytes(b"data")

    # The run has to have recorded something, or the absent trace is the finding.
    @mep_tool(kind="analysis")
    def unrelated(output_filepath: str) -> str:
        return "ok"

    unrelated(output_filepath=str(tmp_path / "elsewhere.npy"))

    assert "VERDICT=WARN" in check_tools.check_provenance(str(target), run_id="test")


# ------------------------------------------------------------------- rubric


def _trace(tmp_path, packets):
    """Write packets as a trace and score them."""
    path = tmp_path / "mep" / "test" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p) for p in packets))
    return mep_rubric.score(mep_rubric.load("test"))


def _packet(step, tool, **kw):
    base = {
        "run_id": "test", "step": step, "packet_id": f"p{step}", "ts": 1000.0 + step,
        "duration_s": 0.1, "actor": "main", "session": {}, "tool": tool,
        "kind": "analysis", "args": {}, "why": "because", "status": "ok",
        "error": None, "artifact": "", "artifact_files": [], "inputs": [],
        "verification": [],
    }
    base.update(kw)
    return base


def test_error_recovery_flags_a_failed_check_that_was_then_ignored(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "check_alignment_residual", kind="check",
                artifact="CHECK check_alignment_residual VERDICT=FAIL TARGET=/data/design.json"),
        _packet(2, "detect_lattice_defects",
                inputs=[{"path": "/data/design.json", "size": 1, "mtime": 0, "edge_sha256": "a"}]),
    ])
    flag = result["flags"]["Error Awareness & Recovery"]
    assert flag["violated"]
    assert "used it anyway" in flag["evidence"][0]


def test_error_recovery_is_clean_when_the_failed_input_is_not_reused(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "check_alignment_residual", kind="check",
                artifact="CHECK check_alignment_residual VERDICT=FAIL TARGET=/data/design.json"),
        _packet(2, "refit_lattice_registration",
                inputs=[{"path": "/data/other.json", "size": 1, "mtime": 0, "edge_sha256": "b"}]),
    ])
    assert not result["flags"]["Error Awareness & Recovery"]["violated"]


def test_state_tracking_flags_a_dropped_registration_correction(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "measure_lattice_iou", args={"correction_filepath": "/data/correction.json"}),
        _packet(2, "detect_lattice_defects", actor="nde-report-generator",
                args={"correction_filepath": ""}),
    ])
    flag = result["flags"]["State Tracking Consistency"]
    assert flag["violated"]
    assert "no correction_filepath" in flag["evidence"][0]


def test_state_tracking_flags_a_mask_rewritten_by_a_second_actor(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "segment_ct_dataset", artifact_files=["/out/mask.tif"]),
        _packet(2, "segment_ct_dataset", actor="threshold-optimizer",
                artifact_files=["/out/mask.tif"]),
    ])
    assert result["flags"]["State Tracking Consistency"]["violated"]


def test_intent_alignment_flags_a_step_with_no_stated_reason(run, tmp_path):
    result = _trace(tmp_path, [_packet(1, "segment_ct_dataset", why="")])
    assert result["flags"]["Intent Alignment"]["violated"]


def test_a_check_marks_the_step_that_shares_its_target_as_verified(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "check_coordinate_frame", kind="check",
                artifact="CHECK check_coordinate_frame VERDICT=PASS TARGET=/data/design.json"),
        _packet(2, "measure_lattice_iou",
                inputs=[{"path": "/data/design.json", "size": 1, "mtime": 0, "edge_sha256": "a"}]),
    ])
    assert result["states"][2] == "pass"
    assert result["attached"][2][0]["check"] == "check_coordinate_frame"


def test_an_unchecked_analysis_step_is_reported_unverified(run, tmp_path):
    result = _trace(tmp_path, [
        _packet(1, "measure_lattice_iou",
                inputs=[{"path": "/data/design.json", "size": 1, "mtime": 0, "edge_sha256": "a"}]),
    ])
    assert result["states"][1] == "unverified"
    assert not result["trustworthy"]


# ------------------------------------------------------------------- render


def test_explain_renders_the_chain_and_writes_a_report(run, tmp_path):
    _trace(tmp_path, [
        _packet(1, "check_coordinate_frame", kind="check",
                artifact="CHECK check_coordinate_frame VERDICT=PASS TARGET=/data/design.json"),
        _packet(2, "measure_lattice_iou", actor="nde-report-generator",
                why="IoU for the report",
                inputs=[{"path": "/data/design.json", "size": 1, "mtime": 0, "edge_sha256": "a"}]),
    ])
    report = tmp_path / "report.html"
    text = mep_rubric.explain("test", str(report))

    assert "RUN test" in text
    assert "nde-report-generator" in text
    assert "IoU for the report" in text
    assert "RUBRIC" in text and "VERDICT" in text

    html = report.read_text()
    assert html.startswith("<!doctype html>")
    assert "measure_lattice_iou" in html
    # Both themes are styled, since the viewer's is unknown.
    assert "prefers-color-scheme" in html and "data-theme" in html


def test_explain_reports_a_clean_run_as_audit_ready(run, tmp_path):
    _trace(tmp_path, [
        _packet(1, "check_coordinate_frame", kind="check",
                artifact="CHECK check_coordinate_frame VERDICT=PASS TARGET=/data/design.json"),
        _packet(2, "measure_lattice_iou",
                inputs=[{"path": "/data/design.json", "size": 1, "mtime": 0, "edge_sha256": "a"}]),
    ])
    assert "Every analysis step rests on a checked input" in mep_rubric.explain("test")


def test_explain_raises_for_an_unrecorded_run(run):
    with pytest.raises(FileNotFoundError):
        mep_rubric.explain("no-such-run")


# ---------------------------------------------------------------- the story view


def _classified(tmp_path, extra_packets, counts=None, excluded=None, with_flag=True):
    """A run that classified struts, with strut_classes.csv written where it says.

    ``excluded`` are struts the detector could not measure: boundary caps, struts buried
    in the build plates, clipped windows. They carry a label but must never be counted.
    """
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    counts = counts or {"missing": 5, "broken": 3, "thin": 10, "thick": 8,
                        "necked": 1, "nominal": 73}
    excluded = excluded or {}
    with open(results / "strut_classes.csv", "w", newline="") as handle:
        handle.write("strut_id,label,measurable\n" if with_flag else "strut_id,label\n")
        i = 0
        for label, n in counts.items():
            for _ in range(n):
                handle.write(f"{i},{label},1\n" if with_flag else f"{i},{label}\n")
                i += 1
        for label, n in excluded.items():
            for _ in range(n):
                handle.write(f"{i},{label},0\n" if with_flag else f"{i},{label}\n")
                i += 1
    packets = [
        _packet(1, "begin_analysis", kind="goal",
                args={"user_prompt": "find the anomalies and tell me what kinds",
                      "goal": "classify every strut"}),
    ] + extra_packets + [
        _packet(90, "detect_lattice_defects", artifact_files=[str(results)],
                args={"sections": 25, "tolerance_fraction": 0.25, "use_cache": True}),
    ]
    return _trace(tmp_path, packets), packets


def _failed_check(step, name, target="/data/design.json"):
    return _packet(step, name, kind="check",
                   artifact=f"CHECK {name} VERDICT=FAIL TARGET={target}")


def test_a_failed_threshold_check_spares_missing_but_not_the_radius_classes(run, tmp_path):
    _classified(tmp_path, [_failed_check(2, "check_threshold_sensitivity")])
    packets = mep_rubric.load("test")
    trust = mep_narrative.class_trust(packets)

    # `missing` is a count against zero with a wide empty gap below the next strut,
    # so the cut cannot move it. Everything measured off the mask moves.
    assert trust["missing"]["trusted"]
    for name in ("thin", "thick", "necked", "broken"):
        assert not trust[name]["trusted"], name


def test_a_failed_registration_check_undermines_every_class(run, tmp_path):
    _classified(tmp_path, [_failed_check(2, "check_alignment_residual")])
    trust = mep_narrative.class_trust(mep_rubric.load("test"))
    assert not any(info["trusted"] for info in trust.values())


def test_all_classes_stand_when_every_check_passes(run, tmp_path):
    _classified(tmp_path, [
        _packet(2, "check_coordinate_frame", kind="check",
                artifact="CHECK check_coordinate_frame VERDICT=PASS TARGET=/data/design.json"),
    ])
    trust = mep_narrative.class_trust(mep_rubric.load("test"))
    assert all(info["trusted"] for info in trust.values())


def test_a_recheck_that_passes_supersedes_its_own_earlier_failure(run, tmp_path):
    # Otherwise a run can never recover: the fixed problem keeps condemning the classes
    # it used to touch, and the report tells the reader to fix what is already fixed.
    _classified(tmp_path, [
        _failed_check(2, "check_alignment_residual"),
        _packet(3, "check_alignment_residual", kind="check",
                artifact="CHECK check_alignment_residual VERDICT=PASS "
                         "TARGET=/data/design.json"),
    ])
    packets = mep_rubric.load("test")
    trust = mep_narrative.class_trust(packets)

    assert all(info["trusted"] for info in trust.values())
    assert mep_narrative.recovery_path(packets) == []


def test_a_recheck_on_a_different_target_does_not_supersede(run, tmp_path):
    _classified(tmp_path, [
        _failed_check(2, "check_alignment_residual", target="/data/a.json"),
        _packet(3, "check_alignment_residual", kind="check",
                artifact="CHECK check_alignment_residual VERDICT=PASS TARGET=/data/b.json"),
    ])
    trust = mep_narrative.class_trust(mep_rubric.load("test"))
    assert not any(info["trusted"] for info in trust.values())


def test_recovery_path_ranks_the_check_that_frees_the_most_classes(run, tmp_path):
    _classified(tmp_path, [
        _failed_check(2, "check_alignment_residual"),
        _failed_check(3, "check_threshold_sensitivity"),
    ])
    ranked = mep_narrative.recovery_path(mep_rubric.load("test"))

    # Registration is the sole obstacle to `missing`; the threshold check frees
    # nothing by itself because registration blocks those classes too.
    assert ranked[0]["check"] == "check_alignment_residual"
    assert ranked[0]["recovers"] == ["missing"]
    assert ranked[1]["recovers"] == []


def test_the_story_reports_counts_the_request_and_the_agents_own_settings(run, tmp_path):
    _classified(tmp_path, [_failed_check(2, "check_threshold_sensitivity")])
    report = tmp_path / "story.html"
    text = mep_narrative.story("test", str(report))

    assert "find the anomalies and tell me what kinds" in text  # the request, verbatim
    assert "18" not in text.split("WHAT WAS FOUND")[0]          # counts are not invented
    assert "missing" in text and "nominal" in text
    assert "thin/thick band width" in text                       # a discretionary choice
    assert "Believe: missing" in text                            # per-class verdict

    page = report.read_text()
    assert "How this defect analysis was done" in page
    assert "prefers-color-scheme" in page and "data-theme" in page


def test_the_story_says_so_when_the_request_was_never_recorded(run, tmp_path):
    _classified(tmp_path, [], counts={"missing": 1, "nominal": 9})
    # Drop the begin_analysis packet to simulate an agent that skipped it.
    path = tmp_path / "mep" / "test" / "trace.jsonl"
    kept = [l for l in path.read_text().splitlines() if "begin_analysis" not in l]
    path.write_text("\n".join(kept))

    assert "did not call begin_analysis" in mep_narrative.story("test")


def test_unmeasurable_struts_are_kept_out_of_the_tally(run, tmp_path):
    # A strut fused into a build plate leaves no signature that separates absence from
    # burial, so counting it inflates `missing` and `thick`. This is the real specimen's
    # shape: 89 measurable missing become 412 if the excluded rows are counted.
    _classified(
        tmp_path, [],
        counts={"missing": 89, "thick": 350, "nominal": 100},
        excluded={"missing": 323, "thick": 1028},
    )
    classes = mep_narrative.read_classes(mep_rubric.load("test"))

    assert classes["counts"]["missing"] == 89
    assert classes["counts"]["thick"] == 350
    assert classes["total"] == 539
    assert classes["designed"] == 1890
    assert classes["excluded"] == 1351
    assert classes["population_known"]


def test_a_results_file_without_the_measurable_column_is_flagged_not_silently_counted(
    run, tmp_path
):
    _classified(tmp_path, [], counts={"missing": 412, "nominal": 100}, with_flag=False)
    classes = mep_narrative.read_classes(mep_rubric.load("test"))

    assert not classes["population_known"]
    assert classes["excluded"] == 0
    # The reader has to be told the rate is an upper bound rather than left to assume.
    assert "upper bounds only" in mep_narrative.story("test")


def test_a_crashed_step_is_not_rendered_as_a_completed_stage(run, tmp_path):
    # A step that threw still has a `why`, so rendering it beside the steps that worked
    # reads as though the validation happened. It did not.
    _classified(tmp_path, [
        _packet(2, "validate_against_stl", status="exception", why="score against 0.5.stl",
                error="TypeError: unexpected keyword argument 'design_filepath'"),
    ])
    report = tmp_path / "s.html"
    text = mep_narrative.story("test", str(report))

    assert "DID NOT RUN" in text
    assert "TypeError" in text
    page = report.read_text()
    assert "did not run" in page and "struck" in page


def test_step_numbering_continues_across_processes(run, tmp_path):
    @mep_tool(kind="analysis")
    def step(output_filepath: str) -> str:
        return "ok"

    step(output_filepath=str(tmp_path / "a.npy"))
    step(output_filepath=str(tmp_path / "b.npy"))

    # A second server process starts with a fresh counter but the same run id; it must
    # continue the trace rather than restart at 1 and collide.
    mep._STEP = 0
    step(output_filepath=str(tmp_path / "c.npy"))

    steps = [json.loads(l)["step"] for l in open(mep.trace_path())]
    assert steps == [1, 2, 3]


def test_the_story_handles_a_run_that_classified_nothing(run, tmp_path):
    _trace(tmp_path, [_packet(1, "segment_ct_dataset")])
    text = mep_narrative.story("test")
    assert "No strut classification in this run" in text
    assert "nothing to trust or doubt" in text
