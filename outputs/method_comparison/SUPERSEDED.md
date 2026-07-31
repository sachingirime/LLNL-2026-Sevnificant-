# Superseded — do not build reports on this directory

Everything under `outputs/method_comparison/` comes from three detectors that were
implemented, measured, and **withdrawn**. The CSVs still parse and still look
authoritative. They are not the current answer.

This file exists because an agent asked to "find the anomalies" read these CSVs and wrote
a plausible HTML report out of them, having never run the current detector.

## What is in here, and why it was withdrawn

| Directory | Method | Why it was dropped |
|---|---|---|
| `tube_connectivity/` | Occupancy in a radius-3 tube about the design centreline | Correct definition of a gap, but its calls acquire a strong x-gradient tracking the registration drift, so its rate is not a part-wide rate |
| `edt_radius/` | Local maximum EDT radius along the design-guided strut | Hit the resolution floor. Healthy median 2.236 vox; the robust 3σ missing cutoff came out at **−0.006 vox**. Any retune manufactures calls |
| `strut_aligned_2d_projection/` | Local 33×7 longitudinal/transverse projection per strut | A max projection hides an out-of-plane gap, and it inherits the same x-dependent registration error |

The 2D and 3D methods agreed on 89.78% of verdicts. That is **not** independent
confirmation — both inherit the same design-registration error.

## The numbers in here, so they are recognisable if quoted back

Over an 18,468-strut design, filtered to a 13,932-strut interior population:
277 missing (1.99%), 1,008 disconnected (7.24%), 5,836 uncertain.

If you see 277 / 1008, or 157 / 397 for the 2D–3D agreed subset, or a population of
13,932 — the source is this directory, and the result is superseded.

## Use instead

`detect_lattice_defects` in `src/mcp_server.py`, which classifies all 18,468 struts
(16,733 measurable) into `missing`, `broken`, `thin`, `thick`, `necked`, `nominal`. Rules
in `scripts/classify_strut_defects.classify`. See `AGENTS.md` for the full path and the
checks to run first.

Kept rather than deleted because a withdrawn method with the number that killed it is a
result worth keeping. It is not an input.
