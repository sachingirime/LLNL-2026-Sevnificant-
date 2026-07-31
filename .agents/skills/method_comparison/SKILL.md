---
name: method-comparison
description: Implements the defect-detection methods shortlisted by the literature review, runs each on the registered real CT plus its design JSON, compares them on a common per-strut footing, and saves results and visualizations. Covers both axis-aligned 2D projection methods and 3D volumetric methods.
---

# Method Implementation and Comparison Protocol

You are the **Coding Agent**. You take the shortlist produced by the `literature-review`
skill, implement each method against the registered real data, and report how they compare
on identical struts with identical output columns.

Your product is a **comparison**, not a champion. A method that fails is a result; write it
down with the number that killed it. Three methods have already been withdrawn from this
project because nobody checked whether their measurement could support their claim.

## Identify yourself on every MCP tool call

Pass `actor="method-comparison"` to every MCP tool you call, and `why="<one line>"` saying what that
call is meant to establish. Both are recorded in the run's explanation trace
(`outputs/mep/<run_id>/trace.jsonl`), which `explain_run()` renders into the audit report.

This matters because MCP carries no caller identity: one server process serves the whole
session and a subagent shares its parent's connection, so a call you make with the default
`actor="main"` is indistinguishable from one the top-level agent made. The provenance audit
then cannot tell whether a mask this skill produced is the one a later step measured
against -- which is the failure mode the report exists to catch.

---

## Step 1: Read the shortlist first

Read `outputs/literature/METHODS_REVIEW.md`. For each method it ranks **implement** or
**test-first**, extract:

- the defect classes it claims to address
- the inputs it needs
- **its falsifiable prediction** — the statement the review committed to

You implement in the review's ranked order. If that file does not exist, stop and say so;
do not invent a method list.

Also read `.agents/skills/literature_review/SKILL.md` for the measured constraint table and
the eight-item failure-mode checklist. Do not restate those numbers from memory.

## Step 2: Use exactly this data

**Registered pair** — the design JSON is already in the CT's voxel frame, so `scale = 1.0`
and `offset = 0,0,0`. No *global* re-registration is required or wanted. A small **residual
correction** is a different matter and IS required — see below.

| role | path |
| :--- | :--- |
| CT volume | `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` |
| registered design graph | `data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json` |
| Otsu mask (threshold **40127**) | `data/9x9x9_octet_lattice/segmentation/mask.tif` |

Verified facts you can rely on, so do not re-derive them:

- The CT above is **byte-identical** to `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`
  (same md5). Either path is the same scan.
- Volume `(761, 815, 837)` uint16; Otsu threshold **40127**; foreground 11.24%.
- **18,468 struts** = 17,496 interior + 972 boundary caps (caps identified by
  `unit_cell_edge_idx` values occurring in under half the modal count; about half are never
  printed, so report them separately or exclude them).
- Junction `position` is `(x, y, z)`; reorder `[2,1,0]` to index `vol[z, y, x]`.
- **A residual node offset remains, and it grows along x.** Measured against the mask: each
  graph node sits a median **1.19 vox** from the material it names, rising **0.70 → 2.35 vox**
  from the near to the far x face. Applying `outputs/registration/correction.json` flattens
  this to **0.48 vox, uniform** (slope +0.01 vox across the whole span). Skipping it is not
  free — measured on this specimen it flips **1 strut in 11** at the far face, inflates
  `broken` from 107 to 283, and drops 2,327 struts out of the measurable population. Always
  pass `correction_filepath`, and sample a tube of radius 2–3 vox rather than a single voxel.
- Solid end plates occupy z < ~95 and z > ~665 (34.6% / 24.3% material vs ~5% in the
  lattice). Exclude them from any population statistic or it will be contaminated.

For calibration and for the classes the real specimen cannot resolve, use the labelled
simulated set: `data/PacificVis Datasets/octet unit cell with defects/` — six volumes,
17.8 µm/vox, **one defect per volume with the class in the filename**.

For the real specimen there is one ground truth: the `0.stl` vs `0.5.stl` diff gives the
designed-missing set exactly, scored by the `validate_against_stl` tool (which takes the
**nominal** design, `data/missing_struts/octet_truss_9x9x9.json`, not the registered one).
Use it for per-strut precision and recall — never validate on an aggregate percentage
alone. The 8×8×8 PacificVis files are simulated, unregistered, and their labels were never
derived; they are not a second validated specimen.

## Step 3: Reuse what exists — do not rebuild it

| asset | what it already does |
| :--- | :--- |
| `scripts/measure_struts.py` | per-strut table: radius profile (EDT local max), material coverage, intensity, tortuosity. Tiled, memory-safe |
| `scripts/frangi3d.py` | tiled 3D vesselness with a **global** two-pass γ (never per-tile) |
| `scripts/strut_graph.py` | skeleton → nodes/edges, two node definitions, degree histogram |
| `scripts/rasterize_design.py` | design JSON → voxel volume |
| `scripts/overlay_webgl.py` | design lines over the solid mask; `--table` optional for class colouring |
| `scripts/mesh_webgl.py` | any volume → self-contained WebGL2 mesh viewer |
| `scripts/graph_webgl.py` | strut graph → WebGL2 line viewer with per-class toggles |
| MCP tools in `src/mcp_server.py` | see the full list below — it is longer than this skill originally assumed |

Note `segment_ct_dataset` takes an explicit threshold — it has no Otsu inside — and
`skeletonize` accepts `.npy` only. Use the `threshold-optimizer` skill when a threshold is
unknown; simulated volumes need ~0.0058, the real CT needs 40127.

### The MCP tools, in full

An earlier version of this skill listed seven tools and omitted the detector. An agent
following it read stale CSVs out of `outputs/method_comparison/` and wrote a report from
them without running anything. The current set:

| Group | Tools |
| :--- | :--- |
| Goal and explanation | `begin_analysis`, `explain_run` |
| Segmentation | `segment_ct_dataset`, `filter_ct_volume` |
| Design and registration | `rasterize_lattice`, `refit_lattice_registration` |
| **Detection** | **`detect_lattice_defects`**, `measure_lattice_iou`, `detect_missing_nodes`, `detect_missing_nodes_2d`, `skeletonize` |
| Ground truth | `validate_against_stl` |
| Verification | `check_provenance`, `check_alignment_residual`, `check_cache_staleness`, `check_coordinate_frame`, `check_threshold_sensitivity`, `check_detector_agreement` |
| Visualisation | `visualize_slice`, `compare_slices`, `visualize_lattice_3d`, `visualize_strut_classes`, `export_lattice_html` |

**`detect_lattice_defects` is the current strut detector.** It classifies all 18,468
struts into `missing`, `broken`, `thin`, `thick`, `necked`, `nominal`; the rules are in
`scripts/classify_strut_defects.classify`. Any new method is compared *against* it, not
in place of it.

Call `begin_analysis` before anything else and `explain_run` at the end, and pass `actor=`
and `why=` on every call. Run the relevant checks before quoting a rate — `AGENTS.md`
lists which check guards which claim.

### Superseded results in `outputs/method_comparison/`

The `tube_connectivity/`, `edt_radius/` and `strut_aligned_2d_projection/` directories
hold **withdrawn** methods. Their CSVs parse and look authoritative. Read
`outputs/method_comparison/SUPERSEDED.md` before touching them, and never assemble a
report from them. If you see 277 missing / 1,008 disconnected, or a 13,932-strut
population, you are reading superseded output.

## Step 4: Implement both families

**3D volumetric.** Operates on the volume or mask directly — vesselness, medial axis /
distance transform, connectivity within a tube about the design centreline, skeleton graph
topology.

**2D axis-aligned projection.** Project the volume **along** a strut family's direction so
those struts appear end-on as compact spots; a missing strut is then a missing spot in a
regular 2D grid. Octet struts lie along ⟨110⟩, so a small set of projections puts every
family end-on in turn.

- Derive the projection directions from the design graph's own strut vectors — cluster the
  unit vectors, do not hard-code them.
- Report **both** sum and maximum projections; they behave differently over a gap.
- State explicitly what superposition costs: ~9 struts stack along one ray through this
  specimen, so quantify how much a single missing strut changes the ray integral before
  claiming the method works.
- Per-slice 2D on axis-aligned slices is **out of scope** — struts cut those planes
  obliquely and every such attempt on this project measured tilt rather than health.

## Step 5: One common output schema

Every method writes the same per-strut table so results are comparable rather than merely
adjacent. One row per strut, all 18,468, in `outputs/method_comparison_<date>/<method>/struts.csv`
— a **fresh** directory. Do not write into `outputs/method_comparison/`: it holds withdrawn
results that must stay distinguishable from new ones.

```
strut_id, is_boundary_cap, in_plate_region, midpoint_z, midpoint_y, midpoint_x,
verdict, confidence, <method-specific columns...>
```

`verdict` ∈ `nominal | missing | disconnected | thin | dross | bent | uncertain`.
**Emit `uncertain` rather than forcing a call** — near a threshold, or where local
registration is unreliable. An honest abstention is worth more than a guess.

Keep the raw per-sample measurements alongside as `.npz` so thresholds can be re-cut later
without recomputing anything.

## Step 6: Compare

Produce `outputs/method_comparison_<date>/README.md` containing:

1. **Method table** — one row per method: inputs used, runtime, defect classes it produced,
   counts and rates per class, and whether **its stated prediction from the review held**.
2. **Agreement matrix** — pairwise, over interior non-plate struts: how many struts each
   pair of methods labels the same, and where they disagree most. Two methods agreeing is
   evidence; two methods agreeing because both inherit the same registration error is not,
   so check whether disagreement correlates with x position.
3. **Reference comparison** — the rates against Tran et al. (0.57% missing, 4.97%
   disconnected) for this specimen, plus rates restricted to the well-registered region.
   Report both and say which you trust.
4. **Visual gallery** — embed the figures and link the WebGL viewers.
5. **What failed** — every method whose prediction did not hold, with the measurement that
   showed it.

## Step 7: Visualizations

- One `overlay_webgl.py` viewer per method that produces per-strut verdicts, with the
  measurement table passed so struts are coloured by class over the solid Otsu mask.
- One matplotlib panel per method: metric distribution, and **flag rate versus x** — that
  curve is the drift detector. A rate that rises monotonically along x is registration
  error, not a defect population.
- Keep viewers to one per method. They run 20–60 MB each; do not accumulate dozens.

## Rules

- **Never tune a threshold until the rate matches a published number.** Derive it from the
  specimen's own healthy population (median and a robust σ via MAD, not the standard
  deviation, which the plates and junction blobs inflate). Report the sweep, not one point.
- **Reference the measured healthy population, never the CAD nominal.** As-built struts are
  ~74% of the 350 µm design diameter; referencing the design classifies every strut as thin.
- **Per-strut attribution is mandatory.** A whole-volume statistic cannot see one bad strut
  among thousands — measured: a single thin strut moved the global median radius by exactly
  zero.
- **Respect the resolution floor.** The EDT returns only ~7 distinct radius values on the
  real specimen (√n voxels). Do not claim thickness grading there; test thin and dross on
  the 17.8 µm unit cells instead.
- **Verify each metric on the unit cell that isolates its defect** before it enters the
  comparison. If a metric does not fire on its own labelled volume, it does not ship.
- **Report in markdown.** No PDF conversion.
- Clean up throwaway scripts; anything worth keeping goes in `scripts/` with a docstring
  saying what it does and why.
