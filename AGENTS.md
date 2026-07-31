# Working in this repository

X-ray CT defect analysis of additively manufactured octet-truss lattices, for the LLNL
Data Science Challenge 2026. Read this before starting work. It is the routing document:
it tells you which path is current and which looks current but is not.

Codex CLI reads this file. Claude Code reads it too if symlinked as `CLAUDE.md`.

## The one rule that matters

**Defect detection goes through the MCP tools in `src/mcp_server.py`.** Do not
reimplement it in a shell script, and do not assemble a report out of CSVs you find in
`outputs/`. Several directories there hold results from methods that were measured,
found unsound, and withdrawn; they are not labelled as wrong in their own filenames, and
a report built from them looks entirely plausible and is not.

If the MCP server is not connected, say so and stop. Do not route around it.

## The canonical path

Call these in order. Every one of them takes `actor=` and `why=`; pass both.

1. **`begin_analysis(user_prompt=..., goal=...)`** — first, always. Pass the user's
   request verbatim. The server never sees the conversation, so without this the run
   explanation cannot show what was asked, and cannot separate settings the user chose
   from settings you chose.
2. **`segment_ct_dataset`** — only if no mask exists. A supplied Otsu mask is at
   `data/9x9x9_octet_lattice/segmentation/mask.tif` (threshold 40127, 11.24% foreground).
3. **`refit_lattice_registration`** — produces `correction.json`. Every per-strut tool
   below takes `correction_filepath`. Omitting it is not a minor loss of accuracy: the
   defect population is swamped by registration error and the counts stop meaning
   anything.
4. **`detect_lattice_defects`** — the strut detector. Returns the class table and writes
   `strut_classes.csv`. Classes are `missing`, `broken`, `thin`, `thick`, `necked`,
   `nominal`; the rules live in `scripts/classify_strut_defects.classify`.
5. **`detect_missing_nodes`** / **`detect_missing_nodes_2d`** — the node detectors, if
   nodes are in scope.
6. **`validate_against_stl`** — scores the `missing` calls against the designed-missing
   set. Takes the **nominal** design (`data/missing_struts/octet_truss_9x9x9.json`), not
   the registered one.
7. **`explain_run(style="story", report_filepath=...)`** — last, always. Renders the
   account of what was done and how much of it to believe.

### When the user asks for an HTML report

`explain_run(style="story", report_filepath="outputs/<name>/report.html")` **is** the
deliverable. Do not hand-write a second HTML alongside it. The page it produces already
contains the request verbatim, the per-class counts over the measurable population, which
classes the checks support and which they do not, every figure the run's `visualize_*`
steps produced (inlined, so the file travels on its own), the stages in order with the
reason given for each, the settings the agent chose that the user did not, and the
shortest route back to a reportable answer.

So generate the figures **before** calling it — `visualize_slice`, `visualize_strut_classes`,
`visualize_lattice_3d` — and they are picked up automatically. A separate hand-written
report duplicates this, drifts from it, and carries none of the verification.

## Verify before you believe

Six checks are available as tools. They are not run automatically, by design: whether you
checked before asserting is part of what the run explanation reports.

| Check | Run it before |
|---|---|
| `check_coordinate_frame` | measuring anything against a design JSON |
| `check_alignment_residual` | quoting any per-strut or per-node rate |
| `check_cache_staleness` | `detect_lattice_defects` with `use_cache=True` |
| `check_threshold_sensitivity` | trusting a hand-picked segmentation cut |
| `check_provenance` | using a mask or results directory you did not just produce |
| `check_detector_agreement` | claiming two methods confirm each other |

A check that fails and is then ignored is worse than no check: it puts the refutation in
the record and the number in the report. If one fails, either fix the cause or scope the
claim to the classes it does not touch — `explain_run` prints which those are.

## Superseded — do not build on these

`outputs/method_comparison/` holds tube-connectivity, EDT-radius and strut-aligned-2D
results. **All three methods were withdrawn.** See `outputs/method_comparison/SUPERSEDED.md`.
Their CSVs are still on disk and still parse. A report assembled from them will quote
~277 missing and ~1008 disconnected over a 13,932-strut filtered population, which is not
the current answer and is not defensible. The current detector reports over all 18,468
struts with 16,733 measurable.

Two further dead ends, both already paid for:

- **Per-2D-slice defect analysis.** The lattice is oblique to every slice plane, so struts
  cut slices as ellipses whose shape encodes *tilt*. Any 2D ridge, coverage or area metric
  tracks tilt as much as health. This killed 2D Dice, `strut_contour_view`, and the
  dashboard's Frangi/high-pass/FFT panel.
- **3D Frangi vesselness as a classifier.** It collapses at junctions, so it cannot carry
  a graph. Fine as a visualisation aid, not as a detector.

## Standing constraints

- **The node grid drifts along x, and the fix is already available.** The design JSON is a
  graph of junction positions; the printed lattice came out ~1.2% smaller in x than the
  design places those nodes (`scale_correction_zyx` x = 0.9876), which accumulates to ~8.9
  voxels of position error across the 715-voxel span. This is about WHERE struts are, not
  how thick they are. Against ~4-voxel-wide struts, the far face measures the gap beside
  the strut, so calls climb with x and read as "the part gets worse to the right".
  `refit_lattice_registration` corrects it and leaves a sub-voxel residual
  (`residual_rms_zyx` = 0.27, 0.55, 0.32) — so **pass `correction_filepath` and the drift
  is handled.** Quoting a rate without it, or without `check_alignment_residual`, is what
  is uninterpretable. Note the residual is one whole-part RMS and cannot show whether what
  remains is still structured in x.
- **Use the mask, not raw CT,** for per-strut work. Raw background sits near 32k of 65k,
  so per-sub-volume Otsu is unstable.
- **As-built struts are ~2 voxels in radius** against a nominal 3.01. Thickness and shape
  classes sit close to the resolution floor; treat `thin`/`thick`/`necked` as weaker
  evidence than `missing`/`broken`.
- **Never choose a cutoff to reproduce a published rate.** Tran et al. report 0.57%
  missing and 4.97% disconnected. That is a comparison, never a target. A threshold tuned
  to hit it has measured nothing.
- **Never validate on an aggregate percentage alone.** The `0.stl` vs `0.5.stl` diff gives
  the designed-missing set exactly, so per-strut precision and recall are available.

## Data

| | |
|---|---|
| CT volume | `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif` (761×815×837 uint16) — the same file as `data/missing_struts/tif_stacks/210127_..._0point5dash1 1 Slices.tif` |
| Supplied mask | `data/9x9x9_octet_lattice/segmentation/mask.tif` |
| Registered design | `data/missing_struts/registered_jsons/210127_..._0point5dash1 1 Slices.json` — already in CT voxel coordinates |
| Nominal design | `data/missing_struts/octet_truss_9x9x9.json` — for `validate_against_stl` only |
| Ground truth | `data/missing_struts/stls/0.5.stl` |
| Resolution | 58.1 µm/voxel; 4.56 mm unit cell; 350 µm nominal strut |

There is **one** CT volume with **one** ground truth. The 8×8×8 PacificVis files are
simulated, unregistered, and their labels were never derived — do not present them as a
second validated specimen.

## Sub-agents

Skills in `.agents/skills/` run as sub-agents. MCP carries no caller identity — one server
process serves the whole session and a sub-agent shares its parent's connection — so each
skill must pass its own name as `actor=`. A skill that forgets is filed as `main`, and the
provenance audit is then wrong without looking wrong.

## Housekeeping

- Run tests with `python -m pytest tests/ -q` before claiming anything works.
- Write new results to a fresh directory under `outputs/`; do not overwrite a prior run's
  outputs, since another step may already have read them.
- A method that fails is a result. Write it down with the number that killed it. Three
  detectors have been withdrawn from this project because nobody checked whether their
  measurement could support their claim.
