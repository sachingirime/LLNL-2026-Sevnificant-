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
and `offset = 0,0,0`. No registration fitting is required or wanted.

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
- Registration is exact globally but drifts **0 → 3 vox along x**. Sample a tube of radius
  2–3 vox, never a single voxel.
- Solid end plates occupy z < ~95 and z > ~665 (34.6% / 24.3% material vs ~5% in the
  lattice). Exclude them from any population statistic or it will be contaminated.

For calibration and for the classes the real specimen cannot resolve, use the labelled
simulated set: `data/PacificVis Datasets/octet unit cell with defects/` — six volumes,
17.8 µm/vox, **one defect per volume with the class in the filename**. This is the only
ground truth in the repository.

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
| MCP tools in `src/mcp_server.py` | `segment_ct_dataset`, `rasterize_lattice`, `visualize_slice`, `compare_slices`, `visualize_lattice_3d`, `export_lattice_html`, `skeletonize` |

Note `segment_ct_dataset` takes an explicit threshold — it has no Otsu inside — and
`skeletonize` accepts `.npy` only. Use the `threshold-optimizer` skill when a threshold is
unknown; simulated volumes need ~0.0058, the real CT needs 40127.

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
adjacent. One row per strut, all 18,468, in `outputs/method_comparison/<method>/struts.csv`:

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

Produce `outputs/method_comparison/README.md` containing:

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
