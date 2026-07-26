# Findings — strut defect detection on the registered 9×9×9 CT

Working log for the DSC 2026 final project. Session of **2026-07-25**.

Everything below is measured on this repository's data. Where a number came from a single
session it says so, and it should be treated as challengeable rather than settled.

---

## 1. Measured constraints

These govern which methods can work at all, and most of the day's conclusions follow from
them rather than from any algorithm.

| quantity | value | consequence |
| :--- | :--- | :--- |
| CT volume | (761, 815, 837) uint16, 1.04 GB | — |
| voxel size | 58.1 µm | — |
| Otsu threshold | **40127** (settled in 2 iterations, 11.24% foreground) | segmentation is not the hard part |
| nominal strut | 350 µm dia = 3.01 vox radius | — |
| **as-built strut** | **~2 vox radius, 232–260 µm dia = 74% of nominal** | a strut is ~4 voxels across |
| **EDT radius quantisation** | only √n: 1.00, 1.41, 1.73, 2.00, 2.24, 2.45, 2.83, 3.00 | **~7 usable levels — thin/dross are below the resolution floor** |
| strut length | 55.8 vox (4.56 mm cell ÷ √2) | — |
| strut count | 18,468 = 17,496 interior + 972 boundary caps | ~half the caps are never printed |
| solid end plates | z < ~95 and z > ~665; 34.6% / 24.3% material vs ~5% in the lattice | contaminate any global statistic |
| **registration drift** | exact globally; local offset **0 → 3 vox rising monotonically along x** | dominates every design-referenced number |
| published rates (Tran 2023) | 0.57% missing, 4.97% disconnected | target, never a tuning goal |
| the two CT files | `9x9x9_octet_lattice.tif` and the `0point5dash1` tif_stack are **byte-identical** (md5 `a61434…`) | one scan, two paths |

The registration figure is the single most consequential measurement of the day, and it was
initially overstated: sampling whether the *exact voxel* was material gave 0.09 material
fraction in one corner and looked like a broken transform. With a 2–3 voxel tolerance the
same region reads 98% within 2 vox over most of the specimen and ~3 vox at worst. The
transform is fine; single-voxel sampling was not.

## 2. Methods withdrawn

Three detectors were removed after inspection showed the verdict did not follow from the
measurement. Reasoning is preserved in commit `72a35a2`; the code is not in any git object.

**`detect_struts`** — measured distance to nearby material and called it connectivity. A
strut severed by a hairline crack has every centreline sample inside material, so it read
"present". Of 1,301 interior "broken", 916 had max distance ≤ 5 vox, which is mild warp at
~2 vox radius plus ~1.4 vox scatter; only 107 reached the search cap. Its 0.514% missing
became 0.37% when the cutoff moved 4.0 → 4.5 vox, so agreement with the published 0.57% was
a property of the threshold.

**`strut_contour_view`** — `cross_radius_factor` multiplied the strut *radius*, giving a
29 px strut inside a 41 px window. 33 of 49 interior slices had the blob touching the
border, and `find_contours` emits open polylines there, which is why the "Contour View"
rendered as straight lines. The centre-component filter was also a no-op wherever a strut
fuses with neighbours at a junction: 23 of 49 slices reported area above the 641 px nominal,
peaking at 1,593 px = 95% of the window.

**The dashboard's Frangi/high-pass/FFT panel** — 420 pre-baked images, no generator in the
repo, and per-slice 2D on a lattice that is oblique to every slice plane.

## 3. Methods evaluated

**3D Frangi vesselness** (`scripts/frangi3d.py`) — works, but as a ranking cue rather than a
data feed. On a validated 192³ crop it separates strut from background at **AUC 0.90**, and
suppresses junction blobs. It retains more than expected: response magnitude correlates with
true EDT radius at **+0.760**, spatial extent at **+0.800**. But it erases plate-like
material (94% / 91% / 100% on the final slice) and discards 74% of the metal, so it cannot
serve as the shape representation. Jerman/Cui-style contrast-invariant vesselness would make
this worse, not better: it saturates thickness by design, removing the thin class.

**Skeleton → graph** (`scripts/strut_graph.py`) — unsolved. On the *defect-free* unit cell,
where the answer is 14 nodes and 36 edges, neither node definition recovers it. Skeleton
junction clustering gives 21 nodes / 43 edges; an EDT-maxima cut gives 12 / 30, and a sweep
of the cut from 6.5 to 10.0 never reaches 14/36 — corner nodes carry 3 struts and face
centres carry 8, so a single global radius cut cannot capture both. The design graph makes
this optional, which is why it was set aside rather than pursued.

**Per-strut measurement** (`scripts/measure_struts.py`) — the working primitive. Per-sample
radius from a local-max EDT, material coverage, intensity, tortuosity, over all 18,468
struts, tiled for memory. Two things it taught us: pooled statistics are blind (one thin
strut among ~13,000 moved the global median radius by **exactly zero**, ratio 1.000), and
tortuosity computed along a *design* centreline is identically 1.0 by construction, so bend
detection needs the as-built centreline.

## 4. Agent pipeline and its result

Two skills were written and then executed by the Codex agent.

- `.agents/skills/literature_review/SKILL.md` — screens candidate methods against the
  constraint table above and an eight-item failure-mode checklist drawn from §2, and requires
  a falsifiable prediction per method.
- `.agents/skills/method_comparison/SKILL.md` — implements the shortlist on the registered
  pair, on a common per-strut schema, with verification on the labelled unit cells.

The agent produced `outputs/literature/METHODS_REVIEW.md` and
`outputs/method_comparison/README.md`. **All three implemented methods were reported as
failing their own stated predictions**, which is the pipeline working rather than failing:

| method | verdict | why |
| :--- | :--- | :--- |
| tube-constrained 3D connectivity | failed as a specimen-level detector | topology test is sound, but missing/disconnected rates rise strongly with x |
| EDT local-radius | failed as a missing comparator | healthy median 2.236 vox, robust σ 0.747, so the pre-registered 3σ cutoff is **−0.006 vox** — no physically possible radius qualifies |
| strut-aligned 2D max projection | useful visualisation, failed as independent confirmation | agrees with 3D on 89.78%, but carries the same high-x excess |

Rates, all on 13,932 interior non-plate struts:

| region | tube missing | tube disconnected |
| :--- | ---: | ---: |
| all eligible | 1.99% | 7.24% |
| low-drift subset (x ≤ 300) | 0.72% | 1.10% |
| high-drift check (x > 500) | 4.39% | 17.21% |

The agent declined to promote the low-drift subset to a validated rate, and declined to tune
any cutoff toward 0.57% / 4.97%. It also reasoned correctly that 89.78% agreement between
the 2D and 3D methods is **not** independent validation, since both are design-referenced
and inherit the same alignment field.

Its projection screen concluded that ~9 struts superimpose along one ray through a 9-cell
part, so one missing strut perturbs a sum projection by about one ninth — a fast screen, not
a per-strut detector. That matches the physics and is a genuine answer to the question.

## 5. What is established, and what is not

**Established.** Otsu at 40127 is adequate. As-built struts are 74% of nominal diameter — a
process finding in its own right. Thin, dross and bent are **not** measurable on the 58.1 µm
specimen; they belong to the 17.8 µm labelled unit cells. Missing and wide disconnections are
the only supportable automatic labels here. Design-referenced measurement is viable but
bounded by local alignment.

**Not established.** Any defect *rate* for this specimen. Every method's rate varies by
6–17× across x, tracking the known drift direction, so no number survives as a
whole-specimen claim.

**The one blocking problem** is local registration refinement. The drift is smooth and
monotonic in x, which is what a small residual scale or shear term looks like. If a
per-region offset fit flattens the flag-rate-versus-x curve, the existing methods become
usable as they stand; if it does not, the design-referenced route should be abandoned for
graph topology — which in turn requires solving skeleton → graph.

## 6. Process notes

Four times this session a wrong conclusion came from a broken *measurement* rather than a
broken system, and each looked convincing at first: per-tile γ normalisation in the Frangi
tiler, unpadded marching cubes leaving cut struts open, threshold-then-max-pool inflating
4.2% foreground to 19.0%, and exact-voxel registration sampling. Every one was caught by
checking against a case with a known answer.

Hence the discipline now encoded in both skills: **verify a metric on data where the answer
is known before believing what it says about data where it is not.** The six labelled unit
cells (one defect per volume, class in the filename, 17.8 µm) are the only ground truth in
the repository and were under-used for most of the day.

## 7. Reproducing

```bash
# per-strut measurement on the registered pair (JSON is already in CT voxel space)
python scripts/measure_struts.py \
  "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \
  "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \
  -o outputs/measure/struts_real_body.npz \
  --scale 1.0 --offset 0,0,0 --threshold 40127 --t-range 0.25,0.75 --samples 21

# the agent's method comparison
MPLCONFIGDIR=/tmp/mpl_defect python scripts/compare_defect_methods.py

# design over as-built mask, single WebGL viewer
python scripts/overlay_webgl.py data/9x9x9_octet_lattice/segmentation/mask.tif \
  "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \
  -o outputs/webgl_design_vs_asbuilt.html --max-dim 256
```

Generated volumes, viewers and GIFs are gitignored — they are large and reproducible from
`scripts/`. The exception is `outputs/dashboard_0point5dash1.html`, which has no generator in
the repository and cannot be rebuilt.
