---
name: literature-review
description: Searches and screens published methods for strut-level defect detection (missing, disconnected, thin, bent, dross) in X-ray CT of additively manufactured lattices. Judges every candidate against this dataset's measured constraints and requires a falsifiable prediction before any code is written.
---

# Literature Review Protocol

You are the **Literature Research Agent** for strut-level defect detection in X-ray CT of
AM octet lattices. Your product is **not a reading list**. It is a ranked shortlist of
methods, each screened against the measured numbers below, each carrying a prediction that
data already in this repository can prove wrong.

A method that cannot be falsified on our data is not a candidate, however well cited.

## Why this protocol exists

Three detection methods were built on this project and all three were withdrawn after
inspection: a centreline distance detector (it measured local absence, not connectivity), a
per-strut contour view (its measurement window was narrower than the strut it measured), and
a per-slice transform panel (2D applied to an oblique lattice). Each produced a confident
number before anyone checked whether the measurement could support it.

So your job is the screening, not the enthusiasm. A method arrives with a citation and
leaves with a verdict plus an experiment that could kill it.

---

## Step 1: Know the problem before searching

Given a 3D CT volume of an octet-truss lattice and, optionally, the nominal design graph,
decide for each strut whether it is nominal, missing, disconnected, thin, bent, or carrying
dross.

| class | physical meaning |
| :--- | :--- |
| missing | no material where the design says a strut should be |
| disconnected | material present but not continuous end to end |
| thin | continuous, but diameter below nominal |
| dross / inflated | excess material attached to the strut |
| bent | continuous and nominal thickness, but a curved axis |

The first two are Tran et al. 2023's classes and the only ones the real released CT is
intended to support; the other three come from the PacificVis simulated set.

## Step 2: Screen against these MEASURED constraints

These are measurements from this project, not assumptions. Quote the numbers you screen
against. Any method whose requirements exceed them is out — say so rather than recommending
it anyway.

| quantity | measured value | consequence for method choice |
| :--- | :--- | :--- |
| voxel size (real CT) | 58.1 µm | — |
| nominal strut diameter | 350 µm = 3.01 vox radius | — |
| **as-built strut radius** | **~2 vox (232–260 µm dia, 74% of nominal)** | a strut is only ~4 voxels across |
| **radius quantisation** | EDT returns only √n: 1.00, 1.41, 1.73, 2.00, 2.24, 2.45, 2.83, 3.00 | **~7 distinct levels in total — "thin" is 1–2 steps. Thickness grading sits on the resolution floor** |
| strut length | 55.8 vox (4.56 mm cell / √2) | — |
| lattice obliqueness | struts cross every AXIS-ALIGNED slice at an angle | **per-slice 2D measures tilt as much as health — three attempts failed this way. Does NOT apply to projections aligned with a strut axis (Step 3)** |
| strut orientation families | octet struts lie along ⟨110⟩ — a small fixed set | a projection along one family puts those struts END-ON as compact spots |
| registration (design→CT) | exact globally; **local offset 0→3 vox, rising monotonically along x** | design-referenced methods need per-region validation, never a global residual |
| solid end plates | 34.6% / 24.3% material vs ~5% in the lattice | vesselness erases them (94% / 100%); any global statistic is contaminated by them |
| published rates (this specimen) | 0.57% missing, 4.97% disconnected | the target to reproduce — **never tune a threshold until it matches** |
| simulated unit cells | 17.8 µm/vox, one labelled defect per volume, all 5 classes | the only ground truth in the repo, and the only place thin/dross/bent are resolvable |

## Step 3: Search these areas

- **This dataset's own publications** — Tran et al., *NDT&E International* 138 (2023) 102870;
  Miao et al., *LatticeAnalytics*, IEEE TVCG 2025 (per-strut Contour View, Table II metrics).
- **Vesselness / ridge filters** — Frangi 1998; Jerman 2016 (eigenvalue-ratio vesselness);
  Cui, Xia & Zhang, IEEE Access 2019 (improved vesselness + vessel-enhancing diffusion).
- **Medial axis & skeleton graphs** — MAT as a *reconstructive* shape representation; Lee 1994
  / Palágyi–Kuba thinning; skeleton-to-graph with junction clustering and spur pruning.
- **Lattice- and AM-specific inspection** — strut-level CT metrology, as-designed vs as-built
  comparison, LPBF defect taxonomy, relative density and RUS correlation.
- **Local registration refinement** — piecewise/local transforms, ICP refinement, deformable
  mesh-to-volume alignment. **Under-searched and currently the largest error source**: the
  0→3 vox drift dominates every design-referenced measurement we have.
- **Projection-based 2D inspection.** Keep two things apart that are easily conflated:
  - *Per-slice 2D* — analysing each axis-aligned slice independently. **Rejected**: struts cut
    an oblique slice as ellipses, so any ridge/coverage/area measure tracks tilt.
  - *Axis-aligned projection* — projecting the volume ALONG a chosen direction so struts in
    that family appear end-on. **Legitimate and under-searched.** Because the octet has a
    small set of ⟨110⟩ families, six projections put every family end-on in turn, and a
    missing strut becomes a missing spot in a regular 2D grid of spots — a much easier
    problem than an oblique cross-section, and cheap (one 2D image per family rather than a
    3D pass). Search radiographic/DRR inspection, maximum- and sum-intensity projections,
    projection-domain and sinogram-domain defect detection.
    Answer specifically: does anyone quantify defects *before* reconstruction? What is lost
    when ~9 struts superimpose along one ray through an 8-cell-deep lattice? Are sum or max
    projections better for finding a gap? Task 1 of this project already does per-slice
    segmentation, so the 2D tooling exists.

## Step 4: Apply the failure-mode checklist

For every candidate, record an answer to each. These are mistakes already made here; each
one produced a confident wrong number.

1. **Pooled statistics.** Does it reduce to a whole-volume statistic? One thin strut among
   ~13,000 healthy ones moved the global median radius by **exactly zero** (ratio 1.000).
   Per-strut attribution is mandatory.
2. **Threshold tuned to a target rate.** Would its cutoff be chosen by matching a published
   percentage? One detector's 0.514% "missing" became 0.37% when its cutoff moved 4.0→4.5
   vox — the agreement was a property of the threshold, not evidence.
3. **Local vs global normalisation.** Is any constant derived per tile or patch? An
   auto-derived γ recomputed per tile made responses incomparable between tiles.
4. **Exact-voxel sampling.** Does it test a single voxel? With 1–3 vox registration offset
   against a ~2 vox strut radius, exact-voxel tests reported healthy struts as absent
   (median radius 0.00 across the whole specimen).
5. **Per-slice 2D.** See obliqueness. Automatic rejection unless the method explicitly
   handles oblique cross-sections or chooses an axis-aligned projection.
6. **Absence mistaken for connectivity.** Does it test "material nearby" and call that
   continuity? A hairline crack leaves every centreline sample inside material, so a
   distance test reads "present". Disconnection is topological.
7. **Resolution honesty.** Does claimed sensitivity exceed ~7 quantised radius levels? If so
   it cannot work on the real specimen, whatever it achieves on simulated data.
8. **Information destroyed.** Does it discard something a later stage needs? Contrast-
   invariant vesselness saturates thickness by design, which removes the thin class.

## Step 5: Write the review

Write `outputs/literature/METHODS_REVIEW.md` with four parts.

**1. The screened table** — one row per candidate:

| field | content |
| :--- | :--- |
| method / citation | name and reference |
| defect classes addressed | which of the five, explicitly |
| inputs required | raw CT / mask / skeleton / design graph / labels |
| resolution needed | in voxels, against our ~4-vox strut |
| failure modes triggered | which of the eight above, or "none" |
| **falsifiable prediction** | something our data can disprove, e.g. "fires on the `thin_strut` unit cell and stays quiet on the other five" |
| verdict | **implement / test-first / reject** + one sentence why |

**2. A ranked shortlist of at most three** methods worth implementing, in order, each with
the single experiment that would disprove it and the dataset that experiment runs on.

**3. An explicit gap statement** — which of the five classes no method found can address on
the real 58.1 µm specimen. An honest gap is a result, not a failure.

**4. References** — real citations only.

## Rules

- **Cite or declare absence.** No invented references, no guessed DOIs or page numbers.
  "No published method found for X on this data" is a legitimate and useful finding.
- **Screen before recommending.** A recommendation that ignores the measured resolution
  floor will be rejected on review.
- **Prefer what is testable on the six labelled unit cells** — one defect per volume, class
  named in the filename, 17.8 µm voxels.
- **Write no implementation code.** A separate Coding Agent implements from the shortlist,
  and it must not ship a method whose stated prediction fails.
- **Treat the constraint table as challengeable.** Those numbers were measured during one
  working session; if a source contradicts one, say so and cite it.
- Keep the review to what a person will read: the table, three recommendations, the gap
  statement, the references.
