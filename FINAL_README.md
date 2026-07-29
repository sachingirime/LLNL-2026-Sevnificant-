# 9×9×9 octet-lattice XCT: literature-guided defect screen

## Outcome

This report applies published, strut-level XCT inspection ideas to the 9×9×9 octet lattice in `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`.  The scan is the same file as the registered Tran specimen at `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`.

The analysis produces useful **missing/disconnected-strut candidates**, but does **not** establish a defensible whole-part defect rate.  Candidate calls rise sharply with x, matching the measured 0–3 voxel design-to-CT registration drift.  This is a registration artifact warning, not evidence that defects physically increase across the part.

## Dataset and analysis basis

| Item | Value |
|---|---|
| CT volume | 761 × 815 × 837, uint16 |
| Segmentation | supplied Otsu mask, threshold 40,127; 11.24% foreground |
| Design graph | 18,468 struts, already registered in CT voxel coordinates |
| Reported population | 13,932 interior, non-end-plate struts (boundary caps and plate regions excluded) |
| Physical resolution | 58.1 µm/voxel; as-built struts are roughly 4 voxels across |
| Main limitation | local registration offset increases from ~0 to ~3 voxels along x |

The strut width is too close to the voxel scale to make reliable real-scan thin/dross/bent classifications: the local EDT radius has only about seven useful quantized levels.  Those classes should instead be validated on the labelled, higher-resolution simulated unit cells in `data/PacificVis Datasets/octet unit cell with defects/`.

## Related literature and how it was used

| Source | Relevance | Applied decision |
|---|---|---|
| Tran *et al.* (2023), [*Resonant ultrasound spectroscopy measurement and modeling of additively manufactured octet truss lattice cubes*](https://doi.org/10.1016/j.ndteint.2023.102870) | XCT-based strut-continuity reference for this specimen family. | Used only as a reference comparison (0.57% missing, 4.97% disconnected), never to tune thresholds. |
| Miao *et al.* (2025), [*LatticeAnalytics*](https://doi.org/10.1109/TVCG.2025.3593230) | Design-guided, strut-specific XCT inspection; local strut views/Contour View. | Motivated registered per-strut measurements and strut-aligned 2-D projections rather than axis-aligned CT slices. |
| Frangi *et al.* (1998), [*Multiscale vessel enhancement filtering*](https://doi.org/10.1007/BFb0056195) | Hessian/vesselness enhancement of tubular structures. | Screened as a visualization/preprocessing aid, rejected as a classifier: it does not establish topology and discards thickness information. |
| Jerman *et al.* (2016), [*Enhancement of vascular structures in 3D and 2D angiographic images*](https://doi.org/10.1109/TMI.2016.2550102) | Documents scale/contrast and junction limitations of Hessian vesselness. | Reinforced the decision not to use vesselness for junction-heavy, ~4-voxel-wide struts. |
| Oosterbeek & Jeffers (2022), [*StrutSurf*](https://doi.org/10.1016/j.softx.2022.101043) | Per-strut morphology and surface-roughness analysis from micro-CT. | Supports local strut attribution, while also highlighting that diameter/ellipticity need more resolution than this scan provides. |

The full screened review, including falsifiable predictions and rejected methods, is in [METHODS_REVIEW.md](outputs/literature/METHODS_REVIEW.md).

## Methods applied to this dataset

All methods use the same registered graph, Otsu mask, population, and 33 longitudinal samples per strut.  No cutoff was selected to match a published rate.

| Method | What it measures | Result on 13,932 eligible struts | Interpretation |
|---|---|---:|---|
| Tube-constrained 3-D connectivity | Occupancy in a radius-3 voxel tube; an internal empty run with material at both ends is required for `disconnected`. | 277 missing (1.99%); 1,008 disconnected (7.24%); 1,300 uncertain (9.33%). | Correct measurement definition for a gap, but its calls acquire a strong x-gradient, so it is not a valid part-wide rate yet. |
| Per-strut EDT radius | Local maximum EDT radius and material support along the design-guided strut. | 0 missing; 4,876 uncertain (35.00%). | The healthy median is 2.236 voxels and the robust 3σ missing cutoff becomes −0.006 voxels. This demonstrates the resolution floor; retuning would manufacture calls. |
| Strut-aligned 2-D maximum projection | A local 33 × 7 longitudinal/transverse projection through each strut, avoiding oblique axis-aligned slices. | 214 missing (1.54%); 557 disconnected (4.00%); 1,151 uncertain (8.26%). | Good candidate-triage visualization, but a max projection can hide an out-of-plane gap and retains the same x-dependent registration error. |

The 2-D and 3-D methods agree on 12,508 / 13,932 verdicts (89.78%), but this is not independent confirmation because both inherit the same design-registration error.

## Quality-control finding

| Region | Tube missing | Tube disconnected | Conclusion |
|---|---:|---:|---|
| Low-drift subset, midpoint x ≤ 300 vox | 0.72% | 1.10% | Less confounded, but still not externally validated. |
| Whole eligible population | 1.99% | 7.24% | Includes the drift-confounded high-x region. |
| High-drift check, midpoint x > 500 vox | 4.39% | 17.21% | The increase is incompatible with a credible spatially uniform defect rate. |

For comparison only, Tran *et al.* report 0.57% missing and 4.97% disconnected.  Neither the low-drift subset nor the global result should be claimed as reproducing those numbers: the registration diagnostic is more informative than numerical proximity.

## Deliverables

- [Complete comparison report](outputs/method_comparison/README.md)
- [Literature screen and shortlist](outputs/literature/METHODS_REVIEW.md)
- [Tube-connectivity candidates](outputs/method_comparison/tube_connectivity/struts.csv)
- [EDT measurements](outputs/method_comparison/edt_radius/struts.csv)
- [Strut-aligned 2-D candidates](outputs/method_comparison/strut_aligned_2d_projection/struts.csv)
- [2-D vs 3-D agreement figure](outputs/method_comparison/2d_vs_3d/agreement_and_x_rates.png)
- [Tube diagnostic](outputs/method_comparison/tube_connectivity/diagnostics.png) and [interactive overlay](outputs/method_comparison/tube_connectivity/viewer.html)

## Reproduction

The comparison that produced these files can be rerun from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl_defect python scripts/compare_defect_methods.py
python scripts/compare_2d_3d.py
python scripts/project_struts_2d.py
```

## Recommended next step

Locally refine the design-to-mask alignment before re-running the unchanged tube-topology rule.  Accept a defect-rate estimate only if the missing/disconnected flag-rate-versus-x plot no longer shows the present monotonic high-x increase.  Keep 2-D strut-aligned views for review, and use the labelled simulated unit cells to test thin, dross, and bent detectors separately.
