# Strut-aligned 2-D projection screen

This method transfers each registered nominal strut into a local `(longitudinal, transverse-1, transverse-2)` frame.  It maximum-projects the Otsu mask through `transverse-2` over ±3 voxels, producing one 33 × 7 binary image per strut.  This is a strut-aligned reformat inspired by published Contour View / per-strut XCT workflows; it is not an analysis of axis-aligned CT slices.

| Eligible struts | Nominal | Missing | Disconnected | Uncertain |
|---:|---:|---:|---:|---:|
| 13,932 | 12,010 (86.20%) | 214 (1.54%) | 557 (4.00%) | 1,151 (8.26%) |

The result is a candidate screen, not a validated rate.  The high-x region (`x > 500` vox) has 3.15% missing and 9.08% disconnected calls, versus 0.69% and 0.94% in the lower-drift `x <= 300` region.  This confirms that changing to a 2-D strut-local space removes the oblique-slice problem but does not remove the known local graph-to-mask drift.

![Coverage and x-rate diagnostic](projection_diagnostics.png)

![Examples of local projections](projection_gallery.png)

![Raw-CT candidate audit](candidate_raw_ct_audit.png)

![Normal-plane maps by graph-derived direction family](orientation_projection_maps.png)

Files: [per-strut results](struts.csv), [raw profiles](raw_measurements.npz), and [method metadata](metadata.json).  The [direct 2-D versus 3-D agreement figure](../2d_vs_3d/agreement_and_x_rates.png) and full interpretation are in [the parent report](../README.md).
