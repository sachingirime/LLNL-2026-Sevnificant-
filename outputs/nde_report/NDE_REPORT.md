# NDE Report — Unit-Cell Reconstruction

**Source volume:** `data/unitcell/unitcell.npy`  
**Report scope:** The unit-cell is the only native volumetric `.npy` dataset in `data/`. The 9×9×9 lattice is supplied as a TIFF stack and the other directories contain design/support files, so they are not included in this 3D mask/skeleton analysis.

## Summary

| Category | Metric | Result |
|---|---|---:|
| Volume | Array shape / datatype | 256 × 256 × 256 / float32 |
| Volume | Total voxels | 16,777,216 |
| Volume | Mean intensity | 0.000539 |
| Volume | Intensity range | −0.003129 to 0.015258 |
| Mask | Segmentation threshold | 0.005000 intensity units |
| Mask | Foreground volume | 721,774 voxels (4.302% of volume) |
| Mask | Mean foreground intensity | 0.011661 |
| Mask | Connected regions (26-neighbor) | 1 |
| Skeleton | Skeleton length proxy | 3,173 skeleton voxels |
| Skeleton | Connected components (26-neighbor) | 1 |
| Skeleton | Endpoints | 47 |
| Skeleton | Branch voxels (at least 3 neighbors) | 168 |
| Skeleton | Maximum local degree | 8 |

Physical voxel spacing was not supplied, so volume and skeletal length are reported in voxel units.

## Visual Gallery

**View A — elevation 30°, azimuth 45°**

![Unit-cell isosurface, View A](unitcell_view_a.png)

**View B — elevation 60°, azimuth 45°**

![Unit-cell isosurface, View B](unitcell_view_b.png)

## Analysis

The mask, source volume, and skeleton have matching 256³ shapes. The supplied derived foreground mask uses an intensity threshold of 0.005; its mean intensity (0.011661) is strongly separated from the background mean (0.000039), so it isolates the high-intensity lattice struts cleanly. It forms one connected region, consistent with the continuous lattice shown in both renderings.

The skeleton is also one connected component and retains a branched network. Endpoint and branch-voxel counts are voxel-topology descriptors: they can be elevated by discretization near strut junctions, so they should be interpreted as complexity indicators rather than an exact count of design struts or nodes.

## Method and provenance

- Original intensity volume: `data/unitcell/unitcell.npy`
- Segmentation used: `outputs/task1/unitcell_segmentation_threshold_0.005.npy`
- Skeleton used: `outputs/task3/unitcell_skeleton.npy`
- Views were rendered using the prescribed 3D visualizer at elevations/azimuths of (30°, 45°) and (60°, 45°), with 4× downsampling for rendering only. The visualizer normalizes its input before extracting its 0.5 isosurface; this corresponds to a raw intensity of approximately 0.00605 and is close to the analysis mask threshold.
