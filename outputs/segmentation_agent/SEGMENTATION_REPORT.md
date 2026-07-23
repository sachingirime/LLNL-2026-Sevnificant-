# Segmentation agent report

## Scope

Two CT volumes found under `data/` were segmented without modifying either
source file.  Outputs are stored under `outputs/segmentation_agent/` and can be
recreated with:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python outputs/segmentation_agent/run_segmentation.py
```

## Threshold selection and QC

Thresholds were selected using sampled intensity percentiles, binary trial
masks, and a side-by-side raw/mask quality-control slice.  For the lattice,
slice 380 trial thresholds of 36,000, 40,000, 40,500, 42,000, and 44,000 were
reviewed.  `40,500` retained the left-edge strut network and the regular node
pattern while avoiding the additional low-intensity background admitted at
36,000.  Its slice pattern visually agrees with the supplied
`ground_truth_segmentation_slice_380.png` reference.  The unit-cell `0.005`
threshold separates its high-density strut ring from the reconstruction
background.

## Results

| Dataset | Source shape / dtype | Threshold | Foreground voxels | Background voxels | Foreground fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `unitcell` | `(256, 256, 256)` / `float32` | `0.005` | 721,774 | 16,055,442 | 4.3021% |
| `9x9x9_octet_lattice` | `(761, 815, 837)` / `>u2` | `40,500` | 56,741,896 | 462,378,059 | 10.9304% |

## Deliverables

| Dataset | Binary mask | QC visualization |
| --- | --- | --- |
| Unit cell | `unitcell/unitcell_segmentation.npy` | `unitcell/unitcell_slice_128_axis0_qc.png` |
| 9x9x9 octet lattice | `9x9x9_octet_lattice/9x9x9_octet_lattice_segmentation.npy` | `9x9x9_octet_lattice/9x9x9_octet_lattice_slice_380_axis0_qc.png` |

## Verification

Both output arrays were reopened after writing.  Their shapes matched their
respective inputs, their dtypes were `uint8`, and the only observed values were
`0` and `1`.  The TIFF mask was written one axial plane at a time to avoid
loading or modifying the 1.04 GB source volume.
