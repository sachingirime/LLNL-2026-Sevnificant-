# Data Inventory

Total files under data/: 16

## Files by extension

- `.json`: 4
- `.stl`: 4
- `.txt`: 3
- `.tif`: 2
- `.png`: 2
- `.npy`: 1

## Dataset folders (4)

- 9x9x9_octet_lattice/
- missing_struts/
- octet_truss_8x8x8/
- unitcell/

## JSON graph files (4)

- `missing_struts/octet_truss_9x9x9.json`: 10206 junctions, 18468 struts, 729 unit_cells, junction position range=((0.0, 0.0, 0.0), (18.0, 18.0, 18.0))
- `missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json`: 10206 junctions, 18468 struts, 729 unit_cells, junction position range=((58.76, 48.57, 24.5), (773.74, 764.94, 737.85))
- `octet_truss_8x8x8/octet_truss_8x8x8.json`: 7168 junctions, 13056 struts, 512 unit_cells, junction position range=((0.0, 0.0, 0.0), (16.0, 16.0, 16.0))
- `unitcell/polyhedron_1x1x1.json`: 14 junctions, 12 struts, 1 unit_cells, junction position range=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

## STL mesh files (4)

- `missing_struts/stls/0.1.stl`: 3511458 triangles
- `missing_struts/stls/0.5.stl`: 3498656 triangles
- `missing_struts/stls/0.stl`: 3514642 triangles
- `missing_struts/stls/1.stl`: 3482368 triangles

## Volumetric datasets (3)

- `9x9x9_octet_lattice/9x9x9_octet_lattice.tif`: shape=(761, 815, 837), dtype=uint16, min=0, max=65535, mean=34296.1
  -> histogram saved to outputs/exploration/9x9x9_octet_lattice__9x9x9_octet_lattice.tif.histogram.png
- `missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`: IDENTICAL to `9x9x9_octet_lattice/9x9x9_octet_lattice.tif` (md5 a61434d647...) — skipping duplicate histogram
- `unitcell/unitcell.npy`: shape=(256, 256, 256), dtype=float32, min=-0.00312875, max=0.0152577, mean=0.000539067
  -> histogram saved to outputs/exploration/unitcell__unitcell.npy.histogram.png
