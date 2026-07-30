"""Declarative description of every MCP tool in src/mcp_server.py.

The frontend builds its forms purely from GET /api/tools, so this is the one
place that needs to change if a tool's signature changes. Each field mirrors
a parameter of the underlying function; "type" drives both the input widget
and the coercion applied before the call.
"""
from __future__ import annotations

TOOLS = [
    {
        "name": "segment_ct_dataset",
        "stage": "1. Segment",
        "summary": "Threshold a raw CT volume (.npy/.tif) into a binary material mask.",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": "Raw CT volume (.npy or .tif/.tiff)"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Where to save the mask (.npy)"},
            {"name": "threshold", "type": "float", "required": True, "default": 0.5, "help": "Density cutoff; voxels >= threshold become material"},
        ],
    },
    {
        "name": "visualize_slice",
        "stage": "2. Slice",
        "summary": "Save a 2D image of one slice through a 3D volume.",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": ".npy or .tif/.tiff volume"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Output image (.png)"},
            {"name": "slice_index", "type": "int", "required": True, "default": 0, "help": "Index along the chosen axis"},
            {"name": "axis", "type": "int", "required": False, "default": 0, "help": "0, 1, or 2"},
        ],
    },
    {
        "name": "skeletonize",
        "stage": "3. Skeletonize",
        "summary": "Reduce a binary mask to its 1-voxel-wide centerline skeleton.",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": "Binary mask (.npy)"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Where to save the skeleton (.npy)"},
        ],
    },
    {
        "name": "visualize_lattice_3d",
        "stage": "4. 3D preview",
        "summary": "Render a binary volume to a static PNG from a chosen camera angle (PyVista, offscreen).",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": "Binary volume (.npy or .tif/.tiff)"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Output image (.png)"},
            {"name": "azimuth", "type": "float", "required": False, "default": 30.0},
            {"name": "elevation", "type": "float", "required": False, "default": 20.0},
            {"name": "color", "type": "str", "required": False, "default": "lightgray"},
            {"name": "background", "type": "str", "required": False, "default": "white"},
            {"name": "image_size", "type": "int", "required": False, "default": 900},
            {"name": "max_dim", "type": "int", "required": False, "default": 256, "help": "Downsample volumes larger than this before meshing"},
        ],
    },
    {
        "name": "export_lattice_html",
        "stage": "4. 3D preview",
        "summary": "Export a rotatable, zoomable WebGL scene of a volume as a self-contained HTML file.",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": "Binary volume (.npy or .tif/.tiff)"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Output scene (.html)"},
            {"name": "color", "type": "str", "required": False, "default": "lightgray"},
            {"name": "background", "type": "str", "required": False, "default": "white"},
            {"name": "max_dim", "type": "int", "required": False, "default": 160},
        ],
    },
    {
        "name": "rasterize_lattice",
        "stage": "5. Design",
        "summary": "Rasterize a lattice design graph (JSON) into a binary voxel volume, painting struts as capsules.",
        "fields": [
            {"name": "input_filepath", "type": "path", "required": True, "help": "Lattice design .json (junctions + struts)"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Output volume (.npy or .tif/.tiff)"},
            {"name": "voxel_size_um", "type": "float", "required": False, "default": 58.1},
            {"name": "strut_diameter_um", "type": "float", "required": False, "default": 350.0},
            {"name": "cell_edge_mm", "type": "float", "required": False, "default": 4.56},
            {"name": "shape_z", "type": "int", "required": False, "default": 0, "help": "0 = auto-size to bounding box"},
            {"name": "shape_y", "type": "int", "required": False, "default": 0},
            {"name": "shape_x", "type": "int", "required": False, "default": 0},
        ],
    },
    {
        "name": "compare_slices",
        "stage": "6. Registration QA",
        "summary": "Slice-by-slice design-vs-CT coverage montage; flags the worst candidate-missing-strut slices.",
        "fields": [
            {"name": "design_filepath", "type": "path", "required": True, "help": "Rasterized design volume, same shape as the CT mask"},
            {"name": "ct_mask_filepath", "type": "path", "required": True, "help": "Binary CT segmentation mask"},
            {"name": "output_montage", "type": "path", "required": True, "help": "Output montage (.png)"},
            {"name": "axis", "type": "int", "required": False, "default": 0},
            {"name": "worst_n", "type": "int", "required": False, "default": 9},
            {"name": "tolerance_vox", "type": "int", "required": False, "default": 2},
            {"name": "min_design_voxels", "type": "int", "required": False, "default": 200},
            {"name": "tile_size", "type": "int", "required": False, "default": 256},
            {"name": "slice_indices", "type": "str", "required": False, "default": "", "help": "Optional comma-separated slices instead of worst-by-coverage"},
        ],
    },
    {
        "name": "refit_lattice_registration",
        "stage": "6. Registration QA",
        "summary": "RUN THIS BEFORE ANY MEASUREMENT. Fits the design-to-CT affine correction (the shipped registration carries a ~1% scale error that swamps every defect count).",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True, "help": "Binary segmentation .tif/.npy (z,y,x)"},
            {"name": "design_filepath", "type": "path", "required": True, "help": "Registered lattice .json"},
            {"name": "output_directory", "type": "path", "required": True, "help": "Writes correction.json + design_corrected.json here"},
            {"name": "block", "type": "int", "required": False, "default": 170},
            {"name": "grid", "type": "int", "required": False, "default": 4},
        ],
    },
    {
        "name": "measure_lattice_iou",
        "stage": "7. Measure",
        "summary": "Per-strut / per-node intersection-over-union against the nominal design; writes struts.csv, nodes.csv, distribution plots.",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True},
            {"name": "design_filepath", "type": "path", "required": True},
            {"name": "output_directory", "type": "path", "required": True},
            {"name": "correction_filepath", "type": "path", "required": False, "default": "", "help": "Strongly recommended: correction.json from refit_lattice_registration"},
            {"name": "strut_diameter_um", "type": "float", "required": False, "default": 350.0},
            {"name": "cell_mm", "type": "float", "required": False, "default": 4.56},
            {"name": "trim_fraction", "type": "float", "required": False, "default": 0.20},
            {"name": "envelope_factor", "type": "float", "required": False, "default": 1.6},
            {"name": "stations", "type": "int", "required": False, "default": 12},
            {"name": "embedded_fraction", "type": "float", "required": False, "default": 0.80},
        ],
    },
    {
        "name": "detect_lattice_defects",
        "stage": "8. Classify defects",
        "summary": "End-to-end strut classifier: missing / broken / thin / thick / necked / nominal. Expensive (~10 min cold, seconds cached).",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True},
            {"name": "design_filepath", "type": "path", "required": True},
            {"name": "output_directory", "type": "path", "required": True},
            {"name": "correction_filepath", "type": "path", "required": False, "default": ""},
            {"name": "strut_diameter_um", "type": "float", "required": False, "default": 350.0},
            {"name": "cell_mm", "type": "float", "required": False, "default": 4.56},
            {"name": "sections", "type": "int", "required": False, "default": 25},
            {"name": "tolerance_fraction", "type": "float", "required": False, "default": 0.25},
            {"name": "use_cache", "type": "bool", "required": False, "default": True},
        ],
    },
    {
        "name": "detect_missing_nodes",
        "stage": "8. Classify defects",
        "summary": "Sphere-fill test for un-printed junctions, cross-checked against strut connectivity.",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True},
            {"name": "design_filepath", "type": "path", "required": True},
            {"name": "correction_filepath", "type": "path", "required": False, "default": ""},
            {"name": "results_directory", "type": "path", "required": False, "default": "", "help": "A detect_lattice_defects output dir, to enable the cross-check"},
            {"name": "strut_diameter_um", "type": "float", "required": False, "default": 350.0},
            {"name": "cell_mm", "type": "float", "required": False, "default": 4.56},
            {"name": "radius_factor", "type": "float", "required": False, "default": 1.467},
        ],
    },
    {
        "name": "detect_missing_nodes_2d",
        "stage": "8. Classify defects",
        "summary": "Design-free node check: finds the lattice from the mask's own periodicity, no registration required.",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True},
            {"name": "r_thr_vox", "type": "float", "required": False, "default": 0.0, "help": "0 = read off the EDT histogram valley"},
            {"name": "match_frac", "type": "float", "required": False, "default": 0.25},
            {"name": "plate_frac", "type": "float", "required": False, "default": 0.20},
        ],
    },
    {
        "name": "visualize_strut_classes",
        "stage": "9. Review evidence",
        "summary": "Draws the voxels behind each strut's classification -- lateral view + cross-sections, one row per strut.",
        "fields": [
            {"name": "mask_filepath", "type": "path", "required": True},
            {"name": "design_filepath", "type": "path", "required": True},
            {"name": "results_directory", "type": "path", "required": True, "help": "A detect_lattice_defects output dir"},
            {"name": "output_filepath", "type": "path", "required": True, "help": "Output gallery (.png)"},
            {"name": "correction_filepath", "type": "path", "required": False, "default": ""},
            {"name": "strut_ids", "type": "str", "required": False, "default": "", "help": "Comma-separated strut ids (overrides classes/per_class)"},
            {"name": "classes", "type": "str", "required": False, "default": "", "help": "Comma-separated class names, default all"},
            {"name": "per_class", "type": "int", "required": False, "default": 1},
            {"name": "n_sections", "type": "int", "required": False, "default": 8},
            {"name": "seed", "type": "int", "required": False, "default": 0},
        ],
    },
    {
        "name": "validate_against_stl",
        "stage": "10. Ground truth",
        "summary": "Scores detected missing struts against a defect STL's actually-removed struts -- the one check that isn't just re-reading the same scan.",
        "fields": [
            {"name": "nominal_design_filepath", "type": "path", "required": True, "help": "UNregistered octet_truss_9x9x9.json"},
            {"name": "stl_filepath", "type": "path", "required": True, "help": "A defect STL, e.g. 0.5.stl (not the complete 0.stl)"},
            {"name": "results_directory", "type": "path", "required": True, "help": "A detect_lattice_defects output dir"},
            {"name": "orientation", "type": "str", "required": False, "default": "", "help": 'Optional "perm|signs" e.g. "2,0,1|-1,-1,-1" to skip the search'},
            {"name": "absent_tol_mm", "type": "float", "required": False, "default": 0.7},
        ],
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}
