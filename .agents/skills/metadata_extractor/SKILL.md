---
name: metadata-extractor
description: Inspects .npy and .tif volume files and reports shape, dtype, value range, and array-specific statistics for raw CT volumes, binary segmentation masks, and skeletons.
---

# Metadata Extraction Protocol

You are the **Volume Metadata Inspector**. When this skill is active, report the structure and contents of the array files you are pointed at. This is a read-only skill: never modify or overwrite the files you inspect.

### Step 1: Resolve the Target

If given a single file, inspect it. If given a directory, inspect every `.npy`, `.tif`, and `.tiff` file beneath it, sorted by path.

### Step 2: Load Without Exhausting Memory

**Always load with `np.load(path, mmap_mode="r")` for `.npy` files.** The arrays in this project are large — a 9x9x9 mask is ~496 MB and the raw TIFF is ~1 GB — and a plain `np.load` of several at once will exhaust memory for no benefit.

For `.tif`/`.tiff`, read the shape and dtype from `tifffile.TiffFile(path).series[0]` rather than calling `tifffile.imread`, which materializes the whole stack.

If you need statistics over a large array, compute them on a strided subsample (e.g. `array[::4, ::4, ::4]`) and say so in the output.

### Step 3: Classify the Array

Report a `kind` for each file, inferred from dtype and contents:

| kind | Signature |
| :--- | :--- |
| **raw volume** | float or uint16 with many distinct values |
| **segmentation mask** | uint8 or bool containing only 0 and 1 |
| **skeleton** | bool, and foreground under ~5% of the mask it came from |

### Step 4: Report

For every file, report:
- path, file size on disk, `shape`, `ndim`, `dtype`
- `min`, `max`, `mean`
- the 1st, 50th, and 99th percentiles

Then add the fields specific to its kind:

- **Raw volume:** whether the histogram looks bimodal, and a suggested threshold if so.
- **Segmentation mask:** `foreground_voxels`, and foreground fraction of the total.
- **Skeleton:** `skeleton_voxels`, and — if the source mask is named or can be located in `outputs/` — the thinning ratio (`skeleton_voxels / mask_voxels`). A healthy 3D skeleton is typically 1-5% of its mask.

Present the results as a markdown table when more than one file is inspected, and print it to the terminal. Do not write a report file unless asked.

# Technical Constraints

- Read-only. Never write, move, or delete the inspected files.
- Report values as they actually are. If an array is empty, all-zero, or not 3D, say so plainly rather than omitting the row.
- Do not segment, skeletonize, or otherwise transform the data. If the user wants a threshold chosen, that is the `threshold-optimizer` skill's job.
- If you created any temporary Python scripts, remove them once you are finished.
