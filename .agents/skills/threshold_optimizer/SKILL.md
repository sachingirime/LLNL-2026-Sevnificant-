---
name: threshold-optimizer
description: Sweeps a range of segmentation thresholds over a CT volume, renders a slice from each result, and recommends the best threshold by comparing foreground fraction and visual quality.
---

# Threshold Sweep Protocol

You are the **Segmentation Threshold Optimizer**. Picking a density threshold by hand is guesswork; this skill runs a controlled sweep and reports which value best separates lattice material from background.

### Step 1: Profile the Input

Before choosing any thresholds, load the input `.npy` or `.tif` and report:
- `shape`, `dtype`, `min`, `max`, `mean`
- the 1st, 25th, 50th, 75th, and 99th percentiles of the intensity values

**Do not assume the data is normalized to 0-1.** These volumes are not:
- `data/unitcell/unitcell.npy` is float32 spanning roughly -0.003 to 0.015
- `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif` is uint16 spanning 0 to 65535

A fixed set of thresholds like 0.3/0.5/0.7 is meaningless against a uint16 volume. Derive the sweep from the percentiles you just measured.

### Step 2: Choose the Sweep

Select **five** thresholds spanning the informative part of the histogram — the region between the background peak and the material peak. The 50th through 99th percentiles are a reasonable spread. Percentiles adapt to the shape of the histogram, so the same sweep is meaningful on any volume regardless of its dtype or scale.

State the five values, each with its percentile label and its resolved absolute value, before running anything.

If the user asks for specific absolute thresholds instead, honour that, but first check each value falls between the volume's `min` and `max` and warn about any that do not — a threshold outside the range yields an all-zero or all-one mask. On `data/unitcell/unitcell.npy`, whose maximum is about 0.0153, a value like 0.1 segments nothing at all.

### Step 3: Run the Sweep

For each threshold, call the MCP tool `segment_ct_dataset()`, writing to:

```
outputs/threshold_sweep/<dataset>/<dataset>_<label>.npy
```

where `<label>` is the percentile that produced it — `p50`, `p62`, `p75`, `p87`, `p99`. Labelling by percentile keeps the files sorted in sweep order, which a bare threshold value would not.

Record the `foreground_voxels` count the tool returns for each run. Never write into `data/` — it is read-only input.

### Step 4: Visualize

For each threshold, call the MCP tool `visualize_slice()` on the resulting mask at the **mid-slice** (`shape[axis] // 2`) along axis 0, writing to:

```
outputs/threshold_sweep/<dataset>/<dataset>_<label>_slice.png
```

using the same `<label>` as the mask it was rendered from, so each image pairs unambiguously with its `.npy`.

Use the same slice index for every threshold so the images are directly comparable.

### Step 5: Report

Write `outputs/threshold_sweep/<dataset>/report.md` containing:

1. **Profile table:** shape, dtype, and the percentiles from Step 1.
2. **Sweep table:** one row per threshold, in ascending order, with its percentile label, resolved absolute threshold, `foreground_voxels`, and foreground fraction (`foreground_voxels / total_voxels`).
3. **Visual gallery:** embed every slice image from Step 4 inline, in ascending threshold order, each captioned with its threshold and foreground fraction:

   ```markdown
   ### p75 — threshold 0.001204, foreground 24.30%
   ![Segmentation at the 75th percentile, threshold 0.001204](unitcell_p75_slice.png)
   ```

   Write the image paths **relative to the report file**. The report and the images both live in `outputs/threshold_sweep/<dataset>/`, so a bare filename is correct — an absolute path or one starting `outputs/` will render as a broken image.
4. **Recommendation:** the single best threshold, with reasoning, and embed its slice image again directly beneath the recommendation so the conclusion is readable on its own. Foreground fraction alone is not sufficient — a threshold that is too low floods the volume (fraction approaching 1.0) and one that is too high fragments the struts. Look for the value where the fraction stabilizes between neighbouring steps, and confirm it against the slice images.
5. **Ground truth check:** if a `ground_truth_segmentation_*.png` exists alongside the input in `data/`, note the slice index it refers to, render that same index for the recommended threshold, and embed the two images one after the other under headings "Ground truth" and "Recommended threshold" so they can be compared directly.

# Technical Constraints

- Segmentation masks are full-size. A single 9x9x9 mask is ~496 MB, so a five-point sweep writes ~2.5 GB. Confirm with the user before sweeping any volume larger than 256^3, and default to `data/unitcell/unitcell.npy` when no dataset is named. If the large volume is swept, offer to delete each mask after recording its voxel count and rendering its slice.
- Report the tool's returned status string verbatim for each run. If a call returns a string starting with `Error:`, stop and surface it rather than continuing the sweep.
- Use the MCP tools `segment_ct_dataset()` and `visualize_slice()` directly. Do not reimplement thresholding in a local script.
- If you created any temporary Python scripts, remove them once you are finished.
