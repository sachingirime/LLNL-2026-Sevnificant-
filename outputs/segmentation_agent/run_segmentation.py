#!/usr/bin/env python3
"""Create reproducible binary CT masks and QC slices for the project datasets.

Thresholds were selected by inspecting intensity histograms and trial masks.  The
9x9x9 threshold is additionally checked against its supplied slice-380 reference.
The TIFF is processed one axial plane at a time, so the source volume is never
loaded or modified in full.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent

JOBS = (
    {
        "name": "unitcell",
        "input": ROOT / "data/unitcell/unitcell.npy",
        "threshold": 0.005,
        "slice_index": 128,
    },
    {
        "name": "9x9x9_octet_lattice",
        "input": ROOT / "data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif",
        "threshold": 40500,
        "slice_index": 380,
    },
)


def load_source(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.memmap(path)
    raise ValueError(f"Unsupported volume type: {path}")


def save_qc_slice(raw: np.ndarray, mask: np.ndarray, threshold: float, path: Path) -> None:
    """Save a side-by-side raw and binary-mask visual quality-control image."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    lo, hi = np.percentile(raw, [1, 99])
    axes[0].imshow(raw, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Raw CT slice")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Binary segmentation (>= {threshold:g})")
    for axis in axes:
        axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def segment(job: dict[str, object]) -> dict[str, object]:
    name = str(job["name"])
    source = load_source(Path(job["input"]))
    threshold = float(job["threshold"])
    slice_index = int(job["slice_index"])
    destination = OUTPUT_ROOT / name
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / f"{name}_segmentation.npy"

    # NPY input masks can be formed directly.  TIFF inputs are streamed plane by
    # plane into an NPY memmap to keep memory bounded and preserve the source.
    if Path(job["input"]).suffix == ".npy":
        mask = (source >= threshold).astype(np.uint8)
        np.save(mask_path, mask)
        foreground = int(mask.sum())
        qc_mask = mask[slice_index]
    else:
        mask = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=np.uint8, shape=source.shape
        )
        foreground = 0
        qc_mask = None
        for index in range(source.shape[0]):
            plane = np.asarray(source[index]) >= threshold
            mask[index] = plane
            foreground += int(plane.sum())
            if index == slice_index:
                qc_mask = plane
        mask.flush()
        del mask
        assert qc_mask is not None

    qc_path = destination / f"{name}_slice_{slice_index}_axis0_qc.png"
    save_qc_slice(np.asarray(source[slice_index]), qc_mask, threshold, qc_path)
    voxels = int(np.prod(source.shape))
    return {
        "name": name,
        "input": str(job["input"]),
        "mask": str(mask_path),
        "qc": str(qc_path),
        "shape": tuple(int(value) for value in source.shape),
        "dtype": str(source.dtype),
        "threshold": threshold,
        "foreground": foreground,
        "background": voxels - foreground,
        "foreground_fraction": foreground / voxels,
    }


if __name__ == "__main__":
    for job in JOBS:
        result = segment(job)
        print(result)
