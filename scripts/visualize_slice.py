"""
Visualize a 2D slice from a 3D CT volume (.npy or .tif/.tiff).

Usage:
    python scripts/visualize_slice.py <input_path> <output_path> <slice_index> [--axis 0]

Note: input_path should point into data/ (read-only, never written to).
Save output_path outside data/, e.g. under outputs/.

Example:
    python scripts/visualize_slice.py \\
        data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif \\
        outputs/9x9x9_octet_lattice/slice_380.png \\
        380 --axis 0
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile


def load_volume(input_path: str) -> np.ndarray:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".npy":
        return np.load(input_path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(input_path)
    raise ValueError(f"Unsupported file type '{ext}'. Expected .npy, .tif, or .tiff.")


def visualize_slice(input_path: str, output_path: str, slice_index: int, axis: int = 0) -> None:
    volume = load_volume(input_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2 (got {axis})")
    if not (0 <= slice_index < volume.shape[axis]):
        raise ValueError(
            f"slice_index {slice_index} out of range for axis {axis} "
            f"with size {volume.shape[axis]}"
        )

    slice_2d = np.take(volume, slice_index, axis=axis)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap="gray")
    plt.title(f"Slice {slice_index} (axis={axis}) of {os.path.basename(input_path)}")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved slice {slice_index} (axis={axis}) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 2D slice from a 3D CT volume.")
    parser.add_argument("input_path", help="Path to the .npy or .tif/.tiff volume.")
    parser.add_argument("output_path", help="Path to save the output image (e.g. .png).")
    parser.add_argument("slice_index", type=int, help="Index of the slice to visualize.")
    parser.add_argument("--axis", type=int, default=0, choices=[0, 1, 2], help="Axis to slice along (default 0).")
    args = parser.parse_args()

    visualize_slice(args.input_path, args.output_path, args.slice_index, args.axis)
