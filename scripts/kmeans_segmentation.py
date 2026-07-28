"""
Segment a 3D CT volume (.npy or .tif/.tiff) into k clusters using k-means on
voxel intensity.

Because the input is a single-channel grayscale volume, clustering is run on
the (value, count) intensity histogram rather than on all voxels directly.
This is mathematically identical to running k-means on every voxel (each
histogram bin is just many identical 1-D points, weighted by how many times
that value occurs) but runs in milliseconds and avoids materializing a huge
feature matrix in memory.

data/ is treated as read-only: this script only reads from it.
All outputs are written under outputs/kmeans_segmentation/.

Usage:
    python scripts/kmeans_segmentation.py <input_path> [--k 5] [--slice-axis 0] [--slice-index N]

Example:
    python scripts/kmeans_segmentation.py \\
        data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif --k 5
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "kmeans_segmentation")


def load_volume(input_path: str) -> np.ndarray:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".npy":
        return np.load(input_path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(input_path)
    raise ValueError(f"Unsupported file type '{ext}'. Expected .npy, .tif, or .tiff.")


def weighted_kmeans_1d(values: np.ndarray, weights: np.ndarray, k: int,
                        n_init: int = 10, max_iter: int = 100, seed: int = 0):
    """Weighted 1-D Lloyd's algorithm with kmeans++-style seeding.

    values/weights describe a histogram: `weights[i]` points sit at `values[i]`.
    Returns (centers, labels_per_value) with centers sorted ascending and
    labels_per_value re-mapped to match that ascending order.
    """
    rng = np.random.default_rng(seed)
    total_weight = weights.sum()
    best_centers, best_labels, best_inertia = None, None, np.inf

    for _ in range(n_init):
        # kmeans++ seeding, weighted by histogram count.
        centers = np.empty(k, dtype=np.float64)
        first_idx = rng.choice(len(values), p=weights / total_weight)
        centers[0] = values[first_idx]
        for c in range(1, k):
            d2 = np.min((values[:, None] - centers[None, :c]) ** 2, axis=1)
            probs = d2 * weights
            probs_sum = probs.sum()
            if probs_sum <= 0:
                centers[c] = rng.choice(values)
            else:
                centers[c] = values[rng.choice(len(values), p=probs / probs_sum)]

        for _ in range(max_iter):
            dists = (values[:, None] - centers[None, :]) ** 2
            labels = dists.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers[j] = np.average(values[mask], weights=weights[mask])
            if np.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        dists = (values[:, None] - centers[None, :]) ** 2
        labels = dists.argmin(axis=1)
        inertia = float(np.sum(weights * dists[np.arange(len(values)), labels]))
        if inertia < best_inertia:
            best_inertia, best_centers, best_labels = inertia, centers, labels

    order = np.argsort(best_centers)
    rank_of = np.empty(k, dtype=np.int64)
    rank_of[order] = np.arange(k)
    sorted_centers = best_centers[order]
    remapped_labels = rank_of[best_labels]
    return sorted_centers, remapped_labels, best_inertia


def run_kmeans_segmentation(input_path: str, k: int = 5, slice_axis: int = 0,
                             slice_index: int | None = None) -> None:
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading volume from {input_path} ...")
    volume = load_volume(input_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")
    if volume.dtype.kind != "u" or volume.dtype.itemsize > 2:
        raise ValueError(f"Expected an unsigned <=16-bit integer volume, got dtype {volume.dtype}")

    print(f"Volume shape={volume.shape}, dtype={volume.dtype}")

    n_bins = int(np.iinfo(volume.dtype).max) + 1
    print(f"Building intensity histogram ({n_bins} bins) ...")
    counts = np.bincount(volume.ravel(), minlength=n_bins).astype(np.float64)
    values = np.arange(n_bins, dtype=np.float64)

    nonzero = counts > 0
    print(f"Running weighted k-means (k={k}) on {int(nonzero.sum())} distinct intensity values ...")
    centers, labels_per_value, inertia = weighted_kmeans_1d(
        values[nonzero], counts[nonzero], k=k
    )
    print(f"Converged. Inertia={inertia:.4g}")
    print("Cluster centers (sorted ascending, in original intensity units):")
    for i, c in enumerate(centers):
        print(f"  cluster {i}: center={c:.1f}")

    # Full lookup table: intensity value (0..n_bins-1) -> cluster label.
    lut = np.zeros(n_bins, dtype=np.uint8)
    lut[nonzero] = labels_per_value.astype(np.uint8)

    print("Applying lookup table to full volume ...")
    label_volume = lut[volume]

    labels_path = os.path.join(out_dir, f"labels_k{k}.npy")
    np.save(labels_path, label_volume)
    print(f"Saved cluster label volume to {labels_path}")

    # Per-cluster voxel stats.
    cluster_counts = np.array([int(np.sum(label_volume == j)) for j in range(k)])
    total_voxels = label_volume.size
    summary_lines = [f"K-means segmentation summary: {os.path.basename(input_path)}", ""]
    summary_lines.append(f"k={k}, volume shape={volume.shape}, dtype={volume.dtype}")
    summary_lines.append(f"inertia={inertia:.6g}")
    summary_lines.append("")
    summary_lines.append(f"{'cluster':>7} | {'center':>10} | {'voxels':>12} | {'fraction':>9}")
    for j in range(k):
        frac = cluster_counts[j] / total_voxels
        summary_lines.append(f"{j:>7} | {centers[j]:>10.1f} | {cluster_counts[j]:>12} | {frac:>9.4%}")
    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(out_dir, f"summary_k{k}.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary_text + "\n")
    print(summary_text)
    print(f"Saved summary to {summary_path}")

    # Histogram with cluster boundaries.
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    plt.figure(figsize=(7, 4))
    plt.hist(volume.ravel(), bins=200, color="steelblue")
    plt.yscale("log")
    for b in boundaries:
        plt.axvline(b, color="red", linestyle="--", linewidth=1)
    for c in centers:
        plt.axvline(c, color="black", linestyle=":", linewidth=1)
    plt.title(f"Intensity histogram with k={k} cluster boundaries\n{os.path.basename(input_path)}")
    plt.xlabel(f"{volume.dtype} value")
    plt.ylabel("voxel count (log scale)")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"histogram_k{k}.png")
    plt.savefig(hist_path, dpi=120)
    plt.close()
    print(f"Saved histogram to {hist_path}")

    # Slice comparison: original grayscale vs cluster labels.
    if slice_index is None:
        slice_index = volume.shape[slice_axis] // 2
    orig_slice = np.take(volume, slice_index, axis=slice_axis)
    label_slice = np.take(label_volume, slice_index, axis=slice_axis)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(orig_slice, cmap="gray")
    axes[0].set_title(f"Original (slice {slice_index}, axis={slice_axis})")
    axes[0].axis("off")

    cmap = plt.get_cmap("viridis", k)
    im = axes[1].imshow(label_slice, cmap=cmap, vmin=-0.5, vmax=k - 0.5)
    axes[1].set_title(f"k-means clusters (k={k})")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(k), fraction=0.046, pad=0.04)
    cbar.set_label("cluster")

    fig.suptitle(os.path.basename(input_path))
    plt.tight_layout()
    slice_path = os.path.join(out_dir, f"slice_{slice_index}_k{k}_comparison.png")
    plt.savefig(slice_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved slice comparison to {slice_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means intensity segmentation of a 3D CT volume.")
    parser.add_argument("input_path", help="Path to the .npy or .tif/.tiff volume.")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (default 5).")
    parser.add_argument("--slice-axis", type=int, default=0, choices=[0, 1, 2],
                         help="Axis to slice along for the comparison image (default 0).")
    parser.add_argument("--slice-index", type=int, default=None,
                         help="Slice index for the comparison image (default: middle slice).")
    args = parser.parse_args()

    run_kmeans_segmentation(args.input_path, k=args.k, slice_axis=args.slice_axis,
                             slice_index=args.slice_index)
