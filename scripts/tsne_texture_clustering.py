"""
Nonlinear (t-SNE) clustering of CT voxels using local texture features, as a
complement to the 1-D intensity k-means in kmeans_segmentation.py.

Plain intensity k-means only sees a single number per voxel, so it can only
ever produce clusters that are intervals of intensity. t-SNE instead embeds
a multi-dimensional per-voxel feature vector (intensity, local mean/std,
gradient magnitude, local material-density context) into 2D based on
nonlinear neighborhood structure, then a clustering algorithm is run on that
embedding. This can separate voxel types that overlap in raw intensity but
differ in local texture (e.g. a strut-surface/partial-volume voxel vs. a
noisy background voxel at the same intensity).

t-SNE does not scale to hundreds of millions of voxels, so this samples a
stratified subset (equal counts per existing k-means intensity cluster) and
extracts a small local patch around each sample for feature computation.

data/ and prior outputs/ are read-only inputs.

Usage:
    python scripts/tsne_texture_clustering.py <tif_path> <kmeans_labels_npy_path>
        [--samples-per-cluster 3000] [--patch-radius 3] [--n-clusters 5]
        [--perplexity 30] [--z-range Z0 Z1]

Example:
    python scripts/tsne_texture_clustering.py \\
        data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif \\
        outputs/kmeans_segmentation/9x9x9_octet_lattice/labels_k5.npy
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "tsne_texture_clustering")

FEATURE_NAMES = ["intensity", "patch_mean", "patch_std", "gradient_mag",
                  "local_density_label", "patch_range"]


def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path)
    raise ValueError(f"Unsupported file type '{ext}'.")


def stratified_sample_coords(cluster_labels, margin, samples_per_cluster, rng, z_range=None,
                              batch_size=200_000, max_batches=200):
    """Sample up to `samples_per_cluster` voxel coordinates per cluster label
    (assumed to be dense integers 0..max), via rejection sampling.

    With ~500M voxels, `np.argwhere(cluster_labels == c)` would materialize
    every matching coordinate just to immediately subsample it -- for a
    cluster covering 70% of the volume that's a multi-GB throwaway array.
    Instead, draw random coordinate batches and keep the ones that land on
    the target cluster; each batch touches only `batch_size` voxels.
    """
    shape = cluster_labels.shape
    z_lo, z_hi = margin, shape[0] - margin
    if z_range is not None:
        z_lo, z_hi = max(z_lo, z_range[0]), min(z_hi, z_range[1])
    y_lo, y_hi = margin, shape[1] - margin
    x_lo, x_hi = margin, shape[2] - margin

    n_clusters = int(cluster_labels.max()) + 1
    coords_by_cluster = {}
    for c in range(n_clusters):
        found = []
        n_found = 0
        batches_used = 0
        while n_found < samples_per_cluster and batches_used < max_batches:
            zs = rng.integers(z_lo, z_hi, size=batch_size)
            ys = rng.integers(y_lo, y_hi, size=batch_size)
            xs = rng.integers(x_lo, x_hi, size=batch_size)
            vals = cluster_labels[zs, ys, xs]
            hit = vals == c
            if hit.any():
                found.append(np.stack([zs[hit], ys[hit], xs[hit]], axis=1))
                n_found += int(hit.sum())
            batches_used += 1
        chosen = np.concatenate(found, axis=0)[:samples_per_cluster] if found else np.empty((0, 3), dtype=int)
        coords_by_cluster[c] = chosen
        print(f"  cluster {c}: sampled {len(chosen)} (target {samples_per_cluster}) in {batches_used} batches")
    return coords_by_cluster


def extract_features(volume, cluster_labels, coords, radius):
    """Per-sample feature vector from a local (2r+1)^3 patch around each coord.

    Only the small per-sample patches are cast to float, never the full
    volume -- with ~500M voxels, an eager `volume.astype(float64)` would
    burn several GB for no benefit since we only ever touch a sparse set of
    small windows.
    """
    n = len(coords)
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i, (z, y, x) in enumerate(coords):
        patch = volume[z - radius:z + radius + 1, y - radius:y + radius + 1,
                        x - radius:x + radius + 1].astype(np.float64)
        label_patch = cluster_labels[z - radius:z + radius + 1, y - radius:y + radius + 1,
                                      x - radius:x + radius + 1]
        gx = float(volume[z, y, x + 1]) - float(volume[z, y, x - 1])
        gy = float(volume[z, y + 1, x]) - float(volume[z, y - 1, x])
        gz = float(volume[z + 1, y, x]) - float(volume[z - 1, y, x])
        grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) / 2.0
        features[i] = [
            float(volume[z, y, x]),
            patch.mean(),
            patch.std(),
            grad_mag,
            label_patch.mean(),
            patch.max() - patch.min(),
        ]
    return features


def run(tif_path, labels_path, samples_per_cluster, patch_radius, n_clusters,
        perplexity, z_range, seed=0):
    name = os.path.splitext(os.path.basename(tif_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Loading volume from {tif_path} ...")
    volume = load_volume(tif_path)
    print(f"Loading k-means cluster labels from {labels_path} ...")
    cluster_labels = load_volume(labels_path)
    assert volume.shape == cluster_labels.shape, "volume/labels shape mismatch"

    print("Stratified sampling by existing k-means cluster ...")
    coords_by_cluster = stratified_sample_coords(
        cluster_labels, margin=patch_radius + 1, samples_per_cluster=samples_per_cluster,
        rng=rng, z_range=z_range,
    )
    old_labels = np.concatenate([np.full(len(v), c) for c, v in coords_by_cluster.items()])
    coords = np.concatenate(list(coords_by_cluster.values()), axis=0)
    print(f"Total samples: {len(coords)}")

    print(f"Extracting features (patch radius={patch_radius}) ...")
    features = extract_features(volume, cluster_labels, coords, patch_radius)
    del volume

    print("Standardizing features ...")
    X = StandardScaler().fit_transform(features)

    print(f"Running t-SNE (perplexity={perplexity}) on {X.shape[0]} points, {X.shape[1]} features ...")
    embedding = TSNE(n_components=2, perplexity=perplexity, init="pca",
                      learning_rate="auto", random_state=seed).fit_transform(X)

    print(f"Clustering the t-SNE embedding with k-means (k={n_clusters}) ...")
    new_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embedding)

    # Cross-tabulation: how do the new nonlinear clusters relate to the old
    # intensity clusters?
    table = np.zeros((len(coords_by_cluster), n_clusters), dtype=int)
    for old_c, new_c in zip(old_labels, new_labels):
        table[old_c, new_c] += 1
    summary_lines = ["Cross-tabulation: rows=old intensity k-means cluster, cols=new t-SNE+kmeans cluster", ""]
    header = "old\\new".rjust(8) + "".join(f"{c:>8}" for c in range(n_clusters))
    summary_lines.append(header)
    for old_c in range(len(coords_by_cluster)):
        row = f"{old_c:>8}" + "".join(f"{table[old_c, c]:>8}" for c in range(n_clusters))
        summary_lines.append(row)
    summary_lines.append("")
    summary_lines.append("Feature order: " + ", ".join(FEATURE_NAMES))
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(summary_text + "\n")

    np.savez(
        os.path.join(out_dir, "tsne_texture_data.npz"),
        coords=coords, features=features, old_labels=old_labels,
        new_labels=new_labels, embedding=embedding, feature_names=FEATURE_NAMES,
    )
    print(f"Saved raw data to {out_dir}/tsne_texture_data.npz")

    # Embedding scatter: old vs. new cluster coloring, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=old_labels, cmap="viridis", s=4)
    axes[0].set_title("t-SNE embedding, colored by\noriginal 1D intensity k-means cluster")
    fig.colorbar(sc0, ax=axes[0], label="old cluster")
    sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=new_labels, cmap="tab10", s=4)
    axes[1].set_title("t-SNE embedding, colored by\nnew texture-based k-means cluster")
    fig.colorbar(sc1, ax=axes[1], label="new cluster")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    embedding_path = os.path.join(out_dir, "tsne_embedding_comparison.png")
    plt.savefig(embedding_path, dpi=140)
    plt.close()
    print(f"Saved embedding comparison to {embedding_path}")

    # 3D spatial scatter: old vs. new cluster coloring, side by side.
    fig = plt.figure(figsize=(13, 6.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.scatter(coords[:, 2], coords[:, 1], coords[:, 0], c=old_labels, cmap="viridis", s=3)
    ax0.set_title("Samples in 3D space,\ncolored by original intensity cluster")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(coords[:, 2], coords[:, 1], coords[:, 0], c=new_labels, cmap="tab10", s=3)
    ax1.set_title("Samples in 3D space,\ncolored by new texture cluster")
    for ax in (ax0, ax1):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    plt.tight_layout()
    spatial_path = os.path.join(out_dir, "spatial_comparison.png")
    plt.savefig(spatial_path, dpi=140)
    plt.close()
    print(f"Saved spatial comparison to {spatial_path}")

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE + clustering of CT voxel texture features.")
    parser.add_argument("tif_path", help="Path to the raw .tif/.tiff/.npy CT volume.")
    parser.add_argument("labels_path", help="Path to the k-means intensity cluster label .npy volume.")
    parser.add_argument("--samples-per-cluster", type=int, default=3000,
                         help="Voxels sampled per existing k-means cluster (default 3000).")
    parser.add_argument("--patch-radius", type=int, default=3,
                         help="Local patch half-size for texture features (default 3, i.e. 7^3 patch).")
    parser.add_argument("--n-clusters", type=int, default=5,
                         help="Number of clusters for k-means on the t-SNE embedding (default 5).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (default 30).")
    parser.add_argument("--z-range", type=int, nargs=2, default=None, metavar=("Z0", "Z1"),
                         help="Restrict sampling to axis-0 range [Z0:Z1) (e.g. to exclude end caps).")
    args = parser.parse_args()

    z_range = tuple(args.z_range) if args.z_range is not None else None
    run(args.tif_path, args.labels_path, args.samples_per_cluster, args.patch_radius,
        args.n_clusters, args.perplexity, z_range)
