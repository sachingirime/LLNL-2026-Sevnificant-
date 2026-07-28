"""
Extract a node/edge (junction/strut) graph from a full 3D CT volume, with real
3D spatial coordinates for every node and edge.

Unlike a slice-by-slice approach, this works on the whole 3D array in one
pass:
  1. Foreground mask from k-means cluster labels (see kmeans_segmentation.py).
  2. True 3D skeletonization (skimage.morphology.skeletonize on the full 3D
     array performs genuine 3D thinning -- it is not applied slice-by-slice).
  3. The skeleton is a graph of voxels; skan.Skeleton turns it into a
     path graph (branches between junction/endpoint pixels).
  4. Spurious short spurs (skeletonization noise) are pruned in batched
     rounds.
  5. Clusters of adjacent junction pixels that represent a single physical
     node (a strut intersection is rarely exactly 1 voxel) are merged via
     union-find into one node, positioned at the centroid of its members.

Output: outputs/lattice_graph/<name>/graph_k<k>.json with the same
junctions/struts shape as the design graphs under data/ (id/position for
nodes, id/junction0/junction1/... for edges), plus PNG visualizations.

data/ and prior outputs/ are read-only inputs; nothing here is written back
into data/.

Usage:
    python scripts/extract_lattice_graph.py <labels_npy_path> [--min-cluster 1]
        [--spur-prune-voxels 8] [--junction-merge-voxels 8] [--max-prune-rounds 4]

Example:
    python scripts/extract_lattice_graph.py \\
        outputs/kmeans_segmentation/9x9x9_octet_lattice/labels_k5.npy
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from skimage.morphology import skeletonize, ball
from skan import Skeleton, summarize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "lattice_graph")


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def clean_mask(mask, closing_radius=2):
    """Keep the largest connected component and smooth its surface.

    The raw k-means mask has a rough, voxel-noisy surface (struts are only
    ~2-4 voxels in radius here), which makes the raw skeleton extremely
    jagged: it fragments each real strut into dozens of spurious short
    branches instead of one clean edge. A morphological closing+opening
    smooths that surface noise before skeletonization. Isolated noise specks
    disconnected from the main lattice are dropped.
    """
    labeled, n_components = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n_components > 1:
        sizes = ndi.sum(mask, labeled, range(1, n_components + 1))
        largest = np.argmax(sizes) + 1
        dropped = int(mask.sum() - sizes[largest - 1])
        print(f"  {n_components} connected components; keeping largest, "
              f"dropping {dropped} voxels ({dropped / mask.sum():.4%}) as noise specks")
        mask = labeled == largest
    struct = ball(closing_radius)
    mask = ndi.binary_closing(mask, structure=struct)
    mask = ndi.binary_opening(mask, structure=struct)
    return mask


def prune_spurs(skel_obj, spur_voxels, max_rounds):
    """Batched removal of short endpoint-to-junction spurs (skeletonization noise)."""
    for round_idx in range(max_rounds):
        df = summarize(skel_obj, separator="_")
        spur_mask = (df["branch_type"] == 1) & (df["branch_distance"] < spur_voxels)
        spur_idx = df.index[spur_mask].to_numpy()
        if len(spur_idx) == 0:
            print(f"  prune round {round_idx}: no more spurs under {spur_voxels} voxels")
            break
        print(f"  prune round {round_idx}: removing {len(spur_idx)} spurs "
              f"(of {len(df)} branches)")
        skel_obj = skel_obj.prune_paths(spur_idx)
    return skel_obj


def merge_junction_clusters(df, merge_voxels):
    """Union-find merge of adjacent junction pixels into single physical nodes.

    Returns (node_positions: {merged_id: (z,y,x)}, edges: list of dict).
    """
    node_ids = set(df["node_id_src"]).union(df["node_id_dst"])
    uf = UnionFind(node_ids)

    raw_coords = {}
    for _, row in df.iterrows():
        raw_coords[row["node_id_src"]] = (
            row["coord_src_0"], row["coord_src_1"], row["coord_src_2"]
        )
        raw_coords[row["node_id_dst"]] = (
            row["coord_dst_0"], row["coord_dst_1"], row["coord_dst_2"]
        )

    short_junction_junction = (df["branch_type"] == 2) & (df["branch_distance"] < merge_voxels)
    for _, row in df[short_junction_junction].iterrows():
        uf.union(row["node_id_src"], row["node_id_dst"])

    groups = {}
    for nid in node_ids:
        groups.setdefault(uf.find(nid), []).append(nid)

    root_to_merged_id = {root: i for i, root in enumerate(sorted(groups))}
    node_positions = {}
    node_members = {}
    for root, members in groups.items():
        merged_id = root_to_merged_id[root]
        coords = np.array([raw_coords[m] for m in members], dtype=np.float64)
        node_positions[merged_id] = tuple(coords.mean(axis=0))
        node_members[merged_id] = members

    edges = []
    n_self_loops = 0
    for _, row in df[~short_junction_junction].iterrows():
        n0 = root_to_merged_id[uf.find(row["node_id_src"])]
        n1 = root_to_merged_id[uf.find(row["node_id_dst"])]
        if n0 == n1:
            n_self_loops += 1
            continue
        edges.append({
            "node0": n0,
            "node1": n1,
            "length_voxels": float(row["branch_distance"]),
            "euclidean_length_voxels": float(row["euclidean_distance"]),
            "mean_density_label": float(row["mean_pixel_value"]),
            "stdev_density_label": float(row["stdev_pixel_value"]),
            "branch_type": int(row["branch_type"]),
        })
    if n_self_loops:
        print(f"  dropped {n_self_loops} self-loop edges created by junction merging")

    node_degree = {nid: 0 for nid in node_positions}
    for e in edges:
        node_degree[e["node0"]] += 1
        node_degree[e["node1"]] += 1

    return node_positions, node_members, node_degree, edges


def save_graph_json(node_positions, node_degree, edges, out_path, meta):
    junctions = [
        {
            "id": nid,
            "position": [round(p, 3) for p in pos],
            "degree": node_degree[nid],
            "type": "endpoint" if node_degree[nid] <= 1 else "junction",
        }
        for nid, pos in sorted(node_positions.items())
    ]
    struts = [{"id": i, **e} for i, e in enumerate(edges)]
    payload = {"meta": meta, "junctions": junctions, "struts": struts}
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    return len(junctions), len(struts)


def visualize_overview(node_positions, edges, volume_shape, out_path):
    """Top-down (axis0 vs axis1) projection of the whole extracted graph,
    edges colored by axis2 (depth) of their midpoint."""
    xs = np.array([node_positions[nid][1] for nid in node_positions])
    ys = np.array([node_positions[nid][0] for nid in node_positions])
    zs = np.array([node_positions[nid][2] for nid in node_positions])

    segs = []
    seg_depth = []
    for e in edges:
        p0 = node_positions[e["node0"]]
        p1 = node_positions[e["node1"]]
        segs.append([(p0[1], p0[0]), (p1[1], p1[0])])
        seg_depth.append((p0[2] + p1[2]) / 2.0)

    fig, ax = plt.subplots(figsize=(9, 9))
    lc = LineCollection(segs, array=np.array(seg_depth), cmap="viridis", linewidths=0.4)
    ax.add_collection(lc)
    ax.scatter(xs, ys, s=1, c="black", alpha=0.4)
    ax.set_xlim(0, volume_shape[1])
    ax.set_ylim(volume_shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(f"Extracted lattice graph: {len(node_positions)} nodes, {len(edges)} struts\n"
                 f"(top-down projection, color = depth axis 2)")
    cbar = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("axis-2 depth (voxels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_crop_3d(node_positions, node_members, edges, crop_bounds, out_path):
    """3D wireframe of nodes/edges whose position falls within crop_bounds
    ((z0,z1),(y0,y1),(x0,x1)), for a close-up sanity check."""
    (z0, z1), (y0, y1), (x0, x1) = crop_bounds

    def in_crop(pos):
        z, y, x = pos
        return z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1

    keep_nodes = {nid: pos for nid, pos in node_positions.items() if in_crop(pos)}
    segs = []
    for e in edges:
        if e["node0"] in keep_nodes and e["node1"] in keep_nodes:
            p0, p1 = keep_nodes[e["node0"]], keep_nodes[e["node1"]]
            segs.append([p0, p1])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    if keep_nodes:
        pts = np.array(list(keep_nodes.values()))
        ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], s=8, c="crimson")
    if segs:
        segs_xyz = [[(p[2], p[1], p[0]) for p in seg] for seg in segs]
        ax.add_collection3d(Line3DCollection(segs_xyz, colors="steelblue", linewidths=1.2))
    ax.set_xlabel("axis 2 (x)")
    ax.set_ylabel("axis 1 (y)")
    ax.set_zlabel("axis 0 (z)")
    ax.set_title(f"Close-up: {len(keep_nodes)} nodes, {len(segs)} struts\n"
                 f"crop z={z0}-{z1}, y={y0}-{y1}, x={x0}-{x1}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(labels_path, min_cluster, spur_prune_voxels, junction_merge_voxels, max_prune_rounds,
        z_range=None):
    name = os.path.basename(os.path.dirname(labels_path)) or "volume"
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"Loading cluster labels from {labels_path} ...")
    labels = np.load(labels_path)
    print(f"Labels shape={labels.shape}, dtype={labels.dtype}")

    z_offset = 0
    if z_range is not None:
        z0, z1 = z_range
        print(f"Cropping to axis-0 range [{z0}:{z1}] (excludes non-lattice regions, e.g. end caps)")
        labels = labels[z0:z1]
        z_offset = z0
        print(f"  cropped shape={labels.shape}")

    mask = labels >= min_cluster
    foreground_frac = mask.mean()
    print(f"Foreground mask (cluster >= {min_cluster}): {foreground_frac:.4%} of voxels")

    print("Cleaning mask (largest component + morphological smoothing) ...")
    t1 = time.time()
    mask = clean_mask(mask)
    print(f"  done in {time.time()-t1:.1f}s")

    print("Running full 3D skeletonization (whole volume, true 3D thinning) ...")
    t1 = time.time()
    skeleton = skeletonize(mask)
    print(f"  done in {time.time()-t1:.1f}s. Skeleton voxels: {int(skeleton.sum())}")
    del mask

    print("Building skan skeleton graph (source_image=cluster label as density proxy) ...")
    t1 = time.time()
    skel_obj = Skeleton(skeleton, source_image=labels)
    del skeleton, labels
    print(f"  n_paths (raw branches)={skel_obj.n_paths}, done in {time.time()-t1:.1f}s")

    print(f"Pruning spurs shorter than {spur_prune_voxels} voxels (max {max_prune_rounds} rounds) ...")
    t1 = time.time()
    skel_obj = prune_spurs(skel_obj, spur_prune_voxels, max_prune_rounds)
    print(f"  n_paths after pruning={skel_obj.n_paths}, done in {time.time()-t1:.1f}s")

    df = summarize(skel_obj, separator="_")
    print(f"Merging junction pixel clusters within {junction_merge_voxels} voxels into single nodes ...")
    t1 = time.time()
    node_positions, node_members, node_degree, edges = merge_junction_clusters(df, junction_merge_voxels)
    print(f"  final nodes={len(node_positions)}, final struts={len(edges)}, done in {time.time()-t1:.1f}s")

    if z_offset:
        node_positions = {nid: (z + z_offset, y, x) for nid, (z, y, x) in node_positions.items()}

    n_junction = sum(1 for d in node_degree.values() if d >= 2)
    n_endpoint = sum(1 for d in node_degree.values() if d <= 1)
    print(f"  nodes with degree>=2 (junction): {n_junction}, degree<=1 (endpoint/dangling): {n_endpoint}")

    meta = {
        "source_labels": os.path.relpath(labels_path, REPO_ROOT),
        "min_cluster": min_cluster,
        "spur_prune_voxels": spur_prune_voxels,
        "junction_merge_voxels": junction_merge_voxels,
        "z_range": list(z_range) if z_range is not None else None,
        "coordinate_order": "voxel indices (axis0, axis1, axis2) in the ORIGINAL (uncropped) volume's coordinate frame",
        "n_junctions": len(node_positions),
        "n_struts": len(edges),
    }
    suffix = f"_z{z_range[0]}-{z_range[1]}" if z_range is not None else ""
    json_path = os.path.join(out_dir, f"graph_min{min_cluster}{suffix}.json")
    save_graph_json(node_positions, node_degree, edges, json_path, meta)
    print(f"Saved graph JSON to {json_path}")

    return node_positions, node_members, node_degree, edges, out_dir, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a 3D node/edge lattice graph from CT k-means labels.")
    parser.add_argument("labels_path", help="Path to the k-means cluster label .npy volume.")
    parser.add_argument("--min-cluster", type=int, default=1,
                         help="Clusters >= this value are treated as foreground material (default 1).")
    parser.add_argument("--spur-prune-voxels", type=float, default=10.0,
                         help="Remove endpoint spurs shorter than this many voxels (default 10).")
    parser.add_argument("--junction-merge-voxels", type=float, default=10.0,
                         help="Merge junction-junction branches shorter than this into one node (default 10).")
    parser.add_argument("--max-prune-rounds", type=int, default=4,
                         help="Max batched spur-pruning rounds (default 4).")
    parser.add_argument("--z-range", type=int, nargs=2, default=None, metavar=("Z0", "Z1"),
                         help="Restrict processing to axis-0 range [Z0:Z1), e.g. to exclude "
                              "non-lattice regions such as solid end caps.")
    parser.add_argument("--crop", type=int, nargs=6, default=None,
                         metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
                         help="Voxel bounds for the 3D close-up visualization.")
    args = parser.parse_args()

    labels_shape = np.load(args.labels_path, mmap_mode="r").shape
    z_range = tuple(args.z_range) if args.z_range is not None else None
    node_positions, node_members, node_degree, edges, out_dir, meta = run(
        args.labels_path, args.min_cluster, args.spur_prune_voxels,
        args.junction_merge_voxels, args.max_prune_rounds, z_range=z_range,
    )

    suffix = f"_z{z_range[0]}-{z_range[1]}" if z_range is not None else ""
    overview_path = os.path.join(out_dir, f"overview_min{args.min_cluster}{suffix}.png")
    visualize_overview(node_positions, edges, labels_shape, overview_path)
    print(f"Saved overview visualization to {overview_path}")

    if args.crop is None:
        z_lo = z_range[0] if z_range is not None else 0
        z_hi = z_range[1] if z_range is not None else labels_shape[0]
        c = [(z_lo + z_hi) // 2 - 50, labels_shape[1] // 2 - 50, labels_shape[2] // 2 - 50]
        crop_bounds = ((c[0], c[0] + 100), (c[1], c[1] + 100), (c[2], c[2] + 100))
    else:
        z0, z1, y0, y1, x0, x1 = args.crop
        crop_bounds = ((z0, z1), (y0, y1), (x0, x1))
    crop_path = os.path.join(out_dir, f"closeup_min{args.min_cluster}{suffix}.png")
    visualize_crop_3d(node_positions, node_members, edges, crop_bounds, crop_path)
    print(f"Saved close-up visualization to {crop_path}")
