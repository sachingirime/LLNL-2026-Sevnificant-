#!/usr/bin/env python
"""Rank intentionally removed lattice members by comparing a full and defective STL.

The registered JSON and the STL use different units.  Junction IDs are shared with the
nominal JSON, so this script first fits registered-CT coordinates back to the nominal
CAD grid, then maps that grid to the STL bounding box.  It samples each JSON strut away
from junctions and ranks the increase in centreline-to-surface distance from the full
STL to the defective STL.

The default orientation is the nominal CAD-to-STL convention.  A bare octet lattice has
rigid symmetry, so an STL alone cannot resolve an alternative reflected/permuted JSON
labelling without an external datum.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


STL_LO = np.array([-20.743635, -25.520002, -20.748846])
STL_HI = np.array([20.749142, 25.520002, 20.746773])


def stl_vertices(path: Path) -> np.ndarray:
    """Memory-map binary STL vertices without materialising triangle normals."""
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=84)
    records = raw.reshape(-1, 50)
    return np.asarray(np.ndarray((len(records), 3), dtype="<f4", buffer=records,
                                 offset=12, strides=(50, 4)))


def load_graph(path: Path):
    graph = json.loads(path.read_text())
    positions = np.asarray([j["position"] for j in graph["junctions"]], dtype=float)
    pairs = np.asarray([[s["junction0"], s["junction1"]] for s in graph["struts"]],
                       dtype=int)
    return positions, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("nominal_json", type=Path,
                        help="same graph in its 0..18 CAD grid coordinates")
    parser.add_argument("full_stl", type=Path)
    parser.add_argument("defective_stl", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=None,
                        help="optional fixed number of calls; normally use --min-score")
    parser.add_argument("--min-score", type=float, default=1e-6,
                        help="minimum 25th-percentile distance increase (mm) for missing")
    parser.add_argument("--plot-out", type=Path,
                        help="optional PNG: classified registered struts in four views")
    parser.add_argument("--graph-out", type=Path,
                        help="optional .npz for scripts/graph_webgl.py")
    args = parser.parse_args()

    registered, pairs = load_graph(args.registered_json)
    nominal, nominal_pairs = load_graph(args.nominal_json)
    if not np.array_equal(pairs, nominal_pairs):
        raise ValueError("JSON strut IDs/endpoints differ; cannot transfer the registration")

    # The two JSON files have common junction IDs. Fit CT xyz -> nominal grid xyz.
    design_from_registered = np.linalg.lstsq(
        np.c_[registered, np.ones(len(registered))], nominal, rcond=None
    )[0]
    grid = np.c_[registered, np.ones(len(registered))] @ design_from_registered
    stl_positions = STL_LO + grid / 18.0 * (STL_HI - STL_LO)
    a, b = stl_positions[pairs[:, 0]], stl_positions[pairs[:, 1]]
    fractions = np.linspace(0.20, 0.80, 9)
    samples = (a[:, None, :] + (b - a)[:, None, :] * fractions[None, :, None])

    print("building STL surface trees")
    full_tree = cKDTree(stl_vertices(args.full_stl), balanced_tree=False)
    defect_tree = cKDTree(stl_vertices(args.defective_stl), balanced_tree=False)
    print("measuring struts")
    full_d = full_tree.query(samples.reshape(-1, 3), workers=-1)[0].reshape(len(pairs), -1)
    defect_d = defect_tree.query(samples.reshape(-1, 3), workers=-1)[0].reshape(len(pairs), -1)

    # A quarter-quantile requires absence across most of the interior, avoiding a score
    # dominated by one endpoint junction. The full-STL subtraction removes CAD meshing bias.
    score = np.quantile(defect_d - full_d, 0.25, axis=1)
    order = np.argsort(score)[::-1]
    selected = (set(order[:args.count].tolist()) if args.count is not None else
                set(np.flatnonzero(score > args.min_score).tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "rank", "strut_id", "verdict", "score_mm", "full_distance_mm",
            "defective_distance_mm", "junction0", "junction1",
            "registered_junction0_xyz", "registered_junction1_xyz",
            "cad_junction0_xyz", "cad_junction1_xyz",
        ])
        writer.writeheader()
        for rank, idx in enumerate(order, 1):
            writer.writerow({
                "rank": rank,
                "strut_id": idx,
                "verdict": "missing" if idx in selected else "nominal",
                "score_mm": f"{score[idx]:.6f}",
                "full_distance_mm": f"{np.median(full_d[idx]):.6f}",
                "defective_distance_mm": f"{np.median(defect_d[idx]):.6f}",
                "junction0": pairs[idx, 0], "junction1": pairs[idx, 1],
                "registered_junction0_xyz": registered[pairs[idx, 0]].tolist(),
                "registered_junction1_xyz": registered[pairs[idx, 1]].tolist(),
                "cad_junction0_xyz": nominal[pairs[idx, 0]].tolist(),
                "cad_junction1_xyz": nominal[pairs[idx, 1]].tolist(),
            })

    if args.graph_out:
        args.graph_out.parent.mkdir(parents=True, exist_ok=True)
        material_fraction = np.ones(len(pairs), dtype=np.float32)
        material_fraction[list(selected)] = 0.0
        np.savez_compressed(args.graph_out, pairs=pairs,
                            material_fraction=material_fraction)
        print(f"wrote {args.graph_out}")

    if args.plot_out:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        args.plot_out.parent.mkdir(parents=True, exist_ok=True)
        a, b = registered[pairs[:, 0]], registered[pairs[:, 1]]
        normal = np.stack([a, b], axis=1)
        missing = normal[sorted(selected)]
        fig = plt.figure(figsize=(14, 11), constrained_layout=True)
        ax3 = fig.add_subplot(2, 2, 1, projection="3d")
        ax3.set_title(f"Registered design — {len(selected)} missing struts")
        for seg, color, width, alpha in [(normal, "#aeb4bd", 0.25, 0.08),
                                         (missing, "#d94841", 2.0, 1.0)]:
            for line in seg:
                ax3.plot(*line.T, color=color, lw=width, alpha=alpha)
        ax3.set_xlabel("x (vox)"); ax3.set_ylabel("y (vox)"); ax3.set_zlabel("z (vox)")
        ax3.view_init(elev=22, azim=-55)
        views = [(0, 1, "x–y"), (0, 2, "x–z"), (1, 2, "y–z")]
        for ax, (u, v, title) in zip([fig.add_subplot(2, 2, i) for i in (2, 3, 4)], views):
            ax.add_collection(LineCollection(normal[:, :, [u, v]], colors="#aeb4bd",
                                             linewidths=0.25, alpha=0.12))
            ax.add_collection(LineCollection(missing[:, :, [u, v]], colors="#d94841",
                                             linewidths=1.6, alpha=1.0))
            ax.autoscale(); ax.set_aspect("equal")
            ax.set_title(title + " projection")
            ax.set_xlabel(("x", "y", "z")[u] + " (vox)")
            ax.set_ylabel(("x", "y", "z")[v] + " (vox)")
        fig.suptitle("0.5.stl classification: gray = nominal, red = missing", fontsize=15)
        fig.savefig(args.plot_out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.plot_out}")
    print(f"wrote {args.out}; selected {len(selected)} missing struts")
    print("missing IDs:", ", ".join(str(i) for i in order if i in selected))


if __name__ == "__main__":
    main()
