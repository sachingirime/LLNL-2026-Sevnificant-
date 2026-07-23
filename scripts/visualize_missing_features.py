"""Overlay candidate missing struts and nodes on a registered TIFF z-slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def local_material_fraction(mask: np.ndarray, point_xyz: np.ndarray, radius: int) -> float:
    x, y, z = np.rint(point_xyz).astype(int)
    region = mask[
        max(z - radius, 0):z + radius + 1,
        max(y - radius, 0):y + radius + 1,
        max(x - radius, 0):x + radius + 1,
    ]
    return float(region.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("segmentation", type=Path)
    parser.add_argument("flagged_struts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--node-plane-tolerance", type=float, default=6.0)
    parser.add_argument("--node-radius", type=int, default=4)
    parser.add_argument("--node-threshold", type=float, default=0.10)
    args = parser.parse_args()

    volume = tifffile.memmap(args.tiff)
    mask = tifffile.memmap(args.segmentation)
    if volume.shape != mask.shape:
        raise ValueError(f"TIFF and segmentation shapes differ: {volume.shape} vs {mask.shape}")
    if not 0 <= args.slice < volume.shape[0]:
        raise ValueError(f"z-slice must be within 0..{volume.shape[0] - 1}")

    model = json.loads(args.registered_json.read_text())
    report = json.loads(args.flagged_struts.read_text())
    points = {item["id"]: np.asarray(item["position"], dtype=float) for item in model["junctions"]}
    flagged_ids = {item["id"] for item in report["flagged_struts"]}

    missing_strut_points = []
    for strut in model["struts"]:
        if strut["id"] not in flagged_ids:
            continue
        first, second = points[strut["junction0"]], points[strut["junction1"]]
        low, high = sorted((first[2], second[2]))
        if low <= args.slice <= high and high > low:
            fraction = (args.slice - first[2]) / (second[2] - first[2])
            position = first + fraction * (second - first)
            missing_strut_points.append(position[:2])

    missing_nodes = []
    for node_id, point in points.items():
        if abs(point[2] - args.slice) > args.node_plane_tolerance:
            continue
        material_fraction = local_material_fraction(mask, point, args.node_radius)
        if material_fraction < args.node_threshold:
            missing_nodes.append((node_id, point[:2], material_fraction))

    image = volume[args.slice]
    low, high = np.percentile(image, (1, 99.7))
    figure, axis = plt.subplots(figsize=(11, 10), constrained_layout=True)
    axis.imshow(image, cmap="gray", vmin=low, vmax=high, origin="upper")
    if missing_strut_points:
        xy = np.asarray(missing_strut_points)
        axis.scatter(xy[:, 0], xy[:, 1], s=125, facecolors="none", edgecolors="#ff3030", linewidths=2.2,
                     label=f"candidate missing struts ({len(xy)})")
    if missing_nodes:
        xy = np.asarray([item[1] for item in missing_nodes])
        axis.scatter(xy[:, 0], xy[:, 1], s=100, c="#d946ef", marker="X", edgecolors="white", linewidths=0.7,
                     label=f"candidate missing nodes ({len(xy)})")
    axis.set_title(f"Candidate missing features — TIFF z-slice {args.slice}")
    axis.set_xlabel("JSON / TIFF x (pixels)")
    axis.set_ylabel("JSON / TIFF y (pixels)")
    axis.legend(loc="lower right", framealpha=0.85)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved {args.output}; {len(missing_strut_points)} struts, {len(missing_nodes)} nodes")


if __name__ == "__main__":
    main()
