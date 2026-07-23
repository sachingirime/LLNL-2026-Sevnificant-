"""Overlay a registered lattice JSON model on one TIFF-stack slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def _json_position_to_volume_position(position: np.ndarray) -> np.ndarray:
    """Convert a JSON node position from [x, y, z] to TIFF memmap order [z, y, x]."""
    return np.asarray(position, dtype=float)[[2, 1, 0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slice", type=int, default=350)
    parser.add_argument(
        "--axis", choices=("x", "y", "z"), default="z",
        help="Registered JSON/TIFF coordinate normal to the displayed plane.",
    )
    parser.add_argument(
        "--unit-cells", action="store_true",
        help="Overlay JSON unit-cell footprints that intersect the displayed plane.",
    )
    args = parser.parse_args()

    # TIFF axes are Z, Y, X.  Memory mapping permits X/Y orthogonal slices
    # without loading the complete 990 MB stack into RAM.
    volume = tifffile.memmap(args.tiff)
    axis_index = {"z": 0, "y": 1, "x": 2}[args.axis]
    if not 0 <= args.slice < volume.shape[axis_index]:
        raise ValueError(
            f"{args.axis}-slice {args.slice} is outside "
            f"0..{volume.shape[axis_index] - 1}"
        )
    image = np.take(volume, args.slice, axis=axis_index)

    model = json.loads(args.json_file.read_text())
    points = {
        item["id"]: _json_position_to_volume_position(item["position"])
        for item in model["junctions"]
    }

    # JSON node coordinates are stored as [x, y, z], while the TIFF volume is
    # indexed in memmap order [z, y, x]. Convert the graph into that volume
    # axis ordering before slicing and overlaying.
    coordinate_index = axis_index

    # np.take preserves the relative order of the two axes that remain after
    # slicing away axis_index: the smaller-numbered remaining axis becomes
    # the resulting 2D array's row dimension (imshow's vertical/Y), and the
    # larger-numbered one becomes its column dimension (imshow's
    # horizontal/X). plot_axes = [larger, smaller] so plot_axes[0] is usable
    # directly as the x-coordinate and plot_axes[1] as the y-coordinate.
    other_axes = sorted(a for a in (0, 1, 2) if a != axis_index)
    plot_axes = [other_axes[1], other_axes[0]]

    intersections: list[tuple[float, float]] = []
    for strut in model["struts"]:
        first = points[strut["junction0"]]
        second = points[strut["junction1"]]
        low, high = sorted((first[coordinate_index], second[coordinate_index]))
        if low <= args.slice <= high and high > low:
            fraction = (
                (args.slice - first[coordinate_index])
                / (second[coordinate_index] - first[coordinate_index])
            )
            location = first + fraction * (second - first)
            intersections.append((location[plot_axes[0]], location[plot_axes[1]]))

    junctions_on_plane = [
        point
        for point in points.values()
        if abs(point[coordinate_index] - args.slice) <= 1.0
    ]
    junctions_on_plane = [point[plot_axes] for point in junctions_on_plane]
    horizontal_label = f"TIFF/JSON axis {plot_axes[0]} (pixels)"
    vertical_label = f"TIFF/JSON axis {plot_axes[1]} (pixels)"

    unit_cell_boxes: list[tuple[float, float, float, float]] = []
    if args.unit_cells:
        for cell in model["unit_cells"]:
            cell_points = np.asarray(
                [
                    endpoint
                    for strut_id in cell["struts"]
                    for endpoint in (points[model["struts"][strut_id]["junction0"]], points[model["struts"][strut_id]["junction1"]])
                ]
            )
            if cell_points[:, axis_index].min() <= args.slice <= cell_points[:, axis_index].max():
                plane_points = cell_points[:, plot_axes]
                x_min, y_min = plane_points.min(axis=0)
                x_max, y_max = plane_points.max(axis=0)
                unit_cell_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    lo, hi = np.percentile(image, (1, 99.7))
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    if unit_cell_boxes:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        patches = [Rectangle((x, y), width, height) for x, y, width, height in unit_cell_boxes]
        ax.add_collection(
            PatchCollection(
                patches, facecolor="none", edgecolor="#ff4d4d", linewidth=0.75,
                label=f"JSON unit-cell footprints ({len(patches)})",
            )
        )
    if intersections:
        xy = np.asarray(intersections)
        ax.scatter(
            xy[:, 0], xy[:, 1], s=12, facecolors="none", edgecolors="#00d7ff",
            linewidths=0.65, label=f"JSON strut-plane intersections ({len(xy)})",
        )
    if junctions_on_plane:
        xy = np.asarray(junctions_on_plane)
        ax.scatter(
            xy[:, 0], xy[:, 1], s=24, c="#ff4d4d", marker="+", linewidths=1.1,
            label=f"JSON junctions within ±1 voxel ({len(xy)})",
        )
    ax.set_title(f"Registered lattice overlay: TIFF {args.axis}-slice {args.slice}")
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(vertical_label)
    ax.legend(loc="lower right", framealpha=0.82)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(
        f"Saved {args.output}; {len(intersections)} strut intersections, "
        f"{len(junctions_on_plane)} near-plane junctions"
    )


if __name__ == "__main__":
    main()
