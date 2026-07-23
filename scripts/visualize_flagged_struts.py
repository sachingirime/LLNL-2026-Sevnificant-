"""
Overlay flagged (candidate-missing) struts from detect_missing_struts.py onto
one full TIFF slice, alongside the full as-designed unit-cell grid, so
individual defects can be seen in the context of the surrounding lattice.

Usage:
    python scripts/visualize_flagged_struts.py <tiff> <registered_json> <flagged_report_json> <output_png> \\
        [--slice <z>] [--vmin 25000] [--vmax 55000]

If --slice is omitted, the slice with the most flagged (interior) struts
passing through it is chosen automatically.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiff", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("flagged_report", type=Path, help="Output of detect_missing_struts.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--slice", type=int, default=None, help="z-slice to render. Auto-picked if omitted.")
    parser.add_argument("--vmin", type=float, default=25000)
    parser.add_argument("--vmax", type=float, default=55000)
    parser.add_argument("--edge-margin", type=float, default=40.0, help="Exclude struts within this many voxels of the specimen's outer faces (unreliable near scan edges).")
    args = parser.parse_args()

    model = json.loads(args.registered_json.read_text())
    report = json.loads(args.flagged_report.read_text())
    positions = {j["id"]: np.asarray(j["position"], dtype=float) for j in model["junctions"]}
    struts_by_id = {s["id"]: s for s in model["struts"]}
    flagged_ids = {f["id"] for f in report["flagged_struts"]}

    all_mid = np.array([s["midpoint_xyz"] for s in report["all_struts"]])
    lo, hi = all_mid.min(axis=0), all_mid.max(axis=0)

    def near_edge(p: np.ndarray) -> bool:
        return bool(np.any(np.abs(p - lo) < args.edge_margin) or np.any(np.abs(p - hi) < args.edge_margin))

    def midpoint(sid: int) -> np.ndarray:
        s = struts_by_id[sid]
        return (positions[s["junction0"]] + positions[s["junction1"]]) / 2

    interior_flagged = [sid for sid in flagged_ids if not near_edge(midpoint(sid))]

    z_slice = args.slice
    if z_slice is None:
        from collections import Counter
        z_hits: Counter = Counter()
        for sid in interior_flagged:
            s = struts_by_id[sid]
            p0, p1 = positions[s["junction0"]], positions[s["junction1"]]
            zlo, zhi = sorted((p0[2], p1[2]))
            for z in range(int(np.floor(zlo)), int(np.ceil(zhi)) + 1):
                z_hits[z] += 1
        z_slice = z_hits.most_common(1)[0][0]

    volume = tifffile.memmap(args.tiff)
    image = volume[z_slice].astype(float)

    fig, ax = plt.subplots(figsize=(11, 10.5), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=args.vmin, vmax=args.vmax, origin="upper")

    # Full design grid for context: every strut that crosses this plane, thin cyan.
    grid_pts = []
    flagged_pts = []
    for s in model["struts"]:
        p0, p1 = positions[s["junction0"]], positions[s["junction1"]]
        zlo, zhi = sorted((p0[2], p1[2]))
        if not (zlo <= z_slice <= zhi and zhi > zlo):
            continue
        fraction = (z_slice - p0[2]) / (p1[2] - p0[2])
        loc = p0 + fraction * (p1 - p0)  # [x, y, z]
        (flagged_pts if s["id"] in interior_flagged else grid_pts).append((loc[0], loc[1]))

    if grid_pts:
        xy = np.asarray(grid_pts)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, facecolors="none", edgecolors="#00d7ff", linewidths=0.5,
                   label=f"design struts present at this plane ({len(xy)})")
    if flagged_pts:
        xy = np.asarray(flagged_pts)
        ax.scatter(xy[:, 0], xy[:, 1], s=140, facecolors="none", edgecolors="#ff2d2d", linewidths=2.2,
                   marker="o", label=f"flagged / candidate-missing struts ({len(xy)})")

    ax.set_title(f"Flagged struts vs. design lattice grid -- TIFF z-slice {z_slice}")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.legend(loc="lower right", framealpha=0.85)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Saved {args.output}; slice z={z_slice}, {len(flagged_pts)} flagged struts shown, {len(grid_pts)} normal struts shown")


if __name__ == "__main__":
    main()
