"""
Compare a CT-derived segmentation mask against the as-designed lattice model
to flag struts that are present in the design but missing (or badly broken)
in the as-built scan.

For each strut in the registered JSON model, sample points along its middle
span (excluding a margin near each junction, since junction "nodes" have
their own material and would bias every strut toward looking present) and
check a small neighborhood around each sample point for segmented material.
A strut whose samples mostly land on empty voxels is flagged as a defect
candidate.

Note: input paths should point into data/ (read-only, never written to).
Save output_path outside data/, e.g. under outputs/.

Usage:
    python scripts/detect_missing_struts.py <registered_json> <mask_path> <output_json> \\
        [--n-samples 11] [--margin 0.2] [--radius 4] [--flag-threshold 0.5] [--histogram <png_path>]

Example:
    python scripts/detect_missing_struts.py \\
        "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \\
        data/9x9x9_octet_lattice/segmentation/9x9x9_octet_lattice_segmentation.tif \\
        outputs/missing_struts/flagged_struts.json \\
        --histogram outputs/missing_struts/presence_fraction_histogram.png
"""

import argparse
import json
import os

import numpy as np
import tifffile


def load_mask(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r")
    if ext in (".tif", ".tiff"):
        return tifffile.memmap(path)
    raise ValueError(f"Unsupported file type '{ext}'. Expected .npy, .tif, or .tiff.")


def strut_presence_fraction(
    mask: np.ndarray, p0_xyz: np.ndarray, p1_xyz: np.ndarray, n_samples: int, margin: float, radius: int
) -> float:
    # JSON positions are [x, y, z]; the mask is indexed [z, y, x] (TIFF order).
    p0 = p0_xyz[::-1]
    p1 = p1_xyz[::-1]
    hits = 0
    for t in np.linspace(margin, 1.0 - margin, n_samples):
        z, y, x = p0 + t * (p1 - p0)
        zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(zi - radius, 0), min(zi + radius + 1, mask.shape[0])
        y0, y1 = max(yi - radius, 0), min(yi + radius + 1, mask.shape[1])
        x0, x1 = max(xi - radius, 0), min(xi + radius + 1, mask.shape[2])
        if mask[z0:z1, y0:y1, x0:x1].any():
            hits += 1
    return hits / n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag struts present in the as-designed JSON but absent from a CT segmentation mask.")
    parser.add_argument("registered_json", help="As-designed lattice JSON, already registered into the mask's voxel coordinate frame.")
    parser.add_argument("mask_path", help="Binary segmentation mask (.npy or .tif), values 0/1, same voxel frame as the JSON.")
    parser.add_argument("output_json", help="Where to save the per-strut presence report (.json).")
    parser.add_argument("--n-samples", type=int, default=11, help="Sample points per strut span. Default 11.")
    parser.add_argument("--margin", type=float, default=0.2, help="Fraction of the strut length to exclude at each end (avoids junction-node bias). Default 0.2.")
    parser.add_argument("--radius", type=int, default=4, help="Neighborhood half-width (voxels) checked around each sample point. Default 4 (~nominal strut radius for this dataset).")
    parser.add_argument("--flag-threshold", type=float, default=0.5, help="Struts with presence_fraction below this are flagged as defect candidates. Default 0.5.")
    parser.add_argument("--histogram", default=None, help="Optional path to save a PNG histogram of presence_fraction across all struts.")
    args = parser.parse_args()

    model = json.loads(open(args.registered_json).read())
    positions = {j["id"]: np.asarray(j["position"], dtype=float) for j in model["junctions"]}
    mask = load_mask(args.mask_path)
    print(f"Mask shape={mask.shape} dtype={mask.dtype}")
    print(f"Model: {len(model['junctions'])} junctions, {len(model['struts'])} struts")

    results = []
    for strut in model["struts"]:
        p0 = positions[strut["junction0"]]
        p1 = positions[strut["junction1"]]
        fraction = strut_presence_fraction(mask, p0, p1, args.n_samples, args.margin, args.radius)
        results.append(
            {
                "id": strut["id"],
                "junction0": strut["junction0"],
                "junction1": strut["junction1"],
                "presence_fraction": fraction,
                "flagged": fraction < args.flag_threshold,
                "midpoint_xyz": ((p0 + p1) / 2).tolist(),
            }
        )
        if strut["id"] % 3000 == 0:
            print(f"  ...{strut['id']}/{len(model['struts'])}")

    fractions = np.array([r["presence_fraction"] for r in results])
    flagged = [r for r in results if r["flagged"]]

    print(f"\nTotal struts: {len(results)}")
    print(f"Flagged (presence_fraction < {args.flag_threshold}): {len(flagged)} ({100 * len(flagged) / len(results):.3f}%)")
    print(f"presence_fraction percentiles: p1={np.percentile(fractions,1):.2f} p10={np.percentile(fractions,10):.2f} "
          f"median={np.percentile(fractions,50):.2f} p90={np.percentile(fractions,90):.2f}")

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(
            {
                "n_struts": len(results),
                "n_flagged": len(flagged),
                "flag_threshold": args.flag_threshold,
                "params": {"n_samples": args.n_samples, "margin": args.margin, "radius": args.radius},
                "flagged_struts": flagged,
                "all_struts": results,
            },
            fh,
            indent=2,
        )
    print(f"Saved report to {args.output_json}")

    if args.histogram:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(fractions, bins=50, color="#3b82f6", edgecolor="none")
        ax.axvline(args.flag_threshold, color="#ef4444", linestyle="--", label=f"flag threshold ({args.flag_threshold})")
        ax.set_xlabel("presence_fraction (share of sampled points with nearby material)")
        ax.set_ylabel("strut count")
        ax.set_title(f"Strut presence vs. as-designed model ({len(flagged)}/{len(results)} flagged)")
        ax.legend()
        os.makedirs(os.path.dirname(os.path.abspath(args.histogram)), exist_ok=True)
        fig.savefig(args.histogram, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram to {args.histogram}")


if __name__ == "__main__":
    main()
