"""
Explore the data/ directory: inventory datasets and files, summarize JSON
lattice graphs (junctions/struts/unit_cells), summarize STL mesh files, and
plot intensity histograms for volumetric data (.tif/.npy).

data/ is treated as read-only: this script only reads from it.
All outputs (summary.md, histogram PNGs) are written to outputs/exploration/.

Usage:
    python scripts/explore_data.py
"""

import hashlib
import json
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "exploration")


def inventory_files():
    by_ext = {}
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            by_ext.setdefault(ext, []).append(path)
            all_files.append(path)
    return all_files, by_ext


def describe_json(path):
    with open(path) as fh:
        d = json.load(fh)
    n_junctions = len(d.get("junctions", []))
    n_struts = len(d.get("struts", []))
    n_unit_cells = len(d.get("unit_cells", []))
    pos_range = None
    if d.get("junctions"):
        arr = np.array([j["position"] for j in d["junctions"]])
        pos_range = (
            tuple(round(float(v), 2) for v in arr.min(axis=0)),
            tuple(round(float(v), 2) for v in arr.max(axis=0)),
        )
    return n_junctions, n_struts, n_unit_cells, pos_range


def describe_stl(path):
    with open(path, "rb") as fh:
        fh.read(80)  # header
        n_tri = struct.unpack("<I", fh.read(4))[0]
    return n_tri


def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path)
    raise ValueError(f"Unsupported volume type: {ext}")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plot_histogram(volume, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(volume.ravel(), bins=200, color="steelblue")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(f"{volume.dtype} value")
    plt.ylabel("voxel count (log scale)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []

    all_files, by_ext = inventory_files()

    lines.append("# Data Inventory\n")
    lines.append(f"Total files under data/: {len(all_files)}\n")

    lines.append("## Files by extension\n")
    for ext, files in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{ext or '(no ext)'}`: {len(files)}")
    lines.append("")

    top_dirs = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    lines.append(f"## Dataset folders ({len(top_dirs)})\n")
    for d in top_dirs:
        lines.append(f"- {d}/")
    lines.append("")

    json_files = sorted(by_ext.get(".json", []))
    lines.append(f"## JSON graph files ({len(json_files)})\n")
    for jf in json_files:
        rel = os.path.relpath(jf, DATA_DIR)
        n_j, n_s, n_u, pos_range = describe_json(jf)
        lines.append(
            f"- `{rel}`: {n_j} junctions, {n_s} struts, {n_u} unit_cells, "
            f"junction position range={pos_range}"
        )
    lines.append("")

    stl_files = sorted(by_ext.get(".stl", []))
    lines.append(f"## STL mesh files ({len(stl_files)})\n")
    for sf in stl_files:
        rel = os.path.relpath(sf, DATA_DIR)
        n_tri = describe_stl(sf)
        lines.append(f"- `{rel}`: {n_tri} triangles")
    lines.append("")

    vol_files = sorted(by_ext.get(".tif", []) + by_ext.get(".tiff", []) + by_ext.get(".npy", []))
    lines.append(f"## Volumetric datasets ({len(vol_files)})\n")

    hash_to_first = {}
    for vf in vol_files:
        rel = os.path.relpath(vf, DATA_DIR)
        h = md5sum(vf)
        if h in hash_to_first:
            lines.append(f"- `{rel}`: IDENTICAL to `{hash_to_first[h]}` (md5 {h[:10]}...) — skipping duplicate histogram")
            continue
        hash_to_first[h] = rel

        vol = load_volume(vf)
        lines.append(
            f"- `{rel}`: shape={vol.shape}, dtype={vol.dtype}, "
            f"min={vol.min():.6g}, max={vol.max():.6g}, mean={vol.mean():.6g}"
        )
        hist_name = rel.replace(os.sep, "__") + ".histogram.png"
        hist_path = os.path.join(OUTPUT_DIR, hist_name)
        plot_histogram(vol, f"Intensity histogram: {rel}", hist_path)
        lines.append(f"  -> histogram saved to outputs/exploration/{hist_name}")
    lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "summary.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nFull report saved to {os.path.relpath(report_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
