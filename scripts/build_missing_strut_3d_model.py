"""
Build a single colored 3D model (PLY) showing the as-scanned lattice together
with the candidate-missing struts found by detect_missing_struts.py, so
defects can be checked visually in 3D instead of one 2D slice at a time.

- The scanned lattice surface (from marching cubes on the segmentation mask)
  is colored light gray.
- Each flagged strut is drawn as a solid cylinder connecting its two design
  junctions, colored:
    red    = flagged AND away from the specimen's outer faces (higher
             confidence -- see detect_missing_struts.py's edge-effect note)
    orange = flagged but near an outer face (lower confidence; could be a
             scan-edge artifact rather than a real defect)

Output is PLY (binary_little_endian) with per-vertex RGB color, viewable in
MeshLab, Blender, or CloudCompare. Plain STL has no color, which is why this
uses PLY instead.

Usage:
    python scripts/build_missing_strut_3d_model.py <mask_path> <registered_json> <flagged_report_json> <output_ply> \\
        [--step-size 4] [--radius 6] [--edge-margin 40]

Example:
    python scripts/build_missing_strut_3d_model.py \\
        data/9x9x9_octet_lattice/segmentation/9x9x9_octet_lattice_segmentation.tif \\
        "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \\
        outputs/missing_struts/flagged_struts.json \\
        outputs/missing_struts/missing_struts_3d_model.ply
"""

import argparse
import json
import os
import struct

import numpy as np
import tifffile
from skimage import measure


def load_volume(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path)
    raise ValueError(f"Unsupported file type '{ext}'.")


def cylinder_mesh(p0: np.ndarray, p1: np.ndarray, radius: float, color: tuple, n_sides: int = 10):
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=np.uint8)
    axis_n = axis / length
    not_axis = np.array([1.0, 0.0, 0.0]) if abs(axis_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 = np.cross(axis_n, not_axis)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis_n, perp1)

    thetas = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring0 = np.array([p0 + radius * (np.cos(t) * perp1 + np.sin(t) * perp2) for t in thetas])
    ring1 = np.array([p1 + radius * (np.cos(t) * perp1 + np.sin(t) * perp2) for t in thetas])
    verts = np.vstack([ring0, ring1, p0[None, :], p1[None, :]])

    n = n_sides
    cap0, cap1 = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([cap0, j, i])
        faces.append([cap1, n + i, n + j])
    faces = np.asarray(faces, dtype=int)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def write_ply(path: str, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "wb") as fh:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        fh.write(header.encode("ascii"))
        v = verts.astype("<f4")
        c = colors.astype(np.uint8)
        for i in range(len(verts)):
            fh.write(struct.pack("<3f3B", v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
        f = faces.astype("<i4")
        for i in range(len(faces)):
            fh.write(struct.pack("<B3i", 3, f[i, 0], f[i, 1], f[i, 2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_path")
    parser.add_argument("registered_json")
    parser.add_argument("flagged_report")
    parser.add_argument("output_ply")
    parser.add_argument("--step-size", type=int, default=4, help="Marching cubes step size for the base lattice mesh. Default 4 (matches reconstruct_3d.py default).")
    parser.add_argument("--radius", type=float, default=6.0, help="Marker cylinder radius in voxels. Default 6 (slightly thicker than a real strut, for visibility).")
    parser.add_argument("--edge-margin", type=float, default=40.0, help="Struts within this many voxels of an outer face are colored orange (lower confidence) instead of red.")
    args = parser.parse_args()

    print("Loading segmentation mask ...")
    volume = load_volume(args.mask_path)
    print(f"Volume shape={volume.shape}")
    print("Running marching cubes on the scanned lattice ...")
    verts, faces, _n, _v = measure.marching_cubes(volume.astype(np.float32), level=0.5, step_size=args.step_size)
    verts = verts[:, [2, 1, 0]]  # (Z,Y,X) -> (X,Y,Z), matches project convention
    base_colors = np.tile(np.array([210, 210, 210], dtype=np.uint8), (len(verts), 1))
    print(f"Base mesh: {len(verts)} verts, {len(faces)} faces")

    model = json.loads(open(args.registered_json).read())
    report = json.loads(open(args.flagged_report).read())
    positions = {j["id"]: np.asarray(j["position"], dtype=float) for j in model["junctions"]}
    struts_by_id = {s["id"]: s for s in model["struts"]}

    all_mid = np.array([s["midpoint_xyz"] for s in report["all_struts"]])
    lo, hi = all_mid.min(axis=0), all_mid.max(axis=0)

    all_verts = [verts]
    all_faces = [faces]
    all_colors = [base_colors]
    offset = len(verts)
    n_red, n_orange = 0, 0

    for f in report["flagged_struts"]:
        s = struts_by_id[f["id"]]
        p0, p1 = positions[s["junction0"]], positions[s["junction1"]]
        mid = (p0 + p1) / 2
        near_edge = np.any(np.abs(mid - lo) < args.edge_margin) or np.any(np.abs(mid - hi) < args.edge_margin)
        color = (255, 140, 0) if near_edge else (230, 20, 20)
        n_orange += near_edge
        n_red += not near_edge
        cv, cf, cc = cylinder_mesh(p0, p1, args.radius, color)
        if len(cv) == 0:
            continue
        all_verts.append(cv)
        all_faces.append(cf + offset)
        all_colors.append(cc)
        offset += len(cv)

    verts_out = np.vstack(all_verts)
    faces_out = np.vstack(all_faces)
    colors_out = np.vstack(all_colors)
    print(f"Markers: {n_red} red (interior, higher confidence), {n_orange} orange (near edge, lower confidence)")

    out_dir = os.path.dirname(os.path.abspath(args.output_ply))
    os.makedirs(out_dir, exist_ok=True)
    write_ply(args.output_ply, verts_out, faces_out, colors_out)
    print(f"Saved {args.output_ply} ({len(verts_out)} verts, {len(faces_out)} faces)")


if __name__ == "__main__":
    main()
