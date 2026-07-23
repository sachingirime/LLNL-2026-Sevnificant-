"""
Reconstruct a 3D surface mesh (STL) of the printed lattice from a CT volume
or segmentation mask, using marching cubes.

Works on both:
- A binary segmentation (.npy/.tif with values 0/1) -> use the default
  --level 0.5.
- A raw grayscale CT volume (.tif with uint16 density values) -> pass
  --level <density threshold>, e.g. the same threshold used to segment it
  (see data/9x9x9_octet_lattice/segmentation/run_segmentation.py, which used
  40500 for that dataset). Running marching cubes directly on the grayscale
  volume at that iso-level gives a smoother surface than binarizing first.

Note: input_path should point into data/ (read-only, never written to).
Save output_path outside data/, e.g. under outputs/.

Usage:
    python scripts/reconstruct_3d.py <input_path> <output_stl_path> \\
        [--level 0.5] [--step-size 2] [--preview <png_path>]

Examples:
    # Already-segmented binary mask -> mesh
    python scripts/reconstruct_3d.py \\
        data/9x9x9_octet_lattice/segmentation/9x9x9_octet_lattice_segmentation.tif \\
        outputs/9x9x9_octet_lattice/reconstruction.stl \\
        --preview outputs/9x9x9_octet_lattice/reconstruction_preview.png

    # Raw grayscale CT volume, thresholded directly at the iso-surface level
    python scripts/reconstruct_3d.py \\
        "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \\
        outputs/missing_struts/reconstruction.stl \\
        --level 40500 --step-size 2
"""

import argparse
import os
import struct
import time

import numpy as np
import tifffile
from skimage import measure


def load_volume(input_path: str) -> np.ndarray:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".npy":
        return np.load(input_path)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(input_path)
    raise ValueError(f"Unsupported file type '{ext}'. Expected .npy, .tif, or .tiff.")


def write_binary_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh as a binary STL file."""
    tri = verts[faces]  # (n_faces, 3, 3)
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).astype("<f4")
    tri = tri.astype("<f4")

    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", len(faces)))
        for normal, triangle in zip(normals, tri):
            fh.write(struct.pack("<3f", *normal))
            fh.write(struct.pack("<9f", *triangle.reshape(-1)))
            fh.write(struct.pack("<H", 0))


def save_preview(volume: np.ndarray, level: float, output_path: str, preview_step_size: int = 8) -> None:
    """Quick static render for a sanity check, built from its own coarse
    marching-cubes pass (not a subsample of the export mesh's faces --
    randomly dropping faces from a fine mesh scatters isolated triangles
    instead of a coherent surface). Not a substitute for viewing the actual
    STL in a real 3D viewer (MeshLab, ParaView, Blender, etc.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts, faces, _normals, _values = measure.marching_cubes(volume, level=level, step_size=preview_step_size)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.9, edgecolor="none", facecolor="tab:orange")
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(f"Reconstruction preview (coarse, {len(faces)} faces)")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def reconstruct(input_path: str, output_path: str, level: float, step_size: int, preview_path: str = None) -> None:
    if os.path.splitext(output_path)[1].lower() != ".stl":
        raise ValueError("output_path must end with '.stl'")

    print(f"Loading volume from {input_path} ...")
    volume = load_volume(input_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")
    print(f"Volume shape={volume.shape} dtype={volume.dtype} range=[{volume.min()}, {volume.max()}]")

    print(f"Running marching cubes (level={level}, step_size={step_size}) ...")
    start = time.time()
    verts, faces, _normals, _values = measure.marching_cubes(
        volume.astype(np.float32), level=level, step_size=step_size
    )
    print(f"Marching cubes done in {time.time() - start:.1f}s: {len(verts)} verts, {len(faces)} faces")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    write_binary_stl(output_path, verts, faces)
    print(f"Saved mesh to {output_path}")

    if preview_path:
        save_preview(volume.astype(np.float32), level, preview_path)
        print(f"Saved preview render to {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct a 3D surface mesh (STL) from a CT volume via marching cubes.")
    parser.add_argument("input_path", help="Path to the .npy or .tif/.tiff volume (segmentation mask or raw grayscale CT).")
    parser.add_argument("output_path", help="Path to save the output mesh (.stl).")
    parser.add_argument("--level", type=float, default=0.5, help="Iso-surface level. Use 0.5 for a binary 0/1 mask (default), or a density value (e.g. 40500) for a raw grayscale volume.")
    parser.add_argument("--step-size", type=int, default=4, help="Marching cubes step size: 1 = full resolution (very slow, huge mesh e.g. >500MB STL for a dense lattice), higher = faster/smaller/coarser. Default 4 (~100-150MB STL for a full lattice cube).")
    parser.add_argument("--preview", default=None, help="Optional path to save a quick static PNG preview of the mesh.")
    args = parser.parse_args()

    reconstruct(args.input_path, args.output_path, args.level, args.step_size, args.preview)
