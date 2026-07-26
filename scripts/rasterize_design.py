"""
Rasterize the registered nominal lattice (JSON graph) into a voxel volume that
lives in the CT's coordinate frame, so as-designed and as-built can be compared
image-to-image.

Why this exists
---------------
The registered JSON (`registered_jsons/...json`) is the *complete* nominal 9x9x9
octet design (all 18,468 struts present) pushed through a rigid similarity
transform into the CT voxel frame -- verified: fitting nominal -> registered is
exact (residual 0). It carries no defect information; the defects live in the
gap between this design and the CT. Rasterizing turns the graph into an image so
that gap can be measured directly (per-strut sampling, difference volume,
phase-correlation registration refinement, visual overlay).

Coordinate convention
---------------------
JSON `position` is (x, y, z) in CT voxel units; numpy indexes vol[z, y, x], so
positions are reordered to (z, y, x) once at load. Empirically only this
permutation puts 100% of junctions in-bounds and ~88% on segmented material.

Geometry
--------
Nominal strut diameter is 424 um (paper Table/Fig 1) = 3.65 vox at the 58.1 um
CT voxel size; README quotes 350 um = 3.01 vox. Each strut is drawn as a capsule
(cylinder with hemispherical caps) of a chosen radius by thresholding the
distance from each local voxel to the strut's line segment. Rasterization is done
in per-strut local bounding boxes to stay memory-safe on the ~519 M-voxel volume.
"""
import argparse
import json
import os
import time

import numpy as np
import tifffile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_JSON = os.path.join(
    REPO, "data/missing_struts/registered_jsons",
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json",
)
DEFAULT_REF_TIF = os.path.join(REPO, "data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif")
DEFAULT_OUT = os.path.join(REPO, "outputs/design_registered/design_volume.tif")

VOXEL_UM = 58.1
NOMINAL_DIAM_UM = 424.0  # paper Fig. 1; radius = 3.65 vox


def load_lattice(path):
    """Return (positions_zyx float64 (N,3), strut_pairs int64 (M,2), raw dict)."""
    with open(path) as fh:
        raw = json.load(fh)
    pos = np.array([j["position"] for j in raw["junctions"]], dtype=np.float64)[:, [2, 1, 0]]
    struts = np.array([[s["junction0"], s["junction1"]] for s in raw["struts"]], dtype=np.int64)
    return pos, struts, raw


def volume_shape_from_tif(path):
    with tifffile.TiffFile(path) as t:
        return (len(t.pages),) + t.pages[0].shape


def rasterize(pos, struts, shape, radius_vox, progress_every=3000):
    """Paint each strut as a capsule of the given radius into a uint8 volume."""
    vol = np.zeros(shape, dtype=np.uint8)
    Z, Y, X = shape
    pad = int(np.ceil(radius_vox)) + 1
    r2 = radius_vox * radius_vox
    a = pos[struts[:, 0]]
    b = pos[struts[:, 1]]
    t0 = time.time()
    for i in range(len(struts)):
        p0, p1 = a[i], b[i]
        lo = np.floor(np.minimum(p0, p1)).astype(int) - pad
        hi = np.ceil(np.maximum(p0, p1)).astype(int) + pad
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, [Z, Y, X])
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]),
            indexing="ij",
        )
        grid = np.stack([zz, yy, xx], axis=-1).astype(np.float64)  # (...,3)
        seg = p1 - p0
        L2 = seg @ seg
        # projection parameter t of each grid point onto the segment, clamped to [0,1]
        t = ((grid - p0) @ seg) / L2 if L2 > 0 else np.zeros(grid.shape[:-1])
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[..., None] * seg
        d2 = ((grid - closest) ** 2).sum(-1)
        m = d2 <= r2
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[m] = 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i+1}/{len(struts)} struts  ({time.time()-t0:.1f}s)")
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--ref-tif", default=DEFAULT_REF_TIF, help="CT volume that defines the target shape")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--radius-vox", type=float, default=NOMINAL_DIAM_UM / 2 / VOXEL_UM)
    args = ap.parse_args()

    pos, struts, _ = load_lattice(args.json)
    shape = volume_shape_from_tif(args.ref_tif)
    print(f"lattice: {len(pos)} junctions, {len(struts)} struts")
    print(f"target volume: {shape}  radius={args.radius_vox:.2f} vox "
          f"({args.radius_vox*2*VOXEL_UM:.0f} um diameter)")

    vol = rasterize(pos, struts, shape, args.radius_vox)
    frac = vol.mean()
    print(f"design foreground fraction: {frac*100:.3f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tifffile.imwrite(args.out, vol, compression="zlib")
    print(f"saved {os.path.relpath(args.out, REPO)}  ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
