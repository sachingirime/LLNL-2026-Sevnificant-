#!/usr/bin/env python
"""Per-strut measurement table from a CT volume plus a registered design graph.

For every strut in the design graph, samples along its centreline and records:

  radius   local half-thickness from the Euclidean distance transform (voxels)
  material fraction of samples sitting on material (with a tolerance ball)
  intensity mean raw CT value inside the strut -- porosity / lack of fusion, the one
            quantity the binary mask throws away
  tortuosity arc length / chord length of the sampled path (bent struts)

Radius is recorded PER SAMPLE, never pooled over the volume: a single thin strut among
~13000 healthy ones moves a global median by nothing (measured: ratio 1.000), so the
whole point is per-strut attribution.

The volume is processed in tiles because a 1200^3 float32 CT is 6.9 GB and its distance
transform is larger still. EDT is nominally global, but strut radii are a few voxels, so
a tile halo of a few tens of voxels is exact in practice.

    python scripts/measure_struts.py <ct.npy|.tif> <design.json> -o table.npz \
        --scale 49.5 --offset 202,198,198
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_lazy(path):
    if str(path).endswith(".npy"):
        return np.load(path, mmap_mode="r")
    import tifffile
    try:
        return tifffile.memmap(path)
    except (ValueError, OSError):
        return tifffile.imread(path)


def design_samples(design_json, scale, offset, n_samples, t_lo, t_hi):
    """Centreline sample points in voxel coordinates, shape (n_struts, n_samples, 3)."""
    g = json.load(open(design_json))
    pos = np.array([j["position"] for j in g["junctions"]], dtype=np.float64)
    pos = pos[:, [2, 1, 0]] * scale + offset          # (x,y,z) -> (z,y,x), to voxels
    pairs = np.array([[s["junction0"], s["junction1"]] for s in g["struts"]],
                     dtype=np.int64)
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    t = np.linspace(t_lo, t_hi, n_samples)
    pts = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
    return pts, a, b, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("design_json")
    p.add_argument("-o", "--out", required=True, help="output .npz")
    p.add_argument("--scale", type=float, required=True,
                   help="voxels per design unit")
    p.add_argument("--offset", default="0,0,0", help="z,y,x voxel offset")
    p.add_argument("--threshold", type=float, default=None,
                   help="segmentation threshold; default = Otsu on a strided sample")
    p.add_argument("--samples", type=int, default=33)
    p.add_argument("--t-range", default="0.15,0.85",
                   help="fractional span sampled along each strut (skips node blobs)")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--halo", type=int, default=24,
                   help="tile overlap; must exceed the largest radius of interest")
    args = p.parse_args()

    vol = load_lazy(args.volume)
    Z, Y, X = vol.shape
    offset = np.array([float(v) for v in args.offset.split(",")])
    t_lo, t_hi = (float(v) for v in args.t_range.split(","))

    if args.threshold is None:
        from skimage.filters import threshold_otsu
        s = np.asarray(vol[::8, ::8, ::8], dtype=np.float32)
        thr = float(threshold_otsu(s))
        print(f"Otsu threshold (8x strided): {thr:.6g}")
    else:
        thr = args.threshold

    pts, a, b, pairs = design_samples(args.design_json, args.scale, offset,
                                      args.samples, t_lo, t_hi)
    n_struts, n_s = pts.shape[:2]
    print(f"volume {vol.shape} {vol.dtype} | {n_struts} struts x {n_s} samples")

    idx = np.round(pts).astype(np.int64)
    inb = ((idx >= 0) & (idx < np.array([Z, Y, X]))).all(2)
    radius = np.full((n_struts, n_s), np.nan, dtype=np.float32)
    onmat = np.zeros((n_struts, n_s), dtype=bool)
    inten = np.full((n_struts, n_s), np.nan, dtype=np.float32)

    from scipy.ndimage import (binary_dilation, distance_transform_edt,
                               maximum_filter)
    # search ball: radius 2 voxels, covering registration scatter without reaching a
    # neighbouring strut (struts are ~2 vox radius, spaced ~56 vox apart)
    rr = 2
    o = np.arange(-rr, rr + 1)
    gz, gy, gx = np.meshgrid(o, o, o, indexing="ij")
    global _BALL
    _BALL = (gz**2 + gy**2 + gx**2) <= rr * rr

    T, H = args.tile, args.halo
    tiles = [(z, y, x) for z in range(0, Z, T) for y in range(0, Y, T)
             for x in range(0, X, T)]
    print(f"{len(tiles)} tiles of {T}^3 with halo {H}")
    for n, (z0, y0, x0) in enumerate(tiles, 1):
        z1, y1, x1 = min(Z, z0 + T), min(Y, y0 + T), min(X, x0 + T)
        # which samples fall in this tile's interior?
        sel = (inb
               & (idx[..., 0] >= z0) & (idx[..., 0] < z1)
               & (idx[..., 1] >= y0) & (idx[..., 1] < y1)
               & (idx[..., 2] >= x0) & (idx[..., 2] < x1))
        if not sel.any():
            continue
        az0, ay0, ax0 = max(0, z0 - H), max(0, y0 - H), max(0, x0 - H)
        az1, ay1, ax1 = min(Z, z1 + H), min(Y, y1 + H), min(X, x1 + H)
        block = np.asarray(vol[az0:az1, ay0:ay1, ax0:ax1], dtype=np.float32)
        mask = block > thr
        edt = distance_transform_edt(mask).astype(np.float32)
        # The radius IS the EDT ridge value, and the EDT is 0 outside material. A design
        # centreline sits a voxel or two off the true medial axis, so reading one rounded
        # voxel returns 0 for a perfectly healthy strut. Take the local maximum instead:
        # that finds the ridge without assuming the centreline is voxel-exact.
        edt_peak = maximum_filter(edt, footprint=_BALL)
        near = binary_dilation(mask, structure=_BALL)
        loc = idx[sel] - np.array([az0, ay0, ax0])
        radius[sel] = edt_peak[loc[:, 0], loc[:, 1], loc[:, 2]]
        onmat[sel] = near[loc[:, 0], loc[:, 1], loc[:, 2]]
        inten[sel] = block[loc[:, 0], loc[:, 1], loc[:, 2]]
        if n % max(1, len(tiles) // 10) == 0 or n == len(tiles):
            done = np.isfinite(radius).sum()
            print(f"  {n}/{len(tiles)} tiles  samples filled {done}/{radius.size}")

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(radius, 1)
        rmin = np.nanmin(radius, 1)
        rmax = np.nanmax(radius, 1)
    matfrac = onmat.mean(1)
    imean = np.nanmean(np.where(onmat, inten, np.nan), 1)
    chord = np.linalg.norm(b - a, axis=1)
    arc = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
    # normalise by the SAMPLED chord, not the full strut: sampling t in [0.15,0.85] covers
    # only 70% of the chord, which made every straight strut report tortuosity 0.70.
    sampled_chord = np.linalg.norm(pts[:, -1, :] - pts[:, 0, :], axis=1)
    tort = np.where(sampled_chord > 0, arc / np.maximum(sampled_chord, 1e-9), np.nan)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, strut_id=np.arange(n_struts), pairs=pairs,
                        radius=radius, r_median=med, r_min=rmin, r_max=rmax,
                        material_fraction=matfrac, intensity_mean=imean,
                        tortuosity=tort, chord=chord, threshold=thr,
                        scale=args.scale, offset=offset)
    good = np.isfinite(med)
    print(f"\nwrote {args.out}  ({int(good.sum())}/{n_struts} struts measured)")
    print(f"  r_median : p1 {np.nanpercentile(med,1):.2f}  median {np.nanmedian(med):.2f}"
          f"  p99 {np.nanpercentile(med,99):.2f}")
    print(f"  material_fraction: p1 {np.percentile(matfrac,1):.3f}  median "
          f"{np.median(matfrac):.3f}")
    print(f"  tortuosity: median {np.nanmedian(tort):.4f}  max {np.nanmax(tort):.4f}")


if __name__ == "__main__":
    main()
