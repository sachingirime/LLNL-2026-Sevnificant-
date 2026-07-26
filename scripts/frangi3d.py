#!/usr/bin/env python
"""3D Frangi vesselness over a CT volume, tiled so it fits in RAM.

A full 761x815x837 volume needs ~20-40 GB for the Hessian in one pass, so this splits
it into cubic tiles with a halo, filters each independently, and writes back only the
tile interior. Tiles are processed in parallel across cores.

Why 3D and not per-slice: the lattice is oblique to every slice plane, so a strut cuts
a slice as an ellipse and any 2D ridge measure tracks strut TILT as much as strut
health. In 3D the vesselness also gains the plate term R_A = |l2|/|l3|, which is what
separates a strut (line) from a sheet.

    python scripts/frangi3d.py <ct.tif> -o outputs/frangi3d/frangi.tif --sigmas 2
    python scripts/frangi3d.py <ct.tif> -o out.tif --crop 280,300,300,192   # test cube
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

_CT = None  # per-worker cache of the (memmapped) input


def _init(ct_path, crop):
    global _CT
    _CT = _open(ct_path, crop)


def _open(ct_path, crop):
    if ct_path.endswith(".npy"):
        vol = np.load(ct_path, mmap_mode="r")
    else:
        try:
            vol = tifffile.memmap(ct_path)
        except (ValueError, OSError):
            vol = tifffile.imread(ct_path)
    if crop:
        z, y, x, s = crop
        vol = vol[z:z + s, y:y + s, x:x + s]
    return vol


def _block(job, halo, lo, hi):
    """Load one haloed tile, scaled to [0,1] with GLOBAL intensity bounds."""
    (z0, z1), (y0, y1), (x0, x1) = job[:3]
    Z, Y, X = _CT.shape
    az0, az1 = max(0, z0 - halo), min(Z, z1 + halo)
    ay0, ay1 = max(0, y0 - halo), min(Y, y1 + halo)
    ax0, ax1 = max(0, x0 - halo), min(X, x1 + halo)
    blk = np.asarray(_CT[az0:az1, ay0:ay1, ax0:ax1], dtype=np.float32)
    return np.clip((blk - lo) / (hi - lo), 0.0, 1.0), (az0, ay0, ax0)


def _smax(job):
    """Pass 1: max Hessian eigenvalue norm ||lambda|| over this tile.

    skimage sets gamma = s.max()/2 from whatever array it is handed, so left to itself
    every tile gets a DIFFERENT gamma and responses are not comparable between tiles --
    tiles containing the dense specimen boundary get a huge gamma and collapse to zero.
    We reduce s.max() globally here and pass one fixed gamma to pass 2.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    (z0, z1), (y0, y1), (x0, x1), halo, sigmas, lo, hi = job
    block, (az0, ay0, ax0) = _block(job, halo, lo, hi)
    best = 0.0
    for sig in sigmas:
        H = hessian_matrix(block, sigma=sig, mode="reflect",
                           use_gaussian_derivatives=True)
        ev = hessian_matrix_eigvals(H)
        s = np.sqrt((ev ** 2).sum(0))
        # interior only -- halo voxels are another tile's business
        s = s[z0 - az0:z1 - az0, y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
        best = max(best, float(s.max()))
    return best


def _tile(job):
    """Pass 2: filter one tile with the fixed global gamma, return its interior."""
    from skimage.filters import frangi

    (z0, z1), (y0, y1), (x0, x1), halo, sigmas, lo, hi, gamma = job
    block, (az0, ay0, ax0) = _block(job, halo, lo, hi)
    out = frangi(block, sigmas=sigmas, black_ridges=False,
                 gamma=gamma).astype(np.float32)
    return (z0, z1, y0, y1, x0, x1,
            out[z0 - az0:z1 - az0, y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ct")
    p.add_argument("-o", "--out", required=True, help="output .tif (float32, zlib)")
    p.add_argument("--sigmas", default="2",
                   help="comma-separated scales in voxels (default 2; printed struts "
                        "measure ~2 vox radius). Multi-scale costs ~1 pass per sigma.")
    p.add_argument("--tile", type=int, default=128, help="tile edge in voxels")
    p.add_argument("--workers", type=int, default=12,
                   help="parallel tiles; each needs ~6 float32 copies of a haloed tile")
    p.add_argument("--crop", default="", help="z0,y0,x0,size to filter a test cube only")
    p.add_argument("--pct", default="0.5,99.8",
                   help="global percentiles for intensity scaling (default 0.5,99.8)")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="fixed Frangi gamma. Default 0 = derive it in a first pass over "
                        "all tiles (global s.max()/2). NEVER leave skimage to pick it "
                        "per tile: it uses that tile's own s.max()/2, so whole tiles "
                        "collapse to zero and responses are incomparable.")
    p.add_argument("--uncompressed", action="store_true",
                   help="write a plain (memory-mappable) tif; zlib output cannot be "
                        "tifffile.memmap'd and must be read whole (~2.1 GB)")
    args = p.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    crop = [int(v) for v in args.crop.split(",")] if args.crop else None
    halo = int(np.ceil(4 * max(sigmas)))  # Gaussian derivative support

    vol = _open(args.ct, crop)
    Z, Y, X = vol.shape
    lo_p, hi_p = (float(v) for v in args.pct.split(","))
    # Global window from a strided sample -- must not be per-tile (see _tile).
    s = vol[::8, ::8, ::8]
    lo, hi = (float(v) for v in np.percentile(np.asarray(s, dtype=np.float32), [lo_p, hi_p]))
    print(f"volume {vol.shape} {vol.dtype}  sigmas={sigmas}  halo={halo}  "
          f"window=[{lo:.0f},{hi:.0f}]")

    T = args.tile
    jobs = [((z, min(Z, z + T)), (y, min(Y, y + T)), (x, min(X, x + T)),
             halo, sigmas, lo, hi)
            for z in range(0, Z, T) for y in range(0, Y, T) for x in range(0, X, T)]
    print(f"{len(jobs)} tiles of {T}^3, {args.workers} workers")

    out = np.zeros((Z, Y, X), dtype=np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.ct, crop)) as ex:
        if args.gamma > 0:
            gamma = args.gamma
            print(f"pass 1 skipped, gamma={gamma:g} (given)")
        else:
            smax = 0.0
            for n, v in enumerate(ex.map(_smax, jobs), 1):
                smax = max(smax, v)
                if n % max(1, len(jobs) // 5) == 0 or n == len(jobs):
                    el = time.time() - t0
                    print(f"  pass1 {n}/{len(jobs)}  {el:.0f}s  running s.max={smax:.4g}")
            gamma = smax / 2.0 or 1.0
            print(f"pass 1 done: global s.max={smax:.6g} -> gamma={gamma:.6g}  "
                  f"(pass this as --gamma to reproduce)")

        jobs2 = [j + (gamma,) for j in jobs]
        t1 = time.time()
        for n, (z0, z1, y0, y1, x0, x1, blk) in enumerate(ex.map(_tile, jobs2), 1):
            out[z0:z1, y0:y1, x0:x1] = blk
            if n % max(1, len(jobs) // 10) == 0 or n == len(jobs):
                el = time.time() - t1
                print(f"  pass2 {n}/{len(jobs)}  {el:.0f}s  (eta {el/n*(len(jobs)-n):.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.out, out,
                     **({} if args.uncompressed else {"compression": "zlib"}))
    nz = float((out > 1e-4).mean())
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.0f} MB, "
          f"max={out.max():.3f}, {100*nz:.1f}% of voxels > 1e-4, gamma={gamma:.6g})  "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
