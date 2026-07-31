"""Low-pass / high-pass / band-pass filtering of a CT volume, out of core.

Three separations, all built from one Gaussian:

    low-pass   G(v, sigma)                 what is left when detail is removed
    high-pass  v - G(v, sigma)             what the low-pass threw away
    band-pass  G(v, s_lo) - G(v, s_hi)     difference of Gaussians, keeps one scale

WHY GAUSSIAN AND NOT A FOURIER BUTTERWORTH. The 9x9x9 volume is 519 M voxels: 1.04 GB as
uint16, 2.08 GB as float32. An FFT filter needs the whole array resident and a complex
copy besides -- 4 GB minimum before any workspace, on a machine that had 4 GB free. A
Gaussian is separable and local, so it can be run on z slabs with a halo and streamed
straight to disk, which is what `filter_volume` does. The price is a soft roll-off instead
of a sharp cutoff; for looking at structure that is not a real loss, and a sharp cutoff
would ring at every strut edge anyway.

THE OUTPUT DTYPE IS NOT A DETAIL, IT IS THE WHOLE CORRECTNESS QUESTION. A high-pass is
SIGNED -- it is a residual about zero, and roughly half its voxels are negative. Writing it
to uint16 does not compress it, it deletes the negative half. So:

    low-pass    uint16 is safe: the output is a weighted mean of the input, so it stays
                inside the input's range. Same 1.04 GB as the source, opens in ImageJ.
    high-pass   float32 or int16-with-offset. Nothing unsigned is honest.
    band-pass   same as high-pass; it is a difference.

`filter_volume` refuses to write a signed result to an unsigned type rather than clipping
it quietly.

TIFF vs NPY. Both are supported for input and output and neither is more accurate. .tif is
the better default because ImageJ/Fiji opens it directly and every other file in this
project is one; .npy is marginally faster to memory-map from numpy and carries dtype and
shape with no convention to agree on. Float32 doubles the file either way -- a filtered
9x9x9 lands at 2.08 GB.
"""
from __future__ import annotations

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

# Beyond ~4 sigma a Gaussian contributes below the 16-bit quantisation of this data, so a
# 4-sigma halo makes the slabbed result identical to filtering the whole volume at once.
HALO_SIGMAS = 4.0
SIGNED_MODES = ("highpass", "bandpass")
# Float32 bytes for one slab INCLUDING its halo. `gaussian_filter` allocates its own
# output and works in a temporary, so the resident set runs a few times this; 256 MB keeps
# the whole thing under ~1 GB, which is what makes a 519 M-voxel volume filterable on a
# machine with a few GB free. Sizing the slab from the halo alone does not: at sigma 6 that
# gives 192 slices, a 655 MB working array and a 5 GB peak.
SLAB_BUDGET_BYTES = 256 << 20


def _open_output(path, shape, dtype):
    """A writable on-disk array, so the result never has to be resident."""
    if str(path).lower().endswith(".npy"):
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return tifffile.memmap(path, shape=shape, dtype=dtype)


def plan(shape, mode, sigma, sigma_high=None, out_dtype="float32", slab=None,
         budget_bytes=SLAB_BUDGET_BYTES):
    """Everything decided before a voxel is touched, so it can be reported or refused."""
    if mode not in ("lowpass", "highpass", "bandpass"):
        raise ValueError(f"mode must be lowpass, highpass or bandpass (got {mode!r})")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma})")
    if mode == "bandpass":
        if sigma_high is None or sigma_high <= sigma:
            raise ValueError("bandpass needs sigma_high > sigma (the wider Gaussian)")
    signed = mode in SIGNED_MODES
    dt = np.dtype(out_dtype)
    if signed and dt.kind == "u":
        raise ValueError(
            f"{mode} is a residual about zero -- about half its voxels are negative -- so "
            f"it cannot be written as {dt.name} without deleting them. Use float32, or "
            f"int16 if you want the smaller file.")
    widest = sigma_high if mode == "bandpass" else sigma
    halo = int(np.ceil(HALO_SIGMAS * widest))
    plane = int(shape[1]) * int(shape[2]) * 4          # one z slice as float32
    if slab is None:
        # Fit slab + its two halos inside the budget. A big sigma buys a big halo, which
        # eats the budget, so the slab shrinks and there are more passes -- the right
        # trade, since exceeding memory is fatal and extra passes are merely slower.
        slab = int(budget_bytes // max(plane, 1)) - 2 * halo
        slab = max(slab, 8)
    slab = int(slab)
    return {"mode": mode, "signed": signed, "sigma": float(sigma),
            "sigma_high": None if sigma_high is None else float(sigma_high),
            "halo": halo, "slab": slab, "dtype": dt.name,
            "out_bytes": int(np.prod(shape)) * dt.itemsize,
            "slab_bytes": (slab + 2 * halo) * plane,
            "n_passes": int(np.ceil(int(shape[0]) / max(slab, 1)))}


def filter_volume(volume, out_path, mode="lowpass", sigma=2.0, sigma_high=None,
                  out_dtype="float32", slab=None, log=print):
    """Filter `volume` into `out_path` in z slabs. Returns (plan, stats).

    `volume` may be a memmap; only one slab plus its halo is ever resident.
    """
    p = plan(volume.shape, mode, sigma, sigma_high, out_dtype, slab)
    out = _open_output(out_path, volume.shape, np.dtype(p["dtype"]))
    nz = volume.shape[0]
    halo, step = p["halo"], p["slab"]
    lo_hits = hi_hits = 0
    vmin, vmax, vsum, vsq, n = np.inf, -np.inf, 0.0, 0.0, 0

    for z0 in range(0, nz, step):
        z1 = min(z0 + step, nz)
        a, b = max(0, z0 - halo), min(nz, z1 + halo)
        chunk = np.asarray(volume[a:b], np.float32)
        if mode == "lowpass":
            res = gaussian_filter(chunk, sigma)
        elif mode == "highpass":
            res = chunk - gaussian_filter(chunk, sigma)
        else:
            res = gaussian_filter(chunk, sigma) - gaussian_filter(chunk, sigma_high)
        res = res[z0 - a:z1 - a]

        info = np.iinfo(np.dtype(p["dtype"])) if np.dtype(p["dtype"]).kind in "iu" else None
        if info is not None:
            lo_hits += int((res < info.min).sum())
            hi_hits += int((res > info.max).sum())
            res = np.clip(res, info.min, info.max)
        out[z0:z1] = res.astype(p["dtype"], copy=False)

        vmin = min(vmin, float(res.min()))
        vmax = max(vmax, float(res.max()))
        vsum += float(res.sum())
        vsq += float(np.square(res, dtype=np.float64).sum())
        n += res.size
        log(f"  z {z0}-{z1} of {nz}")

    if hasattr(out, "flush"):
        out.flush()
    mean = vsum / max(n, 1)
    stats = {"min": vmin, "max": vmax, "mean": mean,
             "std": float(np.sqrt(max(vsq / max(n, 1) - mean * mean, 0.0))),
             "clipped_low": lo_hits, "clipped_high": hi_hits}
    return p, stats
