#!/usr/bin/env python
"""Turn a CT tif stack into an animated GIF that flips through its slices.

Reads pages lazily (never loads the whole 1 GB volume), windows intensities with
global percentiles sampled from a few pages, downsamples in-plane, and writes a
palettised GIF.

    python scripts/tif_to_gif.py data/missing_struts/tif_stacks/*.tif \
        -o outputs/gifs/ct_z.gif --step 4 --max-dim 512
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def window(vol_sample, lo_pct, hi_pct):
    lo, hi = np.percentile(vol_sample, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def to_u8(sl, lo, hi):
    x = (sl.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample(sl, max_dim):
    f = max(1, int(np.ceil(max(sl.shape) / max_dim)))
    if f == 1:
        return sl
    h, w = (s // f * f for s in sl.shape)
    return sl[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tif")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--axis", type=int, default=0, choices=[0, 1, 2],
                   help="slicing axis; 0 = tif pages (default)")
    p.add_argument("--step", type=int, default=4, help="keep every Nth slice")
    p.add_argument("--max-dim", type=int, default=512, help="max in-plane pixels")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--lo-pct", type=float, default=0.5)
    p.add_argument("--hi-pct", type=float, default=99.8)
    p.add_argument("--bounce", action="store_true", help="play forward then back")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(args.tif) as tf:
        series = tf.series[0]
        shape, dtype = series.shape, series.dtype
        print(f"volume {shape} {dtype}")

        if args.axis == 0:
            # page-wise: read only the pages we need
            idx = range(0, shape[0], args.step)
            probe = np.stack([series.asarray(key=k)
                              for k in np.linspace(0, shape[0] - 1, 8, dtype=int)])
            lo, hi = window(probe, args.lo_pct, args.hi_pct)
            frames = [to_u8(downsample(series.asarray(key=k), args.max_dim), lo, hi)
                      for k in idx]
        else:
            # reslicing needs the full volume in memory
            vol = series.asarray()
            lo, hi = window(vol[::16], args.lo_pct, args.hi_pct)
            vol = np.moveaxis(vol, args.axis, 0)
            frames = [to_u8(downsample(vol[k], args.max_dim), lo, hi)
                      for k in range(0, vol.shape[0], args.step)]

    print(f"window [{lo:.1f}, {hi:.1f}] -> {len(frames)} frames of {frames[0].shape}")

    imgs = [Image.fromarray(f, mode="L").convert("P", palette=Image.ADAPTIVE, colors=256)
            for f in frames]
    if args.bounce:
        imgs += imgs[-2:0:-1]

    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / args.fps)), loop=0, optimize=True)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
