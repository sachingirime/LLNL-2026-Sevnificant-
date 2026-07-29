#!/usr/bin/env python
"""Refit the supplied design->CT registration, which carries a residual scale error.

The registered JSON shipped with the dataset is globally close but NOT good enough for
per-strut sampling: its scale is off by about 1%, so the design drifts away from the CT
as you move across the specimen -- up to ~9 vox at the far corners, against an as-built
strut radius of only ~2.2 vox. Sampled on the supplied coordinates the median strut
centreline sits at EDT depth 0.26 vox, i.e. essentially OUTSIDE the material, and 58% of
strut-interior samples miss material entirely. That is a registration artefact, not a
defect rate, and any per-strut detector run on the raw coordinates measures it.

Method. Split the volume into blocks; in each, take points along the interior of every
strut and find the shift maximising the median SIGNED distance field
(EDT(inside) - EDT(outside)) sampled trilinearly. Signed depth is used rather than a
ball centroid because a ball truncates the strut and biases the estimate toward zero
(it under-reports a real 2-4 vox offset as 0.5 vox), and rather than a binary
on/off-material count because depth is smooth and gives a sub-voxel optimum. Then fit
offset(p) = A p + t by least squares over the blocks.

Typical result on 0point5dash1: scale correction -0.82% z, -0.35% y, -1.15% x, residual
0.23-0.41 vox rms -- i.e. after correction the design sits well inside the strut
everywhere, and strut-interior samples land on material ~99% of the time.

    python scripts/refit_registration.py --out outputs/registration/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates

DEF_MASK = "data/9x9x9_octet_lattice/segmentation/mask.tif"
DEF_JSON = ("data/missing_struts/registered_jsons/"
            "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")


def load_design(path):
    """Junction positions in ARRAY order (z,y,x) plus the strut endpoint index pairs.

    JSON `position` is (x,y,z); reversing is the only permutation that puts every
    junction in bounds for vol[z,y,x].
    """
    J = json.load(open(path))
    pos = np.array([j["position"] for j in J["junctions"]], float)[:, ::-1]
    struts = np.array([(s["junction0"], s["junction1"]) for s in J["struts"]])
    return J, pos, struts


def block_offset(signed, pts, coarse=6.0, step=1.0, fine=0.8, fstep=0.2):
    """Shift maximising median signed depth at `pts`; coarse scan then refinement."""
    def depth(sh):
        return float(np.median(map_coordinates(signed, (pts + sh).T,
                                               order=1, mode="nearest")))
    g = np.arange(-coarse, coarse + 1e-9, step)
    v, a, b, c = max((depth(np.array([a, b, c])), a, b, c)
                     for a in g for b in g for c in g)
    g2 = np.arange(-fine, fine + 1e-9, fstep)
    v, a, b, c = max((depth(np.array([a + i, b + j, c + k])), a + i, b + j, c + k)
                     for i in g2 for j in g2 for k in g2)
    return np.array([a, b, c]), v


def refit(mask_path, json_path, block=170, grid=4, min_struts=40, verbose=True):
    _, pos, struts = load_design(json_path)
    shape = tifffile.TiffFile(mask_path).series[0].shape
    ts = np.linspace(.25, .75, 6)

    # block origins spread over the specimen, clipped so every block is full size
    origins = [np.linspace(0.06 * s, 0.94 * s - block, grid).astype(int)
               for s in shape]
    rows = []
    for z0 in origins[0]:
        slab = tifffile.imread(mask_path, key=range(z0, z0 + block)) > 0
        for y0 in origins[1]:
            for x0 in origins[2]:
                lo = np.array([z0, y0, x0])
                m = slab[:, y0:y0 + block, x0:x0 + block]
                if m.shape != (block,) * 3 or m.mean() < 0.02:
                    continue
                ins = ((pos >= lo + 8) & (pos < lo + block - 8)).all(1)
                sel = struts[ins[struts[:, 0]] & ins[struts[:, 1]]]
                if len(sel) < min_struts:
                    continue
                pa, pb = pos[sel[:, 0]] - lo, pos[sel[:, 1]] - lo
                pts = np.concatenate([pa + t * (pb - pa) for t in ts])
                signed = (distance_transform_edt(m)
                          - distance_transform_edt(~m)).astype(np.float32)
                d, v = block_offset(signed, pts)
                rows.append((lo + block / 2.0, d, v, len(sel)))
                if verbose:
                    c = rows[-1][0]
                    print(f"  block ({c[0]:4.0f},{c[1]:4.0f},{c[2]:4.0f}) "
                          f"struts={len(sel):4d}  offset "
                          f"({d[0]:+5.1f},{d[1]:+5.1f},{d[2]:+5.1f})  depth {v:+.2f}")
    if len(rows) < 8:
        raise RuntimeError(f"only {len(rows)} usable blocks -- cannot fit an affine")

    C = np.array([r[0] for r in rows])
    D = np.array([r[1] for r in rows])
    X = np.hstack([C, np.ones((len(C), 1))])
    A = np.zeros((3, 3))
    t = np.zeros(3)
    resid = np.zeros(3)
    for i in range(3):
        coef, *_ = np.linalg.lstsq(X, D[:, i], rcond=None)
        A[i], t[i] = coef[:3], coef[3]
        resid[i] = (D[:, i] - X @ coef).std()
    return A, t, resid, C, D


def apply_correction(pos, A, t):
    """Corrected = p + A p + t."""
    return pos + pos @ A.T + t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", default=DEF_MASK)
    p.add_argument("--json", dest="json_path", default=DEF_JSON)
    p.add_argument("--block", type=int, default=170)
    p.add_argument("--grid", type=int, default=4, help="blocks per axis")
    p.add_argument("--out", default="outputs/registration",
                   help="directory for correction.json and the corrected design JSON")
    args = p.parse_args()

    A, t, resid, C, D = refit(args.mask, args.json_path, args.block, args.grid)
    print(f"\n{len(C)} blocks")
    for i, n in enumerate("zyx"):
        print(f"  d{n} = {A[i,0]:+.5f}*z {A[i,1]:+.5f}*y {A[i,2]:+.5f}*x "
              f"{t[i]:+7.3f}   residual {resid[i]:.2f} vox rms")
    sc = 1 + np.diag(A)
    print(f"\nscale correction (z,y,x) = {np.round(sc,5)}  "
          f"= {', '.join(f'{100*v:+.2f}%' for v in np.diag(A))}")

    J, pos, struts = load_design(args.json_path)
    pc = apply_correction(pos, A, t)
    shift = np.linalg.norm(pc - pos, axis=1)
    print(f"junction displacement: median {np.median(shift):.2f} vox, "
          f"max {shift.max():.2f} vox")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"A_zyx": A.tolist(), "t_zyx": t.tolist(),
               "residual_rms_zyx": resid.tolist(),
               "scale_correction_zyx": sc.tolist(),
               "note": "corrected_zyx = p + A @ p + t, p in (z,y,x) voxel coords"},
              open(out / "correction.json", "w"), indent=2)

    for j, q in zip(J["junctions"], pc):
        j["position"] = [float(v) for v in q[::-1]]      # back to (x,y,z)
    json.dump(J, open(out / "design_corrected.json", "w"))
    print(f"wrote {out/'correction.json'} and {out/'design_corrected.json'}")


if __name__ == "__main__":
    main()
