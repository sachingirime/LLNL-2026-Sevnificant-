#!/usr/bin/env python
"""Compare two resolution-appropriate, design-referenced XCT strut detectors.

The first detector samples a radius-3 voxel tube around each registered design
centreline and tests the *run length* of empty cross-sections.  This distinguishes
gross absence from an internal gap; it is intentionally not just a nearest-material
test.  The second detector uses the existing per-strut EDT ridge profile as an
independent gross-absence score.  At this scan's four-voxel strut diameter it never
emits thin/dross/bent labels.

The script writes one common CSV/NPZ table per method plus diagnostic figures.  It
does not tune a cutoff to a published defect rate: cutoffs are fixed geometric rules
and the EDT comparison uses a robust healthy-population reference.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def lattice(json_path):
    g = json.loads(Path(json_path).read_text())
    pos = np.asarray([j["position"] for j in g["junctions"]], float)[:, [2, 1, 0]]
    pairs = np.asarray([[s["junction0"], s["junction1"]] for s in g["struts"]], int)
    edge = np.asarray([s.get("unit_cell_edge_idx", -1) for s in g["struts"]])
    _, counts = np.unique(edge, return_counts=True)
    modal = counts.max()
    rare = set(np.unique(edge)[counts < modal * .5])
    caps = np.asarray([x in rare for x in edge])
    return pos, pairs, caps


def sample_tube(mask, points, radius=3):
    """Boolean occupancy in a spherical tube for points (strut,sample,zyx)."""
    q = np.rint(points).astype(np.int32)
    dz, dy, dx = np.mgrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    offsets = np.c_[dz.ravel(), dy.ravel(), dx.ravel()]
    offsets = offsets[(offsets**2).sum(1) <= radius * radius]
    flat = q.reshape(-1, 3)
    out = np.zeros(len(flat), bool)
    shape = np.asarray(mask.shape)
    # Chunking bounds peak temporary memory even for the 519M-voxel mask.
    for start in range(0, len(flat), 12000):
        p = flat[start:start+12000]
        loc = p[:, None, :] + offsets[None, :, :]
        good = ((loc >= 0) & (loc < shape)).all(2)
        loc = np.clip(loc, 0, shape - 1)
        out[start:start+len(p)] = np.any(mask[loc[..., 0], loc[..., 1], loc[..., 2]] & good, 1)
    return out.reshape(q.shape[:2])


def longest_false_run(x):
    best = cur = 0
    for v in x:
        if not v:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def classify_connectivity(occ, eligible):
    cov = occ.mean(1)
    runs = np.array([longest_false_run(x) for x in occ])
    verdict = np.full(len(cov), "uncertain", dtype="U12")
    # Geometric, preregistered rules: 33 samples over 70% of a 55.8-voxel strut;
    # 4 empty samples correspond to about 4.7 voxels.  Both end samples occupied
    # is required for a disconnected call, making the test topological rather than
    # a local absence score.
    missing = eligible & (cov <= .12)
    disconnected = eligible & ~missing & occ[:, 0] & occ[:, -1] & (runs >= 4)
    nominal = eligible & ~missing & ~disconnected & (cov >= .97)
    verdict[missing] = "missing"
    verdict[disconnected] = "disconnected"
    verdict[nominal] = "nominal"
    confidence = np.maximum(np.abs(cov - .12) / .12, np.abs(cov - .97) / .03)
    confidence = np.clip(confidence, 0, 1)
    return verdict, confidence, cov, runs


def classify_edt(measure, eligible):
    r = measure["r_median"].astype(float)
    cov = measure["material_fraction"].astype(float)
    healthy = eligible & (cov >= .99) & np.isfinite(r) & (r > 0)
    med = np.median(r[healthy]); mad = np.median(np.abs(r[healthy] - med))
    sigma = max(1.4826 * mad, .25)  # the .25-voxel floor avoids a degenerate quantised MAD
    cutoff = med - 3 * sigma
    verdict = np.full(len(r), "uncertain", dtype="U12")
    missing = eligible & (r < cutoff) & (cov <= .20)
    nominal = eligible & (r >= cutoff) & (cov >= .97)
    verdict[missing] = "missing"; verdict[nominal] = "nominal"
    confidence = np.clip(np.abs(r-cutoff) / max(sigma, .25), 0, 1)
    return verdict, confidence, r, cutoff, med, sigma


def write_table(path, ids, caps, plates, mid, verdict, confidence, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["strut_id", "is_boundary_cap", "in_plate_region", "midpoint_z", "midpoint_y", "midpoint_x", "verdict", "confidence"] + list(extra)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for i in range(len(ids)):
            w.writerow([int(ids[i]), int(caps[i]), int(plates[i]), *[f"{v:.3f}" for v in mid[i]], verdict[i], f"{confidence[i]:.3f}", *[f"{extra[k][i]:.5g}" for k in extra]])


def figure(path, title, x, metric, verdict, eligible, ylabel):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].hist(metric[eligible & np.isfinite(metric)], bins=60, color="#718096")
    ax[0].set(title=title, xlabel=ylabel, ylabel="struts")
    # Bins make x-dependent registration drift visible, without implying local thresholds.
    bins = np.linspace(x.min(), x.max(), 19); ctr = (bins[1:] + bins[:-1]) / 2
    for lab, col in [("missing", "#e76f51"), ("disconnected", "#457b9d"), ("uncertain", "#e9c46a")]:
        rate = [np.mean(verdict[eligible & (x >= a) & (x < b)] == lab) for a, b in zip(bins[:-1], bins[1:])]
        ax[1].plot(ctr, rate, marker="o", ms=3, label=lab, color=col)
    ax[1].set(title="Flag rate versus x (drift diagnostic)", xlabel="midpoint x [vox]", ylabel="fraction")
    ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="data/9x9x9_octet_lattice/segmentation/mask.tif")
    ap.add_argument("--design", default="data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
    ap.add_argument("--measure", default="outputs/measure/struts_real_0point5dash1.npz")
    ap.add_argument("--out", default="outputs/method_comparison")
    args = ap.parse_args(); t0 = time.time(); out = Path(args.out)
    pos, pairs, caps = lattice(args.design); n = len(pairs); ids = np.arange(n)
    mid = (pos[pairs[:, 0]] + pos[pairs[:, 1]]) / 2
    plates = (mid[:, 0] < 95) | (mid[:, 0] > 665)
    eligible = ~caps & ~plates
    measure = np.load(args.measure)
    print(f"reading mask {args.mask}")
    mask = tifffile.imread(args.mask).astype(bool)
    t = np.linspace(.15, .85, 33)
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    points = a[:, None, :] + (b-a)[:, None, :] * t[None, :, None]
    occ = sample_tube(mask, points); del mask
    vc, cc, cov, run = classify_connectivity(occ, eligible)
    ve, ce, rad, cutoff, med, sigma = classify_edt(measure, eligible)
    common = dict(strut_id=ids, is_boundary_cap=caps, in_plate_region=plates, midpoint=mid)
    for name, v, c, extra in [
        ("tube_connectivity", vc, cc, {"tube_occupancy": cov, "longest_empty_run_samples": run}),
        ("edt_radius", ve, ce, {"radius_median_vox": rad, "material_fraction": measure["material_fraction"]}),
    ]:
        d = out / name
        write_table(d / "struts.csv", ids, caps, plates, mid, v, c, extra)
        np.savez_compressed(d / "raw_measurements.npz", **common, pairs=pairs, verdict=v, confidence=c, **extra)
        metric = cov if name == "tube_connectivity" else rad
        ylabel = "tube occupancy" if name == "tube_connectivity" else "EDT radius [vox]"
        figure(d / "diagnostics.png", name.replace("_", " "), mid[:, 2], metric, v, eligible, ylabel)
    agree = (vc == ve) & eligible
    np.savez_compressed(out / "comparison_arrays.npz", eligible=eligible, tube_verdict=vc, edt_verdict=ve, midpoint=mid,
                        tube_occupancy=cov, longest_empty_run=run, edt_radius=rad, edt_cutoff=cutoff)
    (out / "run_metadata.json").write_text(json.dumps({"n_struts": int(n), "eligible": int(eligible.sum()), "samples": 33,
        "tube_radius_vox": 3, "edt_healthy_median_vox": float(med), "edt_robust_sigma_vox": float(sigma), "edt_missing_cutoff_vox": float(cutoff),
        "elapsed_seconds": time.time()-t0}, indent=2))
    print(f"eligible={eligible.sum()} agreement={agree.sum()}/{eligible.sum()} EDT cutoff={cutoff:.3f}; wrote {out}")

if __name__ == "__main__":
    main()
