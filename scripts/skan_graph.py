#!/usr/bin/env python
"""As-built lattice graph from a CT stack, via skeletonisation + skan.

Produces nodes and edges from the SEGMENTATION alone -- no design file, no
registration. That is the point of it: everything else in this repo measures the
CT against the registered design, so this is the one representation that can
disagree with the design rather than inherit it.

    python scripts/skan_graph.py <ct.tif> -o outputs/skan_graph/graph.json
    python scripts/skan_graph.py <ct.tif> -o out.json --mask <mask.tif>   # skip Otsu

Three things this does that a naive skeletonize->skan does not, each of which
changed the answer when it was added (see notes at each step):

1.  Skeletonises in z-slabs with a 64-voxel halo. The medial axis is local -- a
    voxel is on it iff it has two or more nearest boundary points, which depends
    only on its r-neighbourhood, and r here is ~7 voxels. A 64-voxel halo is an
    order of magnitude past that, so slabbing is exact for this geometry and
    keeps peak RAM under ~4 GB instead of OOMing on 519 M voxels.
2.  Prunes short endpoint branches iteratively. The raw skeleton is a hairball
    of surface-roughness spurs: ~460 paths per 75 struts before pruning.
3.  Clusters junction endpoints into nodes with a KD-tree + connected
    components, NOT a single global EDT radius cut. The octet has degree-3
    corners and degree-8 face centres, and no single radius serves both --
    that is what left scripts/strut_graph.py unsolved.

The build plates at both z ends are solid metal and skeletonise to a sheet, not
a curve, which buries the lattice graph in ~10^5 junk paths. They are detected
by per-slice foreground fraction and excluded; --zrange overrides.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HALO = 64          # >> max strut radius (~7 vox); see module docstring
MAX_RADIUS = 24    # EDT halo; nothing in this part is thicker than this


def log(t0, msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- input -----
def load_mask(ct_path, mask_path, threshold, t0):
    if mask_path:
        m = tifffile.imread(mask_path) > 0
        log(t0, f"mask {mask_path} {m.shape} fg={m.mean() * 100:.2f}%")
        return m
    vol = tifffile.imread(ct_path)
    if threshold <= 0:
        from skimage.filters import threshold_otsu
        threshold = float(threshold_otsu(vol[::4, ::4, ::4]))
        log(t0, f"Otsu on 4x-strided subsample -> {threshold:.0f}")
    m = vol >= threshold
    del vol
    log(t0, f"mask from {Path(ct_path).name} thr={threshold:.0f} "
            f"fg={m.mean() * 100:.2f}%")
    return m


def find_lattice_z(mask, factor, t0):
    """Trim the solid build plates: keep slices whose foreground is within
    `factor` of the lattice-level median. The plate taper is oblique, so this
    is a conservative cut and some plate material survives at the ends."""
    f = mask.reshape(mask.shape[0], -1).mean(1)
    med = np.median(f)
    ok = np.flatnonzero(f <= factor * med)
    z0, z1 = int(ok.min()), int(ok.max()) + 1
    log(t0, f"per-slice fg: median {med * 100:.2f}%, "
            f"ends {f[0] * 100:.1f}%/{f[-1] * 100:.1f}%  -> lattice z=[{z0},{z1})")
    return z0, z1


# ------------------------------------------------------------ skeleton ------
def skeletonize_slabs(mask, slab, t0):
    """Slab-wise 3D thinning with a halo; see module docstring for why exact."""
    from skimage.morphology import skeletonize
    Z = mask.shape[0]
    out = np.zeros_like(mask)
    for z0 in range(0, Z, slab):
        z1 = min(Z, z0 + slab)
        a0, a1 = max(0, z0 - HALO), min(Z, z1 + HALO)
        sk = skeletonize(np.ascontiguousarray(mask[a0:a1]))
        out[z0:z1] = sk[z0 - a0:z1 - a0]
        log(t0, f"  skeleton z[{z0}:{z1}) of {Z}  ({sk.sum():,} vox in slab)")
        del sk
    return out


def edt_slabs(mask, slab, t0):
    """Same trick for the EDT -- exact wherever the true distance < MAX_RADIUS,
    which covers every strut and node in this part."""
    Z = mask.shape[0]
    out = np.zeros(mask.shape, np.float32)
    for z0 in range(0, Z, slab):
        z1 = min(Z, z0 + slab)
        a0, a1 = max(0, z0 - MAX_RADIUS), min(Z, z1 + MAX_RADIUS)
        d = ndi.distance_transform_edt(mask[a0:a1]).astype(np.float32)
        out[z0:z1] = d[z0 - a0:z1 - a0]
        del d
    log(t0, f"EDT done, max radius {out.max():.2f} vox")
    return out


def prune(sk, min_len, t0, rounds=8):
    """Drop junction-to-endpoint branches shorter than min_len, repeatedly:
    removing a spur can expose another. Converges in ~3 rounds here."""
    from skan import Skeleton, summarize
    for i in range(rounds):
        S = Skeleton(sk)
        df = summarize(S, separator="_")
        bad = np.flatnonzero((df["branch_type"] == 1) &
                             (df["branch_distance"] < min_len))
        log(t0, f"  prune round {i}: {len(df):,} paths, {len(bad):,} spurs")
        if len(bad) == 0:
            return sk, S, df
        sk = S.prune_paths(bad).skeleton_image.astype(bool)
    S = Skeleton(sk)
    return sk, S, summarize(S, separator="_")


# --------------------------------------------------------------- graph ------
def build_graph(S, df, cluster_r, t0):
    """Junction-to-junction paths become edges; their endpoints, clustered
    within cluster_r, become nodes."""
    jj = df[df["branch_type"] == 2]
    src = jj[["coord_src_0", "coord_src_1", "coord_src_2"]].to_numpy(float)
    dst = jj[["coord_dst_0", "coord_dst_1", "coord_dst_2"]].to_numpy(float)
    pts = np.vstack([src, dst])
    if len(pts) == 0:
        raise SystemExit("no junction-to-junction paths -- nothing to build")

    # single-linkage clustering, done as connected components of the
    # within-radius pair graph. scipy's linkage() is O(n^2) memory and dies
    # well before the ~10^5 endpoints a full lattice produces.
    tree = cKDTree(pts)
    pairs = tree.query_pairs(cluster_r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                   shape=(len(pts), len(pts)))
    n, lab = connected_components(g, directed=False)
    cen = np.zeros((n, 3))
    np.add.at(cen, lab, pts)
    cen /= np.bincount(lab, minlength=n)[:, None]
    log(t0, f"{len(jj):,} junction-junction paths -> {n:,} nodes "
            f"(from {len(pts):,} endpoints)")

    a, b = lab[:len(src)], lab[len(src):]
    idx = jj.index.to_numpy()
    keep = a != b                                   # drop self-loops
    a, b, idx = a[keep], b[keep], idx[keep]

    # collapse parallel paths between the same node pair to the shortest --
    # a skeleton loop around a junction is one strut, not two. Rare (~0.1%)
    # but it would show up as phantom extra connectivity.
    order = np.argsort(df.loc[idx, "branch_distance"].to_numpy(), kind="stable")
    pair = np.sort(np.c_[a, b], axis=1)[order]
    _, first = np.unique(pair, axis=0, return_index=True)
    sel = np.sort(order[first])
    if len(sel) < len(a):
        log(t0, f"  collapsed {len(a) - len(sel):,} parallel paths")
    return cen, a[sel], b[sel], idx[sel]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ct")
    p.add_argument("-o", "--out", required=True, help="output .json")
    p.add_argument("--mask", default="", help="precomputed mask.tif (skips Otsu)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="fixed CT threshold; 0 = Otsu on a strided subsample")
    p.add_argument("--min-object", type=int, default=64,
                   help="drop connected components below this many voxels")
    p.add_argument("--prune", type=float, default=10.0,
                   help="spur length cut in voxels. 5/10/15 give the same graph "
                        "on this part; below ~3 the hairball survives")
    p.add_argument("--cluster", type=float, default=6.0,
                   help="junction-endpoint clustering radius in voxels; should "
                        "be about the as-built node inscribed radius (6.0 here)")
    p.add_argument("--slab", type=int, default=128, help="z-slab height")
    p.add_argument("--zrange", default="", help="z0,z1 to override plate detection")
    p.add_argument("--plate-factor", type=float, default=2.0,
                   help="a slice is plate if its foreground exceeds this times "
                        "the median slice foreground")
    p.add_argument("--voxel-um", type=float, default=58.196,
                   help="voxel size for the micron columns (derived elsewhere "
                        "from strut length vs the known octet cell)")
    p.add_argument("--margin", type=int, default=60,
                   help="nodes within this many voxels of the cropped volume "
                        "face are flagged boundary=true. Must be at least ONE "
                        "STRUT LENGTH (55.4 vox here), not a few voxels: a node "
                        "closer than that has struts running out of the crop, "
                        "so it reads a false low degree and would be counted as "
                        "a defect. At margin=10 the degree histogram picked up "
                        "a spurious tail at 7-11.")
    args = p.parse_args()

    t0 = time.time()
    mask = load_mask(args.ct, args.mask, args.threshold, t0)
    shape_full = mask.shape

    if args.zrange:
        z0, z1 = (int(v) for v in args.zrange.split(","))
        log(t0, f"z range [{z0},{z1}) (given)")
    else:
        z0, z1 = find_lattice_z(mask, args.plate_factor, t0)
    mask = np.ascontiguousarray(mask[z0:z1])

    lab, nlab = ndi.label(mask, np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    # LUT, not np.isin -- isin over 4x10^8 voxels against ~10^5 labels is a
    # sort per element and takes minutes
    mask = (sizes >= args.min_object)[lab]
    log(t0, f"{nlab:,} components -> {int((sizes >= args.min_object).sum()):,} "
            f"kept (>={args.min_object} vox), largest holds "
            f"{sizes.max() / sizes.sum() * 100:.1f}% of material")
    del lab, sizes

    sk = skeletonize_slabs(mask, args.slab, t0)
    log(t0, f"skeleton {sk.sum():,} voxels")
    edt = edt_slabs(mask, args.slab, t0)
    del mask

    sk, S, df = prune(sk, args.prune, t0)
    cen, ea, eb, pidx = build_graph(S, df, args.cluster, t0)

    # ---- per-edge geometry -------------------------------------------------
    dist = df["branch_distance"].to_numpy()
    eucl = df["euclidean_distance"].to_numpy()
    rad_mid, rad_min = np.zeros(len(pidx)), np.zeros(len(pidx))
    for k, i in enumerate(pidx):
        c = np.round(S.path_coordinates(i)).astype(int)
        r = edt[c[:, 0], c[:, 1], c[:, 2]]
        lo, hi = int(0.2 * len(r)), max(int(0.8 * len(r)), int(0.2 * len(r)) + 1)
        rad_mid[k] = float(np.median(r[lo:hi]))     # trim 20% -- nodes are 3x
        rad_min[k] = float(r[lo:hi].min())          # fatter and would dominate
    log(t0, f"per-edge radii done ({len(pidx):,} edges)")

    deg = np.bincount(np.concatenate([ea, eb]), minlength=len(cen))
    rn = np.round(cen).astype(int)
    rad_node = edt[rn[:, 0], rn[:, 1], rn[:, 2]]
    lo_m = np.array([0, 0, 0]) + args.margin
    hi_m = np.array([z1 - z0, shape_full[1], shape_full[2]]) - args.margin
    bnd = ~np.all((cen >= lo_m) & (cen < hi_m), axis=1)

    # positions written (x,y,z) to match the design JSONs in data/
    absz = cen + np.array([z0, 0, 0])
    out = {
        "source": str(Path(args.ct).name),
        "produced_by": "scripts/skan_graph.py",
        "coordinate_order": "position is (x, y, z) in CT voxels of the FULL "
                            "volume, matching data/**/octet_truss_*.json",
        "volume_shape_zyx": list(shape_full),
        "z_range": [int(z0), int(z1)],
        "voxel_um": args.voxel_um,
        "params": {"segmentation": (f"mask:{args.mask}" if args.mask else
                                    (args.threshold or "otsu")),
                   "min_object": args.min_object, "prune": args.prune,
                   "cluster": args.cluster, "margin": args.margin},
        "counts": {"junctions": int(len(cen)), "struts": int(len(ea)),
                   "interior_junctions": int((~bnd).sum()),
                   "interior_struts": int((~bnd[ea] & ~bnd[eb]).sum())},
        "junctions": [
            {"id": int(i), "position": [float(absz[i, 2]), float(absz[i, 1]),
                                        float(absz[i, 0])],
             "degree": int(deg[i]), "radius_vox": float(rad_node[i]),
             "radius_um": float(rad_node[i] * args.voxel_um),
             "boundary": bool(bnd[i])}
            for i in range(len(cen))],
        "struts": [
            {"id": int(k), "junction0": int(ea[k]), "junction1": int(eb[k]),
             "length_vox": float(dist[pidx[k]]),
             "length_um": float(dist[pidx[k]] * args.voxel_um),
             "chord_vox": float(eucl[pidx[k]]),
             "tortuosity": float(dist[pidx[k]] / max(eucl[pidx[k]], 1e-9)),
             "radius_vox": float(rad_mid[k]),
             "diameter_um": float(2 * rad_mid[k] * args.voxel_um),
             "radius_min_vox": float(rad_min[k]),
             "boundary": bool(bnd[ea[k]] or bnd[eb[k]])}
            for k in range(len(ea))],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    ii = ~bnd
    ie = ~bnd[ea] & ~bnd[eb]
    log(t0, f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    print(f"\n  junctions {len(cen):,}  ({int(ii.sum()):,} interior)   "
          f"struts {len(ea):,}  ({int(ie.sum()):,} interior)")
    if ii.any():
        print(f"  interior node degree: median {np.median(deg[ii]):.0f}, "
              f"mean {deg[ii].mean():.2f}, "
              f"deg==12 {int((deg[ii] == 12).sum()):,}")
    if ie.any():
        print(f"  interior strut length {np.median(dist[pidx[ie]]):.1f} vox "
              f"({np.median(dist[pidx[ie]]) * args.voxel_um:.0f} um), "
              f"diameter {np.median(2 * rad_mid[ie]) * args.voxel_um:.0f} um, "
              f"tortuosity median "
              f"{np.median(dist[pidx[ie]] / np.maximum(eucl[pidx[ie]], 1e-9)):.3f}")


if __name__ == "__main__":
    main()
