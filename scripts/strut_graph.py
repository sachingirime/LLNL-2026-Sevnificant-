#!/usr/bin/env python
"""Turn a skeletonised lattice into a graph of nodes (junctions) and edges (struts).

Consumes the output of the existing ``skeletonize`` MCP tool. A skeleton volume says
which voxels are centreline but not which strut they belong to, and every per-strut
metric (radius profile, tortuosity, connectivity) needs that attribution.

Two node definitions are implemented so they can be compared on labelled data:

  skeleton : cluster skeleton voxels with >=3 neighbours into junctions (classic).
             Fragile -- surface bumps sprout spurs, junction blobs fragment.
  edt      : junctions are the THICKEST material, so take connected components of
             (radius > cut) as nodes. Sidesteps clustering and spur pruning.

Expected degrees are geometry, not a guess. An octet unit cell has 8 corner nodes of
degree 3 and 6 face-centre nodes of degree 4 (8*3 = 6*4 = 24 struts). Interior nodes of
a large lattice have degree 12.
"""
import argparse
from collections import deque

import numpy as np

_OFFSETS = np.array([(dz, dy, dx)
                     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dz, dy, dx) != (0, 0, 0)])


def _neighbour_index(coords, shape):
    """For each skeleton voxel, the indices of its 26-neighbours that are also skeleton."""
    lookup = -np.ones(shape, dtype=np.int64)
    lookup[tuple(coords.T)] = np.arange(len(coords))
    nbrs = []
    for c in coords:
        q = c + _OFFSETS
        ok = ((q >= 0) & (q < np.array(shape))).all(1)
        idx = lookup[tuple(q[ok].T)]
        nbrs.append(idx[idx >= 0])
    return nbrs


def _trace_edges(coords, nbrs, node_of):
    """Segment-based edge extraction: each connected run of NON-node skeleton voxels is
    one strut, and its endpoints are the nodes touching that run.

    Walking outward from every node voxel instead (the obvious approach) traces each edge
    twice, once from each end -- which shows up as an all-even degree histogram.

    Returns (edges, spurs, ambiguous) where each entry is (n0, n1, path); spurs touch one
    node, ambiguous runs touch three or more (a junction that failed to cluster).
    """
    n = len(coords)
    seen = np.zeros(n, dtype=bool)
    edges, spurs, ambiguous = [], [], []

    # runs of non-node voxels
    for i in range(n):
        if node_of[i] >= 0 or seen[i]:
            continue
        comp, touching, q = [], set(), deque([i])
        seen[i] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for m in nbrs[v]:
                if node_of[m] >= 0:
                    touching.add(int(node_of[m]))
                elif not seen[m]:
                    seen[m] = True
                    q.append(m)
        t = sorted(touching)
        if len(t) == 2:
            edges.append((t[0], t[1], comp))
        elif len(t) <= 1:
            spurs.append((t[0] if t else -1, -1, comp))
        else:
            ambiguous.append((t[0], t[1], comp))

    # direct node-to-node contact (a strut so short it has no free voxel)
    pairs = set()
    for i in range(n):
        a = node_of[i]
        if a < 0:
            continue
        for m in nbrs[i]:
            b = node_of[m]
            if b >= 0 and b != a:
                pairs.add((min(a, b), max(a, b)))
    for a, b in sorted(pairs):
        edges.append((a, b, []))
    return edges, spurs, ambiguous


def build_graph(skeleton, radius=None, method="skeleton", edt_cut=None,
                spur_frac=0.3, dissolve=True):
    """Return (nodes, edges, info). nodes: (N,3) centroids. edges: list of dicts."""
    sk = skeleton > 0
    coords = np.array(np.nonzero(sk)).T
    if len(coords) == 0:
        raise ValueError("skeleton is empty")
    nbrs = _neighbour_index(coords, sk.shape)
    deg = np.array([len(n) for n in nbrs])

    if method == "skeleton":
        is_node_vox = deg >= 3
    elif method == "edt":
        if radius is None:
            raise ValueError("method 'edt' needs the radius field")
        r_sk = radius[tuple(coords.T)]
        cut = edt_cut if edt_cut else 1.3 * float(np.median(r_sk))
        is_node_vox = r_sk > cut
    else:
        raise ValueError(f"unknown method '{method}'")

    # cluster node voxels into nodes via BFS over the skeleton adjacency
    node_of = -np.ones(len(coords), dtype=np.int64)
    nid = 0
    for i in np.where(is_node_vox)[0]:
        if node_of[i] >= 0:
            continue
        q = deque([i])
        node_of[i] = nid
        while q:
            v = q.popleft()
            for n in nbrs[v]:
                if is_node_vox[n] and node_of[n] < 0:
                    node_of[n] = nid
                    q.append(n)
        nid += 1
    if nid == 0:
        raise ValueError("no junction voxels found -- check the node definition/cut")

    centroids = np.array([coords[node_of == k].mean(0) for k in range(nid)])
    raw, spur_runs, ambig = _trace_edges(coords, nbrs, node_of)

    def arclen(path):
        if len(path) < 2:
            return 0.0
        p = coords[path].astype(float)
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    # node-centre separation is the honest strut length; the skeleton run stops short of
    # both node centroids, so arc length alone underestimates it
    real = []
    for a, b, path in raw:
        real.append(dict(n0=a, n1=b, path=path, arc=arclen(path),
                         span=float(np.linalg.norm(centroids[a] - centroids[b]))))
    spurs = [dict(n0=a, path=path, arc=arclen(path)) for a, _, path in spur_runs]

    med = float(np.median([e["span"] for e in real])) if real else 0.0
    spurs_kept = [e for e in spurs if med and e["arc"] >= spur_frac * med]

    # A degree-2 "junction" is not a junction -- it is a spurious cluster in the middle of
    # a strut (a locally thick spot for method 'edt', a skeleton wobble for 'skeleton').
    # Dissolve it and splice its two edges into one, repeatedly. Degree-1 nodes terminate
    # a spur and are dropped with their edge.
    if dissolve:
        alive = np.ones(nid, dtype=bool)
        changed = True
        while changed:
            changed = False
            inc = {k: [] for k in range(nid) if alive[k]}
            for ei, e in enumerate(real):
                inc[e["n0"]].append(ei)
                inc[e["n1"]].append(ei)
            for k, eis in inc.items():
                if len(eis) != 2:
                    continue
                e1, e2 = real[eis[0]], real[eis[1]]
                far1 = e1["n1"] if e1["n0"] == k else e1["n0"]
                far2 = e2["n1"] if e2["n0"] == k else e2["n0"]
                if far1 == far2 or far1 == k or far2 == k:
                    continue                       # would make a self-loop; leave it
                merged = dict(n0=far1, n1=far2,
                              path=e1["path"] + list(np.where(node_of == k)[0]) + e2["path"],
                              arc=e1["arc"] + e2["arc"],
                              span=float(np.linalg.norm(centroids[far1] - centroids[far2])))
                real = [e for i, e in enumerate(real) if i not in (eis[0], eis[1])]
                real.append(merged)
                alive[k] = False
                changed = True
                break
        # drop spur-terminating degree-1 nodes and their edge
        for _ in range(nid):
            inc = {}
            for ei, e in enumerate(real):
                inc.setdefault(e["n0"], []).append(ei)
                inc.setdefault(e["n1"], []).append(ei)
            ones = [k for k, v in inc.items() if len(v) == 1]
            if not ones:
                break
            drop = {inc[k][0] for k in ones}
            for k in ones:
                alive[k] = False
            real = [e for i, e in enumerate(real) if i not in drop]
        keep = np.where(alive)[0]
        remap = {old: new for new, old in enumerate(keep)}
        real = [dict(e, n0=remap[e["n0"]], n1=remap[e["n1"]])
                for e in real if e["n0"] in remap and e["n1"] in remap]
        centroids = centroids[keep]
        nid = len(keep)
        med = float(np.median([e["span"] for e in real])) if real else 0.0

    degree = np.zeros(nid, dtype=int)
    for e in real:
        degree[e["n0"]] += 1
        degree[e["n1"]] += 1

    info = dict(skeleton_voxels=len(coords), n_nodes=nid, n_edges=len(real),
                n_spurs=len(spurs), n_spurs_kept=len(spurs_kept),
                n_ambiguous=len(ambig), median_span=med,
                median_arc=float(np.median([e["arc"] for e in real])) if real else 0.0,
                degree=degree)
    return centroids, real, info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume", help="CT volume (.npy/.tif) -- will be Otsu-segmented")
    p.add_argument("--method", choices=["skeleton", "edt", "both"], default="both")
    p.add_argument("--edt-cut", type=float, default=0.0)
    args = p.parse_args()

    import tifffile
    from scipy.ndimage import distance_transform_edt
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize as sk3d

    vol = (np.load(args.volume) if args.volume.endswith(".npy")
           else tifffile.imread(args.volume)).astype(np.float32)
    t = threshold_otsu(vol)
    mask = vol > t
    radius = distance_transform_edt(mask)
    skel = sk3d(mask)
    print(f"otsu={t:.5f}  fg={100*mask.mean():.2f}%  skeleton={int(skel.sum())} vox  "
          f"radius on skeleton: median={np.median(radius[skel]):.2f} max={radius[skel].max():.2f}")

    for m in (["skeleton", "edt"] if args.method == "both" else [args.method]):
        print(f"\n--- node definition: {m} ---")
        try:
            _, edges, info = build_graph(skel, radius, m,
                                         args.edt_cut or None)
        except ValueError as e:
            print(f"  failed: {e}")
            continue
        d = info["degree"]
        print(f"  nodes={info['n_nodes']}  edges={info['n_edges']}  "
              f"spurs={info['n_spurs']} (kept {info['n_spurs_kept']})  "
              f"ambiguous runs={info['n_ambiguous']}")
        print(f"  strut length: node-to-node span median={info['median_span']:.1f} vox, "
              f"skeleton arc median={info['median_arc']:.1f} vox")
        vals, cnt = np.unique(d, return_counts=True)
        print("  degree histogram: " + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnt)))
        print("  expected for one octet cell: 8 nodes of degree 3, 6 of degree 4, 24 edges")


if __name__ == "__main__":
    main()
