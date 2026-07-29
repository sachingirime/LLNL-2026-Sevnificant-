"""Design-free missing-node detection, one node plane at a time, in 2D.

Everything else in this project that names a node does so by looking a design junction up
in the CT. That makes the answer only as good as the registration, and it cannot see a
node the design does not mention. This module never opens the design JSON. It finds the
nodes in the mask, works out the lattice they sit on from the nodes themselves, and asks
which sites of that lattice have nothing at them.

Why 2D is the right shape for *nodes*, when it was a dead end for struts. An octet strut
runs at 45 degrees, so no plane contains one and every per-slice strut statistic on this
dataset has had to fight obliqueness. Nodes are the opposite case: they occupy 19 discrete
z planes 39.2 vox apart, and the specimen tilt is only ~3 vox across its whole 700 vox
width, so a node plane is essentially one slice. Measured on this specimen, a single slice
at a node plane yields 165-178 node blobs against the 181 the design puts there.

The four steps, and what makes each one safe:

1. FIND THE PLANES.  Count connected components of `EDT >= r_thr` in every slice. At a
   node plane that count is 165-178; between planes it is 0-2, because an oblique cut
   through a strut has an inscribed radius of ~4.2 vox and a node has ~8. The signal is a
   square wave, so the planes are read off it rather than computed from a cell size.
   Build-plate slices are excluded first by foreground fraction (~0.70 in the plates
   against ~0.05 in the lattice); inside them nothing is separable and the fit explodes.

2. FIND THE NODES.  A node is a locally fat place: `EDT >= r_thr` with r_thr read off the
   VALLEY of the bimodal EDT histogram, not asserted. In-plane struts weld the whole slice
   into one connected component, so labelling the mask itself would return a single blob --
   labelling the *distance transform above a threshold* is what separates them.

3. FIND THE LATTICE.  Cluster the nearest-neighbour vectors of the blob centroids into two
   directions, then alternate `round()` indexing and least-squares refit until the basis
   settles. This recovers |v| = 55.3 vox at 89.4 degrees on this specimen with a 0.74 vox
   median residual -- the octet prediction is cell/sqrt2 = 55.41 vox at 90 degrees, which
   the module never told it. That agreement is the check that the fit is real.

4. ASK WHICH SITES ARE EMPTY.  Enumerate the integer lattice sites inside the CONVEX HULL
   of the observed indices and match each to a blob. The hull is the load-bearing choice:
   it interpolates and never extrapolates, so the detector cannot invent nodes past the
   edge of the printed part. The cost is that it is blind to a whole absent outer row --
   on this specimen the design's y > 750 face has no material behind it, and the hull
   simply declines to claim those 181 sites. That is the correct behaviour for a
   design-free method, and it is why this does not replace the design-referenced test:
   an extent mismatch is exactly the thing only a design comparison can see.

A site the hull calls interior and no blob answers is a missing-node candidate. On the
0.5%-defect specimen that is 3 of ~2,240 interior sites across 17 usable planes, and the
cross-check in `scripts/node_planes_2d.py` splits them: 2 are truly absent (mask material
0.000 in a 17^3 box, 12 of 12 incident struts dead, no skeleton junction within 55 vox),
and 1 is present but undersized -- 570 um inscribed against a 708 um median, below the 1st
percentile, with 3 of 12 struts dead and skeleton degree 9. So this instrument flags
"no node-sized object here", which is absence OR severe undersizing, and the two are
separated afterwards by measuring the site rather than by moving the threshold.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay, cKDTree

# Same palette as src/lattice_iou.py so the figures read as one set.
_BLUE, _ORANGE, _RED = "#2a78d6", "#eb6834", "#e34948"
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
_GREEN = "#2e9e63"

# Foreground fraction above which a slice is inside a build plate rather than the lattice.
PLATE_FRAC = 0.20


def slice_edt(slab):
    """2D distance transform, zero-padded so a blob touching the frame is not measured
    against the frame. Without the pad the EDT of a slab that is solid to the edge has no
    background to work from and returns nonsense."""
    return ndi.distance_transform_edt(np.pad(slab, 1))[1:-1, 1:-1]


def blob_centroids(slab, r_thr):
    """Centroids and peak inscribed radius of every locally-fat region in one slice."""
    ed = slice_edt(slab)
    bw = ed >= r_thr
    lab, n = ndi.label(bw)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    idx = range(1, n + 1)
    cen = np.array(ndi.center_of_mass(bw, lab, idx), float).reshape(-1, 2)
    peak = np.array(ndi.maximum(ed, lab, idx), float).reshape(-1)
    return cen, peak


def edt_radius_histogram(mask, z_list, bins=60, r_max=12.0):
    """Pooled histogram of per-pixel EDT over the node planes.

    Bimodal: struts at ~2-4 vox, nodes at ~6-8. `r_thr` belongs in the valley, and this is
    what it is read off. Returns (bin_centres, counts, valley_radius).
    """
    counts = np.zeros(bins)
    edges = np.linspace(0.0, r_max, bins + 1)
    for z in z_list:
        ed = slice_edt(mask[z])
        v = ed[ed > 0.5]
        counts += np.histogram(v, bins=edges)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    # valley = lowest count between the two dominant modes
    smooth = ndi.uniform_filter1d(counts, 3)
    lo = int(np.argmax(smooth[centres < 4.5]))
    hi_mask = centres > 5.0
    hi = int(np.flatnonzero(hi_mask)[np.argmax(smooth[hi_mask])])
    valley = lo + int(np.argmin(smooth[lo:hi])) if hi > lo else lo
    return centres, counts, float(centres[valley])


def find_node_planes(mask, r_thr, plate_frac=PLATE_FRAC, log=print):
    """z of every node plane, from the periodicity of the blob count. No design input.

    Returns (planes, blob_count_per_z, usable_mask_per_z).
    """
    fg = mask.reshape(len(mask), -1).mean(axis=1)
    usable = fg < plate_frac
    n_blob = np.zeros(len(mask), int)
    for z in np.flatnonzero(usable):
        _, n = ndi.label(slice_edt(mask[z]) >= r_thr)
        n_blob[z] = n

    if not n_blob.any():
        raise ValueError("no locally-fat regions anywhere; is this the right mask?")
    # A node plane is a run above half the typical plane height. The square wave is deep
    # (165-178 on, 0-2 off) so the level is not delicate.
    level = 0.5 * np.percentile(n_blob[n_blob > 0], 95)
    on = (n_blob > level) & usable
    planes = []
    z = 0
    while z < len(on):
        if on[z]:
            z1 = z
            while z1 < len(on) and on[z1]:
                z1 += 1
            planes.append(int(z + np.argmax(n_blob[z:z1])))
            z = z1
        else:
            z += 1
    log(f"  {len(planes)} node planes in z {planes[0]}..{planes[-1]}"
        f"  (spacing median {np.median(np.diff(planes)):.1f} vox)"
        if planes else "  no node planes found")
    return planes, n_blob, usable


def fit_lattice_basis(pts):
    """Two primitive vectors of the 2D lattice the points sit on.

    Nearest-neighbour vectors, restricted to a half plane so +v and -v do not cancel and
    to the short family so second neighbours do not drag the median.
    """
    if len(pts) < 12:
        raise ValueError(f"too few blobs ({len(pts)}) to fit a lattice")
    _, i = cKDTree(pts).query(pts, k=5)
    vec = (pts[i[:, 1:]] - pts[:, None, :]).reshape(-1, 2)
    vec = vec[vec[:, 0] > 1e-6]
    mag = np.linalg.norm(vec, axis=1)
    vec = vec[mag < np.percentile(mag, 60)]
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    near = np.abs(np.angle(np.exp(1j * (ang - np.median(ang))))) < np.pi / 4
    v1, v2 = np.median(vec[near], axis=0), np.median(vec[~near], axis=0)
    if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6:
        raise ValueError("degenerate lattice basis")
    return v1, v2


def index_points(pts, v1, v2, rounds=4):
    """Assign integer indices to the points and refit the basis to them, alternately."""
    origin = pts.mean(0)
    ij = None
    for _ in range(rounds):
        basis = np.column_stack([v1, v2])
        ij = np.round(np.linalg.solve(basis, (pts - origin).T).T)
        design = np.column_stack([ij, np.ones(len(ij))])
        sol, *_ = np.linalg.lstsq(design, pts, rcond=None)
        v1, v2, origin = sol[0], sol[1], sol[2]
    resid = np.linalg.norm(pts - (ij @ np.vstack([v1, v2]) + origin), axis=1)
    return ij.astype(int), v1, v2, origin, resid


def sites_in_hull(ij, v1, v2, origin):
    """Every integer lattice site inside the convex hull of the observed indices.

    Interpolation only. Nothing outside the printed footprint is ever proposed, which is
    what stops the detector from reporting a design overhang as missing nodes.
    """
    hull = ConvexHull(ij)
    tri = Delaunay(ij[hull.vertices].astype(float))
    gi, gj = np.meshgrid(np.arange(ij[:, 0].min(), ij[:, 0].max() + 1),
                         np.arange(ij[:, 1].min(), ij[:, 1].max() + 1), indexing="ij")
    grid = np.column_stack([gi.ravel(), gj.ravel()])
    sites = grid[tri.find_simplex(grid.astype(float)) >= 0]
    xy = sites @ np.vstack([v1, v2]) + origin
    # A site missing any of its four lattice neighbours sits on the boundary ring, where
    # an empty site is an extent question rather than a defect claim.
    present = {tuple(s) for s in sites}
    edge = np.array([sum((tuple(s + d) in present)
                         for d in ((1, 0), (-1, 0), (0, 1), (0, -1))) < 4
                     for s in sites])
    return sites, xy, edge


def analyse_plane(mask, z, r_thr, match_frac=0.25):
    """Blobs, lattice, sites and matches for one node plane. Returns a dict."""
    cen, peak = blob_centroids(mask[z], r_thr)
    v1, v2 = fit_lattice_basis(cen)
    ij, v1, v2, origin, resid = index_points(cen, v1, v2)
    sites, xy, edge = sites_in_hull(ij, v1, v2, origin)

    pitch = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    dist, nearest = cKDTree(cen).query(xy)
    found = dist <= match_frac * pitch
    return {
        "z": int(z), "centroids": cen, "peak_radius": peak,
        "v1": v1, "v2": v2, "origin": origin, "resid": resid,
        "pitch_vox": float(pitch),
        "basis_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        "sites_ij": sites, "sites_xy": xy, "edge": edge,
        "dist": dist, "nearest": nearest, "found": found,
        "site_radius": np.where(found, peak[nearest] if len(peak) else 0.0, 0.0),
    }


def plane_is_sane(res, max_resid=3.0, max_site_ratio=1.4):
    """Reject a plane whose lattice fit did not converge on a lattice.

    A build-plate slice produces hundreds of merged blobs, a meaningless basis and
    thousands of sites; without this guard those planes dominate the totals.
    """
    n_blob = len(res["centroids"])
    reasons = []
    if not 70.0 <= res["basis_angle_deg"] <= 110.0:
        reasons.append(f"basis angle {res['basis_angle_deg']:.0f} deg")
    if np.median(res["resid"]) > max_resid:
        reasons.append(f"median residual {np.median(res['resid']):.1f} vox")
    if len(res["sites_ij"]) > max_site_ratio * max(n_blob, 1):
        reasons.append(f"{len(res['sites_ij'])} sites for {n_blob} blobs")
    return (not reasons), "; ".join(reasons)


def run(mask, r_thr=None, plate_frac=PLATE_FRAC, match_frac=0.25, log=print):
    """Detect every node plane and every empty interior site. Returns a result dict."""
    planes_probe, _, _ = find_node_planes(mask, 5.0, plate_frac, log=lambda *_: None)
    if not planes_probe:
        raise ValueError("no node planes found")

    centres, counts, valley = edt_radius_histogram(mask, planes_probe)
    if r_thr is None:
        r_thr = valley
        log(f"  r_thr {r_thr:.2f} vox read off the EDT histogram valley")
    else:
        log(f"  r_thr {r_thr:.2f} vox (given; histogram valley is at {valley:.2f})")

    planes, n_blob, usable = find_node_planes(mask, r_thr, plate_frac, log=log)

    results, rejected = [], []
    for z in planes:
        try:
            res = analyse_plane(mask, z, r_thr, match_frac)
        except (ValueError, Exception) as err:            # noqa: BLE001 - report, skip
            rejected.append((int(z), str(err)))
            continue
        ok, why = plane_is_sane(res)
        (results if ok else rejected).append(res if ok else (int(z), why))

    for z, why in rejected:
        log(f"  plane z={z} rejected: {why}")
    return {
        "r_thr": float(r_thr), "planes": results, "rejected": rejected,
        "edt_hist": (centres, counts, valley),
        "n_blob_per_z": n_blob, "usable_z": usable,
    }


def threshold_sweep(mask, planes, radii, match_frac=0.25):
    """Interior-miss count as a function of r_thr, so the cut is shown and not asserted."""
    out = []
    for r in radii:
        n_int = n_miss = 0
        for z in planes:
            try:
                res = analyse_plane(mask, z, r, match_frac)
            except Exception:                              # noqa: BLE001
                continue
            if not plane_is_sane(res)[0]:
                continue
            interior = ~res["edge"]
            n_int += int(interior.sum())
            n_miss += int((interior & ~res["found"]).sum())
        out.append((float(r), n_int, n_miss))
    return out
