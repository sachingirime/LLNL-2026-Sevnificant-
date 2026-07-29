"""Which struts were DESIGNED out, read from the STL, and how the detector scored.

Every missing-strut number in this project has so far been checked against another
measurement of the same scan. That tests consistency, not correctness -- three instruments
agreeing on 0.5% would look identical whether or not those are the struts the CAD actually
removed. The `stls/` folder settles it: `0.stl` is the design with every strut present and
`0.5.stl` / `1.stl` are the same design with 0.5% and 1% deliberately deleted. Comparing a
strut's midspan against the mesh gives the exact list of which ones, and that list is a
ground truth no measurement of the CT can talk itself into.

    absent  <=>  no triangle within `tol` mm of the strut's midspan

The midspan matters. Sampling near the ends finds the neighbouring junctions -- which are
still there -- so a removed strut looks supported; over t in 0.40-0.60 the nearest mesh of
an absent strut is a full half-cell away and the two populations separate by ~0.25 mm.

THE COORDINATE FRAME IS THE HARD PART, AND IT HAS A TRAP. The dataset's own file_names.txt
says the STL, JSON and TIF are in three different frames. Scale and centring come straight
off the bounding boxes (both are nominal CAD, so this is exact, not a fit): the STL is
centred on the origin and its two lattice axes span 18 design units, while its third axis
is longer because it carries the build plates. What is left is one of the cube's 48 signed
axis permutations -- and THAT CANNOT BE FIXED BY MATCHING `0.stl`, because a lattice with
every strut present is invariant under all 48. Fitting against the complete design gives
100% support for every one of them and silently picks the wrong one; doing that produced a
ground truth that overlapped the detector at exactly chance (1 strut of 89).

`find_orientation` resolves it against a *defect* STL, where the removed struts break the
symmetry, and reports the margin so the identification can be judged rather than trusted.
On the 0.5% specimen the correct orientation matches 88 struts and the runner-up 12, out
of a chance expectation of 0.5. It also carries an independent check that costs nothing:
the winning orientation must map the STL's plate axis onto the design's build axis, and on
this data it does.
"""
from __future__ import annotations

import itertools
import struct

import numpy as np
from scipy.spatial import cKDTree

# Sample only the middle of the strut: nearer the ends the junctions supply mesh whether
# or not the strut is there.
MIDSPAN = np.linspace(0.40, 0.60, 5)
DESIGN_UNITS = 18.0          # the nominal JSON spans 0..18 in every axis
ABSENT_TOL_MM = 0.7          # ~2x the strut radius, ~4x the worst support in 0.stl


def read_stl_centroids(path, max_triangles=None):
    """Triangle centroids of a binary STL, as (n, 3) float32 in the STL's own frame."""
    with open(path, "rb") as fh:
        header = fh.read(84)
        if len(header) < 84:
            raise ValueError(f"{path} is too short to be a binary STL")
        n = struct.unpack("<I", header[80:84])[0]
        if n == 0:
            raise ValueError(f"{path} declares 0 triangles; ASCII STL is not supported")
        take = n if max_triangles is None else min(n, max_triangles)
        buf = np.frombuffer(fh.read(take * 50), dtype=np.uint8)
    if buf.size < take * 50:
        raise ValueError(f"{path} is truncated: {buf.size} bytes for {take} triangles")
    verts = buf.reshape(take, 50)[:, 12:48].copy().view("<f4").reshape(take, 3, 3)
    return verts.mean(axis=1).astype(np.float32), n


def design_to_stl_scale(centroids):
    """mm per design unit, from the two lattice axes of the STL bounding box.

    The third axis is longer because it carries the build plates, so it is excluded by
    taking the two smallest spans.
    """
    span = centroids.max(0) - centroids.min(0)
    lattice = float(np.sort(span)[:2].mean())
    return lattice / DESIGN_UNITS, span


def strut_support(centroids, pos, pairs, perm, signs, scale, tol=ABSENT_TOL_MM,
                  samples=MIDSPAN):
    """Farthest-from-mesh distance over each strut's midspan, and the absent mask."""
    d = ((pos - DESIGN_UNITS / 2.0) * scale)[:, list(perm)] * np.asarray(signs, float)
    a, b = d[pairs[:, 0]], d[pairs[:, 1]]
    tree = cKDTree(centroids)
    worst = np.zeros(len(pairs))
    for t in samples:
        worst = np.maximum(worst, tree.query(a + t * (b - a))[0])
    return worst, worst > tol


def find_orientation(centroids, pos, pairs, reference, scale, tol=ABSENT_TOL_MM):
    """The signed axis permutation aligning the design to a DEFECT stl.

    `reference` is a boolean mask over struts -- the detector's own missing set is the
    usual choice. Only the removed struts carry orientation information, so this cannot
    be run against a complete lattice; see the module docstring.

    Returns (perm, signs, ranked) where `ranked` is every candidate as
    (matched, perm, signs, n_absent), best first. Judge the identification on the margin
    between the first two, not on the winner alone.
    """
    ref = np.asarray(reference, bool)
    ranked = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            _, absent = strut_support(centroids, pos, pairs, perm, signs, scale, tol)
            m = int((absent[:len(ref)] & ref).sum())
            ranked.append((m, perm, signs, int(absent.sum())))
    ranked.sort(key=lambda r: -r[0])
    return ranked[0][1], ranked[0][2], ranked


def plate_axis_check(span, perm, signs):
    """Does this orientation put the STL's plate axis on the design's build axis?

    Free, and independent of anything the detector said: the STL's longest axis is the one
    carrying the build plates, and in the CT the plates sit at the two z ends. If the
    winning orientation disagrees, the alignment is wrong however well it scored.
    """
    stl_long = int(np.argmax(span))
    design_axis = list(perm)[stl_long]      # design column feeding that STL axis
    names = ("z", "y", "x")                 # load_design returns (z, y, x)
    return names[design_axis], design_axis == 0


def score(pred, truth, usable=None):
    """Precision / recall / F1 of a predicted mask against the designed-out mask."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if usable is None:
        usable = np.ones(len(pred), bool)
    usable = np.asarray(usable, bool)
    tp = int((pred & truth & usable).sum())
    fp = int((pred & ~truth & usable).sum())
    fn = int((~pred & truth & usable).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12), "n_pred": int((pred & usable).sum())}
