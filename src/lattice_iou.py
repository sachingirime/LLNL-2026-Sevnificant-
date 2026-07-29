"""Per-strut and per-node agreement between the nominal lattice design and the CT mask.

The idea is deliberately simple and fully 3D, which is what earlier per-slice attempts
on this dataset lacked. Every strut in the registered design is a line segment between
two junctions. Around that segment we paint the *nominal* part: a cylinder (capsule) of
the design diameter, 350 um. We then ask how much of that nominal cylinder is actually
filled by segmented material, and how much material sits nearby that the nominal
cylinder does not explain. That ratio is an intersection-over-union:

    IoU = |C & M| / |C| + |M & E| - |C & M|

where C is the nominal cylinder, M the CT segmentation, and E a slightly fatter
co-axial envelope that localises the union so a neighbouring strut cannot pollute it.

Two things make this behave:

* The strut ends are **trimmed** by `trim_frac` of the length before painting. Junctions
  in this specimen are far fatter than the struts they join (inscribed radius ~6 vox
  against a strut radius of ~2.2), so an untrimmed cylinder measures the node, not the
  strut, and every strut would look healthy at its ends.
* The cylinder is also diced into `stations` bands along its axis, giving a **profile**
  of fill fraction as you traverse the edge, which a single scalar cannot carry.

Which of these decides what has changed, and the docstrings below say so per function.
As it stands, the only class this file's cylinder fill decides is *missing*, through the
count `n_matched == 0`. Severed struts are decided by `measure_connectivity`, a geodesic
through material inside a tube about the axis, because continuity is topological and
neither a fill fraction nor a stack of independent bands can see it. Strut thickness is
decided by `src.strut_sections`, on planes cut perpendicular to each strut's own axis.
`scripts/classify_strut_defects.classify` is where the rules actually live.

Nodes get a sphere. `measure_node_sphere_fill` is the missing-node test -- fill is 0.000
at an absent junction and 1.000 everywhere else, a gap 0.90 wide -- while `measure_nodes`
adds IoU and two size readings, of which the 90%-fill radius is the one to threshold.
Because the design JSON stores junctions per unit cell, the same physical node appears up
to 8 times; node measurements are made on de-duplicated positions (3430 of the 10206
entries for 9x9x9).

Registration matters more than any parameter here. The shipped registration carries a
~1% residual scale error, which at a 2.2 vox strut radius is enough to walk the design
off the material at the far corners. Pass the correction from
`scripts/refit_registration.py` (`--correction outputs/registration/correction.json`)
unless you specifically want to measure the uncorrected case.

CLI:
    python -m src.lattice_iou --mask <mask.tif> --design <registered.json> \
        --correction outputs/registration/correction.json --out outputs/lattice_iou
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

# Design constants for the octet-truss specimens in this challenge (README p.307):
# 4.56 mm unit cell, 350 um strut diameter.
CELL_MM = 4.56
STRUT_DIAMETER_UM = 350.0
# Node / strut radius ratio measured on the clean simulated unit cell (8.80 / 6.00).
NODE_RADIUS_RATIO = 1.467
# The 12 unit_cell_edge_idx values carried only by the 9^2 boundary-cap struts.
BOUNDARY_EDGE_IDX = frozenset({8, 9, 10, 11, 16, 17, 18, 19, 32, 33, 34, 35})

# Plot palette: validated categorical slots 1/2/8 plus neutral chrome (light surface).
_BLUE, _ORANGE, _RED = "#2a78d6", "#eb6834", "#e34948"
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"


# --------------------------------------------------------------------------- design

def load_design(design_path, correction_path=None):
    """Return (positions_zyx, strut_pairs, edge_idx, correction_applied).

    JSON `position` is (x, y, z); reversing is the only permutation that puts every
    junction in bounds for a `vol[z, y, x]` lookup.
    """
    doc = json.loads(Path(design_path).read_text())
    pos = np.array([j["position"] for j in doc["junctions"]], float)[:, ::-1]
    pairs = np.array([(s["junction0"], s["junction1"]) for s in doc["struts"]], int)
    edge_idx = np.array([s.get("unit_cell_edge_idx", -1) for s in doc["struts"]], int)

    applied = False
    if correction_path:
        c = json.loads(Path(correction_path).read_text())
        A = np.asarray(c["A_zyx"], float)
        t = np.asarray(c["t_zyx"], float)
        pos = pos + pos @ A.T + t          # corrected = p + A p + t
        applied = True
    return pos, pairs, edge_idx, applied


def dedupe_junctions(pos, pairs, quantum=0.25):
    """Collapse the per-unit-cell junction duplicates onto physical nodes.

    Returns (unique_pos_zyx, node_of_junction, degree). Positions of duplicates are
    bit-identical in these files, so a coarse quantisation is only a safety net.
    """
    key = np.round(pos / quantum).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    upos = pos[first]
    degree = np.bincount(inverse[pairs].ravel(), minlength=len(upos))
    return upos, inverse, degree


def lattice_geometry(pos, pairs, cell_mm=CELL_MM, strut_diameter_um=STRUT_DIAMETER_UM):
    """Voxel size and nominal radii, derived from the registered strut length.

    An octet strut spans half a face diagonal, so |strut| = cell / sqrt(2). Solving that
    against the known 4.56 mm cell gives the voxel pitch without trusting any header.
    """
    lengths = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    strut_len_vox = float(np.median(lengths))
    cell_vox = strut_len_vox * np.sqrt(2.0)
    um_per_vox = cell_mm * 1000.0 / cell_vox
    r_strut = strut_diameter_um / 2.0 / um_per_vox
    return {
        "strut_length_vox": strut_len_vox,
        "strut_length_spread_vox": float(lengths.max() - lengths.min()),
        "cell_edge_vox": float(cell_vox),
        "um_per_voxel": float(um_per_vox),
        "nominal_strut_radius_vox": float(r_strut),
        "nominal_strut_diameter_um": float(strut_diameter_um),
        "nominal_node_radius_vox": float(r_strut * NODE_RADIUS_RATIO),
    }


# ------------------------------------------------------------------------- geometry

def _axis_frame(a, b, r_env, shape):
    """Local bbox around segment a->b plus the axial coordinate s and radial distance."""
    lo = np.floor(np.minimum(a, b) - r_env - 1.0).astype(int)
    hi = np.ceil(np.maximum(a, b) + r_env + 1.0).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    if np.any(hi <= lo):
        return None
    grids = np.meshgrid(*[np.arange(lo[i], hi[i], dtype=np.float32) for i in range(3)],
                        indexing="ij")
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    w = [grids[i] - a[i] for i in range(3)]
    s = w[0] * u[0] + w[1] * u[1] + w[2] * u[2]
    s_clamped = np.clip(s, 0.0, length)
    radial = np.sqrt(sum((w[i] - s_clamped * u[i]) ** 2 for i in range(3)))
    return lo, hi, s_clamped, radial, length


def measure_connectivity(mask, pos, pairs, r_strut, tube_factor=2.0,
                         progress_every=2000, log=print):
    """Is there a path through material from one node to the other, along this strut?

    This is the instrument for *disconnected*, and it is a different kind of measurement
    from everything else in this file. Cross-sections and cylinder fill both ask how much
    material sits near the design axis; neither can answer whether that material is
    *joined*, because independent planes have no access to a topological property. A strut
    severed by a crack narrower than the section spacing reads full on every section.

    Method: restrict the mask to a tube of `tube_factor` x r_nom about the design axis,
    then take the geodesic distance through that material from node A to node B. Three
    readings come out of it:

        reachable == False   no path exists -- severed
        detour ~ 1.0         the path is essentially the straight strut -- healthy
        detour >> 1.0        material connects the nodes but only by wandering

    Why the tube is what makes this work. A plain connected-component test is useless
    here: the whole lattice is one component, so every strut in a healthy part and every
    strut in a severed one both report "connected". Confining the search to the tube means
    a path that leaves this strut has to re-enter near the far node, and the neighbouring
    struts diverge from this one (~9.5 vox separation at mid-span against a 6 vox tube),
    so they are outside it. The detour ratio catches whatever squeaks through.

    The tube deliberately includes the junctions rather than trimming them: the question
    is node-to-node continuity, and a strut that has parted right at its root is exactly
    the case the trim would hide.
    """
    from skimage.graph import MCP_Geometric

    n = len(pairs)
    r_tube = tube_factor * r_strut
    out = {"geo_len_vox": np.full(n, np.inf, np.float32),
           "span_vox": np.zeros(n, np.float32),
           "detour": np.full(n, np.inf, np.float32),
           "tube_material_frac": np.zeros(n, np.float32)}
    out["reachable"] = np.zeros(n, bool)
    out["clipped"] = np.zeros(n, bool)
    out["no_seed"] = np.zeros(n, bool)

    t0 = time.time()
    for i, (i0, i1) in enumerate(pairs):
        a, b = pos[i0], pos[i1]
        frame = _axis_frame(a, b, r_tube, mask.shape)
        if frame is None:
            out["clipped"][i] = True
            continue
        lo, hi, _, radial, length = frame
        out["span_vox"][i] = length

        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        tube = radial <= r_tube
        material = sub & tube
        out["tube_material_frac"][i] = float(material.sum()) / max(int(tube.sum()), 1)
        if not material.any():
            continue

        # Seeds are the node centres. They sit inside the junction blob, which is 3x the
        # strut diameter, so the exact voxel is not delicate -- but a node that failed to
        # print leaves the centre empty, and then the nearest material inside the tube is
        # taken instead so the strut is still tested rather than silently dropped.
        idx = np.array(np.nonzero(material)).T
        seeds = []
        for p in (a, b):
            c = np.round(p).astype(int) - lo
            if (c >= 0).all() and (c < np.array(material.shape)).all() and material[tuple(c)]:
                seeds.append(tuple(c))
            else:
                dist = np.linalg.norm(idx - (p - lo), axis=1)
                k = int(np.argmin(dist))
                seeds.append(tuple(idx[k]))
        if seeds[0] == seeds[1]:
            out["no_seed"][i] = True
            continue

        costs = np.where(material, 1.0, np.inf)
        cum, _ = MCP_Geometric(costs).find_costs([seeds[0]], [seeds[1]])
        g = float(cum[seeds[1]])
        if np.isfinite(g):
            out["reachable"][i] = True
            out["geo_len_vox"][i] = g
            out["detour"][i] = g / max(length, 1e-6)

        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            log(f"  connectivity {done}/{n}  ({time.time() - t0:.0f}s, "
                f"eta {(time.time() - t0) / done * (n - done):.0f}s)")

    return out


def measure_struts(mask, pos, pairs, r_strut, trim_frac=0.20, env_factor=1.6,
                   dross_factor=2.2, stations=12, progress_every=2000, log=print):
    """Per-strut geometry against the nominal cylinder: fill, IoU, radius, gaps, excess.

    Everything is computed on the trimmed span a..b (`trim_frac` off each end, because a
    junction blob is ~3x the strut diameter and an untrimmed cylinder would measure the
    node), in `_axis_frame`'s coordinates: axial s along the segment, radial rho to it.

        C = {rho <= r_strut}                    the nominal 350 um cylinder
        E = {rho <= env_factor * r_strut}       envelope, localises the union so a
                                                neighbouring strut cannot pollute it
        A = {r_strut < rho <= dross_factor * r_strut} & {0.25L <= s <= 0.75L}

        n_matched   = |C & M|                             M = the CT mask
        fill        = |C & M| / |C|
        iou         = |C & M| / (|C| + |E & M| - |C & M|)
        excess_frac = |A & M| / |A|

    A is restricted to the middle half of the span because nearer the ends the junction's
    other struts pass legitimately through that annulus.

    `stations` bands along s carry fill and IoU as profiles; `gap_stations` is the longest
    run of bands under 2% fill. `radius_*_vox` is the true local radius -- an exact EDT of
    the cropped material, read on the centreline (rho <= 1) and reduced per band -- not
    the sqrt(fill) estimate, which is only valid while the as-built strut is thinner than
    nominal and concentric with it.

    WHAT OF THIS ACTUALLY DECIDES A CLASS, as the pipeline now stands:

    * `n_matched` is the only input from this function to `missing`, together with
      `empty_sections` from `src.strut_sections`. It is a count, so the rule is `== 0`
      and carries no threshold. See `scripts/classify_strut_defects.classify`.
    * `env_material_frac` sets `embedded` in `run()` (>= 0.80), excluding struts buried
      in the build plates, where an absent strut leaves no signature, from every tally.

    Everything else here is measured, written out and plotted, and classifies nothing.
    Three of them used to drive classes and were each replaced by a better instrument, so
    do not wire them back in without reading why:

    * `radius_med_vox` / `radius_min_vox` drove *thin* / *necked*. Superseded by the
      equivalent-circle radius sqrt(area/pi) of planes cut perpendicular to the strut's
      own axis in `src.strut_sections`; this crop is axis-aligned and its EDT inherits
      the obliqueness that defeats every 2D metric on this dataset.
    * `gap_stations` / `gap_interior` drove *broken*. Superseded by
      `measure_connectivity`: a stack of independent bands has no access to a topological
      property, and 65 of the 155 severed struts have no empty band at all.
    * `excess_frac` drove a *dross* class that no longer exists.

    `profile_min` and `station_iou_min` are still consumed, but by the visualisation
    scripts only (`inspect_strut_crops`, `visualize_lattice_iou`, `lattice_iou_webgl`).

    `mask` is boolean in (z, y, x). Returns per-strut arrays plus (n, stations) profiles
    of fill, IoU and radius.
    """
    n = len(pairs)
    r_env = env_factor * r_strut
    r_dross = dross_factor * r_strut
    out = {k: np.zeros(n, np.float32) for k in
           ("iou", "fill", "profile_min", "profile_mean", "profile_std",
            "station_iou_min", "diameter_est_um", "env_material_frac",
            "radius_med_vox", "radius_min_vox", "radius_std_vox", "excess_frac")}
    for k in ("n_nominal", "n_matched", "n_material_env", "gap_stations"):
        out[k] = np.zeros(n, np.int64)
    out["clipped"] = np.zeros(n, bool)
    out["gap_interior"] = np.zeros(n, bool)
    prof_fill = np.full((n, stations), np.nan, np.float32)
    prof_iou = np.full((n, stations), np.nan, np.float32)
    prof_rad = np.full((n, stations), np.nan, np.float32)

    t0 = time.time()
    for i, (i0, i1) in enumerate(pairs):
        p0, p1 = pos[i0], pos[i1]
        seg = p1 - p0
        trim = trim_frac * seg
        a, b = p0 + trim, p1 - trim

        frame = _axis_frame(a, b, max(r_env, r_dross), mask.shape)
        if frame is None:
            out["clipped"][i] = True
            continue
        lo, hi, s, radial, length = frame
        material = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        cyl = radial <= r_strut
        env = radial <= r_env
        n_cyl = int(cyl.sum())
        if n_cyl == 0:
            out["clipped"][i] = True
            continue
        hit = cyl & material
        n_hit = int(hit.sum())
        n_env = int((env & material).sum())

        out["n_nominal"][i] = n_cyl
        out["n_matched"][i] = n_hit
        out["n_material_env"][i] = n_env
        # How solid is the neighbourhood? Near 1 means the strut is buried in bulk metal
        # (the build plates at either z end), where absence of a strut is unobservable.
        out["env_material_frac"][i] = n_env / max(int(env.sum()), 1)
        out["iou"][i] = n_hit / max(n_cyl + n_env - n_hit, 1)
        fill = n_hit / n_cyl
        out["fill"][i] = fill
        out["diameter_est_um"][i] = np.sqrt(fill)

        # --- dross: material in the annulus outside the nominal cylinder. Restricted to
        # the middle half of the span; nearer the ends the neighbouring struts of the
        # junction pass through this annulus and would read as excess.
        mid = (s >= 0.25 * length) & (s <= 0.75 * length)
        ann = mid & (radial > r_strut) & (radial <= r_dross)
        n_ann = int(ann.sum())
        out["excess_frac"][i] = (float((ann & material).sum()) / n_ann) if n_ann else 0.0

        # --- traverse the edge: bin by axial position
        edges = np.clip((s / length * stations).astype(np.int32), 0, stations - 1)
        cb = edges[cyl]
        tot = np.bincount(cb, minlength=stations).astype(np.float64)
        got = np.bincount(cb, weights=material[cyl].astype(np.float64), minlength=stations)
        eb = edges[env & material]
        env_tot = np.bincount(eb, minlength=stations).astype(np.float64)
        good = tot > 0
        f = np.where(good, got / np.maximum(tot, 1), np.nan)
        u = np.where(good, got / np.maximum(tot + env_tot - got, 1), np.nan)
        prof_fill[i] = f
        prof_iou[i] = u
        out["profile_min"][i] = np.nanmin(f)
        out["profile_mean"][i] = np.nanmean(f)
        out["profile_std"][i] = np.nanstd(f)
        out["station_iou_min"][i] = np.nanmin(u)

        # --- true radius along the edge: EDT of the crop, read on the centreline.
        # The crop always reaches r_dross past the axis, so background exists in every
        # radial direction and the transform is not truncated for a normal strut.
        if n_hit:
            edt = distance_transform_edt(material)
            core = radial <= 1.0
            rad = np.full(stations, np.nan)
            cst = edges[core]
            cval = edt[core]
            for k in range(stations):
                sel = cval[cst == k]
                if sel.size:
                    rad[k] = sel.max()
            prof_rad[i] = rad
            if np.isfinite(rad).any():
                out["radius_med_vox"][i] = np.nanmedian(rad)
                out["radius_min_vox"][i] = np.nanmin(rad)
                out["radius_std_vox"][i] = np.nanstd(rad)

        # --- longest run of empty stations, and whether material bounds it both sides
        empty = ~(f > 0.02)
        best_len = best_start = 0
        run = start = 0
        for k in range(stations):
            if empty[k]:
                if run == 0:
                    start = k
                run += 1
                if run > best_len:
                    best_len, best_start = run, start
            else:
                run = 0
        out["gap_stations"][i] = best_len
        out["gap_interior"][i] = bool(
            0 < best_len < stations and best_start > 0
            and best_start + best_len < stations)

        if progress_every and (i + 1) % progress_every == 0:
            log(f"  struts {i + 1}/{n}  ({time.time() - t0:.0f}s)")

    return out, prof_fill, prof_iou, prof_rad


def measure_nodes(mask, node_pos, r_node, half=12, env_factor=1.6, size_fill=0.9,
                  size_min_voxels=27, progress_every=1000, log=print):
    """Nominal-sphere IoU / fill plus two size measurements per physical node.

    This function sizes nodes; it does not detect them. `iou` here is a *size proxy* and
    nothing else -- it is a tight spike at 0.250 that anticorrelates with node size,
    because as-built junctions (~937 um) are far larger than the nominal sphere (~513 um)
    so the union is dominated by material outside it. Do not threshold it. The
    missing-node test is `measure_node_sphere_fill`, and `fill` below is the same
    quantity at the nominal radius.

    Two sizes, because they fail in different ways:

    * `inscribed_radius_vox` -- exact EDT of a local box, read within one voxel of the
      junction centre. It is the honest max-inscribed-sphere radius, but an EDT on a
      voxel grid only takes values sqrt(integer), so the resulting histogram is coarsely
      quantised (the 9x9x9 specimen piles ~1600 nodes onto the single value 6.00 vox).
    * `radius_fill_vox` -- the largest r at which a ball of radius r about the junction
      is still `size_fill` material, found by interpolating the fill-vs-radius curve.
      Sub-voxel and smooth, so a threshold can actually be placed on its distribution.
      This is the one to threshold; the inscribed radius is the sanity check.

    The box is zero-shelled so the EDT stays defined where the box is otherwise solid;
    that caps a reported radius at `half` and sets the `saturated` flag. `box_material_frac`
    reports how solid the neighbourhood is, flagging nodes buried in the build plates.
    """
    n = len(node_pos)
    out = {k: np.zeros(n, np.float32) for k in
           ("iou", "fill", "inscribed_radius_vox", "radius_fill_vox", "diameter_um",
            "diameter_fill_um", "box_material_frac")}
    out["saturated"] = np.zeros(n, bool)
    r_env = env_factor * r_node
    box_shape = (2 * half + 1,) * 3
    offs = np.arange(-half, half + 1, dtype=np.float32)
    dz, dy, dx = np.meshgrid(offs, offs, offs, indexing="ij")
    r_edges = np.arange(0.0, half + 0.25, 0.25)
    r_centres = r_edges[1:]

    t0 = time.time()
    for i, p in enumerate(node_pos):
        base = np.round(p).astype(int)
        lo = base - half
        hi = lo + 2 * half + 1
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, mask.shape)
        box = np.zeros(box_shape, bool)
        box[lo_c[0] - lo[0]:hi_c[0] - lo[0],
            lo_c[1] - lo[1]:hi_c[1] - lo[1],
            lo_c[2] - lo[2]:hi_c[2] - lo[2]] = mask[lo_c[0]:hi_c[0],
                                                    lo_c[1]:hi_c[1],
                                                    lo_c[2]:hi_c[2]]
        box[0] = box[-1] = False
        box[:, 0] = box[:, -1] = False
        box[:, :, 0] = box[:, :, -1] = False

        frac = p - base                       # sub-voxel offset of the true centre
        dist = np.sqrt((dz - frac[0]) ** 2 + (dy - frac[1]) ** 2 + (dx - frac[2]) ** 2)
        sphere = dist <= r_node
        env = dist <= r_env
        n_sph = int(sphere.sum())
        n_hit = int((sphere & box).sum())
        n_env = int((env & box).sum())
        out["fill"][i] = n_hit / max(n_sph, 1)
        out["iou"][i] = n_hit / max(n_sph + n_env - n_hit, 1)
        out["box_material_frac"][i] = float(box.mean())

        edt = distance_transform_edt(box)
        centre = dist <= 1.0
        r_ins = float(edt[centre].max()) if centre.any() else 0.0
        out["inscribed_radius_vox"][i] = r_ins
        out["saturated"][i] = r_ins >= half - 1.0

        # fill(r) over a fine radius grid, then the crossing of `size_fill`.
        # The innermost bins hold only a handful of grid points (sometimes none at all,
        # since the true centre sits between voxels), so their ratio is noise or 0/0 --
        # start the search once the ball contains enough voxels to mean something.
        tot = np.cumsum(np.histogram(dist, bins=r_edges)[0]).astype(np.float64)
        got = np.cumsum(np.histogram(dist[box], bins=r_edges)[0]).astype(np.float64)
        curve = np.divide(got, tot, out=np.zeros_like(tot), where=tot > 0)
        start = int(np.searchsorted(tot, size_min_voxels))
        if start >= len(curve):
            r_fill = 0.0
        else:
            # first crossing, not the last: bulk material further out can push the curve
            # back above `size_fill`, which says nothing about the size of this node
            bad = np.flatnonzero(curve[start:] < size_fill)
            if bad.size == 0:
                r_fill = float(r_centres[-1])
            elif bad[0] == 0:
                r_fill = float(r_centres[start]) if curve[start] >= size_fill else 0.0
            else:
                k = start + int(bad[0]) - 1
                span = (curve[k] - size_fill) / max(curve[k] - curve[k + 1], 1e-9)
                r_fill = float(r_centres[k] + span * (r_centres[k + 1] - r_centres[k]))
        out["radius_fill_vox"][i] = r_fill

        if progress_every and (i + 1) % progress_every == 0:
            log(f"  nodes {i + 1}/{n}  ({time.time() - t0:.0f}s)")

    return out


def measure_node_sphere_fill(mask, node_pos, r_node, progress_every=1000, log=print):
    """How much of a sphere about each node is material. That is the whole test.

    A missing junction leaves nothing behind -- not the node, not the twelve struts that
    would have met there -- so the sphere is simply empty and the fill is 0.000, while
    every sound node in this specimen reads 1.000. The separation does not depend on the
    radius: sweeping it from the nominal 257 um out to 525 um flags exactly the same two
    nodes every time, with the gap below the healthy population never narrower than 0.55.

    The reading is binary because the thing being measured is. A junction in the design
    JSON is a bare `position` -- no radius, no thickness, only struts carry `thickness` --
    so the node is not a separately printed part but the region where twelve struts
    overlap, plus the fillet the melt pool leaves where their scan vectors converge. It is
    therefore present whole or absent whole: fill stays at or above 0.90 through the loss
    of one, two, three, four struts and reaches 0.000 only where all twelve are gone, with
    nothing in between. That is what makes the 0.90-wide gap, and why no radius in
    1.5-3.0r changes the answer.

    Do not look for a node absent while its struts are present; the struts alone would
    fill the sphere, and there is nothing at a junction that could go missing on its own.
    """
    n = len(node_pos)
    fill = np.zeros(n, np.float32)
    half = int(np.ceil(r_node)) + 1
    offs = np.arange(-half, half + 1, dtype=np.float32)
    dz, dy, dx = np.meshgrid(offs, offs, offs, indexing="ij")

    t0 = time.time()
    for i, p in enumerate(node_pos):
        base = np.round(p).astype(int)
        lo = base - half
        hi = lo + 2 * half + 1
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, mask.shape)
        box = np.zeros((2 * half + 1,) * 3, bool)
        box[lo_c[0] - lo[0]:hi_c[0] - lo[0],
            lo_c[1] - lo[1]:hi_c[1] - lo[1],
            lo_c[2] - lo[2]:hi_c[2] - lo[2]] = mask[lo_c[0]:hi_c[0],
                                                    lo_c[1]:hi_c[1],
                                                    lo_c[2]:hi_c[2]]
        f = p - base                          # sub-voxel offset of the true centre
        sphere = ((dz - f[0]) ** 2 + (dy - f[1]) ** 2 + (dx - f[2]) ** 2) <= r_node ** 2
        fill[i] = int((sphere & box).sum()) / max(int(sphere.sum()), 1)

        if progress_every and (i + 1) % progress_every == 0:
            log(f"  node fill {i + 1}/{n}  ({time.time() - t0:.0f}s)")
    return fill


# ------------------------------------------------------------------ distributions

def describe(values, name, bins=100, lo=None, hi=None):
    """Plain description of a metric's distribution -- no threshold is chosen here.

    Deliberately descriptive only. Picking the cut is a judgement call made by looking
    at the shape, so this reports the shape: percentiles and a binned histogram, both
    dumped to summary.json so a threshold can be read straight off them.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"metric": name, "n": 0}
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {
        "metric": name,
        "n": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "percentiles": {str(p): float(np.percentile(v, p))
                        for p in (0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99)},
        "histogram": {"bin_edges": [float(e) for e in edges],
                      "counts": [int(c) for c in counts]},
    }


# ------------------------------------------------------------------------- plotting

def _style(ax):
    ax.set_facecolor(_SURFACE)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK2, labelsize=8, length=3, width=0.8)


def _marks(ax, marks, span=None):
    """Draw vertical rules, staggering the labels so they never overprint."""
    drawn = 0
    for x, label, col in marks:
        if not np.isfinite(x):
            continue
        if span is not None and not (span[0] <= x <= span[1]):
            continue
        ax.axvline(x, color=col, linewidth=1.6)
        ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -10 - 13 * drawn), textcoords="offset points",
                    color=col, fontsize=8, ha="left", va="top")
        drawn += 1


def _hist(ax, values, bins, rng, color, title, xlabel, marks=(), ylabel="struts"):
    _style(ax)
    ax.hist(values, bins=bins, range=rng, color=color, edgecolor="none")
    _marks(ax, marks, span=rng)
    ax.set_title(title, color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=_INK2, fontsize=8)
    ax.set_ylabel(ylabel, color=_INK2, fontsize=8)


def plot_strut_distributions(strut, prof_fill, geom, out_path, tail_max=0.35):
    """Descriptive only -- no threshold is drawn, the cut is read off these shapes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iou, pmin = strut["iou"], strut["profile_min"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor=_SURFACE)

    _hist(axes[0, 0], iou, 120, (0, 1), _BLUE,
          "Strut IoU vs nominal 350 um cylinder", "IoU")
    _hist(axes[0, 1], iou[iou <= tail_max], int(tail_max * 120), (0, tail_max), _BLUE,
          f"Low-IoU tail (IoU <= {tail_max})   n={int((iou <= tail_max).sum())}", "IoU")
    _hist(axes[0, 2], pmin, 120, (0, 1), _BLUE,
          "Weakest station along the edge", "min fill fraction over stations")

    # How many struts a candidate cut would take, for every candidate cut. Flat stretches
    # are cuts whose answer does not depend on exactly where you put them.
    ax = axes[1, 0]
    _style(ax)
    s_iou = np.sort(iou)
    ax.plot(s_iou, np.arange(1, len(s_iou) + 1), color=_BLUE, linewidth=2)
    ax.set_xlim(0, tail_max)
    ax.set_ylim(0, max(int((iou <= tail_max).sum()) * 1.15, 10))
    ax.set_title("Struts below a candidate cut", color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("candidate IoU cut", color=_INK2, fontsize=8)
    ax.set_ylabel("struts below", color=_INK2, fontsize=8)

    # Whole-strut IoU against weakest station: separates wholly missing from severed
    ax = axes[1, 1]
    _style(ax)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.6)
    hb = ax.hexbin(iou, pmin, gridsize=60, extent=(0, 1, 0, 1), bins="log",
                   cmap="Blues", mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("struts (log)", color=_INK2, fontsize=8)
    cb.ax.tick_params(colors=_INK2, labelsize=7)
    ax.set_title("Whole strut vs weakest station", color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("strut IoU", color=_INK2, fontsize=8)
    ax.set_ylabel("min station fill", color=_INK2, fontsize=8)

    # Fill traversing the edge, banded by IoU decile: the shape at low IoU is what
    # distinguishes a strut that is absent everywhere from one that is broken in a spot.
    ax = axes[1, 2]
    _style(ax)
    t = (np.arange(prof_fill.shape[1]) + 0.5) / prof_fill.shape[1]
    bands = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 1.01)]
    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
    for (a, b), col in zip(bands, ramp):
        m = (iou >= a) & (iou < b)
        if m.sum():
            ax.plot(t, np.nanmedian(prof_fill[m], axis=0), color=col, linewidth=2,
                    label=f"IoU {a:.2f}-{b:.2f}  (n={int(m.sum())})")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK2, loc="center right")
    ax.set_title("Median fill traversing the trimmed edge", color=_INK, fontsize=10,
                 loc="left", pad=8)
    ax.set_xlabel("position along strut (trimmed)", color=_INK2, fontsize=8)
    ax.set_ylabel("fill fraction", color=_INK2, fontsize=8)

    fig.suptitle(
        f"Strut IoU: Otsu mask vs nominal 350 um cylinder   "
        f"n={len(iou)}   r_nom={geom['nominal_strut_radius_vox']:.2f} vox "
        f"({geom['um_per_voxel']:.2f} um/vox)",
        color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


def plot_node_distributions(node, usable, geom, out_path):
    """`usable` selects interior nodes that are not buried in the build plates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    umv = geom["um_per_voxel"]
    nominal = geom["nominal_node_radius_vox"] * 2 * umv
    iou = node["iou"][usable]
    fill = node["fill"][usable]
    dia = node["diameter_fill_um"][usable]
    ins = node["diameter_um"][usable]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=_SURFACE)

    _hist(axes[0, 0], iou, 100, (0, 1), _ORANGE,
          f"Node IoU vs nominal {nominal:.0f} um sphere", "IoU", ylabel="nodes")
    _hist(axes[0, 1], fill, 100, (0, 1), _ORANGE,
          "Nominal node sphere fill fraction",
          "fraction of the nominal sphere that is material", ylabel="nodes")

    hi = float(np.percentile(dia, 99.8)) * 1.06
    _hist(axes[1, 0], dia, 100, (0, hi), _ORANGE,
          "Node size", "node diameter at 90% ball fill (um)",
          [(nominal, f"design {nominal:.0f} um", _INK2),
           (float(np.median(dia)), f"as-built median {np.median(dia):.0f} um", _BLUE)],
          ylabel="nodes")

    # IoU against size: the low ceiling on node IoU is a size effect, not disagreement,
    # so the two belong on one pair of axes.
    ax = axes[1, 1]
    _style(ax)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.6)
    hb = ax.hexbin(dia, iou, gridsize=44, bins="log", cmap="Oranges",
                   mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("nodes (log)", color=_INK2, fontsize=8)
    cb.ax.tick_params(colors=_INK2, labelsize=7)
    ax.axvline(nominal, color=_INK2, linewidth=1.2)
    ax.annotate(f"design {nominal:.0f} um", xy=(nominal, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", color=_INK2, fontsize=8, va="top")
    ax.set_title("Node IoU vs node size", color=_INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel("node diameter at 90% ball fill (um)", color=_INK2, fontsize=8)
    ax.set_ylabel("node IoU", color=_INK2, fontsize=8)

    fig.suptitle(f"Node IoU: Otsu mask vs nominal {nominal:.0f} um sphere   "
                 f"n={int(usable.sum())} measurable interior nodes of {len(node['fill'])}   "
                 f"max-inscribed median {np.median(ins):.0f} um",
                 color=_INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ pipeline

def _write_csv(path, header, columns):
    rows = np.column_stack([np.asarray(c) for c in columns])
    fmt = []
    for c in columns:
        a = np.asarray(c)
        fmt.append("%d" if a.dtype.kind in "iub" else "%.5f")
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="",
               fmt=fmt)


def run(mask_path, design_path, out_dir, correction_path=None, trim_frac=0.20,
        env_factor=1.6, stations=12, node_half=12, strut_diameter_um=STRUT_DIAMETER_UM,
        cell_mm=CELL_MM, embedded_frac=0.80, node_size_fill=0.9,
        max_struts=None, max_nodes=None, make_plots=True, log=print):
    """Measure every strut and node, write CSV/NPZ/JSON/plots, return the summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mask = tifffile.imread(mask_path) > 0
    log(f"mask {mask.shape} loaded in {time.time() - t0:.1f}s "
        f"(foreground {100 * mask.mean():.2f}%)")

    pos, pairs, edge_idx, corrected = load_design(design_path, correction_path)
    geom = lattice_geometry(pos, pairs, cell_mm, strut_diameter_um)
    node_pos, node_of, degree = dedupe_junctions(pos, pairs)
    log(f"design: {len(pairs)} struts, {len(pos)} junction entries -> "
        f"{len(node_pos)} physical nodes, correction_applied={corrected}")
    log(f"geometry: {geom['um_per_voxel']:.3f} um/vox, strut {geom['strut_length_vox']:.2f} vox, "
        f"nominal radius {geom['nominal_strut_radius_vox']:.3f} vox")

    sel = slice(None) if max_struts is None else slice(0, max_struts)
    t0 = time.time()
    strut, prof_fill, prof_iou, prof_rad = measure_struts(
        mask, pos, pairs[sel], geom["nominal_strut_radius_vox"],
        trim_frac=trim_frac, env_factor=env_factor, stations=stations, log=log)
    log(f"struts measured in {time.time() - t0:.1f}s")

    npos = node_pos if max_nodes is None else node_pos[:max_nodes]
    t0 = time.time()
    node = measure_nodes(mask, npos, geom["nominal_node_radius_vox"],
                         half=node_half, env_factor=env_factor,
                         size_fill=node_size_fill, log=log)
    log(f"nodes measured in {time.time() - t0:.1f}s")

    umv = geom["um_per_voxel"]
    strut["diameter_est_um"] = strut["diameter_est_um"] * geom["nominal_strut_diameter_um"]
    node["diameter_um"] = node["inscribed_radius_vox"] * 2 * umv
    node["diameter_fill_um"] = node["radius_fill_vox"] * 2 * umv

    pp = pairs[sel]
    p0, p1 = pos[pp[:, 0]], pos[pp[:, 1]]
    mid = 0.5 * (p0 + p1)
    boundary = np.array([e in BOUNDARY_EDGE_IDX for e in edge_idx[sel]])
    # This specimen is fused into solid build plates at both z ends. A strut whose
    # envelope is nearly all material is inside bulk metal, where a missing strut leaves
    # no signature at all -- such struts are unmeasurable, not healthy, so they are
    # flagged and held out of the statistics rather than silently counted as intact.
    embedded = strut["env_material_frac"] >= embedded_frac
    node_embedded = node["box_material_frac"] >= embedded_frac
    node_interior = degree[:len(npos)] == 12       # 12 = fully surrounded octet node
    node_usable = node_interior & ~node_embedded

    _write_csv(out / "struts.csv",
               ["strut_id", "junction0", "junction1", "node0", "node1",
                "unit_cell_edge_idx", "is_boundary", "embedded",
                "z0", "y0", "x0", "z1", "y1", "x1", "z", "y", "x",
                "iou", "fill", "profile_min", "profile_mean", "profile_std",
                "station_iou_min", "diameter_est_um", "env_material_frac",
                "radius_med_vox", "radius_min_vox", "radius_std_vox", "excess_frac",
                "gap_stations", "gap_interior",
                "n_nominal", "n_matched", "n_material_env", "clipped"],
               [np.arange(len(pp)), pp[:, 0], pp[:, 1], node_of[pp[:, 0]], node_of[pp[:, 1]],
                edge_idx[sel], boundary.astype(int), embedded.astype(int),
                p0[:, 0], p0[:, 1], p0[:, 2], p1[:, 0], p1[:, 1], p1[:, 2],
                mid[:, 0], mid[:, 1], mid[:, 2],
                strut["iou"], strut["fill"], strut["profile_min"], strut["profile_mean"],
                strut["profile_std"], strut["station_iou_min"], strut["diameter_est_um"],
                strut["env_material_frac"],
                strut["radius_med_vox"], strut["radius_min_vox"], strut["radius_std_vox"],
                strut["excess_frac"], strut["gap_stations"],
                strut["gap_interior"].astype(int),
                strut["n_nominal"], strut["n_matched"], strut["n_material_env"],
                strut["clipped"].astype(int)])

    _write_csv(out / "nodes.csv",
               ["node_id", "z", "y", "x", "degree", "is_interior", "embedded",
                "iou", "fill", "inscribed_radius_vox", "diameter_um",
                "radius_fill_vox", "diameter_fill_um", "box_material_frac", "saturated"],
               [np.arange(len(npos)), npos[:, 0], npos[:, 1], npos[:, 2],
                degree[:len(npos)], node_interior.astype(int), node_embedded.astype(int),
                node["iou"], node["fill"], node["inscribed_radius_vox"],
                node["diameter_um"], node["radius_fill_vox"], node["diameter_fill_um"],
                node["box_material_frac"], node["saturated"].astype(int)])

    np.savez_compressed(out / "profiles.npz", fill=prof_fill, iou=prof_iou,
                        radius_vox=prof_rad, strut_id=np.arange(len(pp)))

    interior = ~boundary & ~strut["clipped"] & ~embedded
    distributions = {
        "strut_iou": describe(strut["iou"][interior], "strut_iou", lo=0.0, hi=1.0),
        "strut_fill": describe(strut["fill"][interior], "strut_fill", lo=0.0, hi=1.0),
        "strut_profile_min": describe(strut["profile_min"][interior],
                                      "strut_profile_min", lo=0.0, hi=1.0),
        "strut_diameter_um": describe(strut["diameter_est_um"][interior],
                                      "strut_diameter_um"),
        "node_iou": describe(node["iou"][node_usable], "node_iou", lo=0.0, hi=1.0),
        "node_fill": describe(node["fill"][node_usable], "node_fill", lo=0.0, hi=1.0),
        "node_diameter_um": describe(node["diameter_fill_um"][node_usable],
                                     "node_diameter_um"),
        "node_inscribed_diameter_um": describe(node["diameter_um"][node_usable],
                                               "node_inscribed_diameter_um"),
    }

    summary = {
        "mask": str(mask_path),
        "design": str(design_path),
        "correction": str(correction_path) if correction_path else None,
        "correction_applied": corrected,
        "mask_shape": list(mask.shape),
        "geometry": geom,
        "parameters": {"trim_frac": trim_frac, "env_factor": env_factor,
                       "stations": stations, "node_half": node_half,
                       "cell_mm": cell_mm, "strut_diameter_um": strut_diameter_um,
                       "embedded_frac": embedded_frac, "node_size_fill": node_size_fill},
        "counts": {"struts": int(len(pp)), "boundary_struts": int(boundary.sum()),
                   "embedded_struts": int(embedded.sum()),
                   "measurable_struts": int(interior.sum()),
                   "junction_entries": int(len(pos)), "physical_nodes": int(len(npos)),
                   "interior_nodes": int(node_interior.sum()),
                   "embedded_nodes": int(node_embedded.sum()),
                   "measurable_nodes": int(node_usable.sum())},
        "distributions": distributions,
        "as_built": {
            "median_strut_iou": float(np.median(strut["iou"][interior])),
            "median_strut_fill": float(np.median(strut["fill"][interior])),
            "median_strut_diameter_um": float(np.median(strut["diameter_est_um"][interior])),
            "median_node_diameter_um": float(np.median(node["diameter_fill_um"][node_usable])),
            "median_node_inscribed_diameter_um": float(
                np.median(node["diameter_um"][node_usable])),
            "node_over_strut_diameter_ratio": float(
                np.median(node["diameter_fill_um"][node_usable])
                / max(np.median(strut["diameter_est_um"][interior]), 1e-9)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if make_plots:
        plot_strut_distributions(
            {k: v[interior] for k, v in strut.items()
             if isinstance(v, np.ndarray) and v.shape[:1] == (len(pp),)},
            prof_fill[interior], geom, out / "strut_distributions.png")
        plot_node_distributions(node, node_usable, geom, out / "node_distributions.png")
        log(f"wrote plots to {out}")

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True, help="binary segmentation .tif (z,y,x)")
    p.add_argument("--design", required=True, help="registered lattice .json")
    p.add_argument("--correction", default=None,
                   help="registration correction.json from scripts/refit_registration.py")
    p.add_argument("--out", default="outputs/lattice_iou")
    p.add_argument("--trim-frac", type=float, default=0.20,
                   help="fraction of the strut length cut from each end (node exclusion)")
    p.add_argument("--env-factor", type=float, default=1.6,
                   help="union envelope radius as a multiple of the nominal radius")
    p.add_argument("--stations", type=int, default=12)
    p.add_argument("--node-half", type=int, default=12)
    p.add_argument("--strut-diameter-um", type=float, default=STRUT_DIAMETER_UM)
    p.add_argument("--cell-mm", type=float, default=CELL_MM)
    p.add_argument("--embedded-frac", type=float, default=0.80,
                   help="local material fraction above which a strut/node is treated as "
                        "buried in bulk metal (the build plates) and held out")
    p.add_argument("--node-size-fill", type=float, default=0.9,
                   help="ball fill fraction defining the reported node radius")
    p.add_argument("--max-struts", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()

    s = run(a.mask, a.design, a.out, a.correction, a.trim_frac, a.env_factor,
            a.stations, a.node_half, a.strut_diameter_um, a.cell_mm,
            a.embedded_frac, a.node_size_fill,
            a.max_struts, a.max_nodes, not a.no_plots)
    for name, d in s["distributions"].items():
        if d.get("n"):
            print(f"{name:28s} n={d['n']:6d}  median {d['median']:9.4f}  "
                  f"p1 {d['percentiles']['1']:9.4f}  p5 {d['percentiles']['5']:9.4f}  "
                  f"p95 {d['percentiles']['95']:9.4f}")
    print(json.dumps(s["as_built"], indent=2))


if __name__ == "__main__":
    main()
