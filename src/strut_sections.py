"""Perpendicular cross-sections along each strut, and the profile of how they change.

This is the measurement the defect classes actually want. A strut is resampled on its
own frame -- a stack of planes cut perpendicular to its axis -- and each section is
reduced to a shape: area, equivalent radius, max inscribed radius, ellipse axes, and
how far the material's centroid sits from where the design says the axis is. Reading
those along the edge gives a profile, and the shape of the profile carries most of the
defect classes:

    thin      r_eq_med < the design band          median section radius sits low
    thick     r_eq_med > the design band          over-melt / dross on the surface
    necked    r_eq_min low, r_eq_med normal       a local pinch, not a thin strut
    missing   every section empty (with n_matched == 0 from `src.lattice_iou`)

where r_eq = sqrt(area / pi), the equivalent-circle radius of one section.

Two classes that this file used to decide, and no longer does:

* `broken` is now `src.lattice_iou.measure_connectivity`, a geodesic through material
  from node to node. Independent planes have no access to a topological property: 65 of
  the 155 severed struts have no empty section at all, and moving the old cut from 2
  empty sections to 1 swung the count 62 -> 99, which is a threshold deciding a class
  rather than measuring one.
* `warped` was removed outright. It rested on `offset_max`, and among struts with an
  empty section the median offset was 6.24 vox against a 6.01 vox window radius -- the
  blob being measured was the neighbouring strut, not a displaced one. `offset_max`,
  `tilt_deg` and `bow_vox` are still reported; they define nothing.

Why sections rather than axis-aligned slices. The lattice is oblique to every slice
plane of the volume, so a strut cuts an axis-aligned slice as an ellipse whose shape
encodes its *tilt*. Every 2D metric computed that way tracks tilt as much as health,
which is what defeated the earlier per-slice attempts on this dataset. A plane
perpendicular to the strut's own axis removes that entirely: a healthy strut is a
circle in its own frame regardless of how it is oriented in the volume.

Two failure modes of the earlier `strut_contour_view` are designed against explicitly:

* **The window must be big enough.** Its window was 1.4x the strut radius, so a 29 px
  strut sat in a 41 px frame and the material ran into the border on two thirds of the
  sections; contours traced along the border and came out as straight lines. Here the
  half-width is `window_factor` x the nominal radius (default 2.5, i.e. ~7.5 vox against
  a ~2.2 vox strut), and `border_frac` is reported per strut so a recurrence is visible
  rather than silent.
* **"The component at the centre" is not a filter when the components have fused.** At a
  junction the target and its neighbours are one component, so that rule returns the
  whole blob. Two things prevent it here: the junctions are trimmed off before sampling,
  and the window is deliberately *narrower than the neighbour separation* at the trim
  point (~9.5 vox at the octet's 60 deg), so a neighbouring strut is outside the frame
  instead of fused into it. Sections whose component still touches the border are
  flagged, not silently measured.

Sampling is sub-voxel (`step` = 0.5 vox by default) because an as-built strut is only
about 4.4 voxels across; at whole-voxel resolution a cross-section is ~15 pixels and no
shape statistic computed on it means anything.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates


def _perp_basis(u):
    """Two unit vectors spanning the plane perpendicular to u."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def section_stack(mask, a, b, half, step, n_sections):
    """Sample `n_sections` perpendicular planes between a and b.

    Returns (stack, coords) where stack is (n_sections, m, m) of interpolated mask values
    and coords is the 1-D grid of in-plane offsets in voxels.

    The local bbox is cropped and cast *before* interpolating. Casting the whole volume
    inside the call instead turns a 519 MB mask into a 2 GB float array once per strut,
    which costs about a second each and dominates everything else here by three orders
    of magnitude.
    """
    d = b - a
    length = float(np.linalg.norm(d))
    u = d / length
    e1, e2 = _perp_basis(u)

    g = np.arange(-half, half + 1e-9, step)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    t = np.linspace(0.0, 1.0, n_sections)

    # (n_sections, m, m, 3) sample points, built in one shot so a single interpolation
    # call covers the whole strut -- per-section calls are dominated by their own overhead
    base = a[None, :] + t[:, None] * d[None, :]
    plane = G1[..., None] * e1[None, None, :] + G2[..., None] * e2[None, None, :]
    pts = base[:, None, None, :] + plane[None, ...]
    flat = pts.reshape(-1, 3)

    lo = np.maximum(np.floor(flat.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(flat.max(0)).astype(int) + 2, mask.shape)
    crop = np.ascontiguousarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
                                dtype=np.float32)
    vals = map_coordinates(crop, (flat - lo).T, order=1, mode="constant", cval=0.0)
    return vals.reshape(n_sections, len(g), len(g)), g


def longitudinal_section(mask, p0, p1, r_nom, pad=2.5, margin=0.25, samples_r=81):
    """Sample the mask on a plane *containing* the strut axis -- the lateral view.

    The complement of `section_stack`: that one cuts across the strut and shows what the
    cross-section looks like, this one cuts along it and shows where material starts and
    stops. A gap is a white column here and is obvious by eye, which is exactly what a
    stack of independent cross-sections cannot show you.

    Returns (image, extent) with the axial coordinate running 0..1 over the *untrimmed*
    strut, extended by `margin` at both ends so the junctions stay in frame.
    """
    d = p1 - p0
    length = float(np.linalg.norm(d))
    u = d / length
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v /= np.linalg.norm(v)

    reach = r_nom * pad
    t = np.linspace(-margin, 1 + margin, int(length * (1 + 2 * margin)) * 2)
    r = np.linspace(-reach, reach, samples_r)
    T, R = np.meshgrid(t, r, indexing="xy")
    pts = (p0[:, None, None] + T[None] * d[:, None, None] + R[None] * v[:, None, None])

    flat = pts.reshape(3, -1)
    lo = np.maximum(np.floor(flat.min(1)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(flat.max(1)).astype(int) + 2, mask.shape)
    crop = np.ascontiguousarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
                                dtype=np.float32)
    img = map_coordinates(crop, flat - lo[:, None], order=1, mode="constant",
                          cval=0.0).reshape(T.shape)
    return img, (t[0], t[-1], r[0], r[-1])


def section_shape(img, g, r_max_vox, level=0.5):
    """Reduce one interpolated section to a shape record.

    The region kept is the connected component containing the section centre; if the
    centre is background, the nearest component whose distance to the centre is within
    `r_max_vox` (an off-axis but present strut), else nothing.
    """
    step = g[1] - g[0]
    m = img.shape[0]
    c = m // 2
    binary = img >= level
    out = dict(area_vox2=0.0, r_eq_vox=0.0, r_ins_vox=0.0, offset_vox=0.0,
               c1_vox=np.nan, c2_vox=np.nan,
               axis_major_vox=0.0, axis_minor_vox=0.0, border=False, empty=True)
    if not binary.any():
        return out

    lab, n = label(binary)
    which = lab[c, c]
    if which == 0:
        # centre is empty: accept a nearby component (an off-axis strut is still a strut)
        yy, xx = np.nonzero(binary)
        dist = np.hypot(g[yy], g[xx])
        k = int(np.argmin(dist))
        if dist[k] > r_max_vox:
            return out
        which = lab[yy[k], xx[k]]
    sel = lab == which

    ys, xs = np.nonzero(sel)
    npix = len(ys)
    area = npix * step * step
    py, px = g[ys], g[xs]
    cy, cx = py.mean(), px.mean()

    # second moments -> equivalent ellipse, the paper's cross-section descriptor
    cov = np.cov(np.vstack([py - cy, px - cx])) if npix > 2 else np.zeros((2, 2))
    ev = np.linalg.eigvalsh(cov) if npix > 2 else np.zeros(2)
    ev = np.clip(ev, 0, None)

    # Max inscribed radius inside this section. The zero border is required, not
    # cosmetic: on a section that is solid to the frame edge (a strut inside the build
    # plates) there is no background for the transform to measure against and it returns
    # a meaningless large number. Padding bounds the answer at the window half-width and
    # makes "the section fills the window" read as saturation rather than a giant strut.
    edt = distance_transform_edt(np.pad(sel, 1)) [1:-1, 1:-1] * step

    out.update(
        area_vox2=float(area),
        r_eq_vox=float(np.sqrt(area / np.pi)),
        r_ins_vox=float(edt.max()),
        offset_vox=float(np.hypot(cy, cx)),
        # signed, so a line can be fitted through them -- the magnitude alone cannot tell
        # a strut leaning steadily one way from one wandering back and forth
        c1_vox=float(cy),
        c2_vox=float(cx),
        axis_major_vox=float(2.0 * np.sqrt(ev[1])),
        axis_minor_vox=float(2.0 * np.sqrt(ev[0])),
        border=bool(sel[0].any() or sel[-1].any() or sel[:, 0].any() or sel[:, -1].any()),
        empty=False,
    )
    return out


def measure_sections(mask, pos, pairs, r_strut, trim_frac=0.20, window_factor=2.5,
                     n_sections=16, step=0.5, progress_every=2000, log=print):
    """Cross-section profiles for every strut. Returns (per-strut dict, profile dict)."""
    n = len(pairs)
    half = window_factor * r_strut
    keys = ("r_eq", "r_ins", "area", "offset", "major", "minor", "c1", "c2")
    prof = {k: np.full((n, n_sections), np.nan, np.float32) for k in keys}
    prof["empty"] = np.zeros((n, n_sections), bool)
    prof["border"] = np.zeros((n, n_sections), bool)

    out = {k: np.zeros(n, np.float32) for k in
           ("r_eq_med", "r_eq_min", "r_eq_max", "r_eq_std", "r_eq_cv",
            "r_ins_med", "offset_max", "offset_med", "aspect_med",
            "border_frac", "empty_frac")}
    for k in ("tilt_deg", "bow_vox"):
        out[k] = np.full(n, np.nan, np.float32)
    out["empty_sections"] = np.zeros(n, np.int32)
    # how many sections the centreline fit actually rests on -- a tilt from 3 sections is
    # not the same claim as one from 25, and without this the difference is invisible
    out["tilt_sections"] = np.zeros(n, np.int32)
    out["gap_len"] = np.zeros(n, np.int32)
    out["gap_interior"] = np.zeros(n, bool)
    out["clipped"] = np.zeros(n, bool)

    t0 = time.time()
    for i, (i0, i1) in enumerate(pairs):
        p0, p1 = pos[i0], pos[i1]
        seg = p1 - p0
        a, b = p0 + trim_frac * seg, p1 - trim_frac * seg
        if np.any(np.minimum(a, b) - half - 1 < 0) or \
           np.any(np.maximum(a, b) + half + 1 >= np.array(mask.shape)):
            out["clipped"][i] = True
            continue

        stack, g = section_stack(mask, a, b, half, step, n_sections)
        recs = [section_shape(stack[k], g, r_max_vox=2.0 * r_strut)
                for k in range(n_sections)]
        # axial position of each section in voxels, so the centreline fit below has a
        # slope in vox/vox and its arctangent is a real angle
        t_vox = np.linspace(0.0, float(np.linalg.norm(b - a)), n_sections)

        for k, r in enumerate(recs):
            prof["r_eq"][i, k] = r["r_eq_vox"]
            prof["r_ins"][i, k] = r["r_ins_vox"]
            prof["area"][i, k] = r["area_vox2"]
            prof["offset"][i, k] = r["offset_vox"]
            prof["major"][i, k] = r["axis_major_vox"]
            prof["minor"][i, k] = r["axis_minor_vox"]
            prof["c1"][i, k] = r["c1_vox"]
            prof["c2"][i, k] = r["c2_vox"]
            prof["empty"][i, k] = r["empty"]
            prof["border"][i, k] = r["border"]

        empty = prof["empty"][i]
        live = ~empty
        out["empty_sections"][i] = int(empty.sum())
        out["empty_frac"][i] = float(empty.mean())
        out["border_frac"][i] = float(prof["border"][i].mean())
        if live.any():
            r_eq = prof["r_eq"][i][live]
            out["r_eq_med"][i] = np.median(r_eq)
            out["r_eq_min"][i] = r_eq.min()
            out["r_eq_max"][i] = r_eq.max()
            out["r_eq_std"][i] = r_eq.std()
            out["r_eq_cv"][i] = r_eq.std() / max(np.median(r_eq), 1e-6)
            out["r_ins_med"][i] = np.median(prof["r_ins"][i][live])
            out["offset_max"][i] = prof["offset"][i][live].max()
            out["offset_med"][i] = np.median(prof["offset"][i][live])
            mn = prof["minor"][i][live]
            mj = prof["major"][i][live]
            ok = mn > 1e-6
            out["aspect_med"][i] = np.median(mj[ok] / mn[ok]) if ok.any() else 0.0

            # Fit a straight line through the section centroids -- this is the strut's
            # own centreline, measured -- and compare it to the design axis. Two numbers
            # come out and they mean different things: `tilt_deg` is how far the fitted
            # line leans away from the design direction, `bow_vox` is how far the
            # centroids stray from that fitted line, i.e. curvature. A strut printed
            # straight but at the wrong angle has tilt and no bow; a banana has both.
            #
            # This replaces the `offset_max` scalar the old `warped` class used, which
            # could not tell a strut leaning steadily one way from one wandering back and
            # forth, and which grew with any registration error rather than with the
            # defect. Both quantities here are differential -- a constant offset of the
            # whole strut, which is what a registration shift produces, cancels out.
            # Fit only the longest unbroken run of clean sections. Fitting one line
            # through everything is the wrong model the moment a strut has a gap: the
            # sections past the gap sit on whatever material is on the far side, which is
            # a different object. On strut #434 the centroid climbs 0.91 -> 6.58 (11.6
            # deg) and then the first section after a 3-section break reports 2.10; that
            # one point pulled the least-squares slope down to 8.30 deg. Border-touching
            # sections are excluded for the same reason -- their centroid is this strut
            # plus the neighbouring bulk it has fused with in the window.
            good = live & ~prof["border"][i]
            best = run = 0
            best_end = run_start = 0
            for k in range(n_sections + 1):
                if k < n_sections and good[k]:
                    if run == 0:
                        run_start = k
                    run += 1
                    if run > best:
                        best, best_end = run, k + 1
                else:
                    run = 0
            fit = np.zeros(n_sections, bool)
            fit[best_end - best:best_end] = True
            if best >= 3:
                axial = t_vox[fit]
                res = 0.0
                slopes = []
                for key in ("c1", "c2"):
                    c = prof[key][i][fit]
                    m, b0 = np.polyfit(axial, c, 1)
                    slopes.append(m)
                    res += float(np.sum((c - (m * axial + b0)) ** 2))
                out["tilt_deg"][i] = np.degrees(np.arctan(np.hypot(*slopes)))
                out["bow_vox"][i] = np.sqrt(res / best)
                out["tilt_sections"][i] = best
            else:
                out["tilt_deg"][i] = np.nan
                out["bow_vox"][i] = np.nan

        # longest run of empty sections, and whether material bounds it on both sides
        best = start = run = 0
        best_start = 0
        for k in range(n_sections):
            if empty[k]:
                if run == 0:
                    start = k
                run += 1
                if run > best:
                    best, best_start = run, start
            else:
                run = 0
        out["gap_len"][i] = best
        out["gap_interior"][i] = bool(0 < best < n_sections and best_start > 0
                                      and best_start + best < n_sections)

        if progress_every and (i + 1) % progress_every == 0:
            done = i + 1
            log(f"  sections {done}/{n}  ({time.time() - t0:.0f}s, "
                f"eta {(time.time() - t0) / done * (n - done):.0f}s)")

    return out, prof
