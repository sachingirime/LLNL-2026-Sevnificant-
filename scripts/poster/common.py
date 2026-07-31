"""Shared loading and styling for the poster figures.

Everything here reads *finished* detector output. Nothing re-detects: the class
labels come from `outputs/ct_anomaly_report_20260730/defects/strut_classes.csv`,
written by `detect_lattice_defects` with the refined registration correction
applied. Re-deriving them in a plotting script is exactly the failure mode
CLAUDE.md warns about.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUN = ROOT / "outputs/ct_anomaly_report_20260730"
DEFECTS = RUN / "defects"
DESIGN = ROOT / ("data/missing_struts/registered_jsons/"
                 "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
CORRECTION = RUN / "registration_refined/correction.json"
MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/mask.tif"
CT = ROOT / "data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif"
STL_GT = ROOT / "outputs/stl_missing_struts/0.5_missing_struts.csv"
OUT = ROOT / "outputs/poster"

# ---------------------------------------------------------------- palette

# Dark poster palette. Struts read as cold structure; defects read as heat.
INK = "#080b14"          # page
PANEL = "#0d1220"        # panel fill
GRID = "#1b2337"
TEXT = "#e8edf7"
MUTED = "#7d8aa5"
STRUT = "#3a4a6b"        # healthy lattice, receding
STRUT_HI = "#5b7099"

CLASS_COLOR = {
    "missing": "#ff2d55",
    "broken": "#ff7a1a",
    "thin": "#ffd166",
    "thick": "#22d3ee",
    "dross": "#a78bfa",
    "necked": "#f472b6",
    "node": "#39ff8b",
    "nominal": STRUT,
}

# Sequential ramp for the density contours: page -> ember -> white hot.
HEAT = ["#0d1220", "#1c2748", "#33306b", "#6b2f6e", "#a63258", "#d94f34",
        "#f5921b", "#ffd166", "#fff6c9"]


def heat_cmap(name="poster_heat"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, HEAT)


def rc():
    """Poster-scale matplotlib defaults on the dark page."""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": INK,
        "savefig.facecolor": INK,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------- data

def load_design():
    """(pos_zyx, pairs, geometry) with the run's registration correction applied."""
    import lattice_iou
    pos, pairs, _, applied = lattice_iou.load_design(str(DESIGN), str(CORRECTION))
    if not applied:
        raise RuntimeError("registration correction was not applied")
    geo = json.loads((DEFECTS / "summary.json").read_text())["geometry"]
    return pos, pairs, geo


def load_classes():
    """strut_id-indexed (labels, measurable) from the finished detector run."""
    import csv
    labels, meas = [], []
    with open(DEFECTS / "strut_classes.csv") as fh:
        for row in csv.DictReader(fh):
            labels.append(row["label"])
            meas.append(int(row["measurable"]))
    return np.array(labels), np.array(meas, bool)


def load_stl_truth():
    """Designed-out struts from the 0.stl / 0.5.stl diff, in strut-index space.

    Do NOT read `outputs/stl_missing_struts/0.5_missing_struts.csv` for this. That
    file resolved the STL against the *registered* graph and settled on one of the
    cube's 48 signed axis permutations that is not this one; its 79 ids overlap the
    detector's missing set in exactly 0 places. The orientation has to be resolved
    the way `validate_against_stl` does -- against the nominal design, using the
    detected pattern as the reference, with the margin over the runner-up reported.

    Returns (truth, usable, info).
    """
    import lattice_iou
    import stl_ground_truth as G

    cache = OUT / "stl_truth.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        return z["truth"], z["usable"], json.loads(str(z["info"]))

    pos, pairs, _, _ = lattice_iou.load_design(
        str(ROOT / "data/missing_struts/octet_truss_9x9x9.json"), None)
    cen, _ = G.read_stl_centroids(str(ROOT / "data/missing_struts/stls/0.5.stl"))
    scale, span = G.design_to_stl_scale(cen)

    lab, _ = load_classes()
    st = np.genfromtxt(DEFECTS / "struts.csv", delimiter=",", names=True)
    n = len(lab)
    usable = np.zeros(len(pairs), bool)
    usable[:n] = (st["is_boundary"][:n] == 0) & (st["embedded"][:n] == 0)
    detected = np.zeros(len(pairs), bool)
    detected[:n] = (lab == "missing") & usable[:n]

    perm, signs, ranked = G.find_orientation(cen, pos, pairs, detected, scale)
    _, truth = G.strut_support(cen, pos, pairs, perm, signs, scale)
    s = G.score(detected, truth, usable)
    info = dict(perm=list(perm), signs=list(signs), best=int(ranked[0][0]),
                second=int(ranked[1][0]), plate_axis=G.plate_axis_check(span, perm, signs),
                **{k: float(v) if isinstance(v, float) else int(v) for k, v in s.items()})
    ensure_out()
    np.savez(cache, truth=truth, usable=usable, info=json.dumps(info))
    return truth, usable, info


def strut_midpoints(pos, pairs):
    return 0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])


def density_field(points_2d, extent, bandwidth, grid=320):
    """Gaussian-smoothed count density of scattered points on a regular grid.

    A histogram convolved with a Gaussian rather than a KDE: with fewer than a
    hundred points a KDE's bandwidth rule would pick the bandwidth off the sample
    size, and the field would change shape between defect classes for reasons that
    have nothing to do with the part.
    """
    from scipy.ndimage import gaussian_filter
    (x0, x1, y0, y1) = extent
    h, _, _ = np.histogram2d(
        points_2d[:, 0], points_2d[:, 1], bins=grid,
        range=[[x0, x1], [y0, y1]])
    sigma = bandwidth * grid / (x1 - x0)
    return gaussian_filter(h.T, sigma, mode="constant")


def ensure_out(sub=""):
    d = OUT / sub if sub else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- 3D render

def _tube_poly(seg, radius, sides=14):
    import pyvista as pv
    pts = seg.reshape(-1, 3)
    lines = np.hstack([np.full((len(seg), 1), 2),
                       np.arange(len(seg) * 2).reshape(-1, 2)]).ravel()
    return pv.PolyData(pts, lines=lines).tube(radius=radius, n_sides=sides)


def _camera(pl, bounds, azim, elev, zoom, roll=0.0):
    """Place the camera by orbiting the bounding-box centre at a fixed radius."""
    c = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2])
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    a, e = np.radians(azim), np.radians(elev)
    d = 3.9 * span          # ~1.7x the cube diagonal, so no corner clips the frame
    eye = c + d * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pl.camera_position = (eye.tolist(), c.tolist(), (0, 0, 1))
    pl.camera.roll += roll
    pl.camera.zoom(zoom)


def render_layers(bundles, size=(1800, 1800), azim=38, elev=20, zoom=1.0, roll=0.0,
                  bounds=None):
    """Render each tube bundle as its own image layer on black, same camera.

    Returned separately so they can be composited additively. Compositing beats
    drawing everything into one scene: the defect struts sit *inside* an 18,468-strut
    cage, so in a single pass they are occluded by whatever is in front of them and
    read as broken fragments. Additive blending lets them burn through, which is
    also the honest picture -- the point of the figure is where they are, not which
    of them happens to face the camera.

    `bundles` is a list of dicts: seg (n,2,3 in x,y,z), color, radius, and optional
    glow (float) and sides.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    if bounds is None:
        allpts = np.concatenate([b["seg"].reshape(-1, 3) for b in bundles])
        bounds = (allpts[:, 0].min(), allpts[:, 0].max(), allpts[:, 1].min(),
                  allpts[:, 1].max(), allpts[:, 2].min(), allpts[:, 2].max())

    layers = []
    for b in bundles:
        pl = pv.Plotter(off_screen=True, window_size=list(size))
        pl.set_background("black")
        if len(b["seg"]):
            gl = b.get("glow", 0.0)
            if gl > 0:
                pl.enable_depth_peeling(16)
                for mult, alpha in ((4.0, 0.05), (2.6, 0.09), (1.7, 0.16)):
                    pl.add_mesh(_tube_poly(b["seg"], b["radius"] * mult, 10),
                                color=b["color"], opacity=alpha * gl,
                                ambient=1.0, diffuse=0.0, specular=0.0,
                                lighting=False)
            pl.add_mesh(_tube_poly(b["seg"], b["radius"], b.get("sides", 14)),
                        color=b["color"], smooth_shading=True,
                        ambient=0.30 + 0.5 * gl, diffuse=0.8,
                        specular=0.4 if gl else 0.18, specular_power=20)
        _camera(pl, bounds, azim, elev, zoom, roll)
        layers.append(pl.screenshot(return_img=True).astype(np.float32) / 255.0)
        pl.close()
    return layers


def depth_bins(seg, azim, elev, n=3):
    """Split segments into n slabs by distance along the view direction.

    Rendering the slabs at different gains gives the cage atmospheric perspective.
    Without it an 18,468-strut wireframe composites to a flat grey mat and the cube
    loses its near corner.
    """
    a, e = np.radians(azim), np.radians(elev)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    d = seg.mean(1) @ view
    edges = np.quantile(d, np.linspace(0, 1, n + 1))
    edges[0] -= 1
    return [seg[(d > edges[i]) & (d <= edges[i + 1])] for i in range(n)]


def composite(layers, gains, page=INK, crop=True, margin=0.02):
    """Screen-blend emissive layers over the page colour.

    `crop` trims to the rendered content before the page colour is laid in, so the
    subject fills its panel whatever the camera left as margin. Cropping on the
    pre-page image is the point: once the background is the page colour there is no
    longer a threshold that separates subject from surround.
    """
    from matplotlib.colors import to_rgb
    out = np.zeros_like(layers[0])
    for lay, g in zip(layers, gains):
        out = 1.0 - (1.0 - out) * (1.0 - np.clip(lay * g, 0, 1))

    if crop:
        lum = np.max(np.stack(layers), axis=(0, 3))
        rows = np.where(lum.max(1) > 0.02)[0]
        cols = np.where(lum.max(0) > 0.02)[0]
        if len(rows) and len(cols):
            m = int(margin * max(out.shape[:2]))
            r0, r1 = max(rows[0] - m, 0), min(rows[-1] + m + 1, out.shape[0])
            c0, c1 = max(cols[0] - m, 0), min(cols[-1] + m + 1, out.shape[1])
            out = out[r0:r1, c0:c1]

    bg = np.array(to_rgb(page), np.float32)
    out = bg + (1.0 - bg) * out
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
