"""Three-dimensional renders of the voxels behind a defect call, cut out of the CT.

The 2-D galleries this replaces -- `classify_strut_defects.plot_section_gallery` and
`node_health.plot_node_gallery` -- both draw single planes through the binary mask. That
is enough to *carry* a number and not enough to *show* one: a plane through a mask is a
silhouette, it has no depth, and at whole-voxel resolution its edge is a staircase against
a strut only ~4.4 voxels across. Nothing here measures anything. Every label and every
statistic still comes from the cached `detect_lattice_defects` run; this module draws the
material those numbers were computed from.

Three decisions were made against measurements rather than taste, and the figures depend
on all three:

**The surface comes off the CT, not off the mask.** Marching cubes on a sub-voxel resample
of the raw CT at the segmentation threshold (`ISOLEVEL`, the 40127 Otsu cut that produced
`mask.tif`) gives a smooth surface. Marching cubes on the binary mask gives the same
surface quantised to the voxel grid, which on a 4.4-voxel strut is mostly quantisation.
The mask is still used, as a *gate*: dilated by a couple of voxels and multiplied in, it
stops a neighbouring strut clipped by the window from fusing onto the subject. So the
render is the mask applied over the CT, which is exactly what the detector saw.

**The isosurface is never coloured by CT intensity.** It is tempting -- the greys are
right there -- and it is meaningless: the surface *is* the level set of the CT at
`ISOLEVEL`, so by construction the CT is constant on it to within the interpolation error.
Colouring by it produces a mottled texture that is pure noise and reads, misleadingly, as
density variation. CT greys are shown on the **cut plane** instead, where the value is a
real measurement of what is inside the strut.

**A junction box has to be about one strut length across, not fourteen voxels.** The two
interior nodes this project calls missing (1935, 2189) sit in a region with *zero* mask
voxels out to +-30 voxels, CT maximum 34664 against the 40127 cut. Drawn at the old
+-14 voxel crop they are blank frames, which is true and shows nothing. Drawn at +-1 strut
length the surrounding cage comes into the box and the picture becomes a lattice with a
hole in it, which is the claim.

The camera is parallel-projected and its scale is locked to the *window*, not to the
subject's own bounding box. Fitting each subject to its own frame is what makes a thin
strut and a thick strut come out the same size on the page, i.e. it hides the one quantity
the row exists to show.

Requires pyvista/VTK, which are NOT dependencies of the MCP server -- they are imported
inside the render functions so that importing this module costs nothing and a machine
without VTK fails with a sentence rather than a traceback.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from skimage.measure import marching_cubes

# The Otsu cut that produced data/9x9x9_octet_lattice/segmentation/mask.tif. Surfacing the
# CT here rather than surfacing the mask is what buys the sub-voxel detail; using any other
# level would draw a boundary the detector never used.
ISOLEVEL = 40127.0

# A fixed greyscale window for every raw-CT panel in every figure. The volume's own
# profile (segmentation/report.md): p01 29789, p50 32409, p99 54370, i.e. background sits
# just under 32k of 65k and metal runs to ~54k. Letting each panel autoscale is what makes
# an all-background crop render as pure black -- indistinguishable from "no data" when it
# is in fact the measurement, and the entire content of a missing-node row.
CT_WINDOW = (30000.0, 54000.0)

SURFACE = "#fcfcfb"
METAL = "#c2c5cb"
GHOST = "#e34948"
AXIS = "#2a78d6"
_INK_3D = "#52514e"


def require_pyvista():
    """Import pyvista, or explain what is missing instead of raising ImportError."""
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "3-D rendering needs pyvista and VTK, which are not in requirements.txt. "
            f"Install them with `pip install pyvista` ({error})."
        ) from error
    pv.OFF_SCREEN = True
    return pv


# --------------------------------------------------------------------- sampling

def _perp(u):
    """Two unit vectors spanning the plane perpendicular to u."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def sym_axis(half, res):
    """A sample axis that is *exactly* symmetric about zero, spanning at least +-half.

    `np.arange(-half, half + eps, res)` is not, whenever `half / res` is not an integer:
    the wide node box (half 58.176, res 0.5) came out running -58.176 .. +57.824, so its
    centre sample sat at -0.176 rather than at the junction, and any code reconstructing
    the lower bound as -o[-1] put the isosurface 0.353 voxels away from the sphere and
    spokes drawn at the true origin. Rounding the half-width up to a whole number of steps
    removes the entire class of error rather than correcting it downstream.
    """
    n = int(np.ceil(half / res - 1e-9))
    return np.arange(-n, n + 1) * res


def _sample(vol, pts, shape):
    """Trilinear sample of a memmapped volume on `pts` (3, N), cropped to what is needed.

    The crop is not an optimisation detail. The CT is 1.0 GB and the mask 519 MB against
    about 2.5 GB of free memory on this box, so anything that materialises a whole volume
    as float32 fails outright.
    """
    lo = np.maximum(np.floor(pts.min(1)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(pts.max(1)).astype(int) + 2, vol.shape)
    crop = np.ascontiguousarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    out = map_coordinates(crop, pts - lo[:, None], order=1, mode="constant", cval=0.0)
    del crop
    return out.reshape(shape)


class Frame:
    """A strut resampled into its own frame: axis along index 0, e1 and e2 across it.

    `ct` is the raw CT interpolated onto that grid and `gate` is the dilated mask on the
    same grid. `field` is the two combined -- the CT everywhere the segmentation says
    metal, and hard zero elsewhere -- which is what gets surfaced.
    """

    def __init__(self, ct, gate, t, s, res, length, r_nom):
        self.ct = ct
        self.gate = gate
        self.t = t
        self.s = s
        self.res = res
        self.length = length
        self.r_nom = r_nom
        self.origin = np.array([t[0], s[0], s[0]])
        self.field = np.where(gate, ct, 0.0)

    @property
    def half(self):
        return float(-self.s[0])

    @property
    def bounds(self):
        return (self.t[0], self.t[-1], self.s[0], self.s[-1], self.s[0], self.s[-1])

    def index_at(self, along):
        """Grid index of an axial position given as a fraction of the design length."""
        return int(round((along * self.length - self.t[0]) / self.res))


def strut_frame(ct, mask, p0, p1, r_nom, half_r=3.2, margin=0.22, res=0.22, dilate=2):
    """Resample CT and mask into one strut's own frame.

    `half_r` is the window half-width in nominal strut radii and `margin` how far past
    each junction the window reaches, as a fraction of the strut length -- both junctions
    stay in frame, because where a strut stops is only legible against what it should have
    joined.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.linalg.norm(d))
    u = d / length
    e1, e2 = _perp(u)
    half = half_r * r_nom

    t = np.arange(-margin * length, length * (1 + margin) + 1e-9, res)
    s = sym_axis(half, res)
    T, S1, S2 = np.meshgrid(t, s, s, indexing="ij")
    pts = (np.asarray(p0, float)[:, None, None, None]
           + T * u[:, None, None, None]
           + S1 * e1[:, None, None, None]
           + S2 * e2[:, None, None, None]).reshape(3, -1)

    g = _sample(ct, pts, T.shape)
    m = _sample(mask, pts, T.shape) >= 0.5
    gate = binary_dilation(m, iterations=dilate) if dilate else m
    del T, S1, S2, pts, m
    return Frame(g, gate, t, s, res, length, r_nom)


def node_box(ct, centre, half, res, mask=None, dilate=2):
    """Axis-aligned CT cube about a junction, resampled at `res` voxels.

    No rotation: a junction has no axis of its own, and leaving the box aligned with the
    volume keeps the three struts that run along the grid recognisable between figures.

    Returns (field, raw, o). `field` is gated by the mask when one is given and is what
    gets surfaced; `raw` is the untouched CT and is what the flat panels are drawn from --
    a gated panel shows hard zeros outside the metal, which is a rendering artefact
    masquerading as a measurement of background.
    """
    o = sym_axis(half, res)
    Z, Y, X = np.meshgrid(o, o, o, indexing="ij")
    c = np.asarray(centre, float)
    pts = np.stack([Z + c[0], Y + c[1], X + c[2]]).reshape(3, -1)
    raw = _sample(ct, pts, Z.shape)
    field = raw
    if mask is not None:
        m = _sample(mask, pts, Z.shape) >= 0.5
        gate = binary_dilation(m, iterations=dilate) if dilate else m
        field = np.where(gate, raw, 0.0)
        del m, gate
    del Z, Y, X, pts
    return field, raw, o


# ------------------------------------------------------------------- geometry

def isosurface(field, origin, res, level=ISOLEVEL, smooth=12, pass_band=0.12):
    """Marching-cubes surface of `field`, in the frame's own coordinates.

    Returns None when nothing in the box reaches the isolevel -- which is a result, not an
    error: it is what a missing strut and a missing node both look like.

    Taubin smoothing is a display choice and it shrinks nothing (that is the point of
    Taubin over Laplacian), but it is still a choice, so callers should say so in the
    caption. Thickness is read off `sections.npz`, never off this mesh.
    """
    pv = require_pyvista()
    if not np.isfinite(field).any() or field.max() < level:
        return None
    verts, faces, _, _ = marching_cubes(field, level=level, spacing=(res, res, res))
    verts = verts + np.asarray(origin, float)
    mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    if smooth and mesh.n_points:
        mesh = mesh.smooth_taubin(n_iter=smooth, pass_band=pass_band)
    return mesh


def _slab(frame, k, thick):
    """A closed disc of material around section `k`, for threading along the axis.

    Zero-padded on the axial faces before surfacing so the disc comes out closed. Without
    the pad marching cubes leaves the ends open and the disc renders as a hollow ring lit
    from inside.
    """
    lo = max(k - thick, 0)
    hi = min(k + thick + 1, frame.field.shape[0])
    sub = np.pad(frame.field[lo:hi], ((1, 1), (0, 0), (0, 0)))
    origin = np.array([frame.t[lo] - frame.res, frame.s[0], frame.s[0]])
    return isosurface(sub, origin, frame.res, smooth=6, pass_band=0.2)


# --------------------------------------------------------------------- camera

def _eye_dir(azim, elev):
    """Unit vector from the subject towards the camera, for the given orbit angles."""
    a, e = np.radians(azim), np.radians(elev)
    return np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])


def _place(pl, frame_bounds, half_v, azim, elev, pad=1.10, up=(0, 0, 1)):
    """Parallel camera orbiting the window centre, scale locked to the window.

    `half_v` is half the vertical world extent the panel must contain. Locking it to the
    window rather than calling reset_camera on the mesh is what keeps rows comparable:
    reset_camera would fit each strut to the frame and a thin one would come out the same
    size on the page as a thick one.
    """
    b = frame_bounds
    c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    eye = c + 1000.0 * _eye_dir(azim, elev)
    pl.camera_position = (eye.tolist(), c.tolist(), list(up))
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = half_v * pad


def _plotter(size):
    pv = require_pyvista()
    pl = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])])
    pl.set_background(SURFACE)
    pl.enable_depth_peeling(16)
    return pl


def _metal(pl, mesh):
    if mesh is None or not mesh.n_points:
        return
    pl.add_mesh(mesh, color=METAL, smooth_shading=True, ambient=0.30, diffuse=0.78,
                specular=0.32, specular_power=24)


def _ghost_cylinder(pl, frame, span=(0.0, 1.0), opacity=0.30):
    """The nominal 350 um cylinder, over the span it is actually measured on.

    Front faces are culled, so only the *far* wall of the cylinder is drawn and the strut
    occludes it. Rendered whole it is a solid pink slab across the middle of the panel
    that hides the one thing the panel is for; culled it reads as an envelope the material
    sits inside, and where the material has gone the envelope is visible through the gap.

    It stops at the trim points because that is the span the sections are measured over --
    running it into the junctions would imply the nominal cylinder reaches material that
    is deliberately excluded from every statistic.
    """
    pv = require_pyvista()
    t0, t1 = span[0] * frame.length, span[1] * frame.length
    cyl = pv.Cylinder(center=((t0 + t1) / 2, 0, 0), direction=(1, 0, 0),
                      radius=frame.r_nom, height=t1 - t0, capping=False)
    pl.add_mesh(cyl, color=GHOST, opacity=opacity, lighting=False, culling="front")


def _shot(pl):
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# --------------------------------------------------------------------- panels

def panel_size(frame, width=1500, pad=1.10, vfactor=1.0):
    """Panel pixels matching the window's own aspect, so nothing is padded with white."""
    h_world = 2 * frame.half * pad * vfactor
    w_world = frame.t[-1] - frame.t[0]
    return width, max(120, int(round(width * h_world / w_world)))


def render_lateral(frame, mesh, width=1500, azim=6, elev=13, trim=0.20):
    """The strut side-on: the whole surface, lit, with the nominal cylinder through it."""
    size = panel_size(frame, width, vfactor=1.18)
    pl = _plotter(size)
    _metal(pl, mesh)
    _ghost_cylinder(frame=frame, pl=pl, span=(trim, 1 - trim))
    _place(pl, frame.bounds, frame.half, azim, elev, pad=1.18)
    return _shot(pl)


def render_cutaway(frame, mesh, width=1500, azim=0, elev=30, trim=0.20):
    """The strut opened along its own axis, CT greys painted on the cut face.

    The half nearer the camera is clipped away and the plane it was cut on is textured
    with the CT. Alpha is zero outside metal, so the far half of the surface shows through
    around the cut rather than being hidden behind an opaque rectangle -- which is what
    makes this a cutaway and not just a slice with a mesh behind it.
    """
    pv = require_pyvista()
    size = panel_size(frame, width, vfactor=1.55)
    pl = _plotter(size)

    if mesh is not None and mesh.n_points:
        _metal(pl, mesh.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=True))

    mid = frame.field.shape[2] // 2
    grey = frame.ct[:, :, mid]
    inside = frame.gate[:, :, mid]
    # The common CT_WINDOW, not this strut's own percentiles. Per-panel autoscaling makes
    # every strut come out with the same interior contrast whatever its density, which is
    # the one thing a cut face is worth drawing for.
    lo, hi = CT_WINDOW
    v = np.clip((grey - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = np.zeros(grey.shape + (4,), np.uint8)
    rgba[..., :3] = (v[..., None] * 255).astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0)
    tex = np.ascontiguousarray(rgba.transpose(1, 0, 2))

    plane = pv.Plane(center=((frame.t[0] + frame.t[-1]) / 2, 0, 0), direction=(0, 0, 1),
                     i_size=frame.t[-1] - frame.t[0], j_size=frame.s[-1] - frame.s[0],
                     i_resolution=1, j_resolution=1)
    plane.texture_map_to_plane(inplace=True)
    pl.add_mesh(plane, texture=pv.Texture(tex), lighting=False, opacity=1.0)

    _ghost_cylinder(frame=frame, pl=pl, span=(trim, 1 - trim), opacity=0.13)
    _place(pl, frame.bounds, frame.half, azim, elev, pad=1.55)
    return _shot(pl)


def render_disks(frame, width=1500, n=8, trim=0.20, azim=26, elev=16, thick=2):
    """The cross-sections as solid discs threaded along the axis, in one 3-D scene.

    They recede, so they are not the panel to compare shapes on -- the flat strip below
    the row is. What this one shows that the strip cannot is that the discs are cut from
    one continuous object, and where along that object each of them sits.
    """
    size = panel_size(frame, width, vfactor=1.30)
    pl = _plotter(size)
    for along in np.linspace(trim, 1 - trim, n):
        disc = _slab(frame, frame.index_at(along), thick)
        if disc is not None and disc.n_points:
            pl.add_mesh(disc, color=METAL, smooth_shading=True, ambient=0.32,
                        diffuse=0.75, specular=0.30, specular_power=20)
    _ghost_cylinder(frame=frame, pl=pl, span=(trim, 1 - trim), opacity=0.10)
    _place(pl, frame.bounds, frame.half, azim, elev, pad=1.30)
    return _shot(pl)


def section_images(frame, n=8, trim=0.20):
    """The flat cross-sections: CT greys, the mask boundary, in the strut's own frame.

    Returns (greys, gates, extent) with greys/gates as (n, m, m). Same planes the discs
    are cut on, drawn head-on so their shapes can be compared across rows.
    """
    idx = [frame.index_at(a) for a in np.linspace(trim, 1 - trim, n)]
    idx = [min(max(k, 0), frame.field.shape[0] - 1) for k in idx]
    greys = np.stack([frame.ct[k] for k in idx])
    gates = np.stack([frame.gate[k] for k in idx])
    return greys, gates, (frame.s[0], frame.s[-1], frame.s[0], frame.s[-1])


# ----------------------------------------------------------------------- nodes

def render_node(field, o, r_sphere=None, spokes=None, width=900, azim=38, elev=24,
                pad=1.02, show_sphere=True, open_to_camera=False, inset_half=None,
                triad=True):
    """One junction in 3-D: the material, the sphere the fill is measured in, the spokes.

    `spokes` are the design directions of the struts that should meet here, as (k, 3)
    offsets in the box's own (z, y, x) frame. On a junction that printed they disappear
    inside the metal; on one that did not they hang in the void pointing at nothing, and
    that single contrast is the whole detector drawn.

    `open_to_camera` clips everything nearer the camera than the junction away. Without it
    a wide box is useless for its own purpose: a lattice is opaque at one strut length, so
    the hole the figure exists to show sits behind four or five struts and the row reads as
    an ordinary piece of cage. Cutting the near half puts the junction's own position on
    the cut surface, in the open.
    """
    pv = require_pyvista()
    half = float(o[-1])
    pl = _plotter((width, width))

    mesh = isosurface(field, (o[0], o[0], o[0]), float(o[1] - o[0]), smooth=10,
                      pass_band=0.15)
    if open_to_camera and mesh is not None and mesh.n_points:
        mesh = mesh.clip(normal=_eye_dir(azim, elev).tolist(), origin=(0, 0, 0),
                         invert=True)
    _metal(pl, mesh)

    if show_sphere and r_sphere:
        pl.add_mesh(pv.Sphere(radius=r_sphere, center=(0, 0, 0), theta_resolution=48,
                              phi_resolution=48),
                    color=GHOST, opacity=0.30, lighting=False)
    if spokes is not None and len(spokes):
        seg = []
        for d in np.asarray(spokes, float):
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            seg.append([[0, 0, 0], (d / n * min(half * 0.94, n)).tolist()])
        if seg:
            seg = np.array(seg)
            lines = pv.PolyData(
                seg.reshape(-1, 3),
                lines=np.hstack([np.full((len(seg), 1), 2),
                                 np.arange(len(seg) * 2).reshape(-1, 2)]).ravel())
            pl.add_mesh(lines.tube(radius=max(0.22, half * 0.012)), color=AXIS,
                        opacity=0.9, lighting=True, ambient=0.4)

    # Where the next panel along is looking. Without it the wide and close views are two
    # unrelated pictures at an undeclared zoom ratio, and a feature cannot be carried from
    # one to the other -- which is the complaint a reader makes first.
    if inset_half:
        pl.add_mesh(pv.Box(bounds=(-inset_half, inset_half) * 3).extract_all_edges(),
                    color=_INK_3D, line_width=2, opacity=0.55, lighting=False)

    _place(pl, (-half, half, -half, half, -half, half), half, azim, elev, pad=pad,
           up=(1, 0, 0))
    # Volume axes, not the panel's. Mesh axis 0/1/2 are the volume's z/y/x, and the camera
    # is oblique, so without this marker no direction in a 3-D panel can be matched to a
    # direction in the flat CT panels beside it.
    if triad:
        pl.add_axes(xlabel="z", ylabel="y", zlabel="x", line_width=3, labels_off=False,
                    color=_INK_3D, viewport=(0.0, 0.0, 0.26, 0.26))
    return _shot(pl)


def node_planes(raw, o):
    """The three orthogonal centre planes of a node box, on raw CT greys.

    Kept from the old gallery because they are the view in which "there is nothing here"
    is checkable rather than inferred from a shaded surface -- provided they are drawn on
    the common `CT_WINDOW`, so that background reads as grey and only a genuinely empty
    panel is uniformly flat.
    """
    c = raw.shape[0] // 2
    return [raw[c], raw[:, c], raw[:, :, c]], (o[0], o[-1], o[0], o[-1])
