"""Cut a single plane out of the segmentation, and draw the method's own geometry on it.

Everything here reads the **CT mask** -- the segmentation `segment_ct_dataset` produced --
never the raw CT. The figures built on it therefore show what the detector saw, not what a
human would see in a greyscale slice, which is the whole point: a claim about a strut has
to be legible in the same field the claim was measured in.

Two fields are sampled, both derived from the mask alone:

  `sdf`   signed distance to the segmented surface, positive inside metal, in voxels.
          The 0-level *is* the mask boundary, so one contour set carries both the
          silhouette and a topographic map of how far every point sits from metal. It is
          also the field `measure_struts` reads for `radius_med_vox` -- an EDT of the
          cropped material sampled on the centreline -- so the contours are the detector's
          own local-thickness measurement, drawn rather than reduced to a scalar.
  `occ`   the interpolated mask itself, 0..1. Only used where a hard silhouette is wanted.

The plane is specified by an origin and two in-plane unit vectors, so the same code cuts a
plane *containing* a strut axis (the lateral view, where a gap is a visible column of
nothing) and a plane *perpendicular* to it (the cross-section the shape classes are
measured on). Sampling is sub-voxel and cubic: an as-built strut is ~4.4 voxels across, so
at whole-voxel resolution a contour is a staircase and carries no shape.

Coordinates are (z, y, x) voxels throughout, matching `src.lattice_iou`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ mask access

def open_mask(path):
    """Memory-map the mask. It is 519 MB and this box has ~1 GB free."""
    import tifffile
    return tifffile.memmap(str(path))


def crop(mask, centre, half):
    """Bounded cube of the mask around `centre`, plus the lower corner it starts at."""
    centre = np.asarray(centre, float)
    half = np.broadcast_to(np.asarray(half, float), (3,))
    lo = np.maximum(np.floor(centre - half).astype(int), 0)
    hi = np.minimum(np.ceil(centre + half).astype(int) + 1, mask.shape)
    return np.ascontiguousarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) > 0, lo


class Local:
    """A cropped neighbourhood of the mask and the fields sampled off it."""

    def __init__(self, mask, centre, half, smooth=0.7):
        self.occ, self.lo = crop(mask, centre, half)
        self.smooth = smooth
        self._sdf = None

    @property
    def sdf(self):
        """Signed distance to the mask surface in voxels, + inside.

        Smoothed by `smooth` voxels before use. Without it the discrete transform is a
        staircase of half-voxel steps and every contour line inherits the voxel grid; the
        0-level moves by well under a tenth of a voxel, so the silhouette is unchanged.
        """
        if self._sdf is None:
            if self.occ.any() and not self.occ.all():
                inside = distance_transform_edt(self.occ)
                outside = distance_transform_edt(~self.occ)
                d = inside - outside
            else:
                d = np.where(self.occ, 1.0, -1.0) * 50.0
            self._sdf = gaussian_filter(d.astype(np.float32), self.smooth)
        return self._sdf

    def sample(self, pts, field=None, order=3):
        """Interpolate a field at world (z, y, x) points of shape (..., 3)."""
        f = self.sdf if field is None else field
        flat = np.asarray(pts, float).reshape(-1, 3) - self.lo
        v = map_coordinates(f, flat.T, order=order, mode="nearest")
        return v.reshape(np.asarray(pts).shape[:-1])


# ------------------------------------------------------------------ plane geometry

def perp_basis(u):
    """The same two perpendicular vectors `src.strut_sections` uses.

    Sharing the convention is what lets a cut angle be quoted against the centroid
    offsets the detector already stored in `prof_c1` / `prof_c2`.
    """
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def plane_grid(origin, e_u, e_v, u, v):
    """(len(v), len(u), 3) world points of a regular grid on the plane."""
    U, V = np.meshgrid(u, v, indexing="xy")
    return (origin[None, None, :] + U[..., None] * e_u[None, None, :]
            + V[..., None] * e_v[None, None, :])


def axial_plane(local, p0, p1, phi, reach, margin=0.30, res=0.10):
    """The lateral view: a plane *containing* the strut axis, rolled by `phi` about it.

    Returns (field, extent) with the horizontal axis in voxels along the strut measured
    from node `p0`, and the vertical axis the signed radial offset in the cut direction.
    A missing or severed strut is a column of nothing here, which is exactly what a stack
    of independent cross-sections cannot show.
    """
    d = p1 - p0
    length = float(np.linalg.norm(d))
    u_hat = d / length
    e1, e2 = perp_basis(u_hat)
    e_v = np.cos(phi) * e1 + np.sin(phi) * e2

    s = np.arange(-margin * length, (1 + margin) * length + 1e-9, res)
    r = np.arange(-reach, reach + 1e-9, res)
    pts = plane_grid(p0, u_hat, e_v, s, r)
    return local.sample(pts), (s[0], s[-1], r[0], r[-1])


def cross_plane(local, p0, p1, t, half, res=0.08):
    """A plane perpendicular to the strut axis at fraction `t` of its length.

    This is the frame the shape classes are measured in. A healthy strut is a circle here
    however it is oriented in the volume, which is what defeats every axis-aligned 2D
    metric on this oblique lattice.
    """
    d = p1 - p0
    u_hat = d / np.linalg.norm(d)
    e1, e2 = perp_basis(u_hat)
    g = np.arange(-half, half + 1e-9, res)
    pts = plane_grid(p0 + t * d, e1, e2, g, g)
    return local.sample(pts), (g[0], g[-1], g[0], g[-1])


def free_plane(local, centre, e_u, e_v, half_u, half_v, res=0.10, offset=0.0):
    """An arbitrary plane, optionally pushed `offset` voxels along its normal.

    Used for the nodes, where the informative cut is the one spanned by two of the twelve
    incident strut directions rather than anything referred to a single axis.
    """
    n = np.cross(e_u, e_v)
    n /= np.linalg.norm(n)
    u = np.arange(-half_u, half_u + 1e-9, res)
    v = np.arange(-half_v, half_v + 1e-9, res)
    pts = plane_grid(np.asarray(centre, float) + offset * n, e_u, e_v, u, v)
    return local.sample(pts), (u[0], u[-1], v[0], v[-1])


# ------------------------------------------------------------------ cut selection

def centreline_phi(prof_c1, prof_c2, live):
    """Roll angle whose plane contains the design axis *and* the measured centreline.

    `prof_c1` / `prof_c2` are the signed section centroids `src.strut_sections` fitted,
    in the `perp_basis` frame. Cutting through their mean direction is a choice made by
    the measurement rather than by eye -- the alternative, picking the prettiest angle by
    hand, would make every figure a best case. Falls back to 0 when the strut is
    concentric enough that the direction is noise.
    """
    if not np.any(live):
        return 0.0
    c1 = float(np.nanmean(np.asarray(prof_c1)[live]))
    c2 = float(np.nanmean(np.asarray(prof_c2)[live]))
    return 0.0 if np.hypot(c1, c2) < 0.25 else float(np.arctan2(c2, c1))


def node_plane_basis(node_pos, neighbour_pos):
    """Two incident strut directions spanning the most informative cut through a node.

    An octet junction joins twelve struts. The plane through the two most nearly
    perpendicular of them contains four strut axes, so a sound node reads as an X of metal
    and an absent one as a hole with four stubs pointing at it -- the same picture the
    sphere-fill test reduces to a single number.
    """
    d = np.asarray(neighbour_pos, float) - np.asarray(node_pos, float)
    n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-6] / n[n > 1e-6, None]
    best, pair = 1.0, (0, 1)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            c = abs(float(d[i] @ d[j]))
            if c < best:
                best, pair = c, (i, j)
    e_u = d[pair[0]]
    e_v = d[pair[1]] - (d[pair[1]] @ e_u) * e_u
    return e_u, e_v / np.linalg.norm(e_v)
