"""Draw one mask slice, with the geometry the detector actually used laid over it.

Each panel is a single plane cut out of the segmentation. On top of it go only quantities
that exist in the finished run: the nominal cylinder `measure_struts` paints, the envelope
that localises its union, the tube `measure_connectivity` searches, the annulus
`excess_frac` integrates over, the sphere `measure_node_sphere_fill` tests, and the
per-section equivalent radius `src.strut_sections` measured. Nothing is recomputed and
nothing is drawn that the detector did not use.

Distances on both axes are microns, so a reader can measure the defect off the figure.
"""
from __future__ import annotations

import numpy as np

import maskplane as MP
import style as S

# Contour spacing for the distance field, fixed across every figure so band counts are
# comparable between classes.
BAND_UM = 20.0
LEVELS = np.arange(-420.0, 160.0 + 1e-9, BAND_UM)


def sdf_contours(ax, field_vox, extent_vox, um, color, vmin=-420.0, vmax=150.0):
    """Filled + line contours of the signed distance field, and the surface at zero.

    The heavy contour is the 0-level: it is the segmented surface itself, so it is the one
    line in the figure that is a measurement rather than an annotation.
    """
    f = field_vox * um
    ext = tuple(np.asarray(extent_vox, float) * um)
    cmap = S.class_cmap(color, vmin, vmax)
    cf = ax.contourf(f, levels=LEVELS, cmap=cmap, extent=ext, extend="both",
                     antialiased=True)
    cf.set_edgecolor("face")
    ax.contour(f, levels=LEVELS[::2], extent=ext, colors="#ffffff",
               linewidths=0.28, alpha=0.16)
    ax.contour(f, levels=[0.0], extent=ext, colors="#ffffff", linewidths=1.5)
    ax.set_facecolor(S.VOID[0])
    return cf, ext


def _hline_pair(ax, r_um, s0, s1, **kw):
    for sign in (1, -1):
        ax.plot([s0, s1], [sign * r_um, sign * r_um], **kw)


def strut_geometry(ax, g, show=("cylinder", "trim")):
    """The design geometry for one strut, in microns along and across its axis.

    `g` carries the lengths already converted: L (node to node), trim span, and the radii
    the detector uses -- nominal, envelope 1.6x, tube 2.0x, annulus 2.2x.
    """
    L, a, b = g["L"], g["trim0"], g["trim1"]

    if "tube" in show:
        _hline_pair(ax, g["r_tube"], 0, L, color="#8fa4c8", lw=0.9, ls=(0, (5, 3)),
                    alpha=0.75, zorder=6)
        ax.add_patch(_rect(0, -g["r_tube"], L, 2 * g["r_tube"], "#8fa4c8", 0.05))
    if "envelope" in show:
        _hline_pair(ax, g["r_env"], a, b, color="#7d8aa5", lw=0.8, ls=(0, (1.6, 2.2)),
                    alpha=0.85, zorder=6)
    if "annulus" in show:
        m0, m1 = g["mid0"], g["mid1"]
        for sign in (1, -1):
            lo = min(sign * g["r_nom"], sign * g["r_dross"])
            ax.add_patch(_rect(m0, lo, m1 - m0, abs(g["r_dross"] - g["r_nom"]),
                               g["color"], 0.16, edge=g["color"], lw=0.8, ls=(0, (4, 2))))
    if "cylinder" in show:
        ax.add_patch(_rect(a, -g["r_nom"], b - a, 2 * g["r_nom"], "none", 0,
                           edge="#ffffff", lw=1.15, ls=(0, (5, 3)), alpha=0.9))
    if "trim" in show:
        for x in (a, b):
            ax.axvline(x, color="#ffffff", lw=0.6, ls=(0, (1.5, 2.5)), alpha=0.35,
                       zorder=6)
    for x, lab in ((0, "node A"), (L, "node B")):
        ax.axvline(x, color="#f5921b", lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=6)
        ax.text(x, ax.get_ylim()[1], f" {lab} ", color="#f5921b", fontsize=7.6,
                ha="center", va="top", zorder=26,
                bbox=dict(boxstyle="square,pad=0.18", fc=S.INK, ec="none", alpha=0.75))


def _rect(x, y, w, h, fc, alpha, edge="none", lw=0.0, ls="-"):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=fc, alpha=alpha, edgecolor=edge,
                     linewidth=lw, linestyle=ls, zorder=7)


def radius_trace(ax, g, r_eq_um, empty, color):
    """The detector's own per-section equivalent radius, drawn as a ribbon.

    r_eq = sqrt(area / pi) of a plane cut *perpendicular* to the strut, so it is an
    area-equivalent radius, not the half-width of this particular cut. The two agree for a
    round strut and diverge for a flattened one, which is the point of plotting them on
    the same axes.
    """
    s = np.linspace(g["trim0"], g["trim1"], len(r_eq_um))
    r = np.where(empty, np.nan, r_eq_um)
    ax.plot(s, r, color=color, lw=1.5, zorder=12, solid_capstyle="round")
    ax.plot(s, -r, color=color, lw=1.5, zorder=12, solid_capstyle="round")
    ax.fill_between(s, -r, r, color=color, alpha=0.10, zorder=5)
    if empty.any():
        for k in np.where(empty)[0]:
            ax.plot([s[k]], [0], marker="x", ms=5.5, mew=1.6, color=color, zorder=13)


def strut_slice(ax, local, run, i, color, phi=None, reach_factor=4.0, res=0.10,
                margin=0.30):
    """Sample and draw the lateral view of one strut. Returns the geometry dict."""
    p0, p1 = run.ends(i)
    L_vox = float(np.linalg.norm(p1 - p0))
    if phi is None:
        live = ~run.prof["empty"][i] & ~run.prof["border"][i]
        phi = MP.centreline_phi(run.prof["c1"][i], run.prof["c2"][i], live)

    reach = reach_factor * run.r_nom
    field, ext = MP.axial_plane(local, p0, p1, phi, reach, margin=margin, res=res)
    sdf_contours(ax, field, ext, run.um, color)

    um, r = run.um, run.r_nom
    g = dict(L=L_vox * um, trim0=0.20 * L_vox * um, trim1=0.80 * L_vox * um,
             mid0=0.25 * L_vox * um, mid1=0.75 * L_vox * um,
             r_nom=r * um, r_env=1.6 * r * um, r_tube=2.0 * r * um,
             r_dross=2.2 * r * um, reach=reach * um, phi=phi, color=color)
    ax.set_xlim(ext[0] * um, ext[1] * um)
    ax.set_ylim(-reach * um, reach * um)
    ax.set_aspect("equal")
    S.frame(ax)
    return g


def cross_slice(ax, local, run, i, t, color, half_factor=2.5, res=0.08):
    """One plane cut perpendicular to the strut, with the nominal circle on it."""
    from matplotlib.patches import Circle
    p0, p1 = run.ends(i)
    half = half_factor * run.r_nom
    field, ext = MP.cross_plane(local, p0, p1, t, half, res=res)
    sdf_contours(ax, field, ext, run.um, color)
    ax.add_patch(Circle((0, 0), run.r_nom * run.um, fill=False, edgecolor="#ffffff",
                        lw=1.15, ls=(0, (5, 3)), alpha=0.9, zorder=10))
    lim = half * run.um
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    S.frame(ax)
    return ext


def node_slice(ax, local, run, node_id, color, e_u, e_v, half_vox, res=0.10, offset=0.0):
    """A plane through a junction, spanned by two of its incident strut directions."""
    from matplotlib.patches import Circle
    centre = run.node_pos[node_id]
    field, ext = MP.free_plane(local, centre, e_u, e_v, half_vox, half_vox,
                               res=res, offset=offset)
    sdf_contours(ax, field, ext, run.um, color)
    r_node = run.geo["nominal_node_radius_vox"] * run.um
    ax.add_patch(Circle((0, 0), r_node, fill=False, edgecolor="#ffffff", lw=1.4,
                        ls=(0, (5, 3)), zorder=12))
    lim = half_vox * run.um
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    S.frame(ax)
    return r_node
