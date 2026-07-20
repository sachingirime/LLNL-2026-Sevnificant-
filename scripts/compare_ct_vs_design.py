"""
Side-by-side: as-BUILT CT slice (.tif) vs as-DESIGNED geometry (.stl).

- Left:   one grayscale CT slice from the TIFF volume (real print, has noise/pores/defects).
- Middle: same slice thresholded (solid material vs air) — what segmentation extracts.
- Right:  a cross-section outline of the STL design mesh at a chosen z-plane (perfect geometry).

NOTE: TIFF, STL and JSON live in DIFFERENT coordinate systems (see data note.txt),
so the two cross-sections are NOT registered — this figure is for *conceptual* comparison
of "real vs ideal", not a pixel-aligned overlay.
"""
import os, struct
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from skimage.filters import threshold_otsu

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIF  = os.path.join(REPO, "data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif")
STL  = os.path.join(REPO, "data/missing_struts/stls/0.stl")
OUT  = os.path.join(REPO, "outputs/exploration/ct_vs_design.png")

def read_stl(path):
    with open(path, "rb") as fh:
        fh.read(80); n = struct.unpack("<I", fh.read(4))[0]
        data = np.frombuffer(fh.read(n*50), dtype=np.uint8).reshape(n, 50)
    tris = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3)  # 3 verts x xyz
    return tris

def slice_mesh_z(tris, z0):
    """Return 2D line segments where the mesh crosses plane z=z0."""
    z = tris[:, :, 2]
    above = z > z0
    n_above = above.sum(1)
    crossing = (n_above > 0) & (n_above < 3)
    tris = tris[crossing]; z = z[crossing]; above = above[crossing]
    segs = []
    for tri, zt, ab in zip(tris, z, above):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if ab[a] != ab[b]:
                t = (z0 - zt[a]) / (zt[b] - zt[a])
                pts.append(tri[a] + t * (tri[b] - tri[a]))
        if len(pts) == 2:
            segs.append([pts[0][:2], pts[1][:2]])
    return np.array(segs)

# --- CT slice ---
with tifffile.TiffFile(TIF) as t:
    nz = len(t.pages); zi = nz // 2
    ct = t.pages[zi].asarray()
thr = threshold_otsu(ct)
ct_bin = ct > thr

# --- STL cross-section (mid-height of its own z-range) ---
tris = read_stl(STL)
zmin, zmax = tris[:, :, 2].min(), tris[:, :, 2].max()
z0 = (zmin + zmax) / 2
segs = slice_mesh_z(tris, z0)

fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
ax[0].imshow(ct, cmap="gray"); ax[0].set_title(f"AS-BUILT CT (tif)\nslice z={zi}/{nz}, uint16 grayscale")
ax[1].imshow(ct_bin, cmap="gray"); ax[1].set_title(f"CT thresholded (Otsu={thr:.0f})\nsolid=white, air/pore=black")
from matplotlib.collections import LineCollection
ax[2].add_collection(LineCollection(segs, colors="k", linewidths=0.4))
ax[2].set_aspect("equal"); ax[2].autoscale()
ax[2].set_title(f"AS-DESIGNED STL cross-section\nz={z0:.1f}mm, {len(segs)} edges (perfect geom.)")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=130); print("saved", os.path.relpath(OUT, REPO))
print(f"CT slice: {ct.shape} uint16 [{ct.min()}..{ct.max()}] otsu={thr:.0f}")
print(f"STL z-range: [{zmin:.2f}..{zmax:.2f}] mm, cross-section at z={z0:.2f} -> {len(segs)} segments")
