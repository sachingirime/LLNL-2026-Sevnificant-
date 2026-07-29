"""Self-contained sanity check on the skan graph -- nothing here uses the design."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

ROOT = "/home/sachin/research/LLNL DSC 2026/llnl_data_science_challenge_2026"
g = json.load(open(f"{ROOT}/outputs/skan_graph/graph.json"))
P = np.array([j["position"] for j in g["junctions"]], float)[:, ::-1]  # (z,y,x)
deg = np.array([j["degree"] for j in g["junctions"]])
bnd = np.array([j["boundary"] for j in g["junctions"]])
E = np.array([[s["junction0"], s["junction1"]] for s in g["struts"]])
L = np.array([s["length_vox"] for s in g["struts"]])
eb = np.array([s["boundary"] for s in g["struts"]])

print("=== does this look like an octet lattice, judged only on itself? ===")
print(f"interior node degree: "
      f"{dict(zip(*np.unique(deg[~bnd], return_counts=True)))}")
print(f"interior strut length vox p1/p50/p99: "
      f"{np.percentile(L[~eb], [1, 50, 99]).round(1)}  (design 55.4)")

# strut direction families -- an octet has struts at 45 and 90 deg to the build
# axis and nothing else. This is a topology-free check on the geometry.
v = P[E[:, 1]] - P[E[:, 0]]
v /= np.linalg.norm(v, axis=1)[:, None]
ang = np.degrees(np.arccos(np.abs(v[:, 0])))
h, edges = np.histogram(ang[~eb], bins=18, range=(0, 90))
print("\nangle to build axis (deg) histogram, interior struts:")
for c, lo in zip(h, edges[:-1]):
    print(f"   {lo:4.0f}-{lo+5:3.0f}: {'#' * int(60 * c / h.max())} {c:,}")

# ---- picture: one unit-cell-thick slab of the graph over the CT ------------
Z = 380
sl = tifffile.imread(f"{ROOT}/data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif",
                     key=Z)
near = np.abs(P[:, 0] - Z) < 6
show = np.flatnonzero(near[E[:, 0]] & near[E[:, 1]])

fig, ax = plt.subplots(1, 2, figsize=(17, 8.5))
ax[0].imshow(sl, cmap="gray", vmin=30000, vmax=50000)
for k in show:
    a, b = P[E[k, 0]], P[E[k, 1]]
    ax[0].plot([a[2], b[2]], [a[1], b[1]], "-", color="#4da6ff", lw=1.4)
ax[0].plot(P[near, 2], P[near, 1], "o", color="#ff4d4d", ms=3.5)
ax[0].set_title(f"skan graph within 6 vox of CT slice z={Z}\n"
                f"{near.sum()} nodes, {len(show)} struts drawn on the raw CT")
ax[0].axis("off")

# one node plane only -- projecting all z overplots into an unreadable mush
m = (~bnd) & (np.abs(P[:, 0] - Z) < 6)
sc = ax[1].scatter(P[m, 2], P[m, 1], c=deg[m], s=90, cmap="viridis",
                   vmin=6, vmax=12)
plt.colorbar(sc, ax=ax[1], shrink=0.7, label="node degree")
ax[1].set_title(f"the {m.sum()} interior nodes in this same z slab, by degree\n"
                f"(12 = full octet interior connectivity)")
ax[1].set_aspect("equal"); ax[1].invert_yaxis(); ax[1].axis("off")
fig.tight_layout()
out = f"{ROOT}/outputs/skan_graph/graph_check.png"
fig.savefig(out, dpi=110)
print(f"\nwrote {out}")
