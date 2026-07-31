"""Pick one exemplar per defect class out of the finished detector run.

Nothing here re-detects. The labels are read from `strut_classes.csv`, the metrics from
`struts.csv` / `sections.npz` / `connectivity.npz`, all written by `detect_lattice_defects`
with the registration correction applied.

The selection rules are ranked on the quantity that *defines* the class -- the thinnest
`thin`, the largest excess annulus for `dross` -- and then filtered for legibility only:
away from the build plates, cross-sections not touching their window, both junctions
present. Filtering for legibility after ranking on the defining quantity is the honest
order; picking by eye out of the whole population would make every figure a best case.

`dross` is not one of the detector's six classes and is not presented as one. It is the
top of `excess_frac` -- the fraction of the annulus r_nom < rho <= 2.2 r_nom over the
middle half of the span that is material, which `measure_struts` computes and which drove
a `dross` class that was withdrawn. The figure says so.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/ct_anomaly_report_20260730"
DEFECTS = RUN / "defects"
DESIGN = ROOT / ("data/missing_struts/registered_jsons/"
                 "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
CORRECTION = RUN / "registration_refined/correction.json"
MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/mask.tif"

VOL_SHAPE = np.array([761, 815, 837])
# Solid build plates occupy both z ends and taper obliquely; a strut inside them leaves no
# signature at all, so exemplars are drawn from well clear of that region.
PLATE_MARGIN = 110


class Run:
    """The finished run, loaded once."""

    def __init__(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import lattice_iou

        self.pos, self.pairs, _, applied = lattice_iou.load_design(
            str(DESIGN), str(CORRECTION))
        if not applied:
            raise RuntimeError("registration correction was not applied")
        self.geo = self._geometry()

        self.d = np.genfromtxt(DEFECTS / "struts.csv", delimiter=",", names=True)
        z = np.load(DEFECTS / "sections.npz")
        self.sec = {k[4:]: z[k] for k in z.files if k.startswith("sec_")}
        self.prof = {k[5:]: z[k] for k in z.files if k.startswith("prof_")}
        self.conn = dict(np.load(DEFECTS / "connectivity.npz"))
        self.nodes = np.genfromtxt(DEFECTS / "nodes.csv", delimiter=",", names=True)

        lab, meas = [], []
        with open(DEFECTS / "strut_classes.csv") as fh:
            for row in csv.DictReader(fh):
                lab.append(row["label"])
                meas.append(int(row["measurable"]))
        self.label = np.array(lab)
        self.measurable = np.array(meas, bool)

        self.node_pos, self.node_of, self.degree = lattice_iou.dedupe_junctions(
            self.pos, self.pairs)
        self.n_sections = int(self.prof["r_eq"].shape[1])

    def _geometry(self):
        import json
        return json.loads((DEFECTS / "summary.json").read_text())["geometry"]

    # -------------------------------------------------------------- helpers
    @property
    def um(self):
        return self.geo["um_per_voxel"]

    @property
    def r_nom(self):
        return self.geo["nominal_strut_radius_vox"]

    def ends(self, i):
        return self.pos[self.pairs[i, 0]], self.pos[self.pairs[i, 1]]

    def midpoints(self):
        return 0.5 * (self.pos[self.pairs[:, 0]] + self.pos[self.pairs[:, 1]])

    def clear_of_plates(self):
        m = self.midpoints()
        return np.all((m > PLATE_MARGIN) & (m < VOL_SHAPE - PLATE_MARGIN), axis=1)

    def nodes_present(self):
        """Both junctions of the strut are sound nodes (sphere fill at 1.0)."""
        fill = np.zeros(len(self.node_pos))
        fill[self.nodes["node_id"].astype(int)] = self.nodes["fill"]
        nn = self.node_of[self.pairs]
        return (fill[nn[:, 0]] > 0.5) & (fill[nn[:, 1]] > 0.5)

    def base(self):
        """Measurable, clear of the plates, and joining two junctions that printed."""
        n = len(self.label)
        b = np.zeros(len(self.pairs), bool)
        b[:n] = self.measurable
        return b & self.clear_of_plates() & self.nodes_present()

    def has_label(self, name):
        out = np.zeros(len(self.pairs), bool)
        out[:len(self.label)] = self.label == name
        return out

    def col(self, name):
        """A per-strut column padded to the full strut count.

        The three tables disagree on length -- `sections.npz` covers every strut, the
        CSVs are written per measured strut -- so everything is padded with NaN to the
        design's strut count and indexed by `strut_id` throughout.
        """
        for src in (self.d.dtype.names or (), self.sec, self.conn):
            if name in src:
                v = np.asarray(self.d[name] if src is self.d.dtype.names
                               else src[name], float)
                break
        else:
            raise KeyError(name)
        if len(v) < len(self.pairs):
            v = np.concatenate([v, np.full(len(self.pairs) - len(v), np.nan)])
        return v[:len(self.pairs)]


def rank(mask, score, n=8):
    idx = np.where(mask)[0]
    if not len(idx):
        return idx
    s = np.asarray(score, float)[idx]
    return idx[np.argsort(-np.nan_to_num(s, nan=-np.inf))][:n]


def candidates(run, n=8):
    """Ranked candidate struts per class. Highest first."""
    b = run.base()
    clean = b & (run.col("border_frac") <= 0.0)
    r_med = run.col("r_eq_med")
    cv = run.col("r_eq_cv")
    excess = run.col("excess_frac")

    out = {}
    # Most isolated first: a missing strut reads best where its neighbours are intact,
    # so rank on how full the *envelope* around the absent cylinder is NOT.
    m = b & run.has_label("missing")
    out["missing"] = rank(m, -run.col("env_material_frac"), n)

    # Severed with the break inside the sectioned span, so the gap is in frame. Ranked on
    # gap length; `broken` itself is decided by connectivity, which has no threshold.
    br = b & run.has_label("broken")
    out["broken"] = rank(br, run.col("gap_len") + 0.01 * run.col("r_eq_med"), n)

    out["thin"] = rank(clean & run.has_label("thin"), -r_med, n)
    out["thick"] = rank(clean & run.has_label("thick"), r_med, n)

    # Dross: the largest excess annulus on a strut that is otherwise continuous and not
    # simply fat everywhere -- a lump, so the radius has to vary along the strut.
    intact = clean & (run.col("reachable") > 0.5) & ~run.has_label("missing") \
        & ~run.has_label("broken")
    out["dross"] = rank(intact, excess * np.sqrt(np.clip(cv, 0, None)), n)
    return out


def missing_nodes(run):
    """Interior nodes whose nominal sphere holds no material at all."""
    sel = (run.nodes["fill"] == 0) & (run.nodes["is_interior"] == 1)
    return run.nodes["node_id"][sel].astype(int)


def node_neighbours(run, node_id):
    """Positions of the far ends of every strut incident on this physical node."""
    nn = run.node_of[run.pairs]
    hit = np.where((nn[:, 0] == node_id) | (nn[:, 1] == node_id))[0]
    far = np.where(nn[hit, 0] == node_id, run.pairs[hit, 1], run.pairs[hit, 0])
    return run.pos[far], hit


def report(run):
    lines = []
    cand = candidates(run)
    for name, idx in cand.items():
        lines.append(f"{name}:")
        for i in idx[:6]:
            p0, p1 = run.ends(i)
            mid = 0.5 * (p0 + p1)
            lines.append(
                f"  #{i:6d} {run.label[i]:8s} z{mid[0]:5.0f} y{mid[1]:5.0f} x{mid[2]:5.0f}"
                f"  dia {run.col('r_eq_med')[i] * 2 * run.um:6.1f} um"
                f"  cv {run.col('r_eq_cv')[i]:5.3f}"
                f"  excess {run.col('excess_frac')[i]:5.3f}"
                f"  gap {run.col('gap_len')[i]:.0f}"
                f"  reach {run.col('reachable')[i]:.0f}"
                f"  detour {run.col('detour')[i]:6.3f}")
    for nid in missing_nodes(run):
        p = run.node_pos[nid]
        lines.append(f"node #{nid} z{p[0]:.0f} y{p[1]:.0f} x{p[2]:.0f} "
                     f"degree {run.degree[nid]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(Run()))
