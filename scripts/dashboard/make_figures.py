#!/usr/bin/env python
"""Render every figure for a finished detector run into one directory.

Nothing here detects anything. It calls the repo's existing renderers — the MCP
`visualize_*` tools and `scripts/gallery3d.py`, `scripts/node_health.py`,
`scripts/visualize_lattice_iou.py` — against a run that already exists, and drops the
output where the dashboard's "Figures from this run" panel already looks.

Steps are independent: one that fails is reported and the rest still run, because a
missing `sections.npz` should not cost you the CT slices.

    python scripts/dashboard/make_figures.py --run outputs/fresh_pass_.../defects
    python scripts/dashboard/make_figures.py --run <dir> --json-progress
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CT = ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
DESIGN = ROOT / ("data/missing_struts/registered_jsons/"
                 "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")
SUPPLIED_MASK = ROOT / "data/9x9x9_octet_lattice/segmentation/mask.tif"

EMIT_JSON = False


def emit(**payload):
    if EMIT_JSON:
        print(json.dumps({"type": "progress", **payload}), flush=True)
    else:
        print(f"[{payload.get('status', '..'):7}] {payload.get('step', '')}: "
              f"{payload.get('message', '')}", flush=True)


def run_step(name, fn):
    """Each step is isolated: a failure is a reported result, not the end of the run."""
    emit(step=name, status="start", message="running")
    started = time.time()
    try:
        detail = fn() or ""
        emit(step=name, status="ok", message=str(detail)[:400],
             seconds=round(time.time() - started, 1))
        return True
    except Exception as exc:
        emit(step=name, status="failed",
             message=f"{type(exc).__name__}: {exc}",
             detail=traceback.format_exc()[-800:],
             seconds=round(time.time() - started, 1))
        return False


def script_step(name, argv):
    """Run one of the repo's own plotting scripts as a subprocess."""
    def go():
        proc = subprocess.run([sys.executable, *argv], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(" | ".join(tail[-3:]) or f"exit {proc.returncode}")
        lines = (proc.stdout or "").strip().splitlines()
        return lines[-1] if lines else "done"
    return run_step(name, go)


def main():
    global EMIT_JSON
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="results directory holding strut_classes.csv / struts.csv")
    p.add_argument("--mask", default="", help="defaults to the run's own mask, else the supplied one")
    p.add_argument("--design", default=str(DESIGN))
    p.add_argument("--correction", default="")
    p.add_argument("--ct", default=str(CT))
    p.add_argument("--per-class", type=int, default=3)
    p.add_argument("--json-progress", action="store_true")
    args = p.parse_args()
    EMIT_JSON = args.json_progress

    run_dir = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    if not (run_dir / "strut_classes.csv").is_file():
        emit(step="setup", status="failed", message=f"no strut_classes.csv under {run_dir}")
        sys.exit(1)

    # A run produced by the fresh-pass driver keeps its own mask one level up; otherwise
    # fall back to the supplied Otsu mask, which is what the stored runs were built on.
    mask = Path(args.mask) if args.mask else None
    if mask is None:
        for candidate in (run_dir / "mask.tif", run_dir.parent / "mask.tif", SUPPLIED_MASK):
            if candidate.is_file():
                mask = candidate
                break
    correction = Path(args.correction) if args.correction else None
    if correction is None:
        for candidate in (run_dir.parent / "registration" / "correction.json",
                          run_dir / "correction.json",
                          ROOT / "outputs/registration/correction.json"):
            if candidate.is_file():
                correction = candidate
                break

    emit(step="setup", status="ok",
         message=f"run={run_dir}  mask={mask}  correction={correction}")

    import mcp_server as M

    def mcp(fn, **kw):
        result = fn(actor="main", why="render figures for the dashboard's visualization tab", **kw)
        if str(result).lstrip().startswith("Error:"):
            raise RuntimeError(str(result))
        return result

    # ---- the headline: struts by class and missing nodes, drawn on the CT -----------
    # gallery3d opens the mask with tifffile.memmap, which refuses a compressed file.
    # _save_volume writes zlib, so a mask produced by segment_ct_dataset needs an
    # uncompressed twin. Written once here and removed afterwards — it is ~520 MB.
    import tifffile

    gallery_mask, scratch_mask = mask, None
    try:
        tifffile.memmap(str(mask))
    except (ValueError, OSError):
        scratch_mask = run_dir / "mask_uncompressed.tif"
        emit(step="uncompressed mask", status="start",
             message=f"{mask.name} is compressed; writing a memory-mappable twin")
        tifffile.imwrite(str(scratch_mask), tifffile.imread(str(mask)))
        gallery_mask = scratch_mask
        emit(step="uncompressed mask", status="ok",
             message=f"{scratch_mask.name} ({scratch_mask.stat().st_size/1e6:.0f} MB, temporary)")

    gallery = ["scripts/gallery3d.py", "--dir", str(run_dir), "--out", str(run_dir),
               "--ct", args.ct, "--mask", str(gallery_mask)]
    if correction:
        gallery += ["--correction", str(correction)]

    # gallery3d ranks its strut exemplars on r_eq_min_um, which only the rich
    # strut_classes.csv carries (the one lattice_iou.run writes). detect_lattice_defects
    # writes four columns, so on an MCP run this step cannot pick exemplars. The value is
    # in sections.npz as sec_r_eq_min, in voxels — deriving and writing it here would mean
    # this dashboard editing detector output, so the step is skipped and says so instead.
    header = (run_dir / "strut_classes.csv").read_text().splitlines()[0].split(",")
    if "r_eq_min_um" in header:
        script_step("strut classes on CT (3D)", gallery + ["--struts"])
    else:
        emit(step="strut classes on CT (3D)", status="skipped",
             message="this run's strut_classes.csv has no r_eq_min_um column, which "
                     "gallery3d ranks exemplars on. Runs from scripts/classify_strut_defects.py "
                     "(e.g. outputs/lattice_iou) carry it; detect_lattice_defects writes 4 columns.")

    script_step("missing nodes on CT (3D)", gallery + ["--nodes"])

    if scratch_mask is not None and scratch_mask.is_file():
        scratch_mask.unlink()
        emit(step="uncompressed mask", status="ok", message="temporary twin removed")

    # ---- the interactive one: every class, rotatable, self-contained ----------------
    classes_argv = ["scripts/dashboard/classes3d.py", "--run", str(run_dir),
                    "--design", args.design,
                    "-o", str(run_dir / "strut_classes_interactive.html")]
    if correction:
        classes_argv += ["--correction", str(correction)]
    script_step("interactive 3D classes (WebGL)", classes_argv)

    # The as-built surface with the classes inside it. --max-dim 128 keeps the page near
    # 10 MB; 256 gives 2.5M triangles and a 60 MB file, which is the size that made the
    # older viewers in outputs/ unopenable.
    overlay_argv = ["scripts/dashboard/overlay3d.py", "--run", str(run_dir),
                    "--mask", str(mask), "--design", args.design, "--max-dim", "128",
                    "-o", str(run_dir / "asbuilt_overlay.html")]
    if correction:
        overlay_argv += ["--correction", str(correction)]
    script_step("as-built surface overlay (WebGL)", overlay_argv)

    # ---- per-class cross-sections through the CT -----------------------------------
    run_step("strut class cross-sections", lambda: mcp(
        M.visualize_strut_classes,
        mask_filepath=str(mask), design_filepath=args.design,
        results_directory=str(run_dir),
        output_filepath=str(run_dir / "strut_classes_sections.png"),
        correction_filepath=str(correction) if correction else "",
        per_class=args.per_class))

    # ---- plain CT slices, one per axis ---------------------------------------------
    # Read the shape from the header rather than the pixels: this mask is ~520 MB
    # decompressed and all that is needed is a mid-slice index per axis.
    import tifffile

    with tifffile.TiffFile(str(mask)) as handle:
        shape = handle.series[0].shape
    emit(step="CT slices", status="ok", message=f"mask shape {shape}")

    for axis, name in ((0, "z"), (1, "y"), (2, "x")):
        run_step(f"CT slice ({name})", lambda axis=axis, name=name: mcp(
            M.visualize_slice,
            input_filepath=str(mask),
            output_filepath=str(run_dir / f"ct_slice_{name}.png"),
            slice_index=shape[axis] // 2, axis=axis))

    # ---- design against the scan ----------------------------------------------------
    # compare_slices diffs two VOLUMES, so the design graph has to be rasterised onto the
    # CT grid first; handing it the .json is what failed here the first time.
    raster = run_dir / "design_raster.tif"
    rasterised = run_step("rasterize design", lambda: mcp(
        M.rasterize_lattice,
        input_filepath=args.design, output_filepath=str(raster),
        shape_z=shape[0], shape_y=shape[1], shape_x=shape[2]))

    if rasterised:
        run_step("design vs CT montage", lambda: mcp(
            M.compare_slices,
            design_filepath=str(raster), ct_mask_filepath=str(mask),
            output_montage=str(run_dir / "design_vs_ct.png")))

    # ---- whole-part overview --------------------------------------------------------
    run_step("lattice overview (3D)", lambda: mcp(
        M.visualize_lattice_3d,
        input_filepath=str(mask),
        output_filepath=str(run_dir / "lattice_3d.png")))

    # ---- node health and the run's own defect maps -----------------------------------
    node_argv = ["scripts/node_health.py", "--dir", str(run_dir),
                 "--design", args.design, "--mask", str(mask)]
    if correction:
        node_argv += ["--correction", str(correction)]
    script_step("node health", node_argv)
    script_step("defect / node / quality maps", ["scripts/visualize_lattice_iou.py",
                                                 "--dir", str(run_dir)])

    figures = sorted(p.name for p in run_dir.iterdir()
                     if p.suffix.lower() in {".png", ".gif", ".html"})
    (run_dir / "figures_manifest.json").write_text(json.dumps({
        "run": str(run_dir), "mask": str(mask),
        "correction": str(correction) if correction else None,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "figures": figures,
    }, indent=2))

    emit(step="done", status="ok", message=f"{len(figures)} figures in {run_dir}",
         directory=str(run_dir), figures=figures)


if __name__ == "__main__":
    main()
