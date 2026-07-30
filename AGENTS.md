# Agent instructions for this repository

This repo analyzes X-ray CT scans of additively manufactured octet-lattice structures:
segmentation, skeletonization, registration against the design, and strut/node defect
classification (missing / broken / thin / thick / necked). Read `README.md` and
`FINAL_README.md` first for the science; this file is about how an agent (Codex CLI or
Codex Cloud) should work in this repo specifically.

## Environment setup (needs internet -- runs once, before your task)

Codex Cloud tasks execute in a sandbox with **no internet access**, so anything that
needs the network must happen in the environment's setup script, not mid-task:

```bash
pip install -r requirements.txt
pip install -r platform/requirements.txt
git lfs pull   # the raw CT/STL data in data/ is Git-LFS-tracked and NOT fetched by a
               # normal clone; skip this if your task only touches outputs/ already in git
```

`visualize_lattice_3d` and `export_lattice_html` additionally need `pyvista` + `trame`
(commented out in `platform/requirements.txt` -- uncomment if your task needs a 3D view).

## Running the pipeline

Every analysis tool is a plain Python function in `src/mcp_server.py` -- segment, slice,
skeletonize, rasterize the design, refit registration, measure IoU, classify defects,
detect missing nodes, render strut evidence, validate against a ground-truth STL.
`platform/backend/tool_specs.py` lists every tool's exact parameters and defaults if you
need the full signature. Call them directly:

```python
import sys; sys.path.insert(0, "src")
import mcp_server as m

print(m.segment_ct_dataset("data/foo.tif", "outputs/foo/mask.npy", threshold=0.5))
print(m.refit_lattice_registration("outputs/foo/mask.npy", "data/foo_design.json", "outputs/foo"))
print(m.detect_lattice_defects("outputs/foo/mask.npy", "data/foo_design.json", "outputs/foo",
                                correction_filepath="outputs/foo/correction.json"))
```

**Always run `refit_lattice_registration` before any measurement or defect-detection
call, and pass its `correction.json` to them.** The shipped registrations carry a ~1%
scale error that swamps every defect count if skipped -- this is called out repeatedly
in the tools' own docstrings and is the single most common way to get a meaningless
result here.

If Codex CLI (not Cloud) is running locally, you can also add `src/mcp_server.py` as an
MCP server per the setup in `README.md` and call these as MCP tools instead of Python
directly -- same functions either way.

## Conventions

- Write new outputs under `outputs/<analysis_name>/`, matching the existing subfolders
  (`outputs/method_comparison/`, `outputs/lattice_iou/`, etc.) -- don't scatter result
  files at the repo root.
- Summarize what you ran and why in a short Markdown report (see `FINDINGS.md` and
  `FINAL_README.md` for the existing tone: state the method, the numbers, and the
  caveats -- particularly registration/resolution limits -- rather than a bare number).
- Never quote a defect rate without checking whether the registration correction was
  applied; every relevant tool's return text says so explicitly.

## Publishing results

When you're done, commit the new/changed files under `outputs/` (and any report) and
push to a branch / open a PR against `main`. Once merged to `main`,
`.github/workflows/deploy-pages.yml` automatically rebuilds and republishes the static
results gallery (`platform/build_static_site.py`) to GitHub Pages -- no separate publish
step needed, and no need to touch `_site/` yourself (it's a build artifact, not checked
in).

There's also an optional local, interactive version of this dashboard --
`platform/README.md` -- with live forms and a job runner over these same tools, for
running things by hand outside of Codex.
