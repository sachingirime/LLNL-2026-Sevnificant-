# Lattice NDE Platform

Two ways to see this project's analysis pipeline, sharing one set of tools:

1. **A public, static results gallery on GitHub Pages** -- rebuilt automatically by
   `.github/workflows/deploy-pages.yml` every time `outputs/` changes on `main`. No
   server, no login: just a live page of whatever's already been computed. See
   `AGENTS.md` at the repo root for how Codex (CLI or Cloud) runs the pipeline and
   publishes results here -- that's the intended way to generate new analyses now,
   rather than clicking through a local dashboard.
2. **An optional local, interactive dashboard** (below) with live forms and a job
   runner, for running tools by hand instead of through Codex.

Neither reimplements any analysis. Both call `src/mcp_server.py` directly -- the exact
same functions Codex calls as MCP tools -- so results are identical everywhere.

## The static gallery (GitHub Pages)

`platform/build_static_site.py` scans `outputs/` and the project's `*.md` reports and
turns them into a self-contained static site (`gallery.json` + `tools.json` + a `files/`
mirror), using only the standard library -- no scientific dependencies needed just to
publish PNGs and CSVs. The GitHub Actions workflow calls it on every push to `main` and
publishes the result with `actions/upload-pages-artifact` + `actions/deploy-pages`.

One-time setup on GitHub: repo Settings -> Pages -> Source -> **GitHub Actions**. After
that, merging any branch that touches `outputs/` into `main` republishes the page within
a minute or two -- no manual step.

To preview it locally before pushing:

```bash
python platform/build_static_site.py --out platform/_site
python -m http.server 8080 --directory platform/_site   # open http://localhost:8080
```

## The local interactive dashboard (optional)

## Start it

```bash
cd platform
./run.sh
```

Then open **http://127.0.0.1:8420**. First run creates a virtualenv and installs
`platform/requirements.txt` (a superset of the repo's own `requirements.txt`, plus
`fastapi`/`uvicorn` for the web app).

`visualize_lattice_3d` and `export_lattice_html` additionally need `pyvista` +
`trame` (commented out in requirements.txt since they're heavy and every other
tool works without them) -- uncomment those lines if you want the 3D views.

## What's in the UI

- **Dashboard** -- every image, CSV, JSON, and report already sitting in `outputs/`
  and the project root, browsable without running anything.
- **File browser** -- navigate the whole project tree, preview any file.
- **Pipeline** (left sidebar, grouped by stage) -- one page per tool in
  `mcp_server.py`: segment, slice, skeletonize, 3D preview, rasterize design,
  registration QA, IoU measurement, defect classification, node detection,
  strut-evidence rendering, and STL ground-truth validation. Each page is a form
  generated from that tool's actual parameters; a "Browse…" button next to every
  path field opens a file picker scoped to the project so you don't have to type
  paths by hand.
- **Jobs** -- every run in this session with live status. Runs happen in a
  background thread and are polled, since `detect_lattice_defects` can take up to
  ~10 minutes cold (seconds on a cached re-run).

## Typical flow for a new CT scan

1. **Segment** the raw volume -> binary mask.
2. **Slice** a few indices to sanity-check the mask.
3. **Refit registration** (`refit_lattice_registration`) against the design JSON --
   do this before any measurement; the shipped registration has a ~1% scale error
   that swamps defect counts if skipped.
4. **Measure** (`measure_lattice_iou`) to see the strut/node distributions and pick
   sane cuts.
5. **Classify defects** (`detect_lattice_defects`, then `detect_missing_nodes` /
   `detect_missing_nodes_2d`) for the missing/broken/thin/thick/necked counts.
6. **Review evidence** (`visualize_strut_classes`) before trusting any class count.
7. If you have a defect STL, **validate against it** (`validate_against_stl`) --
   the one check here that isn't just re-reading the same scan.

## Notes

- All file paths in the UI are relative to the project root (one level above
  `platform/`) for portability; the backend refuses to read/write outside it.
- This repo's raw CT/STL data is tracked with Git LFS and kept out of the
  connected folder by default -- `git lfs pull` first if `data/` looks empty and
  you want to run the pipeline end-to-end rather than just browsing `outputs/`.
- Single-user, local-only: jobs are tracked in memory, not persisted across
  restarts.
