# Lattice NDE Platform

Three ways to see this project's analysis pipeline, all sharing the same tools
(`src/mcp_server.py`) so results are identical no matter which path produced them:

1. **A public dashboard on GitHub Pages** that anyone with the link can open, browse
   already-computed results on, and submit a new CT scan to for analysis. Submitting
   doesn't run anything on GitHub Pages itself (it can't -- static hosting only); it
   pushes the scan to the repo and asks **Codex** to run the pipeline against it,
   headlessly, via GitHub Actions. Results land in a pull request for review before
   they reach the public page. This is the intended day-to-day way to use the project
   now, from a phone or a laptop, no local setup.
2. **Codex CLI/Cloud directly** against the repo, for anyone comfortable with that --
   see `AGENTS.md` at the repo root.
3. **An optional local, interactive dashboard** (further down) with live forms and a
   job runner, for running tools by hand on your own machine instead of through Codex.

## The public dashboard (GitHub Pages + Cloudflare Worker + GitHub Actions)

Three pieces, only one of which needs your own setup:

- `platform/frontend/static_index.html` -- the page itself. Built by
  `platform/build_static_site.py`, which scans `outputs/` and the project's `*.md`
  reports into a static `gallery.json` + `tools.json` + `files/` mirror, no scientific
  dependencies required. `.github/workflows/deploy-pages.yml` rebuilds and republishes
  it on every push to `main` that touches `outputs/`.
- `platform/cloudflare-worker/` -- the upload proxy. GitHub Pages can't accept a file
  upload or hold a credential, so this small Worker does both: it takes the upload,
  pushes it to a new branch, and fires the event that starts the analysis. **This is
  the one piece you need to deploy yourself** -- it needs your own GitHub token and
  Cloudflare account; see `platform/cloudflare-worker/README.md` for the ~5 minute
  setup, and the `OPENAI_API_KEY` repo secret it depends on.
- `.github/workflows/analyze-upload.yml` -- runs when the Worker fires its event.
  Checks out the upload branch and runs Codex headlessly (`openai/codex-action`)
  against it per `AGENTS.md`'s "on-demand analysis" section -- using the project's
  existing subagents (`.codex/agents/`) and skills (`.agents/skills/`) where they
  fit, not just raw tool calls -- then opens a PR instead of pushing to `main`
  directly.

This one-shot run can't hold a conversation (see `AGENTS.md` for why, and the upstream
issue tracking a scriptable fix). For asking follow-up questions about a specific
upload, the PR body and the dashboard's status panel both point at Codex Cloud
directly (`chatgpt.com/codex`, connected to this repo, a task against that upload's
branch) -- that's OpenAI's own follow-up-message feature, not something reimplemented
here.

One-time setup on GitHub for the page itself (independent of the Worker): repo
Settings -> Pages -> Source -> **GitHub Actions**. After that, merging anything that
touches `outputs/` into `main` republishes the page within a minute or two.

To preview the static page locally before pushing (gallery browsing only -- the upload
form still needs the deployed Worker to actually do anything):

```bash
python platform/build_static_site.py --out platform/_site
python -m http.server 8080 --directory platform/_site   # open http://localhost:8080
```

## The local interactive dashboard (optional)

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
