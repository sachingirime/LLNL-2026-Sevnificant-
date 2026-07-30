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

## On-demand analysis triggered from the public dashboard

The dashboard published to GitHub Pages (`platform/frontend/static_index.html`) lets a
visitor upload a new scan. That flow is: the dashboard's Cloudflare Worker
(`platform/cloudflare-worker/`) pushes the upload to a fresh branch named
`upload/<request_id>` under `uploads/incoming/<request_id>/`, then fires a
`repository_dispatch` event that runs `.github/workflows/analyze-upload.yml` --
which is the one place Codex is invoked non-interactively (`openai/codex-action`) rather
than through a person typing a prompt. If you're the agent running inside that
workflow, here is what's expected of you, precisely:

- **Inputs**: `github.event.client_payload` gives you `tiff_path`, `design_path`
  (may be absent), `threshold` (may be absent), and `request_id`.
- **Output location**: everything you write goes under
  `outputs/uploads/<request_id>/`, nowhere else. Do not touch any other file.
- **Use what's already built, not just raw functions**: `.codex/agents/segmentation_agent.toml`
  and `.codex/agents/visual_reasoner_agent.toml` are pre-defined subagents for the
  segmentation and visual-inspection steps; `.agents/skills/nde_report_expert/SKILL.md`
  defines this project's own convention for the final report; `.agents/skills/threshold_optimizer`
  and `.agents/skills/metadata_extractor` help when the threshold hint is missing.
  Delegate to these where the invoking context supports it; where it doesn't, follow
  their instructions yourself rather than skipping what they'd do. Everything ultimately
  calls `src/mcp_server.py` for the parts not covered by a subagent/skill.
- **Always run**, regardless of whether a design graph was supplied:
  segment -> a couple of sanity-check slice visualizations -> skeletonize ->
  `detect_missing_nodes_2d` (it needs no design file, so it's your baseline
  defect screen for every submission).
- **Only if a design graph was supplied**: run `refit_lattice_registration` first,
  then `measure_lattice_iou`, `detect_lattice_defects`, and `detect_missing_nodes`,
  passing the fresh `correction.json` through to each. If no design graph came in,
  say so in the report rather than fabricating a registration.
- **Write `REPORT.md`** in that same output directory, following the
  `nde_report_expert` convention (feature-metrics table, small visual gallery,
  analysis section) and the tone of `FINDINGS.md` / `FINAL_README.md` -- numbers
  with their caveats, not bare claims.
- **Commit only `outputs/uploads/<request_id>/`** and let the workflow push the
  branch and open the PR -- do not push directly to `main` yourself from this path.
  This is deliberate: an anonymous public upload should never publish straight to
  the results page without a person reviewing the PR first.

### This is a one-shot run, not a conversation

`openai/codex-action` runs `codex exec` once and exits -- it does not create a Codex
Cloud task and there is no follow-up-message thread attached to it. If someone wants to
keep asking questions about a specific upload after the PR is opened, the honest answer
today is: point them at Codex Cloud directly (`https://chatgpt.com/codex`, connected to
this repo, a new task against the `upload/<request_id>` branch) -- Codex Cloud's own
follow-up-message feature keeps that environment and context alive across turns, which
this one-shot Action deliberately does not try to reimplement. As of this writing there
is no stable, public API for a script to create a Codex Cloud task and hand back a
ready-made "continue this conversation" link (see
[openai/codex#24777](https://github.com/openai/codex/issues/24777) -- environment/task
lifecycle scripting is an open request, not yet shipped), so don't build automation that
assumes one exists; check that issue's status before attempting it.
