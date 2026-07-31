# Codex dashboard

Chat with the Codex agent over this repository and watch the execution trace build as it
happens. This is the agent-side half of the trajectory the poster describes — the MCP
server records its own side in `outputs/mep/`, this records who asked for it.

```
python scripts/dashboard/server.py          # http://127.0.0.1:8787
python scripts/dashboard/server.py --port 9000 --sandbox workspace-write
```

Use the interpreter that has this repo's requirements installed — the dashboard needs
`starlette`, `uvicorn` and `sse_starlette`, all already present in `dssi_env`:

```
/home/sachin/miniconda3/envs/dssi_env/bin/python scripts/dashboard/server.py
```

Nothing here imports from `src/`. The dashboard reads the repo; it does not change how the
repo works, and it does not touch `src/mcp_server.py`.

## How it works

`codex exec --json` prints one JSON event per line on stdout. The server spawns it, forwards
every event to the browser over SSE, and keeps the `thread_id` so the next message resumes
the same conversation instead of starting over.

One wrinkle worth knowing if you edit `build_command`: `codex exec resume` accepts neither
`-C/--cd` nor `-s/--sandbox`. The working directory therefore travels through the subprocess
`cwd`, and the sandbox through a `-c sandbox_mode=` override, on both the first turn and every
resumed one.

Item ids restart at `item_0` on every turn, so the UI keys cards by `turn:item_id`. Keying on
the id alone makes turn 2 overwrite turn 1's cards.

## Sandbox

Defaults to `read-only`: the agent can read every file and run shell commands, but cannot
write. Switch to `workspace-write` in the header when you want it to actually produce
outputs — for example to run the canonical pipeline from `AGENTS.md`. The selector applies
to the next turn.

## Trace capture and replay

Every run's raw events are journalled to `outputs/dashboard_runs/<run_id>.jsonl`, together
with the prompt that caused it. Replay one through the same render path the live stream uses:

```
http://127.0.0.1:8787/?replay=<run_id>          # instant
http://127.0.0.1:8787/?replay=<run_id>&speed=250 # 250ms per event, for demoing
http://127.0.0.1:8787/?replay=<run_id>&theme=dark
```

What you see replayed is what you saw live — same code path, no separate renderer. This is
also the demo safety net: a saved run needs no network and spends no tokens.

## Export trace

**Export trace** downloads the run as JSONL in the same envelope `src/mep.py` writes —
`run_id, step, packet_id, ts, duration_s, actor, session, tool, kind, args, why, status,
error, artifact, artifact_files, inputs, verification` — so both halves can be read by the
same tooling.

The difference is what fills `session`. The MCP server cannot see which agent called it, so
its packets carry `session_id: null, client_id: null`. These are recorded on the agent side,
so they carry the codex thread id and the run id. `actor` is `codex-agent`.

`duration_s` is currently always `0.0` — codex events carry no per-item timing.

## What is verified and what is not

Confirmed against a live `codex-cli 0.144.6`: the event types `thread.started`,
`turn.started`, `item.started`, `item.completed`, `turn.completed`, and the item types
`agent_message` and `command_execution`.

The cards for `mcp_tool_call`, `reasoning`, `file_change`, `patch_apply`, `web_search` and
`error` were written against expected field names and exercised with a fixture, not observed
live. If one renders wrong, its fields are in the collapsed **raw event** fold — unrecognised
item types fall through to a generic card rather than being dropped, because a trace that
silently omits a step is worse than an ugly one.

## Visualizations tab

Read-only. It renders results; it does not compute any.

**Defect classes** reads `strut_classes.csv` from whichever run you pick in the header, and
shows the five defect classes as labelled rows. `nominal` is deliberately not a bar — it is
80% of the lattice and would flatten everything that matters. Clicking a class fills
**What separates this class** with the metrics that actually drive it, taken from
`scripts/classify_strut_defects.classify`:

| class | what decides it |
|---|---|
| `missing` | `iou`, `empty_sections` — zero matched voxels AND every section empty. No threshold. |
| `broken` | `detour`, `reachable` — no geodesic path through material inside a 2× tube. Topological. |
| `thin` / `thick` | `r_eq_med_um` against the tolerance band |
| `necked` | `r_eq_min_um` — a local pinch under a normal median radius |

Runs whose table carries only `label`, `r_eq_med_um` and `measurable` show fewer tiles;
`outputs/lattice_iou/` has the full 16-column table and is the default selection.

Colours come from `classify_strut_defects.CLASS_COLORS`, so this tab and every figure
already in `outputs/` name a class the same way. One caveat worth knowing: that map puts
`missing` (`#e34948`) next to `broken` (`#eb6834`), which is below the palette's
normal-vision separation floor. Colour is therefore never the only channel here — each row
carries a swatch, the class name and the count. Do not add a view that distinguishes those
two by hue alone.

**Everything else in the repo** indexes every `.png`, `.gif` and `.html` under `outputs/`
and `images/` — 138 files at last count. Thumbnails are built only when you open a group,
and pages over 4 MB open in a tab rather than embedding, because the largest viewers here
are 36–67 MB and inlining one makes the page unopenable.

`?tab=viz`, `?class=broken` and `?theme=dark` are honoured on load, which is how the
screenshots were taken.

### The interactive 3D pages

Three WebGL viewers, all self-contained (no CDN, work from `file://`), all reading labels
from a run's `strut_classes.csv` and computing nothing.

| script | page | what it answers |
|---|---|---|
| `classes3d.py` | `strut_classes_interactive.html` (~0.6 MB) | where each class sits in the part |
| `overlay3d.py` | `asbuilt_overlay.html` (~10 MB) | is a `missing` strut really empty metal, or is the design line off it |
| `delta3d.py` | `fresh_vs_stored.html` (~0.6 MB) | what moved between two runs, per class and per direction |

`delta3d` is not in the button — it needs two runs, which does not fit the per-run step
model. Run it directly:

```
python scripts/dashboard/delta3d.py --fresh <run> --stored <run> --include-caps -o out.html
```

Three things worth knowing:

- **The repo's own viewers could not do this.** `lattice_iou_webgl` colours by live IoU
  sliders and `graph_webgl` by material-fraction cuts — neither draws the detector's
  labels. `overlay_webgl`'s `verdict` path has a fixed vocabulary (`disconnected`,
  `dross`, `bent`) with no `broken`, `thick` or `necked`, so feeding it these labels would
  mean renaming `thick` to `dross`. Its isosurface extractor and viewer shell are reused;
  only the class table is new.
- **`overlay3d --max-dim` drives the file size.** 128 gives 372k triangles and ~10 MB;
  256 gives 2.5M and 60 MB, which is the size that made the older viewers in `outputs/`
  unopenable. The surface is block-reduced for display and is **not** a thickness
  measurement — the page says so.
- **Two bugs in the shared viewer shells are patched in these copies only**, leaving
  `graph_webgl.py` and `overlay_webgl.py` untouched: both hardcode `"material "` before
  every rule (right for their own coverage classes, wrong for "no path node-to-node
  through material"), and both print `count`, which for a node cross is three segments —
  2 missing nodes would advertise as 6. Note the element is named `rl` in one shell and
  `r` in the other.

Missing nodes appear in all three as 3-axis crosses, split into **interior** (2) and
**surface** (182). That split is not cosmetic: `detect_missing_nodes` reports 184 of 3430
as its headline and 2 of 2456 restricted to degree-12 interior sites, and 181 of the 184
lie on the single plane y = 759. Rotate the page and that face is visibly a face, which is
the judgment the tool deliberately leaves to you. Never quote the 2 alone.

### Generate figures

**Generate figures** renders everything for the selected run **into that run's own
directory**, so there is one place to go afterwards — the path is shown under the progress
list with a copy button, and the "Figures from this run" grid refreshes when it finishes.

It runs `scripts/dashboard/make_figures.py`, which detects nothing. It calls the repo's
existing renderers:

| step | produces | via |
|---|---|---|
| strut classes on CT (3D) | `strut_classes_3d.png` | `scripts/gallery3d.py --struts` |
| missing nodes on CT (3D) | `missing_nodes_3d.png` | `scripts/gallery3d.py --nodes` |
| strut class cross-sections | `strut_classes_sections.png` | MCP `visualize_strut_classes` |
| CT slices, one per axis | `ct_slice_{z,y,x}.png` | MCP `visualize_slice` |
| design vs CT montage | `design_raster.tif`, `design_vs_ct.png` | MCP `rasterize_lattice` + `compare_slices` |
| lattice overview | `lattice_3d.png` | MCP `visualize_lattice_3d` |
| node health | `node_health.png`, `node_gallery.png` | `scripts/node_health.py` |
| defect / node / quality maps | `defect_map.png`, `node_map.png`, `quality_field.png` | `scripts/visualize_lattice_iou.py` |

plus `figures_manifest.json` listing what was written.

Steps are independent — one that fails is reported in the progress list and the rest still
run, because a missing `sections.npz` should not cost you the CT slices.

Two things it handles that will bite you if you call these tools by hand:

- `gallery3d.py` opens the mask with `tifffile.memmap`, which refuses a compressed file,
  and `segment_ct_dataset` writes zlib. The runner writes an uncompressed twin
  (~520 MB), uses it, and deletes it.
- `compare_slices` diffs two **volumes**. Handing it the design `.json` fails; the design
  has to go through `rasterize_lattice` onto the CT grid first.

It can run for several minutes — `gallery3d` surfaces the 1 GB CT twice.

**It writes into the run's own directory, so re-running overwrites that run's figures.**
`node_health.py` and `visualize_lattice_iou.py` both read from and write to their `--dir`,
so a separate output directory is not available without copying the tables. This is
narrower than the housekeeping rule in CLAUDE.md ("do not overwrite a prior run's
outputs") allows, and it is safe only because every one of these figures is a
deterministic function of tables that are not touched — the CSVs, `sections.npz` and
`connectivity.npz` are read, never written. If you want a pristine copy of a run's
figures, copy the directory before pressing the button.

**`strut classes on CT (3D)` is skipped on runs produced by `detect_lattice_defects`.**
`gallery3d` ranks its exemplars on `r_eq_min_um`, and the MCP tool writes a four-column
`strut_classes.csv` without it. The quantity exists in `sections.npz` as `sec_r_eq_min` in
voxels, but deriving it here would mean the dashboard writing detector output, so the step
reports `skipped` with the reason instead. Runs from `scripts/classify_strut_defects.py`
(`outputs/lattice_iou`) carry the column and render normally.

### Re-running detection

**Run detection via MCP** does not compute anything in this process. It switches to the
Chat tab and hands the agent the canonical prompt from `AGENTS.md` — `begin_analysis`,
`refit_lattice_registration`, `detect_lattice_defects` with `correction_filepath`, then
`explain_run`. Detection stays with the MCP server, which is where the repo's routing
document puts it. The button forces `workspace-write` first, since the read-only sandbox
cannot write a results directory.

**This button has not been run end to end.** The MCP server is registered and enabled
(`codex mcp list` shows `segmentation-tools`), but whether `codex exec` will call a tool
whose config sets `approval_mode = "approve"` in a non-interactive session is untested.
If it stalls, run the pipeline in an interactive `codex` session and reload the tab — the
run selector picks up any directory under `outputs/` holding a `strut_classes.csv`.

## Known noise

Two benign stderr lines appear in **Diagnostics** on every run: codex reports
`Reading additional input from stdin...` because stdin is not a TTY, and a
`failed to load models cache` error from a config-schema mismatch in the CLI itself. Neither
affects the turn. They are shown rather than filtered so that real stderr is never hidden.
