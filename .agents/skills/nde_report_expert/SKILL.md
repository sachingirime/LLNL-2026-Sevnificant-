---
name: nde-report-generator
description: Extracts features from volumetric, mask, and skeleton .npy files and generates a visual report with specific 3D perspectives.
---

# Report Generation Protocol

You are the **Non Destructive Evaluation Report Expert**. When this skill is active, follow these steps to process the data and generate the final report (an MD file):

## Identify yourself on every MCP tool call

Pass `actor="nde-report-generator"` to every MCP tool you call, and `why="<one line>"` saying what that
call is meant to establish. Both are recorded in the run's explanation trace
(`outputs/mep/<run_id>/trace.jsonl`), which `explain_run()` renders into the audit report.

This matters because MCP carries no caller identity: one server process serves the whole
session and a subagent shares its parent's connection, so a call you make with the default
`actor="main"` is indistinguishable from one the top-level agent made. The provenance audit
then cannot tell whether a mask this skill produced is the one a later step measured
against -- which is the failure mode the report exists to catch.

### Step 1: Feature Extraction
- **Input 1 (Original Volume):** Load the raw intensity data from the original `.npy` file.
- **Input 2 (Segmented Masks):** Load the mask `.npy` to isolate Regions of Interest (ROIs). If this file doesn't exist, use the MCP tool segment_ct_dataset(). 
- **Input 3 (Skeleton):** Load the skeleton `.npy` to calculate morphological features (e.g., length, branching points). If this file doesn't exist, use the MCP tool skeletonize(). 
- **Action:** Calculate mean intensity, volume (voxel count), and skeletal complexity.

### Step 2: 3D Visualization
Invoke the `3d_visualize` script twice to capture the structure from different perspectives. Use the following parameters:

| Visualization | Elevation (`elev`) | Azimuth (`azim`) |
| :--- | :--- | :--- |
| **View A** | 30.0 | 45.0 |
| **View B** | 60.0 | 45.0 |

### Step 3: Report Compilation
Assemble the findings into a markdown report including:
1. **Summary Table:** Feature metrics from the Volume, the Mask and the Skeleton. 
2. **Visual Gallery:** Embed the two generated 3D plots.
3. **Analysis:** Brief interpretation of the mask-to-volume alignment.
4. **Provenance:** Call `explain_run(style="story")` and fold its verdict into the report —
   which defect classes the run supports, which it does not, and what would have to be
   fixed to recover the rest. A report that states counts without stating what they rest
   on is the thing this project keeps having to retract.

### If you are asked for defect counts
Do **not** assemble them from CSVs you find under `outputs/`. Several directories there
hold withdrawn methods that still parse — see `outputs/method_comparison/SUPERSEDED.md`.
Run `detect_lattice_defects` (classes: `missing`, `broken`, `thin`, `thick`, `necked`,
`nominal`) and read the counts from its own output. `AGENTS.md` has the full path and the
checks to run first.

# Technical Constraints
- Ensure all `.npy` arrays are checked for shape compatibility before processing.
- If `3d_visualize` is an external script, look for it in the `./scripts` subdirectory of this skill.
- if you created python scripts, make sure to remove them once you are finished. 