---
name: threshold-optimizer
description: Compare, sweep, or optimize segmentation thresholds on the same input CT volume by calling the segment_ct_dataset MCP tool once per threshold and saving separate outputs for comparison. Use for requests such as "try a few thresholds on this scan," "compare segmentation results," or "find the best threshold for this dataset."
---

# Threshold Optimizer

## Instructions
1. Identify the input .npy file the user wants segmented.
2. Choose a set of threshold values to test. If the user doesn't specify values, default to a sweep of 3-5 values spanning the data's value range (check min/max first if unknown).
3. For each threshold value, call the segment_ct_dataset MCP tool with:
   - The same input_filepath
   - output_filepath MUST be exactly:
     outputs/threshold_optimizer/segmentation_threshold_<value>.npy
     Create the outputs/threshold_optimizer/ directory first if it does not already exist. Do NOT write to any other directory (e.g. outputs/task1) and do NOT reuse filenames from other tasks.
   - The threshold value itself
4. Collect the returned status message from each call (each includes foreground_voxels and shape).
5. Summarize the results in a comparison table for the user, showing threshold value vs. foreground voxel count, so they can see how segmentation changes across thresholds.
6. Do not guess at which threshold is "best" - just present the comparison and let the user decide, unless they ask for a recommendation.
