# Metadata Extractor

## Description
Loads a .npy file containing CT scan data (raw, segmented, or skeletonized) and prints basic metadata: shape, data type, min/max values, mean value, and non-zero voxel count.

## When to use this skill
Trigger when the user asks to inspect, summarize, or check metadata for a .npy file - e.g. "what's in this file", "check the shape of this array", "give me metadata for [file]".

## Instructions
1. Identify the .npy file path the user is referring to.
2. Run:
   python .agents/skills/metadata_extractor/scripts/metadata_extractor.py <filepath>
3. Report the printed metadata back to the user in a clear summary.
4. If the file doesn't exist or fails to load, report the error clearly.
