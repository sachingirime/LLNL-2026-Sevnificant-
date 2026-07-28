# Segmentation Evaluation Rubric

## Purpose
Evaluate a segmentation result image against a ground truth image for a
9x9x9 octet lattice CT scan slice. Judge visual and structural fidelity of
the segmented lattice structure.

## Criteria
Assess the result image against the ground truth on the following dimensions:

1. **Structural Integrity**: Does the result capture the connectivity of the
   lattice struts compared to the ground truth? Are struts continuous, or are
   they broken/fragmented where they shouldn't be?

2. **False Positives/Negatives**: Identify over-segmentation (extra noise or
   material not present in the ground truth) or under-segmentation (missing
   struts or material that should be present).

3. **Topology**: Are the nodes (junctions where multiple struts meet)
   preserved? Junction preservation is critical for lattice structures, since
   missing or merged nodes change the structure's connectivity.

4. **Noise and Artifacts**: Does the result image contain noise, speckling,
   or artifacts not present in the clean ground truth?

## Scoring (0-5)
Assign a single integer score based on overall fidelity to the ground truth:

- **5**: Identical to ground truth. No missing structures, no false positives.
- **4**: Excellent with very minor differences.
- **3**: Main topology is correct, but noticeable noise or thin struts are
  missing.
- **2**: Fair, but with significant differences (e.g., large chunks missing).
- **1**: Major structural failure or excessive noise.
- **0**: Blank or unrelated output.

## Output Format
Return ONLY a JSON object with exactly two fields, and no other text before
or after it:

```json
{
  "reasoning": "A concise explanation covering structural integrity, false positives/negatives, topology, and noise/artifacts observed when comparing the result image to the ground truth image.",
  "score": <integer from 0 to 5>
}
```

Do not include markdown code fences, headers, or any commentary outside the
JSON object itself.
