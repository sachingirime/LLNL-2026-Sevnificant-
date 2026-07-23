# Segmentation Evaluation Rubric

You are evaluating a segmentation result image against a ground-truth image for an octet lattice CT slice.

## Objective
Judge how closely the submitted result matches the ground truth in terms of lattice structure preservation, topology, and artifact behavior.

## Criteria
1. Structural Integrity
   - Does the result preserve the connectivity and continuity of the lattice struts compared with the ground truth?
2. False Positives / False Negatives
   - Identify over-segmentation (extra noise or spurious structures) and under-segmentation (missing struts or broken connections).
3. Topology
   - Are key junctions, nodes, and major lattice intersections preserved?
4. Noise and Artifacts
   - Does the result contain extra noise, isolated specks, broken edges, or other artifacts not present in the clean ground truth?

## Scoring Scale
- 5: Identical to ground truth. No missing structures, no false positives.
- 4: Excellent with very minor differences.
- 3: Main topology is correct, but noticeable noise or thin struts are missing.
- 2: Fair, but with significant differences such as large missing chunks.
- 1: Major structural failure or excessive noise.
- 0: Blank or unrelated output.

## Required Response Format
Return only valid JSON with exactly these keys:

```json
{
  "reasoning": "<brief explanation of the comparison>",
  "score": <integer from 0 to 5>
}
```

## Evaluation Guidance
- Compare the result image to the ground truth image only.
- Favor preservation of global lattice connectivity over tiny local pixel differences.
- Penalize false positives and disconnected artifacts strongly.
- When uncertainty exists, prefer the lower score that is still justified by the evidence.
