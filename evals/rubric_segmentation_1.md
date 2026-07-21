# Rubric: Segmentation Quality vs. Ground Truth

You are evaluating a binary segmentation of an X-ray CT scan of a 9x9x9 octet-truss
lattice. Two images are attached:

1. **Ground truth** — the reference segmentation of slice 380.
2. **Result** — the segmentation produced by the segmentation agent for the same slice.

Judge how faithfully the result reproduces the lattice structure shown in the ground
truth, then return a single integer score from 0 to 5.

---

## Before you begin: compare structure, not rendering

The two images were produced by different plotting code and will **not** match
pixel-for-pixel even when the underlying segmentation is identical. The following are
artifacts of how each image was drawn. **Ignore them entirely — they are not
segmentation errors:**

- **Colour and colormap.** The ground truth may be rendered in viridis (dark purple
  background, yellow material) and the result in greyscale (black background, white
  material). In each image, treat the bright regions as material regardless of hue.
- **Figure chrome.** Axes, tick marks, numeric labels, titles, colorbars, borders, and
  surrounding whitespace.
- **Resolution and framing.** The images differ in pixel dimensions, aspect padding, and
  DPI. Do not attempt pixel-level alignment.
- **Edge softness.** The ground truth may be anti-aliased with soft, graded edges, while
  the result may have hard binary edges. This is a rendering difference, not roughness.

Compare only the **segmented structure**: where material is and is not, and how it
connects.

If the result appears rotated, mirrored, or transposed relative to the ground truth, do
not silently penalise it under the criteria below. Score the structure as you see it and
state the discrepancy explicitly in your reasoning — it indicates an axis-ordering bug
rather than a thresholding failure.

---

## How to read this slice

Slice 380 cuts through the lattice at an angle, so the image contains **two visually
different regions, and both are correct**:

- **Connected outlines.** Where struts run roughly parallel to the slice plane, they
  appear as continuous diamond-shaped outlines. Connectivity is meaningful here.
- **Isolated dots.** Where struts run roughly perpendicular to the slice plane, they
  appear only as isolated cross-sections in a regular grid. **These are supposed to be
  disconnected.** A field of separate dots is the correct appearance, not fragmentation.

Judge connectivity only where the ground truth itself shows continuous structure. Never
penalise the result for disconnection that is also present in the ground truth.

**The specimen has intentionally missing struts** (nominally 0.5% of them), so the ground
truth legitimately contains gaps. A gap counts against the result **only if it appears in
the result and not in the ground truth**. Gaps present in both are real features of the
part and must be treated as neutral — matching them is correct behaviour.

---

## Criteria

Weigh all four together. No single criterion determines the score alone.

**1. Structural Integrity.** Does the result capture the connectivity of the lattice
struts? Struts continuous in the ground truth should be continuous in the result, not
broken into fragments. Fragmentation of otherwise-correct struts is a serious defect even
when the total amount of material looks right. Assess this only in the regions where the
ground truth shows continuous outlines.

**2. False Positives / False Negatives.** Identify over- and under-segmentation.
Over-segmentation appears as material where the ground truth shows background: thickened
struts, filled cell interiors, or scattered specks. Under-segmentation appears as missing
or thinned features. Because the lattice is a regular grid, the practical check is
whether every grid position carrying material in the ground truth also carries it in the
result. Distinguish a few marginal thin features from large contiguous regions of missing
structure — the latter is far worse.

**3. Topology.** Are the junctions preserved? Nodes appear as the larger, denser blobs at
the intersections of the lattice grid, distinct from the smaller strut cross-sections.
Nodes are preserved if they occur at the same grid positions, remain single connected
blobs, and have neither eroded away nor merged into neighbouring features. Broken
topology is serious even when individual struts look acceptable.

**4. Noise and Artifacts.** Does the result contain speckle, isolated stray pixels, or
ragged boundaries absent from the clean ground truth? Note whether such noise is confined
to the background or intrudes on the structure. Remember that hard binary edges and
aliasing are rendering differences, not artifacts.

---

## Scoring

Return one integer:

| Score | Meaning |
| :---- | :------ |
| **5** | Identical to the ground truth. No missing structures, no false positives. |
| **4** | Excellent, with very minor differences: slight edge roughness, one or two marginal features. |
| **3** | Main topology is correct, but there is noticeable noise or some thin struts are missing. |
| **2** | Fair, but with significant differences — large chunks of lattice missing, or substantial spurious material. |
| **1** | Major structural failure or excessive noise; the lattice is barely recognisable. |
| **0** | Blank, uniform, or unrelated output; no lattice structure present. |

When the result sits between two levels, choose the lower. Base the score on what is
visible in the images, not on assumptions about how the segmentation was produced.

---

## Output format

Return **only** a JSON object — no prose before or after it, and no markdown code fence.
It must have exactly these two keys:

```json
{
  "reasoning": "Concise assessment addressing Structural Integrity, False Positives/Negatives, Topology, and Noise and Artifacts by name, citing specific visual evidence, and justifying the score against the band definitions.",
  "score": 4
}
```

- `reasoning` — a string. Address all four criteria by name and cite what you actually
  observed in the images, not what you would expect a segmentation to look like.
- `score` — an integer from 0 to 5 inclusive. Not a string, not a float, not a range.
