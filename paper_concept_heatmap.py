"""
paper_concept_heatmap.py

Eye-tracking-style heatmap overlay on a rendered PDF page, in the vein of
https://inforiver.com/insights/heatmaps-in-data-visualization-a-comprehensive-introduction/
(soft green/yellow/orange/red gaussian blobs over a page screenshot, dense
where several relevant words cluster, isolated small blobs elsewhere).

This is honestly labelled: it is NOT real eye-tracking or gaze data, and it
is NOT a neural attention weight. It's a *concept-occurrence* heatmap --
"heat" = density of prompt-relevant words on the page, found by locating each
concept word's bounding box with PyMuPDF and stacking a soft blob at every
hit. Overlapping hits (several concept words near each other) sum before the
blur, which is what produces the bigger hot cores, exactly like the
reference image's cluster around dense, on-topic paragraphs.

Two concept sources:
  - "prompt"  (default) -- content words from PROMPT_FILE (my_literature_query.txt),
    stopword-filtered. Use this against an actual literature-review paper.
  - "auto" -- top frequent content words from the PDF's own text. Falls back
    to this automatically if the prompt-derived concepts get too few hits
    (e.g. the PDF isn't actually about the prompt's topic), so you always get
    a meaningful heatmap instead of a blank page.

Usage:
    python paper_concept_heatmap.py "some_paper.pdf"
    python paper_concept_heatmap.py "some_paper.pdf" --mode auto
    python paper_concept_heatmap.py "some_paper.pdf" --max-pages 5 --out-dir heatmap_pages
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from scipy.ndimage import gaussian_filter

PROMPT_FILE = "my_literature_query.txt"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "about", "as", "by", "from", "that", "this", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "i", "me", "my",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "look", "tell", "give", "use", "uses", "used", "using", "all", "any",
    "other", "others", "do", "does", "did", "have", "has", "had", "not",
    "so", "than", "then", "there", "here", "such", "into", "out", "up",
    "down", "over", "under", "how", "when", "where", "why", "can", "will",
    "would", "should", "could", "may", "might", "must", "also", "each",
    "more", "most", "some", "no", "yes", "one", "two", "three", "new",
    "like", "below", "above", "different", "same", "type", "types",
    "understand", "shows", "show", "several", "example", "examples",
    "often", "much", "many", "very", "well", "just", "still", "even",
    "way", "ways", "make", "made", "making", "take", "taken", "taking",
    "see", "seen", "let", "us", "these", "those", "instead", "rather",
    "usually", "typically", "generally", "simply", "particular",
    "particularly", "certain", "either", "both", "quickly", "easily",
}

_WORD_RE = re.compile(r"[A-Za-z]+")  # hyphens split into separate words, so e.g.
# "threshold-determination" is searchable as "threshold" and "determination"


def load_prompt_concepts(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_tokens = _WORD_RE.findall(text)
    # Also split hyphenated compounds ("threshold-determination") into their
    # parts -- a PDF will almost never contain that exact hyphenated token,
    # but very likely contains "threshold" and "determination" separately.
    tokens = []
    for w in raw_tokens:
        tokens.append(w)
        if "-" in w:
            tokens.extend(w.split("-"))

    seen = []
    for w in tokens:
        # Acronyms (AI, CT, ML, ...) are written ALL-CAPS in normal prose and
        # are exactly the kind of short-but-load-bearing term the length
        # filter below would otherwise drop -- exempt them from it.
        is_acronym = len(w) >= 2 and w.isupper()
        wl = w.lower()
        if wl in _STOPWORDS:
            continue
        if not is_acronym and len(wl) < 3:
            continue
        if wl not in seen:
            seen.append(wl)
    return seen


def auto_extract_concepts(doc, top_n=14):
    counts = Counter()
    for page in doc:
        words = [w.lower() for w in _WORD_RE.findall(page.get_text())]
        counts.update(w for w in words if w not in _STOPWORDS and len(w) >= 5)
    return [w for w, _ in counts.most_common(top_n)]


def _matches(word_clean, concepts):
    for c in concepts:
        if word_clean == c:
            return c
        if len(word_clean) >= 4 and len(c) >= 4:
            if word_clean.startswith(c) or c.startswith(word_clean):
                return c
    return None


def find_hits(page, concepts):
    """Return list of (x0, y0, x1, y1, matched_concept) in PDF point space."""
    hits = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        wc = re.sub(r"[^A-Za-z\-]", "", word).lower()
        if not wc:
            continue
        m = _matches(wc, concepts)
        if m:
            hits.append((x0, y0, x1, y1, m))
    return hits


HEAT_COLORS = [
    (0.00, (30, 180, 30)),    # green
    (0.45, (190, 220, 20)),   # yellow-green
    (0.70, (255, 200, 0)),    # amber
    (0.88, (255, 120, 0)),    # orange
    (1.00, (230, 20, 20)),    # red
]


def _heat_to_rgb(t):
    """t: array in [0,1] -> (..., 3) uint8 array, piecewise-linear over HEAT_COLORS."""
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros(t.shape + (3,), dtype=np.float32)
    for (t0, c0), (t1, c1) in zip(HEAT_COLORS[:-1], HEAT_COLORS[1:]):
        mask = (t >= t0) & (t <= t1)
        span = max(t1 - t0, 1e-6)
        local = np.clip((t[mask] - t0) / span, 0, 1)
        for ch in range(3):
            out[..., ch][mask] = c0[ch] + local * (c1[ch] - c0[ch])
    return out


def render_page_with_heatmap(page, hits, zoom=2.2, blur_sigma_px=30, max_alpha=0.68,
                              hits_for_full_red=4.0):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).astype(np.float32)

    heat = np.zeros((pix.height, pix.width), dtype=np.float32)
    for x0, y0, x1, y1, _concept in hits:
        cx = int((x0 + x1) / 2 * zoom)
        cy = int((y0 + y1) / 2 * zoom)
        # Slightly more weight for longer word/phrase boxes, capped so one very
        # long token (e.g. a URL slug) can't single-handedly read as a hot cluster.
        weight = min(1.8, 1.0 + 0.4 * max(0, (x1 - x0) * zoom - 20) / 40)
        if 0 <= cy < heat.shape[0] and 0 <= cx < heat.shape[1]:
            heat[cy, cx] += weight

    if heat.max() <= 0:
        return Image.fromarray(page_img.astype(np.uint8)), 0

    heat = gaussian_filter(heat, sigma=blur_sigma_px)

    # Calibrate against the analytical peak of ONE isolated hit after blurring
    # (the center height of a unit-mass Gaussian kernel with this sigma), not a
    # per-page percentile -- otherwise a page where every hit is isolated ends
    # up with its "typical" hit near the ceiling and everything reads as hot.
    # `hits_for_full_red` nearby, overlapping hits are needed to reach full red;
    # a single isolated hit should land solidly in the green range.
    single_hit_peak = 1.0 / (2 * np.pi * blur_sigma_px ** 2)
    ceiling = single_hit_peak * hits_for_full_red
    heat_norm = np.clip(heat / max(ceiling, 1e-12), 0, 1)

    alpha = np.clip(heat_norm, 0, 1) * max_alpha
    alpha[heat_norm < 0.10] = 0.0

    color = _heat_to_rgb(heat_norm)
    alpha3 = alpha[..., None]
    composite = page_img * (1 - alpha3) + color * alpha3
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return Image.fromarray(composite), len(hits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path")
    parser.add_argument("--mode", choices=["prompt", "auto"], default="prompt",
                         help="Concept source (default: prompt, falling back to auto if too few hits).")
    parser.add_argument("--concepts-file", default=PROMPT_FILE)
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages processed.")
    parser.add_argument("--min-hits-before-fallback", type=int, default=8,
                         help="If prompt-mode finds fewer total hits than this across the whole doc, "
                              "auto-fall-back to frequency-based concepts (default: 8).")
    parser.add_argument("--out-dir", default="heatmap_pages")
    parser.add_argument("--hits-for-full-red", type=float, default=4.0,
                         help="How many nearby/overlapping concept hits it takes to reach full red "
                              "(default: 4.0). Lower this (e.g. 2.5) for a 'hotter' looking map with "
                              "more orange/red; raise it for a calmer, mostly-green map.")
    parser.add_argument("--blur-sigma", type=float, default=30.0,
                         help="Gaussian blur radius in pixels at the render resolution (default: 30). "
                              "Bigger = larger, softer blobs that merge more readily.")
    args = parser.parse_args()

    doc = fitz.open(args.pdf_path)
    n_pages = doc.page_count if args.max_pages is None else min(args.max_pages, doc.page_count)

    if args.mode == "prompt":
        concepts = load_prompt_concepts(args.concepts_file)
        print(f"Prompt concepts ({args.concepts_file}): {concepts}")
        total_hits = sum(len(find_hits(doc[i], concepts)) for i in range(n_pages))
        if total_hits < args.min_hits_before_fallback:
            print(f"Only {total_hits} hit(s) for prompt concepts across {n_pages} page(s) -- "
                  f"this PDF doesn't look like it's about that topic. Falling back to --mode auto "
                  f"(this document's own frequent terms) so the heatmap isn't blank.")
            concepts = auto_extract_concepts(doc)
            print(f"Auto concepts: {concepts}")
    else:
        concepts = auto_extract_concepts(doc)
        print(f"Auto concepts: {concepts}")

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for i in range(n_pages):
        page = doc[i]
        hits = find_hits(page, concepts)
        img, n_hits = render_page_with_heatmap(
            page, hits, blur_sigma_px=args.blur_sigma, hits_for_full_red=args.hits_for_full_red,
        )
        if n_hits == 0:
            continue
        out_path = os.path.join(args.out_dir, f"page_{i + 1:02d}_heatmap.png")
        img.save(out_path)
        written.append((out_path, n_hits))
        print(f"  page {i + 1}: {n_hits} hit(s) -> {out_path}")

    if not written:
        print("No pages had any concept hits -- nothing written.")
    else:
        print(f"\nWrote {len(written)} heatmap page(s) to {args.out_dir}/")


if __name__ == "__main__":
    main()
