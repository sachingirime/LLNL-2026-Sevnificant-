"""
build_literature_heatmaps.py

Deterministic batch step of the literature-heatmap subagent (see
.agents/skills/literature_heatmap/SKILL.md): runs paper_concept_heatmap.py's
underlying functions against every PDF already resolved and saved under
outputs/literature/papers/, all against the same prompt concepts
(my_literature_query.txt), and writes one manifest ranking every paper's
pages by hit density -- so a human (or the next agent) can jump straight to
the most relevant page of the most relevant paper instead of paging through
every heatmap of every paper one at a time.

This script deliberately does NOT search the web or decide which papers are
worth visualizing -- that judgment call (Steps 2-3 of the skill) belongs to
the live agent, which has web access and can read abstracts; this script
only does the mechanical part: given PDFs that already exist locally,
heatmap all of them and rank the results.

Usage:
    python build_literature_heatmaps.py
    python build_literature_heatmaps.py --papers-dir outputs/literature/papers \\
        --concepts-file my_literature_query.txt
"""

import argparse
import glob
import os

import fitz  # PyMuPDF

from paper_concept_heatmap import (
    load_prompt_concepts,
    auto_extract_concepts,
    find_hits,
    render_page_with_heatmap,
)


def slugify(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_")[:60] or "paper"


def process_paper(pdf_path, concepts, out_root, hits_for_full_red, blur_sigma, min_hits_before_fallback):
    doc = fitz.open(pdf_path)
    slug = slugify(pdf_path)
    out_dir = os.path.join(out_root, slug)

    per_page_hits = [find_hits(doc[i], concepts) for i in range(doc.page_count)]
    total_hits = sum(len(h) for h in per_page_hits)

    used_concepts = concepts
    fallback = False
    if total_hits < min_hits_before_fallback:
        used_concepts = auto_extract_concepts(doc)
        fallback = True
        per_page_hits = [find_hits(doc[i], used_concepts) for i in range(doc.page_count)]

    os.makedirs(out_dir, exist_ok=True)
    page_results = []
    for i, hits in enumerate(per_page_hits):
        if not hits:
            continue
        page = doc[i]
        img, n_hits = render_page_with_heatmap(
            page, hits, blur_sigma_px=blur_sigma, hits_for_full_red=hits_for_full_red,
        )
        out_path = os.path.join(out_dir, f"page_{i + 1:02d}_heatmap.png")
        img.save(out_path)
        page_results.append((i + 1, n_hits, out_path))

    page_results.sort(key=lambda r: -r[1])
    return {
        "pdf": pdf_path,
        "slug": slug,
        "out_dir": out_dir,
        "fallback_to_auto": fallback,
        "concepts_used": used_concepts,
        "pages": page_results,
    }


def write_manifest(manifest_path, prompt_concepts_file, prompt_concepts, results):
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Literature Heatmap Manifest\n\n")
        f.write(f"Prompt: `{prompt_concepts_file}`\n\n")
        f.write(f"Prompt concepts searched for: {', '.join(prompt_concepts)}\n\n")

        ranked_overall = sorted(
            [(r, p) for r in results for p in r["pages"]],
            key=lambda rp: -rp[1][1],
        )
        if ranked_overall:
            f.write("## Strongest pages across all papers\n\n")
            f.write("| paper | page | hits | file |\n|---|---|---|---|\n")
            for r, (page_no, n_hits, path) in ranked_overall[:10]:
                f.write(f"| {os.path.basename(r['pdf'])} | {page_no} | {n_hits} | `{path}` |\n")
            f.write("\n")

        f.write("## Per-paper detail\n\n")
        for r in results:
            f.write(f"### {os.path.basename(r['pdf'])}\n\n")
            if r["fallback_to_auto"]:
                f.write(
                    f"*Too few hits for the prompt's own concepts -- this paper doesn't look like it's "
                    f"about the prompt's topic. Fell back to this paper's own frequent terms instead: "
                    f"{', '.join(r['concepts_used'])}.*\n\n"
                )
            if not r["pages"]:
                f.write("No concept hits found on any page.\n\n")
                continue
            f.write("| page | hits | file |\n|---|---|---|\n")
            for page_no, n_hits, path in r["pages"]:
                f.write(f"| {page_no} | {n_hits} | `{path}` |\n")
            f.write(f"\nStrongest page: **{r['pages'][0][0]}** ({r['pages'][0][1]} hits).\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers-dir", default="outputs/literature/papers")
    parser.add_argument("--concepts-file", default="my_literature_query.txt")
    parser.add_argument("--out-root", default="outputs/literature/heatmaps")
    parser.add_argument("--manifest", default="outputs/literature/heatmap_manifest.md")
    parser.add_argument("--hits-for-full-red", type=float, default=4.0)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--min-hits-before-fallback", type=int, default=8)
    args = parser.parse_args()

    pdfs = sorted(glob.glob(os.path.join(args.papers_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(
            f"No PDFs found in {args.papers_dir}. Resolve papers to open-access PDFs first "
            f"(Steps 2-3 of the literature_heatmap skill) and save them there, then re-run."
        )

    concepts = load_prompt_concepts(args.concepts_file)
    print(f"Prompt concepts ({args.concepts_file}): {concepts}")

    results = []
    for pdf in pdfs:
        print(f"\n{pdf}")
        r = process_paper(
            pdf, concepts, args.out_root, args.hits_for_full_red, args.blur_sigma,
            args.min_hits_before_fallback,
        )
        results.append(r)
        if r["fallback_to_auto"]:
            print(f"  too few prompt-concept hits -- fell back to auto concepts: {r['concepts_used']}")
        for page_no, n_hits, path in r["pages"][:3]:
            print(f"  page {page_no}: {n_hits} hit(s) -> {path}")
        if not r["pages"]:
            print("  no hits on any page")

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    write_manifest(args.manifest, args.concepts_file, concepts, results)
    print(f"\nWrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
