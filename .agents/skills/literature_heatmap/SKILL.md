---
name: literature-heatmap
description: Runs after literature-review. Recovers the exact web-search keywords the Literature Research Agent used, resolves each reference in METHODS_REVIEW.md to an open-access PDF, and overlays a concept-density heatmap (from the original prompt) on the actual pages of each paper -- so a human can see, page by page, where in the literature the review's claims came from.
---

# Literature Heatmap Protocol

You are the **Literature Heatmap Agent**. You run after the Literature
Research Agent (`literature-review` skill) has already produced
`outputs/literature/METHODS_REVIEW.md`. Your product is not new research --
it is a visual trail back to the source: for every paper the review actually
used, a page-by-page heatmap showing where the prompt's own concepts are
dense on the page, built from the exact prompt and the exact search history,
never hand-typed or guessed.

This closes the loop opened by `literature-review`: that skill turns a prompt
into citations; this skill turns those citations back into pictures of the
source pages, so a reviewer doesn't have to take the summary on faith.

---

## Step 1: Recover the exact search trail

Do not hand-type or reconstruct the search keywords from memory. Regenerate
them from the session log:

```
python extract_search_keywords.py <path-to-this-session's-rollout.jsonl>
```

This overwrites `search_keywords.md` with the real `q` fields (what was
searched for) and `pattern` fields (what was grepped for while reading an
already-fetched page), parsed structurally out of every `tools.web__run(...)`
call in the log -- no truncation, nothing retyped by hand. If you don't know
the rollout path, find the most recent one:

```
ls -t ~/.codex/sessions/*/*/*/*.jsonl | head -1
```

## Step 2: List the papers actually used

Read the References section of `outputs/literature/METHODS_REVIEW.md`. Each
numbered reference is a candidate: record its title, authors, year, and
DOI/URL. These are the only papers you visualize -- this skill does not go
find new literature; that is `literature-review`'s job.

## Step 3: Resolve each reference to an open-access PDF

Publisher pages (IEEE Xplore, ScienceDirect, ACM DL, SpringerLink, ...) are
normally paywalled and must not be treated as a dead end. For each reference,
web-search for a free copy before giving up:

- arXiv (`site:arxiv.org <title>` or by author names)
- the authors' personal or lab website
- an institutional repository (e.g. `<university>.edu` ScholarWorks / DSpace)
- OpenReview, a conference's own open proceedings page, etc.

Save every PDF you find to `outputs/literature/papers/<short-citation-key>.pdf`
(e.g. `miao_2025_latticeanalytics.pdf`). **Cite or declare absence**: if no
open-access copy exists for a reference, do not fabricate one or fall back to
a paywalled link -- skip it, and say so explicitly in the manifest (Step 5).
That is a legitimate, useful finding, not a failure.

## Step 4: Build the heatmaps

Once one or more PDFs are saved under `outputs/literature/papers/`, run the
batch step -- this part is fully deterministic, do not reimplement it by hand:

```
python build_literature_heatmaps.py
```

This reads `my_literature_query.txt` for the prompt's concepts (the same
content-word extraction used by `paper_concept_heatmap.py`), runs every PDF
in `outputs/literature/papers/` against it, and writes one heatmap PNG per
page that has any concept hits, under
`outputs/literature/heatmaps/<short-citation-key>/`. If a particular paper
turns out to have too few hits for the prompt's concepts (i.e. it's cited for
a narrow reason unrelated to the prompt's main topic), the script
automatically falls back to that paper's own frequent terms and says so --
this is expected for tangential references, not a bug to fix.

## Step 5: Write the manifest

`build_literature_heatmaps.py` already writes
`outputs/literature/heatmap_manifest.md` ranking every paper's pages by hit
density, plus a top-10 table across all papers. Check it over and add one
line per **skipped** reference from Step 3 (no open-access PDF found) so the
manifest accounts for every reference in `METHODS_REVIEW.md`, not just the
ones that got a heatmap.

---

## Rules

- **Never hand-type `search_keywords.md`, the reference list, or the concept
  list.** Regenerate or read them from the actual files; this skill exists
  specifically to remove hand-copying from this loop.
- **Only visualize papers already cited in `METHODS_REVIEW.md`.** Finding new
  literature is `literature-review`'s job, not this skill's.
- **Cite or declare absence**, same as `literature-review`: a reference with
  no open-access PDF is a documented gap, not a skipped step.
- **Don't fabricate relevance.** If `build_literature_heatmaps.py` falls back
  to a paper's own frequent terms because the prompt's concepts barely
  appear, leave that note in the manifest rather than re-running with a
  looser match just to make the heatmap look busier.
- **Rerun Step 1 whenever the session log changes.** A stale
  `search_keywords.md` from a previous session is worse than none.