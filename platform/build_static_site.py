#!/usr/bin/env python3
"""Builds a static, GitHub-Pages-publishable snapshot of this project's
already-computed outputs.

This does NOT run any analysis. It only looks at what's already on disk
(typically what Codex just committed after running the pipeline) and turns
it into a self-contained static site: index.html + gallery.json + tools.json
+ a `files/` mirror of every referenced output and report. No Python beyond
the standard library is required, on purpose -- the GitHub Actions runner
that calls this should not need the project's scientific dependencies just
to publish a gallery of PNGs and CSVs.

Usage:
    python platform/build_static_site.py --out _site

The GitHub Actions workflow at .github/workflows/deploy-pages.yml calls this
on every push to main (when outputs/ changes) and publishes the result with
actions/upload-pages-artifact + actions/deploy-pages.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLATFORM_DIR.parent
sys.path.insert(0, str(PLATFORM_DIR / "backend"))
from tool_specs import TOOLS  # noqa: E402

GALLERY_EXTS = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".csv": "csv", ".json": "json", ".md": "markdown", ".html": "html", ".txt": "text",
}
SKIP_DIR_NAMES = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
                  ".agents", ".codex", "platform", ".github"}


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))


def _is_skipped(p: Path) -> bool:
    return any(part in SKIP_DIR_NAMES or part.startswith(".") for part in p.relative_to(PROJECT_ROOT).parts)


def scan_gallery() -> tuple[dict, list[Path]]:
    groups: list[dict] = []
    referenced: list[Path] = []
    outputs_dir = PROJECT_ROOT / "outputs"

    if outputs_dir.is_dir():
        candidate_dirs = [d for d in outputs_dir.rglob("*") if d.is_dir() and not _is_skipped(d)]
        candidate_dirs.append(outputs_dir)
        for sub in sorted(set(candidate_dirs), key=_rel):
            items = []
            for f in sorted(sub.iterdir()):
                if f.is_file() and f.suffix.lower() in GALLERY_EXTS:
                    items.append({"name": f.name, "path": _rel(f), "kind": GALLERY_EXTS[f.suffix.lower()]})
                    referenced.append(f)
            if items:
                groups.append({"dir": _rel(sub), "items": items})
    groups.sort(key=lambda g: g["dir"])

    reports: list[dict] = []
    for f in sorted(PROJECT_ROOT.glob("*.md")):
        reports.append({"name": f.name, "path": _rel(f), "kind": "markdown"})
        referenced.append(f)
    for f in sorted(PROJECT_ROOT.glob("*.docx")):
        reports.append({"name": f.name, "path": _rel(f), "kind": "docx"})
        # Not copied/previewed client-side -- listed for visibility only.

    return {"groups": groups, "reports": reports}, referenced


def build(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    gallery, referenced = scan_gallery()
    (out_dir / "gallery.json").write_text(json.dumps(gallery, indent=2))
    (out_dir / "tools.json").write_text(json.dumps(TOOLS, indent=2))

    files_dir = out_dir / "files"
    copied = 0
    for f in referenced:
        if f.suffix.lower() == ".docx":
            continue
        dest = files_dir / _rel(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1

    template = PLATFORM_DIR / "frontend" / "static_index.html"
    shutil.copy2(template, out_dir / "index.html")

    print(
        f"Built static site: {len(gallery['groups'])} output folders, "
        f"{len(gallery['reports'])} reports, {copied} files copied -> {out_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PLATFORM_DIR / "_site"),
                         help="Directory to write the static site into (default: platform/_site)")
    args = parser.parse_args()
    build(Path(args.out).resolve())
