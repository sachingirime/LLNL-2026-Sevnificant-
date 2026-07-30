"""Lattice NDE Platform -- a single local web app over the project's
existing MCP tools (src/mcp_server.py) and its accumulated analysis outputs.

Run with:  python platform/backend/main.py
Then open: http://127.0.0.1:8420

Design notes:
  * Every tool in src/mcp_server.py is a plain function (the @mcp.tool()
    decorator from FastMCP registers it but returns it unchanged), so this
    app imports that module directly and calls the same functions Codex
    would call -- no logic is duplicated or reimplemented.
  * Calls run as background jobs (jobs.py) and are polled from the UI,
    since some tools (detect_lattice_defects) take up to ~10 minutes.
  * All file access is restricted to PROJECT_ROOT to keep the file
    browser/viewer from reaching outside the project.
"""
from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
PLATFORM_DIR = BACKEND_DIR.parent
PROJECT_ROOT = PLATFORM_DIR.parent
FRONTEND_DIR = PLATFORM_DIR / "frontend"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import mcp_server  # noqa: E402  (path insert above must run first)

from tool_specs import TOOLS, TOOLS_BY_NAME  # noqa: E402
from jobs import manager  # noqa: E402

app = FastAPI(title="Lattice NDE Platform")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

GALLERY_EXTS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
                ".csv": "csv", ".json": "json", ".md": "markdown", ".html": "html",
                ".txt": "text"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
             ".agents", ".codex", "platform"}


# ---------------------------------------------------------------- helpers --
def _safe_path(rel_or_abs: str) -> Path:
    """Resolve a user-supplied path against PROJECT_ROOT and refuse escapes.

    Accepts either an absolute path already inside the project, or a path
    relative to the project root (what the file browser hands back).
    """
    p = Path(rel_or_abs)
    candidate = p if p.is_absolute() else (PROJECT_ROOT / p)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise HTTPException(400, f"Path escapes the project root: {rel_or_abs}")
    return resolved


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))


# --------------------------------------------------------------- tool api --
@app.get("/api/tools")
def list_tools() -> list[dict[str, Any]]:
    return TOOLS


class RunRequest(BaseModel):
    args: dict[str, Any] = {}


@app.post("/api/run/{tool_name}")
def run_tool(tool_name: str, body: RunRequest) -> dict[str, Any]:
    spec = TOOLS_BY_NAME.get(tool_name)
    if spec is None:
        raise HTTPException(404, f"Unknown tool '{tool_name}'")
    fn = getattr(mcp_server, tool_name, None)
    if fn is None:
        raise HTTPException(500, f"Tool '{tool_name}' is declared but missing from mcp_server")

    args: dict[str, Any] = {}
    for field in spec["fields"]:
        name = field["name"]
        val = body.args.get(name, field.get("default"))
        if val in (None, "") and field.get("required") and field["type"] != "str":
            raise HTTPException(400, f"Missing required field '{name}'")
        if val in (None, "") and field["type"] == "path" and not field.get("required", True):
            args[name] = ""
            continue
        if field["type"] == "path" and val:
            # Resolve relative to the project root but pass an absolute path
            # string to the tool, which does its own file I/O.
            val = str(_safe_path(val))
        elif field["type"] == "int" and val != "":
            val = int(val)
        elif field["type"] == "float" and val != "":
            val = float(val)
        elif field["type"] == "bool":
            val = bool(val)
        args[name] = val

    job = manager.submit(tool_name, {k: v for k, v in args.items()}, fn)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return job.to_dict()


# ------------------------------------------------------------ file browser --
@app.get("/api/browse")
def browse(path: str = "") -> dict[str, Any]:
    target = _safe_path(path) if path else PROJECT_ROOT
    if not target.exists():
        raise HTTPException(404, f"No such path: {path}")
    if target.is_file():
        target = target.parent
    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
            if child.name in SKIP_DIRS or child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": _rel(child),
                "is_dir": child.is_dir(),
                "ext": child.suffix.lower(),
            })
    except PermissionError:
        pass
    parent = None if target.resolve() == PROJECT_ROOT.resolve() else _rel(target.parent)
    return {"cwd": _rel(target), "parent": parent, "entries": entries}


@app.get("/api/file")
def get_file(path: str):
    target = _safe_path(path)
    if not target.is_file():
        raise HTTPException(404, f"No such file: {path}")
    ext = target.suffix.lower()
    if ext in (".csv", ".json", ".md", ".txt"):
        return PlainTextResponse(target.read_text(errors="replace"))
    mime, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=mime or "application/octet-stream")


# ------------------------------------------------------------------ gallery --
@app.get("/api/gallery")
def gallery() -> dict[str, Any]:
    """Scan outputs/ (and top-level report docs) for anything already
    computed, grouped by directory, so the dashboard has something to show
    even before the user runs a single tool."""
    groups: list[dict[str, Any]] = []
    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.is_dir():
        for sub in sorted(p for p in outputs_dir.rglob("*") if p.is_dir()) + [outputs_dir]:
            items = []
            for f in sorted(sub.iterdir()):
                if f.is_file() and f.suffix.lower() in GALLERY_EXTS:
                    items.append({
                        "name": f.name, "path": _rel(f), "kind": GALLERY_EXTS[f.suffix.lower()],
                    })
            if items:
                groups.append({"dir": _rel(sub), "items": items})
    groups.sort(key=lambda g: g["dir"])

    reports = []
    for f in sorted(PROJECT_ROOT.glob("*.md")) + sorted(PROJECT_ROOT.glob("*.docx")):
        reports.append({"name": f.name, "path": _rel(f), "kind": "markdown" if f.suffix == ".md" else "docx"})

    return {"groups": groups, "reports": reports}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project_root": str(PROJECT_ROOT)}


# ------------------------------------------------------------------ static --
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8420)
