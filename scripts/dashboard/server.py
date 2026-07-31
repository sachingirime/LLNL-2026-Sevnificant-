#!/usr/bin/env python
"""Codex dashboard — chat with the Codex agent and watch what it does, live.

Slice 1: the chat pane and the agent activity timeline. `codex exec --json`
prints one JSON event per line on stdout; this server spawns it, forwards every
event to the browser over SSE, and keeps the thread id so the next message
resumes the same conversation.

Nothing here imports from `src/`. The dashboard reads the repo, it does not
change how the repo works.

    python scripts/dashboard/server.py
    python scripts/dashboard/server.py --port 9000 --sandbox workspace-write
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STATIC = HERE / "static"

# Raw codex events, one run per file. Kept so a run can be replayed into the
# dashboard later without spending tokens — and so a demo does not depend on
# the network being up.
RUNS_DIR = REPO_ROOT / "outputs" / "dashboard_runs"

SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")

# run_id -> live subprocess, so /api/cancel can stop a turn mid-flight.
RUNNING: dict[str, asyncio.subprocess.Process] = {}


def codex_bin() -> str:
    found = shutil.which("codex")
    if not found:
        sys.exit("codex not found on PATH — install the Codex CLI first")
    return found


def build_command(prompt: str, thread_id: str | None, sandbox: str, model: str | None) -> list[str]:
    """Assemble the `codex exec` invocation for one turn.

    `codex exec resume` accepts neither -C/--cd nor -s/--sandbox, so the working
    directory travels through the subprocess cwd and the sandbox through a
    `-c sandbox_mode=` config override. Using that override on both paths keeps
    the first turn and every resumed turn on one code path.
    """
    cmd = [codex_bin(), "exec"]
    if thread_id:
        cmd.append("resume")
    cmd += ["--json", "-c", f'sandbox_mode="{sandbox}"']
    if model:
        cmd += ["-m", model]
    if thread_id:
        cmd.append(thread_id)
    cmd.append(prompt)
    return cmd


async def _pump(stream: asyncio.StreamReader, kind: str, queue: asyncio.Queue) -> None:
    """Forward one output stream into the shared queue, then post an EOF marker."""
    while True:
        raw = await stream.readline()
        if not raw:
            break
        await queue.put((kind, raw.decode("utf-8", "replace").rstrip("\r\n")))
    await queue.put((kind, None))


async def run_turn(prompt: str, thread_id: str | None, sandbox: str, model: str | None):
    """Yield SSE payloads for a single agent turn."""
    run_id = uuid.uuid4().hex[:12]
    cmd = build_command(prompt, thread_id, sandbox, model)

    yield {
        "event": "message",
        "data": json.dumps(
            {"type": "run.started", "run_id": run_id, "command": cmd, "cwd": str(REPO_ROOT)}
        ),
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,  # or codex waits on piped stdin for more prompt
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        yield {"event": "message", "data": json.dumps({"type": "run.failed", "error": str(exc)})}
        return

    RUNNING[run_id] = proc
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    journal = (RUNS_DIR / f"{run_id}.jsonl").open("w", encoding="utf-8")
    # The prompt is not a codex event, but a replay without it shows answers to
    # a question nobody asked.
    journal.write(json.dumps({"type": "user_prompt", "text": prompt}) + "\n")
    queue: asyncio.Queue = asyncio.Queue()
    pumps = [
        asyncio.create_task(_pump(proc.stdout, "stdout", queue)),
        asyncio.create_task(_pump(proc.stderr, "stderr", queue)),
    ]

    try:
        open_streams = 2
        while open_streams:
            kind, line = await queue.get()
            if line is None:
                open_streams -= 1
                continue
            if kind == "stderr":
                if line.strip():
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "stderr", "line": line}),
                    }
                continue
            # stdout is JSONL from codex; anything unparseable is still worth showing.
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "raw", "line": line}
            journal.write(line + "\n")
            yield {"event": "message", "data": json.dumps(event)}

        code = await proc.wait()
        yield {"event": "message", "data": json.dumps({"type": "run.exited", "code": code})}
    finally:
        # Covers the normal end, a cancel, and the browser tab closing mid-turn.
        journal.close()
        for task in pumps:
            task.cancel()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        RUNNING.pop(run_id, None)


async def api_chat(request: Request):
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)

    sandbox = body.get("sandbox") or "read-only"
    if sandbox not in SANDBOX_MODES:
        return JSONResponse({"error": f"bad sandbox mode: {sandbox}"}, status_code=400)

    thread_id = body.get("thread_id") or None
    model = (body.get("model") or "").strip() or None
    return EventSourceResponse(run_turn(prompt, thread_id, sandbox, model))


async def api_cancel(request: Request):
    body = await request.json()
    proc = RUNNING.get(body.get("run_id") or "")
    if proc is None:
        return JSONResponse({"cancelled": False, "reason": "no such run"}, status_code=404)
    proc.terminate()
    return JSONResponse({"cancelled": True})


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


async def api_config(request: Request):
    try:
        version = subprocess.run(
            [codex_bin(), "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = "unknown"

    dirty = _git("status", "--porcelain")
    return JSONResponse(
        {
            "repo": str(REPO_ROOT),
            "repo_name": REPO_ROOT.name,
            "codex_version": version,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("log", "-1", "--format=%h %s"),
            "dirty_files": len([ln for ln in dirty.splitlines() if ln.strip()]),
            "sandbox_modes": list(SANDBOX_MODES),
            "default_sandbox": request.app.state.default_sandbox,
        }
    )


async def api_runs(request: Request):
    """List saved runs, newest first."""
    if not RUNS_DIR.exists():
        return JSONResponse([])
    runs = sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse(
        [{"name": p.stem, "events": sum(1 for _ in p.open()), "mtime": p.stat().st_mtime} for p in runs]
    )


async def api_run(request: Request):
    """Return one saved run's raw events for replay."""
    name = request.path_params["name"]
    path = (RUNS_DIR / f"{name}.jsonl").resolve()
    # Keep the name from escaping RUNS_DIR.
    if RUNS_DIR.resolve() not in path.parents or not path.is_file():
        return JSONResponse({"error": "no such run"}, status_code=404)
    return FileResponse(path, media_type="application/x-ndjson")


ORDER = ["missing", "broken", "thin", "thick", "necked", "nominal"]
# scripts/classify_strut_defects.CLASS_COLORS — the detector's own map, so this tab and
# every figure already in outputs/ name the classes with the same colours.
CLASS_COLORS = {
    "missing": "#e34948", "broken": "#eb6834", "thin": "#2a78d6",
    "thick": "#eda100", "necked": "#1baf7a", "nominal": "#b8b7b2",
}
VIZ_SUFFIXES = {".png", ".gif", ".jpg", ".jpeg", ".svg", ".html"}
# Anything past this is linked, never embedded: the biggest viewers here are 36-67 MB.
EMBED_LIMIT = 4_000_000


def _safe(rel: str) -> Path | None:
    """Resolve a repo-relative path, refusing anything that escapes the repo."""
    try:
        path = (REPO_ROOT / rel).resolve()
    except (OSError, ValueError):
        return None
    root = REPO_ROOT.resolve()
    return path if path == root or root in path.parents else None


async def api_defect_runs(request: Request):
    """Every results directory holding a strut class table, newest first."""
    runs = []
    for table in (REPO_ROOT / "outputs").rglob("strut_classes.csv"):
        directory = table.parent
        runs.append({
            "run": str(directory.relative_to(REPO_ROOT)),
            "name": str(directory.relative_to(REPO_ROOT / "outputs")),
            "mtime": table.stat().st_mtime,
            "bytes": table.stat().st_size,
            "has_summary": (directory / "summary.json").is_file(),
        })
    runs.sort(key=lambda r: -r["mtime"])
    return JSONResponse(runs)


async def api_defects(request: Request):
    """Class counts and per-class metric summaries for one run."""
    rel = request.query_params.get("run", "")
    directory = _safe(rel)
    if directory is None or not (directory / "strut_classes.csv").is_file():
        return JSONResponse({"error": f"no strut_classes.csv under {rel}"}, status_code=404)

    with (directory / "strut_classes.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return JSONResponse({"error": "empty table"}, status_code=422)

    label_key = "label" if "label" in rows[0] else "class"
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[label_key]] = counts.get(row[label_key], 0) + 1

    # detect_lattice_defects writes a 4-column strut_classes.csv, so on an MCP-produced
    # run the metrics that decide each class live next door instead: the shape numbers in
    # struts.csv and the topology in connectivity.npz. Merge them by strut id, otherwise
    # the evidence panel is empty for exactly the runs this dashboard is meant to show.
    merged = [dict(r) for r in rows]
    by_id = {}
    for record in merged:
        try:
            by_id[int(record["strut_id"])] = record
        except (KeyError, TypeError, ValueError):
            pass

    extra_path = directory / "struts.csv"
    if extra_path.is_file():
        with extra_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    target = by_id.get(int(row["strut_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if target is None:
                    continue
                for key, value in row.items():
                    if key not in target:
                        target[key] = value

    conn_path = directory / "connectivity.npz"
    if conn_path.is_file():
        try:
            import numpy as np

            with np.load(conn_path) as arrays:
                for key in arrays.files:
                    column = arrays[key]
                    if column.ndim != 1:
                        continue
                    for index, value in enumerate(column):
                        target = by_id.get(index)
                        if target is not None and key not in target:
                            target[key] = float(value)
        except Exception:  # a malformed cache must not take the whole panel down
            pass

    numeric = []
    for key in (merged[0] if merged else {}):
        if key in (label_key, "strut_id"):
            continue
        try:
            float(merged[0][key])
        except (TypeError, ValueError):
            continue
        numeric.append(key)

    per_class: dict[str, dict] = {}
    for name in counts:
        subset = [r for r in merged if r[label_key] == name]
        stats = {}
        for key in numeric:
            values = []
            for row in subset:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    continue
                # Metrics are NaN where too few sections fitted, and `detour` is +inf for
                # a severed strut (no path at all). Neither is JSON, and neither belongs
                # in a median.
                if math.isfinite(value):
                    values.append(value)
            if values:
                values.sort()
                stats[key] = {
                    "n": len(values),
                    "median": values[len(values) // 2],
                    "min": values[0],
                    "max": values[-1],
                }
        per_class[name] = stats

    summary = {}
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            doc = json.loads(summary_path.read_text())
            summary = {k: doc.get(k) for k in ("mask", "design", "correction",
                                               "correction_applied", "counts", "geometry")}
        except (OSError, json.JSONDecodeError):
            summary = {}

    figures = [
        {"path": str(p.relative_to(REPO_ROOT)), "name": p.name, "bytes": p.stat().st_size,
         "kind": "page" if p.suffix.lower() == ".html" else "image",
         "embeddable": p.stat().st_size <= EMBED_LIMIT}
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in VIZ_SUFFIXES
    ]

    return JSONResponse({
        "run": rel,
        "total": len(rows),
        "label_key": label_key,
        "order": [c for c in ORDER if c in counts] + sorted(set(counts) - set(ORDER)),
        "counts": counts,
        "colors": CLASS_COLORS,
        "numeric_columns": numeric,
        "per_class": per_class,
        "summary": summary,
        "figures": figures,
    })


async def stream_process(cmd: list[str], cwd: Path):
    """Stream a JSONL-emitting subprocess to the browser, one SSE event per line."""
    run_id = uuid.uuid4().hex[:12]
    yield {"event": "message",
           "data": json.dumps({"type": "run.started", "run_id": run_id, "command": cmd})}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        yield {"event": "message", "data": json.dumps({"type": "run.failed", "error": str(exc)})}
        return

    RUNNING[run_id] = proc
    queue: asyncio.Queue = asyncio.Queue()
    pumps = [asyncio.create_task(_pump(proc.stdout, "stdout", queue)),
             asyncio.create_task(_pump(proc.stderr, "stderr", queue))]
    try:
        open_streams = 2
        while open_streams:
            kind, line = await queue.get()
            if line is None:
                open_streams -= 1
                continue
            if kind == "stderr":
                if line.strip():
                    yield {"event": "message",
                           "data": json.dumps({"type": "stderr", "line": line})}
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "raw", "line": line}
            yield {"event": "message", "data": json.dumps(event)}
        code = await proc.wait()
        yield {"event": "message", "data": json.dumps({"type": "run.exited", "code": code})}
    finally:
        for task in pumps:
            task.cancel()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        RUNNING.pop(run_id, None)


async def api_visualize(request: Request):
    """Render every figure for one run, into that run's own directory."""
    body = await request.json()
    directory = _safe(body.get("run", ""))
    if directory is None or not (directory / "strut_classes.csv").is_file():
        return JSONResponse({"error": "not a results directory"}, status_code=422)
    cmd = [sys.executable, str(HERE / "make_figures.py"),
           "--run", str(directory), "--json-progress"]
    return EventSourceResponse(stream_process(cmd, REPO_ROOT))


async def api_gallery(request: Request):
    """Every visualisation already in the repo, grouped by directory."""
    groups: dict[str, list] = {}
    for base in ("outputs", "images"):
        root = REPO_ROOT / base
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in VIZ_SUFFIXES or not path.is_file():
                continue
            rel = str(path.relative_to(REPO_ROOT))
            group = str(path.parent.relative_to(REPO_ROOT))
            size = path.stat().st_size
            groups.setdefault(group, []).append({
                "path": rel,
                "name": path.name,
                "kind": "page" if path.suffix.lower() == ".html" else "image",
                "bytes": size,
                "mtime": path.stat().st_mtime,
                "embeddable": size <= EMBED_LIMIT,
            })
    out = [{"group": g, "items": sorted(v, key=lambda i: i["name"])}
           for g, v in sorted(groups.items())]
    return JSONResponse({"groups": out,
                         "total": sum(len(g["items"]) for g in out),
                         "embed_limit": EMBED_LIMIT})


async def api_file(request: Request):
    """Serve a repo file for the gallery (images and standalone viewers)."""
    path = _safe(request.query_params.get("path", ""))
    if path is None or not path.is_file():
        return JSONResponse({"error": "no such file"}, status_code=404)
    if path.suffix.lower() not in VIZ_SUFFIXES:
        return JSONResponse({"error": "not a viewable file"}, status_code=415)
    return FileResponse(path)


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


class NoStore(BaseHTTPMiddleware):
    """Never let a browser cache the page or its assets.

    Starlette sends ETag and Last-Modified but no Cache-Control, so a browser is free to
    reuse app.js from its own cache without revalidating. Editing the dashboard then looks
    like the edit did nothing: the new index.html renders a new button and the cached old
    script has no listener for it. This is a localhost dev tool; correctness beats caching.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response


def build_app(default_sandbox: str) -> Starlette:
    app = Starlette(
        middleware=[Middleware(NoStore)],
        routes=[
            Route("/", index),
            Route("/api/config", api_config),
            Route("/api/chat", api_chat, methods=["POST"]),
            Route("/api/cancel", api_cancel, methods=["POST"]),
            Route("/api/runs", api_runs),
            Route("/api/runs/{name}", api_run),
            Route("/api/defect-runs", api_defect_runs),
            Route("/api/defects", api_defects),
            Route("/api/gallery", api_gallery),
            Route("/api/file", api_file),
            Route("/api/visualize", api_visualize, methods=["POST"]),
            Mount("/static", StaticFiles(directory=str(STATIC)), name="static"),
        ]
    )
    app.state.default_sandbox = default_sandbox
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--sandbox", default="read-only", choices=SANDBOX_MODES)
    args = parser.parse_args()

    import uvicorn

    print(f"repo   {REPO_ROOT}")
    print(f"codex  {codex_bin()}")
    print(f"open   http://{args.host}:{args.port}")
    uvicorn.run(build_app(args.sandbox), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
