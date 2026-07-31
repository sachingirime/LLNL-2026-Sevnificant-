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
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STATIC = HERE / "static"

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
            yield {"event": "message", "data": json.dumps(event)}

        code = await proc.wait()
        yield {"event": "message", "data": json.dumps({"type": "run.exited", "code": code})}
    finally:
        # Covers the normal end, a cancel, and the browser tab closing mid-turn.
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


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


def build_app(default_sandbox: str) -> Starlette:
    app = Starlette(
        routes=[
            Route("/", index),
            Route("/api/config", api_config),
            Route("/api/chat", api_chat, methods=["POST"]),
            Route("/api/cancel", api_cancel, methods=["POST"]),
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
