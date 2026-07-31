"""Minimal Explanation Packets (MEP) for the CT tool server.

Implements the MEP unit of Chaduvula et al., "From Features to Actions"
(arXiv:2602.06841) section 3.4: every tool call emits a packet bundling

  1. the explanation artifact  -- what the tool reported, and the files it wrote
  2. linked evidence          -- arguments, input fingerprints, timing, actor
  3. verification signals     -- left EMPTY here, filled retroactively by
                                 ``rubric.py`` once the run is complete

Slot 3 is deliberately not filled at call time. Verification in this project is
a set of separate, agent-callable check tools (``check_tools.py``), so whether a
step was verified is a property of the agent's *behaviour* rather than of the
server. A decorator that always verified would make verification a constant and
there would be nothing left to explain.

Packets append to ``outputs/mep/<run_id>/trace.jsonl`` as the run proceeds. The
ordered sequence of packets is the trajectory tau = (s_t, a_t, o_t) of section
3.1 -- the run, not the individual call, is the unit of explanation.
"""

import functools
import hashlib
import inspect
import json
import os
import time
import uuid
from typing import Annotated

from pydantic import Field

# The trace root. Sits under outputs/ alongside the project's other artifacts.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
MEP_ROOT = os.path.join(_PROJECT_ROOT, "outputs", "mep")

# One MCP server process is started per CLI session, so the process start time
# identifies the run. Overridable so a harness can group several processes into
# one logical run, or replay into a fixed directory.
_RUN_ID = os.environ.get("MEP_RUN_ID") or time.strftime("%Y%m%dT%H%M%S", time.localtime())

# Monotonic step counter within the run.
_STEP = 0

# Arguments that name a file the tool WRITES rather than reads. Fingerprinting
# these before the call is pointless (they do not exist yet) and after the call
# they are the artifact, not the evidence.
_OUTPUT_ARG_HINTS = ("output_filepath", "output_directory", "output_path")

# Fingerprint at most this many bytes from each end of a file. The CT volumes
# here reach 6.9 GB and a full digest would dominate the tool's own runtime.
_FINGERPRINT_EDGE_BYTES = 1 << 20


def run_id() -> str:
    """The identifier of the run currently being traced."""
    return _RUN_ID


def run_directory(rid: str | None = None) -> str:
    """Directory holding the trace and report for ``rid`` (default: this run)."""
    return os.path.join(MEP_ROOT, rid or _RUN_ID)


def trace_path(rid: str | None = None) -> str:
    """Path of the append-only JSONL trace for ``rid`` (default: this run)."""
    return os.path.join(run_directory(rid), "trace.jsonl")


def fingerprint_file(path: str) -> dict | None:
    """Identify a file cheaply enough to call on every tool invocation.

    Returns size, mtime, and a digest over the first and last megabyte. This is
    a fingerprint, NOT a content hash: two files agreeing on it are almost
    certainly the same file, but a change confined to the middle of a large
    volume will not be detected. That trade is deliberate -- provenance here
    asks "is this the mask step 1 wrote?", which the fingerprint answers, and
    paying a full digest on a 6.9 GB array to answer it would cost more than
    the tools being traced.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            digest.update(handle.read(_FINGERPRINT_EDGE_BYTES))
            if stat.st_size > _FINGERPRINT_EDGE_BYTES:
                handle.seek(max(0, stat.st_size - _FINGERPRINT_EDGE_BYTES))
                digest.update(handle.read(_FINGERPRINT_EDGE_BYTES))
    except OSError:
        return None

    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "edge_sha256": digest.hexdigest()[:16],
    }


def _looks_like_path(name: str, value) -> bool:
    """Whether an argument plausibly names a file the tool reads."""
    if not isinstance(value, str) or not value:
        return False
    if any(hint in name for hint in _OUTPUT_ARG_HINTS):
        return False
    return "filepath" in name or "path" in name or "directory" in name


def _jsonable(value):
    """Coerce an argument to something json.dump can hold."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _next_step() -> int:
    """The next step number, continuing an existing trace rather than restarting it.

    A run outlives any one server process -- MEP_RUN_ID deliberately lets several
    processes write to one trace, and a long analysis is often driven that way. A
    plain in-process counter restarts at 1 in each, so steps collide and the
    rendered order stops matching the order things happened. On the first write
    of a process, seed from the highest step already recorded.
    """
    global _STEP
    if _STEP == 0:
        try:
            with open(trace_path(), encoding="utf-8") as handle:
                for line in handle:
                    try:
                        _STEP = max(_STEP, json.loads(line).get("step", 0))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    _STEP += 1
    return _STEP


def _write(packet: dict) -> None:
    """Append one packet to the run's trace.

    Never raises: a failure to record an explanation must not take down the
    tool being explained.
    """
    try:
        os.makedirs(run_directory(), exist_ok=True)
        with open(trace_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(packet) + "\n")
    except OSError:
        pass


# Descriptions for the two injected parameters. These are attached with
# Annotated/Field rather than appended to the docstring: FastMCP parses
# Google-style docstrings to pull Args: entries into the schema and discards
# trailing sections it does not recognise, so a docstring append is silently
# lost. A Field description survives into the JSON schema the agent reads.
_ACTOR_DOC = (
    'Which agent is calling. "main" for the top-level agent, or the skill name '
    'for a subagent: "threshold-optimizer", "nde-report-generator", '
    '"method-comparison", "literature-review", "metadata-extractor". MCP carries no '
    "caller identity, so if this is left unset a subagent's work is "
    "indistinguishable from the parent's and the provenance audit in the run "
    "explanation is meaningless."
)

_WHY_DOC = (
    "One line on why this call is being made now, recorded verbatim as this step's "
    "rationale in the run explanation. State the reason for THIS call specifically "
    "(what it is meant to establish, or which earlier result prompted it), not a "
    "restatement of what the tool does."
)

_ACTOR_ANN = Annotated[str, Field(default="main", description=_ACTOR_DOC)]
_WHY_ANN = Annotated[str, Field(default="", description=_WHY_DOC)]


def mep_tool(kind: str = "analysis"):
    """Wrap an MCP tool so each call emits a Minimal Explanation Packet.

    Adds two keyword arguments to the wrapped tool's signature, both of which
    the calling agent is expected to supply:

      ``actor``  which agent made the call -- "main", or a skill name such as
                 "threshold-optimizer". MCP carries no caller identity: one
                 server process serves one stdio connection, and Claude Code
                 subagents share the parent's session, so ``Context.session_id``
                 cannot separate them. It has to be passed. Session fields are
                 recorded anyway, so if that ever stops being true the trace
                 already holds what is needed to exploit it.

      ``why``    the agent's stated reason for the call -- the rationale slot
                 r_t of Selim et al. (2026) section 3.1. MCP's tools/call
                 request carries only a name and arguments, so an argument is
                 the only channel by which a rationale can reach the server.
                 It is SELF-REPORTED, and section 3.2 of Chaduvula et al.
                 applies: it records what the agent claimed, not what caused
                 the call. Treat a mismatch against the trace as a finding.

    ``kind`` tags the packet so the rubric can tell the three roles apart:
    "analysis" (produces a claim), "check" (verifies one), "view" (renders).
    """

    def decorate(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, actor: str = "main", why: str = "", **kwargs):
            step = _next_step()

            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                call_args = dict(bound.arguments)
            except TypeError:
                # Let the wrapped function raise the real binding error.
                call_args = dict(kwargs)

            inputs = [
                fingerprint
                for name, value in call_args.items()
                if _looks_like_path(name, value)
                for fingerprint in [fingerprint_file(value)]
                if fingerprint is not None
            ]

            started = time.time()
            status, error, artifact = "ok", None, None
            try:
                artifact = func(*args, **kwargs)
                # These tools report failure as a returned string, not a raise.
                if isinstance(artifact, str) and artifact.startswith("Error:"):
                    status = "error"
                    error = artifact
                return artifact
            except Exception as exc:
                status = "exception"
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _write(
                    {
                        "run_id": _RUN_ID,
                        "step": step,
                        "packet_id": uuid.uuid4().hex[:12],
                        "ts": started,
                        "duration_s": round(time.time() - started, 3),
                        # --- actor ---
                        "actor": actor,
                        "session": _session_fields(),
                        # --- action ---
                        "tool": func.__name__,
                        "kind": kind,
                        "args": {k: _jsonable(v) for k, v in call_args.items()},
                        "why": why,
                        # --- artifact ---
                        "status": status,
                        "error": error,
                        "artifact": _truncate(artifact),
                        "artifact_files": _declared_outputs(call_args),
                        # --- evidence ---
                        "inputs": inputs,
                        # --- verification: filled by rubric.py, never here ---
                        "verification": [],
                    }
                )

        # Expose actor/why to MCP clients by appending them to the signature.
        # FastMCP builds its schema through pydantic, which reads __annotations__
        # rather than __signature__, and functools.wraps has just copied the
        # undecorated function's annotations over. Both have to be updated or the
        # tool fails to register with KeyError on the new parameters.
        extra = [
            inspect.Parameter(
                "actor",
                inspect.Parameter.KEYWORD_ONLY,
                default="main",
                annotation=_ACTOR_ANN,
            ),
            inspect.Parameter(
                "why",
                inspect.Parameter.KEYWORD_ONLY,
                default="",
                annotation=_WHY_ANN,
            ),
        ]
        wrapper.__signature__ = signature.replace(
            parameters=list(signature.parameters.values()) + extra
        )
        wrapper.__annotations__ = dict(getattr(func, "__annotations__", {}))
        wrapper.__annotations__["actor"] = _ACTOR_ANN
        wrapper.__annotations__["why"] = _WHY_ANN
        return wrapper

    return decorate


def _session_fields() -> dict:
    """Record MCP session identity when a Context happens to be active.

    Best-effort: these do not currently separate a subagent from its parent
    over stdio, and are captured only so the trace is ready if that changes.
    """
    try:
        from fastmcp.server.dependencies import get_context

        ctx = get_context()
        return {
            "session_id": getattr(ctx, "session_id", None),
            "client_id": getattr(ctx, "client_id", None),
            "request_id": str(getattr(ctx, "request_id", None)),
        }
    except Exception:
        return {"session_id": None, "client_id": None, "request_id": None}


def _truncate(artifact, limit: int = 4000):
    """Keep the artifact readable in a trace without storing whole reports."""
    if not isinstance(artifact, str):
        return _jsonable(artifact)
    if len(artifact) <= limit:
        return artifact
    return artifact[:limit] + f"\n... [{len(artifact) - limit} chars truncated]"


def _declared_outputs(call_args: dict) -> list:
    """Paths the tool was told to write, whether or not it succeeded."""
    return [
        os.path.abspath(value)
        for name, value in call_args.items()
        if any(hint in name for hint in _OUTPUT_ARG_HINTS)
        and isinstance(value, str)
        and value
    ]
