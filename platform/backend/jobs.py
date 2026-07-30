"""In-process background job runner.

The MCP tools in src/mcp_server.py can take anywhere from a fraction of a
second (visualize_slice) to ~10 minutes (detect_lattice_defects on an
18,000-strut lattice). The frontend needs to stay responsive either way, so
every tool call is submitted as a job, run in a background thread, and
polled by the UI instead of blocking an HTTP request for ten minutes.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


@dataclass
class Job:
    id: str
    tool: str
    args: Dict[str, Any]
    status: str = "queued"  # queued -> running -> done | error
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """Tracks jobs in memory. One process, one dict, one lock -- this app is
    meant to run locally for a single user, so anything fancier (Redis,
    Celery, ...) would be pure overhead."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, tool: str, args: Dict[str, Any], fn: Callable[..., str]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], tool=tool, args=args)
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            try:
                job.result = fn(**args)
                job.status = "error" if isinstance(job.result, str) and job.result.startswith("Error:") else "done"
            except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
                job.error = f"{exc}\n{traceback.format_exc(limit=6)}"
                job.status = "error"
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=_run, daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]


manager = JobManager()
