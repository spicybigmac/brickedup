from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


class JobStore:
    """Small process-local store suited to a single-instance hackathon demo."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._jobs[job["id"]] = job
            return deepcopy(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id].update(changes)
            return deepcopy(self._jobs[job_id])


store = JobStore()
