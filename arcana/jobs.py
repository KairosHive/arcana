# jobs.py — background work with progress a UI can poll
#
# Indexing a photo library takes minutes to hours, and downloading an encoder
# takes gigabytes. Dash callbacks are request/response, so neither can run
# inside one: the browser would simply hang with no indication of what is
# happening or how long is left.
#
# So long work runs on a worker thread and reports into a registry the UI polls
# with dcc.Interval. Deliberately small: no Celery, no Redis, no diskcache. A
# desktop app has exactly one user.
#
# Jobs run ONE AT A TIME. Indexing is CPU- or GPU-bound and two at once would
# only make both slower while doubling the memory.

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Any

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = (DONE, FAILED, CANCELLED)


class Cancelled(Exception):
    """Raised inside a job when the user asks it to stop."""


@dataclass
class Progress:
    """What a running job reports. Every field is optional to set."""
    fraction: float | None = None     # 0..1, None when the total is unknown
    message: str = ""
    detail: str = ""                  # a second line: current file, MB/s, ETA
    done: int = 0
    total: int = 0


@dataclass
class Job:
    id: str
    kind: str                         # "index" | "download" | ...
    label: str
    status: str = QUEUED
    progress: Progress = field(default_factory=Progress)
    result: Any = None
    error: str = ""
    traceback: str = ""
    log: deque = field(default_factory=lambda: deque(maxlen=200))
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def as_dict(self) -> dict:
        """JSON-safe view for a dcc.Store."""
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "error": self.error,
            "fraction": self.progress.fraction,
            "message": self.progress.message, "detail": self.progress.detail,
            "done": self.progress.done, "total": self.progress.total,
            "log": list(self.log)[-12:],
        }


class _Handle:
    """
    What a job function is given.

    It reports progress through this and checks `raise_if_cancelled()` wherever
    it can stop cleanly -- cancellation is cooperative, because a thread cannot
    be safely killed mid-write.
    """

    def __init__(self, job: Job, lock: threading.Lock):
        self._job = job
        self._lock = lock

    def update(self, fraction: float | None = None, message: str | None = None,
               detail: str | None = None, done: int | None = None,
               total: int | None = None) -> None:
        with self._lock:
            p = self._job.progress
            if fraction is not None:
                p.fraction = max(0.0, min(1.0, float(fraction)))
            if message is not None and message != p.message:
                p.message = message
                self._job.log.append(message)
            if detail is not None:
                p.detail = detail
            if done is not None:
                p.done = int(done)
            if total is not None:
                p.total = int(total)
            if done is not None and total:
                p.fraction = max(0.0, min(1.0, done / total))

    def log(self, line: str) -> None:
        with self._lock:
            self._job.log.append(line)

    @property
    def cancelled(self) -> bool:
        return self._job._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled()


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._worker: threading.Thread | None = None
        self._queue: deque[tuple[Job, Callable]] = deque()

    # -- submission ----------------------------------------------------------
    def submit(self, fn: Callable[[_Handle], Any], *, kind: str, label: str) -> str:
        """
        Queue `fn`, which will be called with a handle for progress reporting.

        Returns the job id immediately; the caller polls with get().
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._queue.append((job, fn))
            needs_worker = self._worker is None or not self._worker.is_alive()
            if needs_worker:
                self._worker = threading.Thread(target=self._run_loop,
                                                name="arcana-jobs", daemon=True)
        if needs_worker:
            self._worker.start()
        return job.id

    # -- inspection ----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job else None

    def active(self) -> dict | None:
        """The job a UI should currently be showing, if any."""
        with self._lock:
            for jid in reversed(self._order):
                j = self._jobs[jid]
                if not j.finished:
                    return j.as_dict()
        return None

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [self._jobs[j].as_dict() for j in reversed(self._order[-limit:])]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.finished:
                return False
            job._cancel.set()
            if job.status == QUEUED:
                job.status = CANCELLED
                job.progress.message = "Cancelled before it started"
            return True

    def clear_finished(self) -> None:
        with self._lock:
            keep = [j for j in self._order if not self._jobs[j].finished]
            for j in self._order:
                if j not in keep:
                    self._jobs.pop(j, None)
            self._order = keep

    # -- the worker ----------------------------------------------------------
    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                job, fn = self._queue.popleft()
                if job.status == CANCELLED:
                    continue
                job.status = RUNNING
            handle = _Handle(job, self._lock)
            try:
                result = fn(handle)
                with self._lock:
                    if job._cancel.is_set():
                        job.status = CANCELLED
                        job.progress.message = "Cancelled"
                    else:
                        job.status = DONE
                        job.result = result
                        job.progress.fraction = 1.0
                        if not job.progress.message:
                            job.progress.message = "Finished"
            except Cancelled:
                with self._lock:
                    job.status = CANCELLED
                    job.progress.message = "Cancelled"
            except BaseException as e:           # noqa: BLE001 -- report anything
                with self._lock:
                    job.status = FAILED
                    job.error = f"{type(e).__name__}: {e}"
                    job.traceback = traceback.format_exc()
                    job.progress.message = "Failed"
                    job.log.append(job.error)


# One manager per process. A desktop app has one user and one queue.
MANAGER = JobManager()
