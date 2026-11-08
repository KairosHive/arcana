# effects.py -- long work, now or as a job, and the job ops
#
# An effect computes (a transfer, a generation, a download) and then records
# what it made (the ledger, a board). The two halves are separate so the
# computing never holds the state lock, and so the same effect can run in the
# caller's call or as a job on jobs.MANAGER's worker:
#
#     colour transfer japan:12 --reference japan:7               waits, returns the picture
#     colour transfer japan:12 --reference japan:7 --background  returns a job id at once
#
# A job's recording runs as its own transaction for the actor who started it,
# so its result is undoable like any other step, and clients learn of it the
# same way they learn of every other change: the state version moves.

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from .registry import Args, Context, Result, op, positional

try:
    from .. import jobs as _jobs
    from .. import state as _state
except ImportError:                                   # pragma: no cover
    import jobs as _jobs
    import state as _state


class JobStarted(Result):
    job: str
    label: str

    def text(self) -> str:
        return (f"Started job {self.job}: {self.label}.\n"
                f"`job wait {self.job}` returns its result, `job show {self.job}` its progress.\n")


def run_effect(ctx: Context, label: str, compute, record, background: bool = False,
               kind: str = "effect") -> Result:
    """
    compute(progress) -> payload does the work. record(tx, payload) -> Result
    writes what it made inside a transaction. With background, both run as a
    job and the caller gets the job's id.
    """
    if not background:
        payload = compute(ctx.progress)
        return record(ctx.tx, payload)

    actor = ctx.actor

    def job(handle):
        def progress(fraction=None, message=None):
            handle.update(fraction=fraction, message=message)
        # Work that can stop cleanly (indexing) polls this between items. It is
        # a flag rather than an exception from progress(), so nothing stops
        # halfway through writing a file.
        progress.cancelled = lambda: handle.cancelled
        payload = compute(progress)
        tx = _state.STORE.begin(actor, label)
        try:
            result = record(tx, payload)
            _state.STORE.commit(tx)
        except BaseException:
            _state.STORE.rollback(tx)
            raise
        return {"text": result.text(), "result": result.model_dump(mode="json")}

    jid = _jobs.MANAGER.submit(job, kind=kind, label=label)
    return JobStarted(job=jid, label=label)


# ─────────────────────────── job ops ───────────────────────────

class JobView(BaseModel):
    id: str
    kind: str
    label: str
    status: str
    fraction: float | None = None
    message: str = ""
    error: str = ""
    result_text: str | None = None

    def line(self) -> str:
        pct = f" {self.fraction:.0%}" if self.fraction is not None and self.status == "running" else ""
        out = f"{self.id}  {self.status}{pct}  {self.label}"
        if self.message and self.status == "running":
            out += f"  ({self.message})"
        if self.error:
            out += f"\n  failed: {self.error}"
        return out


def view(job_id: str) -> JobView:
    j = _jobs.MANAGER.get(job_id)
    if j is None:
        raise ValueError(f"No job {job_id} in this process. `job list` shows the jobs.")
    snap = j.as_dict()
    text = j.result.get("text") if isinstance(j.result, dict) else None
    return JobView(id=j.id, kind=j.kind, label=j.label, status=j.status,
                   fraction=snap["fraction"], message=snap["message"], error=j.error,
                   result_text=text)


class JobResult(Result):
    job: JobView

    def text(self) -> str:
        out = self.job.line()
        if self.job.result_text:
            out += "\n\n" + self.job.result_text.rstrip()
        return out + "\n"


class JobListResult(Result):
    jobs: list[JobView]

    def text(self) -> str:
        if not self.jobs:
            return "No jobs in this process.\n"
        return "\n".join(j.line() for j in self.jobs) + "\n"


class NoArgs(Args):
    pass


@op("job list", "read", NoArgs, JobListResult,
    doc="The jobs in this process, newest first. They are downloads, and transfers and "
        "generations started with --background.",
    examples=("job list",))
def _job_list(a: NoArgs, ctx: Context) -> JobListResult:
    return JobListResult(jobs=[view(d["id"]) for d in _jobs.MANAGER.recent(20)])


class JobIdArgs(Args):
    job: str = positional(0, description="The job's id, as an op that started it printed.")


@op("job show", "read", JobIdArgs, JobResult,
    doc="One job's progress, and its result once it has finished.",
    examples=("job show 3f9c1e2a7b4d",))
def _job_show(a: JobIdArgs, ctx: Context) -> JobResult:
    return JobResult(job=view(a.job))


class JobWaitArgs(Args):
    job: str = positional(0, description="The job's id.")
    timeout: float = Field(120, ge=0, le=3600, description="Seconds to wait at most.")


@op("job wait", "read", JobWaitArgs, JobResult,
    doc="""
    Wait for a job to finish, up to a timeout, then return its state either
    way. A finished job comes back with its result.
    """,
    examples=("job wait 3f9c1e2a7b4d --timeout 60",))
def _job_wait(a: JobWaitArgs, ctx: Context) -> JobResult:
    deadline = time.time() + a.timeout
    v = view(a.job)
    while v.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.2)
        v = view(a.job)
    return JobResult(job=v)


@op("job cancel", "effect", JobIdArgs, JobResult,
    doc="Ask a job to stop. It stops at its next safe point.",
    examples=("job cancel 3f9c1e2a7b4d",), journal=False)
def _job_cancel(a: JobIdArgs, ctx: Context) -> JobResult:
    v = view(a.job)
    if not _jobs.MANAGER.cancel(a.job):
        raise ValueError(f"Job {a.job} has already {v.status}.")
    return JobResult(job=view(a.job))
