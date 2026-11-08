# api/system.py -- what the page asks for on load, and keeps asking
#
#   GET  /api/bootstrap        everything the shell needs once: datasets, boards,
#                              settings, hardware, tokens, the running job
#   GET  /api/state?since=     the state signature, the active job, and the
#                              collection, only when the signature moved
#   GET  /api/jobs/<id>        a job snapshot, with the detail line and log the
#                              job manager keeps and JobView leaves out
#   POST /api/jobs/<id>/cancel
#   GET  /api/settings, POST /api/settings {pixels}

from __future__ import annotations

import os

from flask import jsonify, request

from . import bp, body, call, dump, BadRequest, tokens

try:
    from .. import jobs as _jobs
    from .. import state as _state
    from .. import gpu as _gpu
    from .. import ui_collection as _collection
    from .. import ops as _ops
    from . import datasets as _datasets
except ImportError:                                   # pragma: no cover
    import jobs as _jobs
    import state as _state
    import gpu as _gpu
    import ui_collection as _collection
    import ops as _ops
    from api import datasets as _datasets


def boards() -> list[dict]:
    return [b.model_dump(mode="json") for b in call("board list").boards]


def settings() -> dict:
    s = _ops.archive.settings()
    return {"pixels": s["pixels"], "collection": s["collection"]}


def hardware() -> dict:
    try:
        line = _gpu.describe()
    except Exception:
        line = ""
    note = ""
    try:
        if not _gpu.available():
            idle = _gpu.unused_gpu()
            if idle:
                vram = f" ({idle['vram_mb'] // 1024} GB)" if idle.get("vram_mb") else ""
                note = (f"This build cannot use the {idle['name']}{vram} in this machine: it ships "
                        "CPU-only PyTorch to keep the download near 200 MB rather than 4 GB. Installing "
                        "Arcana from source into a CUDA environment uses the card, worth it for the largest "
                        "encoder and for Inject Poetry, which is about 29x faster on a GPU.")
    except Exception:
        pass
    return {"line": line, "note": note, "gpu": bool(_gpu.available())}


@bp.get("/bootstrap")
def _bootstrap():
    t = tokens()
    return jsonify({
        "version": t["version"],
        "agentToken": t["agent"],
        "datasets": _datasets.picker_options(),
        "boards": boards(),
        "settings": settings(),
        "hardware": hardware(),
        "collection": _collection.snapshot(),
        "signature": _state.STORE.signature(),
        "job": _jobs.MANAGER.active(),
        "dataDir": os.environ.get("ARCANA_DATA_DIR", ""),
    })


@bp.get("/state")
def _state_route():
    since = request.args.get("since") or ""
    sig = _state.STORE.signature()
    out = {"signature": sig, "job": _jobs.MANAGER.active()}
    if sig != since:
        out["boards"] = boards()
        out["collection"] = _collection.snapshot()
        out["settings"] = settings()
    return jsonify(out)


@bp.get("/jobs/<job_id>")
def _job(job_id):
    snap = _jobs.MANAGER.snapshot(job_id)
    if snap is None:
        return jsonify({"error": "No such job."}), 404
    job = _jobs.MANAGER.get(job_id)
    if job is not None and job.finished and isinstance(job.result, dict):
        snap["result"] = job.result.get("result") if "result" in job.result else job.result
        snap["result_text"] = job.result.get("text") if isinstance(job.result, dict) else None
    return jsonify(snap)


@bp.post("/jobs/<job_id>/cancel")
def _cancel(job_id):
    return jsonify({"ok": _jobs.MANAGER.cancel(job_id)})


@bp.get("/jobs")
def _jobs_list():
    return jsonify({"active": _jobs.MANAGER.active(), "recent": _jobs.MANAGER.recent(10)})


@bp.get("/settings")
def _settings():
    return jsonify(settings())


@bp.post("/settings")
def _set_settings():
    data = body()
    args = {}
    if "pixels" in data:
        args["pixels"] = "on" if data["pixels"] else "off"
    if "collection" in data:
        args["collection"] = data["collection"] or ""
    if not args:
        raise BadRequest("Give pixels or collection.")
    res = call("settings set", args)
    return jsonify(dump(res))


@bp.get("/history")
def _history():
    return jsonify(dump(call("history")))


@bp.post("/undo")
def _undo():
    return jsonify(dump(call("undo")))


@bp.post("/redo")
def _redo():
    return jsonify(dump(call("redo")))
