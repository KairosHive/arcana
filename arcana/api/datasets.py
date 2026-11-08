# api/datasets.py -- the Datasets tab's reads
#
# Writes (create, extend, finish, marks, delete, relocate, model download) go
# through /api/rpc as ops. What is here is the reading the wizard and the list
# need: a folder scan, a name suggestion, the encoder catalogue with its time
# estimate, each dataset's health and extras, the label note.

from __future__ import annotations

import os

from flask import jsonify, request

from . import bp, body, call, dump, BadRequest, dataset_id

try:
    from .. import ui_datasets as _dm
    from .. import models as _models
    from .. import paths as _paths
    from .. import jobs as _jobs
    from .. import db as _db
except ImportError:                                   # pragma: no cover
    import ui_datasets as _dm
    import models as _models
    import paths as _paths
    import jobs as _jobs
    import db as _db

_scan_cache: dict = {}


def picker_options() -> list[dict]:
    """One entry per (dataset, layout), for the Dataset menu on every tab."""
    return _dm.picker_options()


def _clean_folder(folder) -> str:
    return os.path.expanduser(str(folder or "").strip().strip('"'))


@bp.get("/datasets")
def _list():
    return jsonify({"options": picker_options(), "rows": _dm.dataset_rows(), "models": _dm.model_rows(),
                    "agentCommand": _dm.agent_command()})


@bp.post("/datasets/scan")
def _scan():
    """Count the media in a folder, detect its type, suggest a name and a time estimate."""
    data = body()
    folder = _clean_folder(data.get("folder"))
    if not folder:
        return jsonify({"folder": "", "exists": False})
    if not os.path.isdir(folder):
        return jsonify({"folder": folder, "exists": False, "message": "Not a folder: " + folder, "tone": "bad"})
    modality = data.get("modality")
    counts = _dm.scan_both(folder)
    detected = "audio" if counts["audio"] > counts["image"] else "image"
    modality = modality if modality in ("image", "audio") else detected
    n = counts[modality]
    other = "audio" if modality == "image" else "image"
    n_other = counts[other]
    noun = "images" if modality == "image" else "audio files"
    if not n:
        message = "No " + noun + " in that folder"
        message += (f", but {n_other:,} {other} files are. Switch the media type above." if n_other else ".")
        tone = "warn"
    else:
        message = f"Found {n:,} {noun}."
        if n_other:
            message += f" ({n_other:,} {other} files here too, index those separately.)"
        tone = "ok"
    sample = counts["sample_image"] if modality == "image" else counts["sample_audio"]
    decode_ms = _models.measure_decode_ms(sample) if (modality == "image" and sample) else None
    _scan_cache[folder] = {"n": n, "decode_ms": decode_ms}
    vocab = _db.folder_vocabulary(folder) if n else []
    return jsonify({"folder": folder, "exists": True, "counts": {"image": counts["image"], "audio": counts["audio"]},
                    "detected": detected, "modality": modality, "n": n, "message": message, "tone": tone,
                    "suggestedName": _dm.suggest_name(folder), "decodeMs": decode_ms,
                    "autoK": _db.auto_k(n) if n else None, "folderWords": vocab,
                    "models": _dm.model_choices(modality, n, decode_ms)})


@bp.get("/datasets/models")
def _models_route():
    modality = request.args.get("modality") or "image"
    folder = _clean_folder(request.args.get("folder"))
    got = _scan_cache.get(folder) or {}
    return jsonify({"models": _dm.model_choices(modality, got.get("n", 0), got.get("decode_ms")),
                    "default": _models.default_for(modality).id})


@bp.post("/datasets/browse")
def _browse():
    """Open a native folder chooser on the machine running Arcana."""
    data = body()
    try:
        from .. import folderpicker
    except ImportError:                               # pragma: no cover
        import folderpicker
    if _dm.picker_busy.get("open"):
        return jsonify({"folder": None, "message": "A folder chooser is already open. Look for it behind this window.",
                        "tone": "warn"})
    _dm.picker_busy["open"] = True
    try:
        picked = folderpicker.pick_folder(_clean_folder(data.get("current")) or None)
    except folderpicker.PickerUnavailable as e:
        return jsonify({"folder": None, "message": f"No folder dialog available here ({e}). Paste the path instead.",
                        "tone": "warn"})
    except TimeoutError:
        return jsonify({"folder": None, "message": "The folder dialog timed out.", "tone": "warn"})
    except Exception as e:
        return jsonify({"folder": None, "message": f"Could not open the folder dialog: {type(e).__name__}: {e}",
                        "tone": "bad"})
    finally:
        _dm.picker_busy["open"] = False
    return jsonify({"folder": picked, "message": "", "tone": "dim"})


@bp.get("/datasets/health")
def _health():
    """Is the chosen dataset's media still where its index says? Drives the relocate panel."""
    name = request.args.get("name") or ""
    modality = request.args.get("modality") or "image"
    if not name:
        raise BadRequest("Give a dataset name.")
    try:
        from ..relocate import dataset_health
    except ImportError:                               # pragma: no cover
        from relocate import dataset_health
    try:
        h = dataset_health(name, modality, sample=200)
    except Exception as e:
        return jsonify({"ok": True, "checked": 0, "missing": 0, "total": 0, "root": "",
                        "error": f"Could not check this dataset: {e}"})
    return jsonify(h)


@bp.post("/datasets/start")
def _start():
    """The wizard's Start: validate, then `dataset create` in the background."""
    data = body()
    folder = _clean_folder(data.get("folder"))
    if not folder:
        raise BadRequest("Choose a folder first.")
    if not os.path.isdir(folder):
        raise BadRequest("Not a folder: " + folder)
    name = (data.get("name") or "").strip()
    if not name:
        raise BadRequest("Give the dataset a name.")
    if not all(c.isalnum() or c in "-_ " for c in name):
        raise BadRequest("Names may only contain letters, numbers, spaces, - and _.")
    if _jobs.MANAGER.active():
        raise BadRequest("Something is already running. Wait for it, or cancel it.")
    feats = set(data.get("features") or [])
    vocab = data.get("labels")
    res = call("dataset create", {
        "name": name, "folder": folder, "modality": data.get("modality") or "image",
        "model": data.get("model") or None, "feature": sorted(feats & {"palette", "style"}),
        "thumbnails": "thumbnails" in feats, "k": int(data.get("k") or 0),
        "labels": vocab if vocab in ("folders", "none") else "builtin",
        "dims": int(data.get("dims") or 2), "background": True})
    _dm.invalidate_dataset_cache()
    return jsonify(dump(res))


@bp.post("/datasets/action")
def _action():
    """Extend, finish (rework), marks, download: one background op, guarded like the Dash tab was."""
    data = body()
    kind = data.get("action")
    if _jobs.MANAGER.active():
        raise BadRequest("Something is already running. Wait for it, or cancel it.")
    if kind == "download":
        res = call("model download", {"model": data.get("model"), "background": True})
        return jsonify(dump(res))
    ds = dataset_id(data.get("name") or "", data.get("modality") or "image")
    if kind == "extend":
        res = call("dataset extend", {"dataset": ds, "background": True})
    elif kind == "finish":
        feats = set(data.get("features") or [])
        vocab = data.get("labels")
        args = {"dataset": ds, "background": True}
        if data.get("rework"):
            args.update({"k": int(data.get("k") or 0), "dims": int(data.get("dims") or 2),
                         "labels": vocab if vocab in ("folders", "none") else "builtin",
                         "feature": sorted(feats & {"palette", "style"}), "thumbnails": "thumbnails" in feats})
        res = call("dataset finish", args)
    elif kind == "marks":
        res = call("dataset marks", {"dataset": ds, "background": True})
    else:
        raise BadRequest("action is extend, finish, marks or download.")
    _dm.invalidate_dataset_cache()
    return jsonify(dump(res))


@bp.post("/datasets/delete")
def _delete():
    data = body()
    ds = dataset_id(data.get("name") or "", data.get("modality") or "image")
    res = call("dataset delete", {"dataset": ds, "dry_run": bool(data.get("dryRun"))})
    if not data.get("dryRun"):
        _dm.invalidate_dataset_cache()
    return jsonify(dump(res))


@bp.post("/datasets/relocate")
def _relocate():
    data = body()
    root = _clean_folder(data.get("root"))
    if not root:
        raise BadRequest("Enter the folder the files live in now.")
    if not os.path.isdir(root):
        raise BadRequest("Not a folder: " + root)
    ds = dataset_id(data.get("name") or "", data.get("modality") or "image")
    res = call("dataset relocate", {"dataset": ds, "root": root, "dry_run": not data.get("apply")})
    if data.get("apply"):
        try:
            from .. import media as _media
        except ImportError:                           # pragma: no cover
            import media as _media
        _media.clear_caches()
        _media.register_media_root(root)
        _dm.invalidate_dataset_cache()
    return jsonify(dump(res))


@bp.post("/datasets/invalidate")
def _invalidate():
    _dm.invalidate_dataset_cache()
    return jsonify({"ok": True})
