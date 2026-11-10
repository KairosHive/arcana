# api/learn.py -- what Alchemy learns from a set: IP labels and texture LoRAs
#
#   GET  /api/labels                     every IP label, with its nearest pictures
#   POST /api/labels/make {name, set | words+dataset+n | picture, crops}   a job
#   POST /api/labels/delete {slug}
#   GET  /api/loras                      LoRAs trained here, with checkpoints and sheets
#   POST /api/loras/train {name, set, steps, every}                       a job

from __future__ import annotations

from flask import jsonify

from . import bp, body, call, dump, BadRequest

try:
    from .. import ip_labels as _labels
    from .. import lora_train as _train
    from .. import media as _media
    from .. import gpu as _gpu
except ImportError:                                   # pragma: no cover
    import ip_labels as _labels
    import lora_train as _train
    import media as _media
    import gpu as _gpu


@bp.get("/labels")
def _labels_list():
    return jsonify({"labels": _labels.listing()})


@bp.post("/labels/make")
def _labels_make():
    data = body()
    name = (data.get("name") or "").strip()
    if not name:
        raise BadRequest("Give the label a name.")
    # Made in the request, not queued: jobs run one at a time, and a label read from an index takes
    # a second, so it should not wait behind a LoRA that trains for a quarter of an hour.
    args = {"name": name, "crops": int(data.get("crops") or 0)}
    for key in ("set", "words", "dataset", "picture"):
        if data.get(key):
            args[key] = data[key]
    if data.get("n"):
        args["n"] = int(data["n"])
    return jsonify(dump(call("label make", args)))


@bp.post("/labels/delete")
def _labels_delete():
    slug = body().get("slug")
    if not slug:
        raise BadRequest("Name the label to delete.")
    call("label delete", {"label": slug})
    return jsonify({"labels": _labels.listing()})


@bp.get("/loras")
def _loras():
    _media.register_media_root(_train.loras_root())
    try:
        gpu = _gpu.device() == "cuda"
    except Exception:
        gpu = False
    return jsonify({"runs": _train.runs(), "gpu": gpu})


@bp.post("/loras/train")
def _loras_train():
    data = body()
    name, src = (data.get("name") or "").strip(), data.get("set")
    if not name or not src:
        raise BadRequest("Give the LoRA a name and the set it trains on.")
    args = {"name": name, "set": src, "background": True}
    for key in ("steps", "every"):
        if data.get(key):
            args[key] = int(data[key])
    return jsonify(dump(call("lora train", args)))


@bp.get("/axes")
def _axes_list():
    return jsonify({"axes": _labels.axis_listing()})


@bp.post("/axes/make")
def _axes_make():
    data = body()
    name, pos, neg = (data.get("name") or "").strip(), (data.get("pos") or "").strip(), (data.get("neg") or "").strip()
    if not name or not pos or not neg:
        raise BadRequest("An axis needs a name and two ends.")
    args = {"name": name, "pos": pos, "neg": neg}
    if data.get("dataset"):
        args["dataset"] = data["dataset"]
    r = call("axis make", args)
    return jsonify({**dump(r), "axes": _labels.axis_listing()})


@bp.post("/axes/delete")
def _axes_delete():
    slug = body().get("slug")
    if not slug:
        raise BadRequest("Name the axis to delete.")
    call("axis delete", {"axis": slug})
    return jsonify({"axes": _labels.axis_listing()})
