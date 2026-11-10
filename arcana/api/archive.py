# api/archive.py -- the Archive tab's reads
#
#   GET  /api/archive                the tree by kind, and the trash
#   GET  /api/archive/show?id=       one thing, with the files it holds
#   POST /api/archive/rename {id, name}   `archive rename`, answering with the new id
#
# Every other change (move, folder add, merge, delete, restore, empty) is an
# `archive` op through /api/rpc.

from __future__ import annotations

from flask import jsonify, request

from . import bp, body, call, dump, BadRequest

try:
    from .. import catalogue as _cat
except ImportError:                                   # pragma: no cover
    import catalogue as _cat


@bp.get("/archive")
def _archive():
    res = call("archive list")
    out = dump(res)
    out["kinds"] = [k.model_dump(mode="json") for k in res.kinds]
    out["titles"] = dict(_cat.KINDS)
    return jsonify(out)


@bp.get("/archive/show")
def _archive_show():
    item_id = request.args.get("id") or ""
    if not item_id:
        raise BadRequest("Give an id.")
    res = call("archive show", {"id": item_id})
    if item_id.startswith("lora:"):
        from .. import lora_train as _train
        from .. import media as _media
        _media.register_media_root(_train.loras_root())
    try:
        from ..alchemy.inputs import resolve
    except ImportError:                               # pragma: no cover
        from alchemy.inputs import resolve
    pictures = []
    for p in res.contents:                            # a film names pictures by handle; show their files
        try:
            pictures.append(resolve(p))
        except ValueError:
            pictures.append(str(p))
    return jsonify({**dump(res), "pictures": pictures})


@bp.post("/archive/rename")
def _archive_rename():
    data = body()
    item_id, name = data.get("id") or "", (data.get("name") or "").strip()
    if not item_id or not name:
        raise BadRequest("Give the thing and its new name.")
    res = call("archive rename", {"id": item_id, "name": name})
    new_id = item_id
    if item_id.startswith("film:"):
        try:
            from ..alchemy.film import slug
        except ImportError:                           # pragma: no cover
            from alchemy.film import slug
        new_id = "film:" + slug(name)
    return jsonify({"id": new_id, "message": res.body})
