# api/sets.py -- sets: pictures gathered by searches (ops/sets.py)
#
#   GET  /api/sets                          every set, newest first
#   GET  /api/sets/<slug>                   its pictures and the searches that gathered them
#   POST /api/sets/save {name, items, dataset, query, k, replace}
#   POST /api/sets/remove {set, items}      take pictures out (pruning)
#   POST /api/sets/export {set, folder}     copy its files into output/search/<folder>
#   POST /api/sets/kind {slug, kind}        a set onto the board shelf, or a board onto the set shelf

from __future__ import annotations

from flask import jsonify

from . import bp, body, call, dump, BadRequest

try:
    from .. import media as _media
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import media as _media
    import paths as _paths


def _rows():
    return [s.model_dump(mode="json") for s in call("set list").sets]


@bp.get("/sets")
def _sets_list():
    return jsonify({"sets": _rows()})


@bp.get("/sets/<slug>")
def _sets_show(slug):
    return jsonify(dump(call("set show", {"set": slug})))


@bp.post("/sets/save")
def _sets_save():
    data = body()
    items = [i for i in (data.get("items") or []) if isinstance(i, str) and i]
    name = (data.get("name") or "").strip()
    if not name:
        raise BadRequest("Give the set a name.")
    if not items:
        raise BadRequest("No pictures to save.")
    args = {"name": name, "items": items, "replace": bool(data.get("replace")),
            "query": data.get("query") or None, "dataset": data.get("dataset") or None,
            "k": int(data.get("k") or 0)}
    if data.get("slug"):
        args["id"] = data["slug"]
    r = call("set save", args)
    return jsonify({**dump(r), "sets": _rows()})


@bp.post("/sets/remove")
def _sets_remove():
    data = body()
    items = [i for i in (data.get("items") or []) if isinstance(i, str) and i]
    if not data.get("set") or not items:
        raise BadRequest("Name the set and the pictures to take out.")
    for start in range(0, len(items), 96):
        call("board remove", {"board": data["set"], "items": items[start:start + 96]})
    return jsonify({**dump(call("set show", {"set": data["set"]})), "sets": _rows()})


@bp.post("/sets/export")
def _sets_export():
    data = body()
    if not data.get("set"):
        raise BadRequest("Name the set to export.")
    r = call("set export", {"set": data["set"], "folder": (data.get("folder") or "").strip() or None})
    _media.register_media_root(_paths.subdir("output"))
    return jsonify(dump(r))


@bp.post("/sets/kind")
def _sets_kind():
    data = body()
    if data.get("kind") not in ("board", "set") or not data.get("slug"):
        raise BadRequest("Give a slug and a kind, board or set.")
    call("board set", {"board": data["slug"], "kind": data["kind"]})
    return jsonify({"sets": _rows()})
