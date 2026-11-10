# api/generate.py -- the Generate tab: new material from places in your archive (ops/material.py)
#
#   POST /api/generate/draw {sources, words, engine, n, seed, size, tileable, push, spread}   a job
#   POST /api/generate/walk {a, b, words, frames, seed, size, tileable, push}                 a job
#   POST /api/generate/keep {draft, collect}
#
# A source is {kind: set | label | picture, id, weight}.

from __future__ import annotations

from flask import jsonify

from . import bp, body, call, dump, BadRequest

try:
    from .. import material_draw as _md
    from .. import media as _media
except ImportError:                                   # pragma: no cover
    import material_draw as _md
    import media as _media


def _sources(items) -> list[str]:
    out = []
    for s in items or []:
        if not isinstance(s, dict) or not s.get("id"):
            continue
        w = float(s.get("weight", 1.0))
        if w > 0:
            out.append(f"{s.get('kind', 'label')}:{s['id']}={w:g}")
    return out


@bp.post("/generate/draw")
def _generate_draw():
    data = body()
    args = {"source": _sources(data.get("sources")), "words": (data.get("words") or "").strip(),
            "engine": "pictures" if data.get("engine") == "pictures" else "place", "background": True}
    for key, cast in (("n", int), ("seed", int), ("size", int), ("push", float), ("spread", float)):
        if data.get(key) is not None:
            args[key] = cast(data[key])
    args["tileable"] = bool(data.get("tileable"))
    if data.get("axis") and data["axis"] != "none":
        args["axis"], args["amount"] = data["axis"], float(data.get("amount") or 0)
    if not args["source"] and not args["words"]:
        raise BadRequest("Choose a source or write the material in words.")
    _media.register_media_root(_md.drafts_dir())
    return jsonify(dump(call("material draw", args)))


@bp.post("/generate/walk")
def _generate_walk():
    data = body()
    a, b = _sources(data.get("a")), _sources(data.get("b"))
    if not a or not b:
        raise BadRequest("A walk needs a place at each end.")
    args = {"a": a, "b": b, "words": (data.get("words") or "").strip(), "background": True,
            "tileable": bool(data.get("tileable"))}
    for key, cast in (("frames", int), ("seed", int), ("size", int), ("push", float)):
        if data.get(key) is not None:
            args[key] = cast(data[key])
    _media.register_media_root(_md.drafts_dir())
    return jsonify(dump(call("material walk", args)))


@bp.post("/generate/keep")
def _generate_keep():
    data = body()
    if not data.get("draft"):
        raise BadRequest("Name the drawn material to keep.")
    return jsonify(dump(call("material keep", {"draft": data["draft"], "collect": bool(data.get("collect"))})))
