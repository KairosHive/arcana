# api/alchemy.py -- the Alchemy tab and its Video subtab
#
#   GET  /api/alchemy/catalog              methods, knobs, presets, finish spec, recipes
#   GET  /api/alchemy/recipe/<name>        a recipe as control values
#   POST /api/alchemy/preview {controls, view, size}   -> after and before as data URLs
#   POST /api/alchemy/keep {controls, size}            -> the kept picture, added to the collection
#   POST /api/alchemy/recipe {controls, name}          -> saves a recipe
#   POST /api/film/meta {spec}             the timeline's lanes, length and labels
#   POST /api/film/add {spec, ref, t}      a picture at the playhead
#   POST /api/film/space {spec, shape, by}  by time or by how much the picture changes
#   POST /api/film/lane {spec, path, t}
#   POST /api/film/look {spec, t, controls}  the Image subtab's step as a look
#   POST /api/film/frame {spec, at, size}  one frame as a data URL
#   POST /api/film/render {spec, size, target}  a background job

from __future__ import annotations

import base64
import hashlib

from flask import jsonify

from . import bp, body, call, dump, BadRequest, output_rel

try:
    from .. import ui_alchemy as _ia
    from .. import ui_film as _film
    from .. import ui_collection as _collection
    from .. import ops as _ops
except ImportError:                                   # pragma: no cover
    import ui_alchemy as _ia
    import ui_film as _film
    import ui_collection as _collection
    import ops as _ops


def _src(att) -> str:
    return "data:" + (att.mime or "image/jpeg") + ";base64," + base64.b64encode(att.data).decode()


@bp.get("/alchemy/catalog")
def _catalog():
    return jsonify(_ia.catalog())


@bp.get("/alchemy/recipes")
def _recipes():
    return jsonify({"recipes": _ia.recipes()})


@bp.get("/alchemy/recipe/<path:name>")
def _recipe(name):
    try:
        return jsonify(_ia.recipe_controls(name))
    except (ValueError, KeyError) as e:
        raise BadRequest(str(e))


def _controls(data: dict) -> dict:
    c = data.get("controls")
    if not isinstance(c, dict) or not c.get("method"):
        raise BadRequest("Send the step's controls.")
    return c


@bp.post("/alchemy/preview")
def _preview():
    data = body()
    controls = _controls(data)
    roles = controls.get("roles") or {}
    if not roles.get("T") and not any(roles.get(r) for r in ("1", "2", "3")):
        raise BadRequest("Give a picture [T] and a material [1] from the collection on the left.")
    args = _ia.settings(controls)
    mask_view = data.get("view") == "mask"
    args.update(view="mask" if mask_view else "compare", size=int(data.get("size") or 640))
    res = call("alchemy preview", args)
    before = None
    if mask_view:
        b = call("alchemy preview", {**args, "view": "before"})
        before = b.attachments[0] if b.attachments else None
    after = res.attachments[0]
    before = before or (res.attachments[1] if len(res.attachments) > 1 else after)
    key = hashlib.blake2b(before.data, digest_size=8).hexdigest()
    bits = [res.method, f"{res.size[0]} x {res.size[1]}", f"{res.ms} ms"]
    if res.mask_share is not None:
        bits.append(f"mask {res.mask_share:.0%}")
    return jsonify({"after": _src(after), "before": None if key == data.get("beforeKey") else _src(before),
                    "beforeKey": key, "info": "  ·  ".join(bits), "view": "mask" if mask_view else "compare"})


@bp.post("/alchemy/keep")
def _keep():
    data = body()
    args = _ia.settings(_controls(data))
    args.update(size=int(data.get("size") or 0))
    res = call("alchemy make", args)
    where = output_rel(res.path)
    try:
        slug = _collection.add([res.id])
    except _ops.Refusal as e:
        return jsonify({"id": res.id, "path": res.path, "size": res.size,
                        "message": f"Kept {res.id} as {where} in output, but could not add it to the collection: "
                                   f"{e.reason}"})
    return jsonify({"id": res.id, "path": res.path, "size": res.size, "collection": _collection.snapshot(),
                    "message": f"Kept {res.size[0]} x {res.size[1]} as {res.id}, in output/{where}, "
                               f"and added it to {slug}."})


@bp.post("/alchemy/recipe")
def _save_recipe():
    data = body()
    name = (data.get("name") or "").strip()
    if not name:
        raise BadRequest("Give the recipe a name.")
    args = _ia.settings(_controls(data), with_pictures=False)
    args = {k: v for k, v in args.items() if v not in ({}, [], None)}
    args["name"] = name
    if data.get("doc"):
        args["doc"] = str(data["doc"])[:300]
    res = call("alchemy recipe save", args)
    return jsonify({"recipe": res.recipe.model_dump(mode="json"), "recipes": _ia.recipes(),
                    "message": f"Saved {res.recipe.name}."})


# ─────────────────────────── film ───────────────────────────

@bp.post("/film/meta")
def _meta():
    return jsonify(_film.meta(body().get("spec") or {}))


@bp.post("/film/add")
def _film_add():
    data = body()
    if not data.get("ref"):
        raise BadRequest("No picture.")
    spec, t = _film.add_picture(data.get("spec") or {}, data["ref"], float(data.get("t") or 0.0))
    return jsonify({"spec": spec, "t": t})


@bp.post("/film/space")
def _space():
    data = body()
    try:
        return jsonify({"spec": _film.spaced(data.get("spec") or {}, data.get("shape") or "even",
                                             data.get("by") or "time")})
    except (ValueError, KeyError, TypeError) as e:
        raise BadRequest(str(e))


@bp.post("/film/lane")
def _lane():
    data = body()
    if not data.get("path"):
        raise BadRequest("Choose a setting to key.")
    return jsonify({"spec": _film.add_lane(data.get("spec") or {}, data["path"], float(data.get("t") or 0.0))})


@bp.post("/film/look")
def _look():
    data = body()
    try:
        look = _film.look_from_controls(_controls(data))
    except (ValueError, _ops.Refusal) as e:
        raise BadRequest(f"Could not take this look: {e}")
    t = round(float(data.get("t") or 0.0), 3)
    return jsonify({"spec": _film.with_look(data.get("spec") or {}, t, look),
                    "message": f"Added this look to the film at {t:.2f} s. The Video subtab shows it on the look lane."})


@bp.post("/film/look-summary")
def _look_summary():
    data = body()
    return jsonify(_film.look_summary(_film.full(data.get("spec") or {}), float(data.get("t") or 0.0)))


@bp.post("/film/look-replace")
def _look_replace():
    """The look at the playhead now holds the Image subtab's step."""
    data = body()
    try:
        look = _film.look_from_controls(_controls(data))
        spec = _film.replace_look(data.get("spec") or {}, float(data.get("t") or 0.0), look)
    except (ValueError, _ops.Refusal) as e:
        raise BadRequest(str(e))
    return jsonify({"spec": spec, "message": "The look here now holds the Image subtab's step."})


@bp.post("/film/look-remove")
def _look_remove():
    data = body()
    try:
        spec, t = _film.remove_look(data.get("spec") or {}, float(data.get("t") or 0.0))
    except ValueError as e:
        raise BadRequest(str(e))
    return jsonify({"spec": spec, "message": f"Removed the look at {t:.2f} s."})


@bp.post("/alchemy/load")
def _load_step():
    """A whole step's settings (a film's look) as control values, for Edit in Image."""
    data = body()
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise BadRequest("Send the look's settings.")
    try:
        return jsonify(_ia.settings_controls(settings, whole=True))
    except (ValueError, KeyError) as e:
        raise BadRequest(str(e))


@bp.post("/alchemy/from-op")
def _from_op():
    """The step an `alchemy preview` or `alchemy make` ran with, as the stage's control values and its
    pictures' files: how an agent's step opens on the person's stage."""
    from ..ops import archive
    from ..ops.alchemy import AlchemyPreviewArgs, settings_of

    data = body()
    args = dict(data.get("args") or {})
    keep = set(AlchemyPreviewArgs.model_fields) - {"view", "size"}
    try:
        parsed = AlchemyPreviewArgs(**{k: v for k, v in args.items() if k in keep})
        out = _ia.settings_controls(settings_of(parsed), whole=True)
        out["picture"] = archive.resolve_item(parsed.picture, "person").path if parsed.picture else None
        out["materials"] = [archive.resolve_item(m, "person").path for m in parsed.material]
        if parsed.weight:
            out["weights"] = list(parsed.weight)
    except (ValueError, KeyError, TypeError) as e:
        raise BadRequest(f"That step cannot open here: {e}")
    return jsonify(out)


@bp.post("/film/frame")
def _frame():
    data = body()
    f, why = _film.normal(_film.full(data.get("spec") or {}))
    if f is None:
        return jsonify({"src": None, "info": "", "why": why})
    res = call("alchemy film preview", {"spec": f, "at": float(data.get("at") or 0.0),
                                        "size": int(data.get("size") or 512)})
    return jsonify({"src": _src(res.attachments[0]) if res.attachments else None,
                    "info": f"{res.size[0]} x {res.size[1]} · {res.ms} ms"})


@bp.post("/film/render")
def _render():
    data = body()
    args = {"spec": data.get("spec") or {}, "size": int(data.get("size") or 768), "background": True}
    if data.get("target"):
        args["target"] = int(data["target"])
    if data.get("fps"):
        args["fps"] = int(data["fps"])
    return jsonify(dump(call("alchemy film make", args)))


@bp.post("/film/shape")
def _film_shape():
    data = body()
    try:
        spec = _film.shape_lane(data.get("spec") or {}, data["path"], data.get("shape") or "rise",
                                float(data["lo"]), float(data["hi"]), float(data["start"]), float(data["end"]),
                                int(data.get("count") or 3), data.get("ease") or "smooth")
    except (ValueError, KeyError, TypeError) as e:
        raise BadRequest(str(e))
    return jsonify({"spec": spec})


@bp.post("/film/series")
def _film_series():
    """Pictures placed one after another from a time, such as a picture's derivatives after it."""
    data = body()
    try:
        spec, t = _film.place_series(data.get("spec") or {}, list(data.get("refs") or []),
                                     float(data.get("t") or 0.0), float(data.get("every") or _film.GAP),
                                     bool(data.get("back")))
    except (ValueError, TypeError) as e:
        raise BadRequest(str(e))
    return jsonify({"spec": spec, "t": t})


@bp.post("/film/derive")
def _film_derive():
    """Derivatives of one picture from the Image subtab's step, as a background job."""
    data = body()
    args = _ia.settings(_controls(data))
    if not data.get("picture"):
        raise BadRequest("Choose the picture to derive from: click it on the picture lane.")
    args["picture"] = data["picture"]
    vary = data.get("vary") or "seed"
    if vary == "recipe":
        args = {"picture": data["picture"], "material": args.get("material") or [],
                "weight": args.get("weight") or []}
    args.update(vary=vary, count=int(data.get("count") or 4), size=int(data.get("size") or 1024),
                background=True)
    for k in ("knob", "start", "end"):
        if data.get(k) not in (None, ""):
            args[k] = data[k]
    if data.get("each"):
        args["each"] = list(data["each"])
    args = {k: v for k, v in args.items() if v not in ([], {}, None)}
    return jsonify(dump(call("alchemy derive", args)))


@bp.get("/film/constants")
def _constants():
    return jsonify({"empty": _film.EMPTY_FILM, "eases": list(_film.EASES), "spaces": _film.SPACES,
                    "shapes": [{"key": k, "label": l, "doc": d} for k, l, d in _film.SHAPES], "gap": _film.GAP})
