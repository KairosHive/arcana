# api/collection.py -- the active collection, and pictures dropped into it
#
#   GET  /api/collection                 slug, name, items, details
#   POST /api/collection/use {slug}      make a board the active collection
#   POST /api/collection/new {name}      a new board, made active
#   POST /api/collection/add {items}     paths or handles onto the active board
#   POST /api/collection/remove {items}
#   POST /api/collection/clear
#   POST /api/upload (multipart file)    an outside picture, kept under output/imports
#                                        and added to the collection

from __future__ import annotations

import hashlib
import os

from flask import jsonify, request

from . import bp, body, call, dump, BadRequest

try:
    from .. import ui_collection as _collection
    from .. import paths as _paths
    from .. import media as _media
except ImportError:                                   # pragma: no cover
    import ui_collection as _collection
    import paths as _paths
    import media as _media


@bp.get("/collection")
def _get():
    return jsonify(_collection.snapshot())


@bp.post("/collection/use")
def _use():
    slug = body().get("slug")
    if slug and slug != _collection.active():
        _collection.use(slug)
    return jsonify(_collection.snapshot())


@bp.post("/collection/new")
def _new():
    name = (body().get("name") or "").strip()
    if not name:
        raise BadRequest("Give the collection a name.")
    slug = call("board create", {"name": name}).board.slug
    _collection.use(slug)
    return jsonify(_collection.snapshot())


@bp.post("/collection/add")
def _coll_add():
    items = [i for i in (body().get("items") or []) if isinstance(i, str) and i]
    if not items:
        raise BadRequest("Nothing to add.")
    slug = _collection.add(items)
    return jsonify({**_collection.snapshot(), "added": len(items), "slug": slug})


@bp.post("/collection/remove")
def _remove():
    items = [i for i in (body().get("items") or []) if isinstance(i, str) and i]
    _collection.remove(items)
    return jsonify(_collection.snapshot())


@bp.post("/collection/clear")
def _clear():
    _collection.clear()
    return jsonify(_collection.snapshot())


@bp.post("/upload")
def _upload():
    """
    Write a browser upload to disk and add it to the collection.

    Content-addressed, so dropping the same picture twice does not litter the
    directory with duplicates. The imports folder is under output/, which the
    media routes serve.
    """
    f = request.files.get("file")
    if f is None:
        raise BadRequest("Send the picture as the form field 'file'.")
    data = f.read()
    if not data:
        raise BadRequest("That file is empty.")
    ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"):
        raise BadRequest("Only image files can be added here.")
    out_dir = _paths.output_dir("imports")
    _media.register_media_root(_paths.subdir("output"))
    path = os.path.join(out_dir, f"ext_{hashlib.md5(data).hexdigest()[:12]}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    if request.form.get("add", "1") != "0":
        _collection.add([path])
    return jsonify({"path": path, **_collection.snapshot()})
