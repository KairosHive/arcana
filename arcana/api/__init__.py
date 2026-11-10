# arcana.api -- the JSON the Svelte page talks to
#
# The page is the person. Everything it changes goes through the ops as actor
# "person" (OPS_PLAN.md), so a terminal and an agent see the same change and
# undo works the same way. What the ops do not cover, because it is a matter of
# drawing rather than of state, is served here as plain reads: the map's
# coordinates, a search grouped into twins, the marks matrix, a folder scan.
#
# Every route under /api needs the page's own token in X-Arcana-UI. The token
# is minted per process and reaches the page only inside the served
# index.html, so an agent started from the Agent tab, which can read
# server.json, still cannot claim to be the person.
#
#   400  bad input          {"error": "..."}
#   409  an op refused      {"error": reason, "help": "...", "op": "..."}
#   500  a crash            {"error": "Type: message"}

from __future__ import annotations

import base64
import os
import secrets
import traceback

from flask import Blueprint, jsonify, request

try:
    from .. import ops as _ops
    from .. import paths as _paths
    from .. import modality as _modality
    from .. import capabilities as _caps
except ImportError:                                   # pragma: no cover
    import ops as _ops
    import paths as _paths
    import modality as _modality
    import capabilities as _caps

UI_TOKEN = secrets.token_urlsafe(24)
HEADER = "X-Arcana-UI"
DEV_ENV = "ARCANA_UI_DEV"              # set while `vite` serves the page from another port

bp = Blueprint("arcana_api", __name__, url_prefix="/api")


class BadRequest(ValueError):
    """The page sent something this route cannot use."""


def dev_mode() -> bool:
    return bool(os.environ.get(DEV_ENV))


def tokens() -> dict:
    try:
        from .. import agent_term as _term
    except ImportError:                               # pragma: no cover
        import agent_term as _term
    try:
        from .. import __version__
    except Exception:                                 # pragma: no cover
        __version__ = ""
    return {"ui": UI_TOKEN, "agent": _term.TOKEN, "version": __version__}


@bp.before_request
def _guard():
    if request.endpoint == "arcana_api._session":
        return None
    # A browser's event stream cannot carry a header, so that one route takes the token in its address.
    if request.endpoint == "arcana_api._events" and request.args.get("ui") == UI_TOKEN:
        return None
    if request.headers.get(HEADER) != UI_TOKEN:
        return jsonify({"error": "This route is for Arcana's own page."}), 403
    return None


@bp.errorhandler(_ops.Refusal)
def _refused(e):
    return jsonify({"error": e.reason, "help": e.help, "op": e.op}), 409


@bp.errorhandler(_caps.Unsupported)
def _unsupported(e):
    """A feature asked to work on media it does not take: the same shape as an op's refusal."""
    return jsonify({"error": str(e), "feature": e.feature, "modality": e.modality}), 409


@bp.errorhandler(BadRequest)
def _bad(e):
    return jsonify({"error": str(e)}), 400


@bp.errorhandler(Exception)
def _crash(e):
    traceback.print_exc()
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.get("/session")
def _session():
    """The tokens, for a development page served by vite on another port only."""
    if not dev_mode():
        return jsonify({"error": "Not available."}), 404
    return jsonify(tokens())


# ─────────────────────────── helpers the routes share ───────────────────────────

def body() -> dict:
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise BadRequest("The request body is a JSON object.")
    return data


def call(op: str, args: dict | None = None, ctx=None):
    """One op, as the person."""
    return _ops.call(op, args or {}, actor="person", ctx=ctx)


def dump(result) -> dict:
    return _ops.render.result_json(result)


def attachments(result) -> list[dict]:
    return [{"caption": a.caption, "mime": a.mime,
             "data": base64.b64encode(a.data).decode("ascii")} for a in result.attachments]


def dataset_id(name: str, modality: str) -> str:
    return _modality.dataset_id(name, modality)


def parse_dataset(value: str) -> tuple[str, int, str]:
    """Split "<name>::<dim>::<modality>" (older values omit the modality)."""
    parts = (value or "").split("::")
    if len(parts) < 2 or not parts[0]:
        raise BadRequest("Choose a dataset first.")
    try:
        dim = int(parts[1])
    except ValueError:
        raise BadRequest("That dataset entry could not be read. Rebuild it from the Datasets tab.")
    return parts[0], dim, (parts[2] if len(parts) == 3 else "image")


def output_rel(path: str) -> str:
    try:
        return os.path.relpath(path, _paths.subdir("output")).replace(os.sep, "/")
    except ValueError:
        return path


# ─────────────────────────── the generic ops transport ───────────────────────────

@bp.post("/rpc")
def _rpc():
    data = body()
    op = data.get("op")
    if not isinstance(op, str):
        raise BadRequest('POST {"op": "board list", "args": {...}}.')
    res = call(op, data.get("args") or {})
    return jsonify({"op": op, "result": dump(res), "text": res.text(), "attachments": attachments(res)})


@bp.post("/exec")
def _exec():
    data = body()
    lines = data.get("lines")
    if not isinstance(lines, list) or not all(isinstance(l, str) for l in lines):
        raise BadRequest('POST {"lines": ["op line", ...]}.')
    ctx = _ops.context("person")
    batch = _ops.exec_lines(lines, ctx=ctx)
    out = _ops.render.batch_json(batch)
    out["text"] = _ops.render.batch_text(batch)
    out["rolled_back"] = batch.rolled_back
    return jsonify(out)


@bp.get("/schema")
def _schema():
    return jsonify(_ops.grammar.schema())


# The feature modules register their routes on `bp` when imported.
from . import system, datasets, search, collection, alchemy, archive    # noqa: E402,F401


def register(flask_app) -> None:
    if "arcana_api" not in flask_app.blueprints:
        flask_app.register_blueprint(bp)
