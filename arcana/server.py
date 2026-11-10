# server.py -- the ops over HTTP, inside the app's own process
#
# While the app is open it owns the encoders, the job queue and the GPU. The
# CLI and the MCP server should reach those rather than load a second copy of
# everything, so the app serves the ops on its own Flask server:
#
#   POST /ops/exec            text lines, run as one batch (the CLI and MCP use this)
#   POST /ops/rpc             one op as JSON
#   GET  /ops/list            the op list
#   GET  /ops/schema.json     every op's JSON Schema
#   GET  /ops/state           the state version, which clients poll
#
# Everything binds to 127.0.0.1. On start the server writes server.json to the
# data dir with its port, pid and a random token, and removes it on exit.
# Clients send the token in X-Arcana-Token. Anything able to read the data dir
# can already edit the boards in it, so the token adds no new trust. It shuts
# out other software on the machine that finds the port.
#
# `arcana --headless` serves the same routes without the page, which is
# how an agent gets a long-lived server with the models kept warm.

from __future__ import annotations

import atexit
import base64
import json
import os
import secrets
import time
import urllib.error
import urllib.request

try:
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths

DISCOVERY = "server.json"
TOKEN_HEADER = "X-Arcana-Token"
REMOTE_ACTORS = ("cli", "agent")      # the person acts through the app, in process

_TOKEN = secrets.token_urlsafe(24)


def discovery_path() -> str:
    return os.path.join(_paths.data_dir(), DISCOVERY)


# ─────────────────────────── the routes ───────────────────────────

def _attachments(result) -> list[dict]:
    return [{"caption": a.caption, "mime": a.mime,
             "data": base64.b64encode(a.data).decode("ascii")} for a in result.attachments]


def run_lines(lines: list[str], actor: str, sheets: bool | None = None) -> dict:
    """A batch as the JSON the routes return. Used by /ops/exec and by tests."""
    from . import ops
    ctx = ops.context(actor, sheets=sheets)
    batch = ops.exec_lines(lines, ctx=ctx)
    out = ops.render.batch_json(batch)
    out["text"] = ops.render.batch_text(batch)
    out["rolled_back"] = batch.rolled_back
    for step, dumped in zip(batch.steps, out["steps"]):
        visible = ops.render.visible_attachments(step.result, ctx) if step.result is not None else []
        dumped["attachments"] = [{"caption": a.caption, "mime": a.mime,
                                  "data": base64.b64encode(a.data).decode("ascii")} for a in visible]
        dumped["text"] = step.result.text() if step.result is not None else None
    return out


def blueprint():
    from flask import Blueprint, jsonify, request

    bp = Blueprint("arcana_ops", __name__)

    @bp.before_request
    def _check_token():
        if request.headers.get(TOKEN_HEADER) != _TOKEN:
            return jsonify({"error": "This route needs the token from server.json "
                                     "in the X-Arcana-Token header."}), 403
        return None

    def _actor(body):
        actor = (body or {}).get("actor", "cli")
        if actor not in REMOTE_ACTORS:
            return None
        return actor

    @bp.post("/ops/exec")
    def _exec():
        body = request.get_json(silent=True) or {}
        actor = _actor(body)
        lines = body.get("lines")
        if actor is None or not isinstance(lines, list) or not all(isinstance(l, str) for l in lines):
            return jsonify({"error": "POST {\"lines\": [\"op line\", ...], \"actor\": \"cli\" "
                                     "or \"agent\"}."}), 400
        return jsonify(run_lines(lines, actor, body.get("sheets")))

    @bp.post("/ops/rpc")
    def _rpc():
        from . import ops
        body = request.get_json(silent=True) or {}
        actor = _actor(body)
        if actor is None or not isinstance(body.get("op"), str):
            return jsonify({"error": "POST {\"op\": \"search text\", \"args\": {...}, "
                                     "\"actor\": \"cli\" or \"agent\"}."}), 400
        ctx = ops.context(actor, sheets=body.get("sheets"))
        try:
            result = ops.call(body["op"], body.get("args") or {}, ctx=ctx)
        except ops.Refusal as e:
            return jsonify({"refused": e.as_dict()})
        return jsonify({"op": body["op"], "result": ops.render.result_json(result),
                        "text": result.text(),
                        "attachments": _attachments(result) if ops.render.visible_attachments(result, ctx) else []})

    @bp.get("/ops/list")
    def _list():
        from . import ops
        return jsonify([{"name": o.name, "kind": o.kind, "summary": o.summary,
                         "needs": sorted(o.needs), "actors": sorted(o.actors),
                         "modalities": o.tags}
                        for o in ops.all_ops()])

    @bp.get("/ops/schema.json")
    def _schema():
        from . import ops
        return jsonify(ops.grammar.schema())

    @bp.get("/ops/state")
    def _state():
        from . import state
        return jsonify({"version": state.STORE.version, "signature": state.STORE.signature(),
                        "pid": os.getpid()})

    return bp


def register(flask_app) -> None:
    """Serve the ops on an existing Flask app (the app's own server). Idempotent."""
    if "arcana_ops" not in flask_app.blueprints:
        flask_app.register_blueprint(blueprint())


# ─────────────────────────── discovery ───────────────────────────

def announce(port: int, *, headless: bool = False) -> str:
    """Write server.json for this process, and remove it again at exit."""
    info = {"port": int(port), "pid": os.getpid(), "token": _TOKEN, "started": time.time(),
            "headless": headless, "vocabulary": vocabulary()}
    path = discovery_path()
    _paths.ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, path)

    def _withdraw():
        try:
            with open(path, encoding="utf-8") as fh:
                if json.load(fh).get("pid") == os.getpid():
                    os.remove(path)
        except (OSError, ValueError):
            pass
    atexit.register(_withdraw)
    return path


def find(timeout: float = 0.5) -> dict | None:
    """
    The running server's discovery record, if one answers.

    The check is a request to the server, never a signal to its pid: on
    Windows os.kill(pid, 0) does not test a process, it terminates it. A
    record left behind by a crash is ignored, and the caller runs ops in its
    own process instead.
    """
    if os.environ.get("ARCANA_NO_SERVER"):
        return None
    try:
        with open(discovery_path(), encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or info.get("pid") == os.getpid():
        return None
    # An app started before an upgrade speaks an older vocabulary. Forwarding
    # to it would refuse ops this client knows, so run them here instead.
    if info.get("vocabulary") != vocabulary():
        import sys
        print("[arcana] The open app runs a different version of Arcana, so this runs here. "
              "Restart the app to share its models again.", file=sys.stderr)
        return None
    try:
        _request(info, "GET", "/ops/state", timeout=timeout)
    except Exception:
        return None
    return info


def vocabulary() -> str:
    """A short fingerprint of the op vocabulary: names and their argument fields."""
    import hashlib
    from . import ops
    spec = sorted((o.name, tuple(sorted(o.args.model_fields))) for o in ops.all_ops())
    return hashlib.sha1(repr(spec).encode("utf-8")).hexdigest()[:12]


def _request(info: dict, method: str, route: str, body: dict | None = None, timeout: float = 600):
    url = f"http://127.0.0.1:{int(info['port'])}{route}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={TOKEN_HEADER: info.get("token", ""),
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def remote_exec(info: dict, lines: list[str], actor: str, sheets: bool | None = None) -> dict:
    """POST a batch to a running server. Returns the same JSON run_lines() builds."""
    return _request(info, "POST", "/ops/exec", {"lines": lines, "actor": actor, "sheets": sheets})


# ─────────────────────────── headless ───────────────────────────

def serve_headless(port: int = 0) -> None:
    """The ops routes without the page, until interrupted."""
    from flask import Flask
    from werkzeug.serving import make_server
    _paths.use_utf8_console()
    app = Flask("arcana-headless")
    register(app)
    srv = make_server("127.0.0.1", int(port), app, threaded=True)
    announce(srv.server_port, headless=True)
    print(f"Arcana ops on http://127.0.0.1:{srv.server_port} (headless). "
          f"Discovery in {discovery_path()}. Ctrl+C stops it.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
