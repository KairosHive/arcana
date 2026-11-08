# agent_term.py -- agents running inside the app, in a terminal
#
# The Agent tab shows an agent's own command line (Claude Code, Codex, a
# shell) running in a pseudo-terminal, the way goofi's agent panel does. The
# agent works in the Arcana workspace (workspace.py), where it finds Arcana's
# MCP server and skills, so it searches, builds boards and recolours with the
# same ops as everything else, as the `agent` actor.
#
# Each running agent is an Instance: the process, a reader thread, and the last
# couple of megabytes of what it printed. The panel streams that over
# Server-Sent Events and sends keystrokes back with small POSTs, so closing the
# tab, or reloading the page, loses nothing: the agent keeps running and the
# next view replays its scrollback.
#
# These routes type into a shell, so they are guarded twice. Every request
# carries a token minted for this process and handed only to the app's own
# page. Every command also needs it in a custom header, which another website's
# script cannot send to this address without a CORS preflight the server never
# grants.

from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import sys
import threading
import time
import uuid

try:
    from . import paths as _paths
    from . import workspace as _workspace
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import workspace as _workspace

TOKEN = secrets.token_urlsafe(24)
HEADER = "X-Arcana-Term"
SCROLLBACK_CHARS = 2_000_000
CONFIG_NAME = "agents.json"

# Offered when they are on PATH. A person adds others in agents.json in the
# data folder: {"agents": [{"name": "...", "command": "...", "args": [...]}]}.
CANDIDATES = (
    {"name": "Claude Code", "command": "claude", "args": []},
    {"name": "Codex", "command": "codex", "args": []},
    {"name": "Gemini", "command": "gemini", "args": []},
)


def _shell() -> dict:
    if sys.platform == "win32":
        return {"name": "Shell", "command": "powershell", "args": ["-NoLogo"]}
    return {"name": "Shell", "command": os.environ.get("SHELL", "/bin/bash"), "args": ["-l"]}


def offered() -> list[dict]:
    """The agents a person can start here: those found on PATH, then their own."""
    out = []
    for c in list(CANDIDATES) + [_shell()]:
        found = shutil.which(c["command"])
        if found:
            out.append({**c, "path": found})
    try:
        with open(os.path.join(_paths.data_dir(), CONFIG_NAME), encoding="utf-8") as fh:
            extra = json.load(fh).get("agents") or []
    except (OSError, ValueError, AttributeError):
        extra = []
    for e in extra:
        if isinstance(e, dict) and e.get("name") and e.get("command"):
            found = shutil.which(e["command"]) or (e["command"] if os.path.isfile(e["command"]) else None)
            if found and all(o["name"] != e["name"] for o in out):
                out.append({"name": e["name"], "command": e["command"],
                            "args": list(e.get("args") or []), "path": found})
    return out


def _environment(ws: str) -> dict:
    env = dict(os.environ)
    env.update(TERM="xterm-256color", COLORTERM="truecolor", ARCANA_ACTOR="agent",
               ARCANA_DATA_DIR=_paths.data_dir(), ARCANA_WORKSPACE=ws)
    # The `arcana` command of this install first on PATH, so the agent's shell
    # reaches the same Arcana the app runs.
    scripts = os.path.dirname(sys.executable)
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    return env


def _spawn(argv: list[str], cwd: str, env: dict, cols: int, rows: int):
    if sys.platform == "win32":
        from winpty import PtyProcess
        return PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=(rows, cols))
    from ptyprocess import PtyProcessUnicode
    return PtyProcessUnicode.spawn(argv, cwd=cwd, env=env, dimensions=(rows, cols))


class Instance:
    def __init__(self, agent: dict, cols: int = 120, rows: int = 32):
        self.id = uuid.uuid4().hex[:10]
        self.name = agent["name"]
        self.started = time.time()
        self.cwd = _workspace.prepare()
        argv = [agent["path"], *agent.get("args", [])]
        if sys.platform == "win32" and argv[0].lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", *argv]
        self.proc = _spawn(argv, self.cwd, _environment(self.cwd), cols, rows)
        self.size = (cols, rows)
        self.text: list[str] = []
        self.chars = 0
        self.exit_code: int | None = None
        self.lock = threading.Lock()
        self.listeners: list[queue.Queue] = []
        threading.Thread(target=self._read, name=f"agent-{self.id}", daemon=True).start()

    @property
    def running(self) -> bool:
        return self.exit_code is None

    def _read(self) -> None:
        while True:
            try:
                chunk = self.proc.read(4096)
            except EOFError:
                break
            except Exception:
                if not self.proc.isalive():
                    break
                time.sleep(0.05)
                continue
            if not chunk:
                if not self.proc.isalive():
                    break
                continue
            with self.lock:
                self.text.append(chunk)
                self.chars += len(chunk)
                while self.chars > SCROLLBACK_CHARS and len(self.text) > 1:
                    self.chars -= len(self.text.pop(0))
                for q in self.listeners:
                    q.put(("d", chunk))
        try:
            code = self.proc.exitstatus if self.proc.exitstatus is not None else self.proc.wait()
        except Exception:
            code = -1
        with self.lock:
            self.exit_code = int(code) if code is not None else -1
            for q in self.listeners:
                q.put(("exit", self.exit_code))

    def subscribe(self) -> tuple[str, queue.Queue]:
        """The scrollback so far, and a queue of what comes next."""
        q: queue.Queue = queue.Queue()
        with self.lock:
            history = "".join(self.text)
            self.listeners.append(q)
            if self.exit_code is not None:
                q.put(("exit", self.exit_code))
        return history, q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)

    def write(self, data: str) -> None:
        if self.running:
            self.proc.write(data)

    def resize(self, cols: int, rows: int) -> None:
        cols, rows = max(20, min(int(cols), 500)), max(5, min(int(rows), 300))
        if self.running and (cols, rows) != self.size:
            self.proc.setwinsize(rows, cols)
            self.size = (cols, rows)

    def stop(self) -> None:
        try:
            self.proc.terminate(force=True)
        except Exception:
            pass

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "running": self.running,
                "exit_code": self.exit_code, "started": self.started, "cwd": self.cwd}


class Manager:
    def __init__(self):
        self._lock = threading.Lock()
        self.instances: dict[str, Instance] = {}

    def start(self, name: str, cols: int = 120, rows: int = 32) -> Instance:
        agent = next((a for a in offered() if a["name"] == name), None)
        if agent is None:
            raise ValueError(f"No agent called {name!r} on this machine.")
        inst = Instance(agent, cols, rows)
        with self._lock:
            self.instances[inst.id] = inst
        return inst

    def get(self, iid: str) -> Instance | None:
        return self.instances.get(iid)

    def roster(self) -> list[dict]:
        with self._lock:
            return [i.info() for i in self.instances.values()]

    def stop(self, iid: str) -> bool:
        inst = self.instances.get(iid)
        if inst is None:
            return False
        inst.stop()
        return True

    def forget(self, iid: str) -> None:
        with self._lock:
            inst = self.instances.pop(iid, None)
        if inst is not None and inst.running:
            inst.stop()

    def stop_all(self) -> None:
        for inst in list(self.instances.values()):
            inst.stop()


MANAGER = Manager()


# ─────────────────────────── routes ───────────────────────────

def blueprint():
    from flask import Blueprint, Response, jsonify, request, stream_with_context

    bp = Blueprint("arcana_agent", __name__)

    @bp.before_request
    def _guard():
        # The stream is an EventSource, which cannot send headers, so it takes
        # the token in the query. Everything that acts needs the header.
        given = request.headers.get(HEADER) or (request.args.get("token")
                                                if request.endpoint == "arcana_agent._stream" else None)
        if given != TOKEN:
            return jsonify({"error": "This needs the app's own page."}), 403
        return None

    @bp.get("/agent/roster")
    def _roster():
        return jsonify({"offered": [{"name": a["name"], "command": a["command"]} for a in offered()],
                        "instances": MANAGER.roster(), "workspace": _workspace.workspace_dir()})

    @bp.post("/agent/start")
    def _start():
        body = request.get_json(silent=True) or {}
        try:
            inst = MANAGER.start(str(body.get("name", "")), int(body.get("cols") or 120),
                                 int(body.get("rows") or 32))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(inst.info())

    def _instance(iid):
        inst = MANAGER.get(iid)
        if inst is None:
            return None, (jsonify({"error": "No such agent."}), 404)
        return inst, None

    @bp.post("/agent/<iid>/input")
    def _input(iid):
        inst, err = _instance(iid)
        if err:
            return err
        data = (request.get_json(silent=True) or {}).get("data", "")
        if isinstance(data, str) and data:
            inst.write(data)
        return jsonify({"ok": True})

    @bp.post("/agent/<iid>/resize")
    def _resize(iid):
        inst, err = _instance(iid)
        if err:
            return err
        body = request.get_json(silent=True) or {}
        inst.resize(body.get("cols") or 120, body.get("rows") or 32)
        return jsonify({"ok": True})

    @bp.post("/agent/<iid>/stop")
    def _stop(iid):
        inst, err = _instance(iid)
        if err:
            return err
        inst.stop()
        return jsonify({"ok": True})

    @bp.post("/agent/<iid>/forget")
    def _forget(iid):
        MANAGER.forget(iid)
        return jsonify({"ok": True})

    @bp.get("/agent/<iid>/stream")
    def _stream(iid):
        inst, err = _instance(iid)
        if err:
            return err

        def events():
            history, q = inst.subscribe()
            try:
                if history:
                    yield "data: " + json.dumps({"d": history}) + "\n\n"
                while True:
                    try:
                        kind, value = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    if kind == "d":
                        yield "data: " + json.dumps({"d": value}) + "\n\n"
                    else:
                        yield "data: " + json.dumps({"exit": value}) + "\n\n"
                        return
            finally:
                inst.unsubscribe(q)

        return Response(stream_with_context(events()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return bp


def register(flask_app) -> None:
    if "arcana_agent" not in flask_app.blueprints:
        flask_app.register_blueprint(blueprint())
        import atexit
        atexit.register(MANAGER.stop_all)
