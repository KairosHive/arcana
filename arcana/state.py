# state.py -- the one owner of Arcana's mutable state
#
# Arcana's mutable state is small and already lives in files in the data dir:
# boards (boards/*.json), the person's agent settings (agent.json) and the
# output ledger (output/ledger.json, the derived pictures). Every op that
# changes one of them runs inside a Transaction from the Store below, which
#
#   * snapshots each file before its first change,
#   * on commit, records one labelled undo entry for the actor who asked,
#   * on rollback, puts every file back as it was,
#   * bumps a version counter that clients poll to learn something changed.
#
# Undo is per actor (person, cli, agent), so an agent cannot undo a person's
# edit and a person's undo leaves the agent's work alone. The history is kept
# on disk (state/history.json), so `arcana undo` in a terminal works whether
# the command ran in the app's process or in its own.
#
# A7 lives here too: an agent may change only boards an agent authored. Asked
# to change a person's board, the transaction refuses and names `board copy`.
# See OPS_PLAN.md, "The owner of state".

from __future__ import annotations

import json
import os
import threading
import time

try:
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths

HISTORY_NAME = "history.json"
LOCK_NAME = "lock"
MAX_UNDO = 40                 # entries kept per actor
LOCK_STALE_S = 30.0           # a lock file older than this was left by a crash


class StateError(ValueError):
    """A write the store refuses: a person's board touched by an agent, a stale undo."""


# ─────────────────────────── a lock across processes ───────────────────────────

class _FileLock:
    """
    One writer at a time across processes, by exclusive creation of a file.

    The app, a terminal and an agent's server can all be open on one data dir.
    Writes are short (a board is a few kilobytes), so a spin with a short sleep
    is enough, and a lock file older than LOCK_STALE_S is taken to be left over
    from a crash.
    """

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        deadline = time.time() + 20
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        os.remove(self.path)
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    raise StateError("Another Arcana process has held the state lock for "
                                     "twenty seconds. Try again in a moment.")
                time.sleep(0.02)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


# ─────────────────────────── files ───────────────────────────

def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def _write(path: str, text: str | None) -> None:
    """Write or remove a file, atomically for the write."""
    if text is None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_json(path: str, data) -> None:
    _write(path, json.dumps(data, indent=1, ensure_ascii=False))


# ─────────────────────────── transactions ───────────────────────────

class Transaction:
    """
    The files one op or one batch changes, with what they held before.

    Ops never write a state file directly. They call touch(path) first, which
    snapshots it, then write. Boards go through touch_board(), which also
    applies A7.
    """

    def __init__(self, store: "Store", actor: str, label: str, locked: bool = True):
        self.store = store
        self.actor = actor
        self.label = label
        self.before: dict[str, str | None] = {}
        self.created: set[str] = set()       # boards this transaction made
        # A write takes the store's lock when it begins. An effect (a colour
        # transfer, a generation) takes it at its first touch, so the seconds
        # it spends computing never block anyone else's write.
        self.locked = locked

    def touch(self, path: str) -> None:
        if not self.locked:
            self.store._thread_lock.acquire()
            self.locked = True
        path = os.path.abspath(path)
        if path not in self.before:
            self.before[path] = _read(path)

    def touch_board(self, slug: str, *, creating: bool = False) -> str:
        """Snapshot a board's file and check this actor may change it. Returns the path."""
        try:
            from . import boards as _boards
        except ImportError:                           # pragma: no cover
            import boards as _boards
        path = _boards._path_for(slug)
        if path is None:
            raise StateError(f"{slug!r} cannot be a board id. Use letters, digits and dashes.")
        current = _read(path)
        if current is not None and self.actor == "agent" and path not in self.created:
            try:
                author = json.loads(current).get("author", "person")
            except ValueError:
                author = "person"
            if author != "agent":
                raise StateError(
                    f"`{slug}` was made by the person, and agents change only boards an agent "
                    f"made. Work on a copy: board copy {slug} --id {slug}-agent")
        if creating:
            if current is not None:
                raise StateError(f"A board called `{slug}` already exists. Pick another --id, "
                                 f"or add to it with `board add {slug} ...`.")
            self.created.add(path)
        self.touch(path)
        return path

    def changes(self) -> dict[str, tuple[str | None, str | None]]:
        out = {}
        for path, before in self.before.items():
            after = _read(path)
            if after != before:
                out[path] = (before, after)
        return out

    def rollback(self) -> None:
        for path, before in self.before.items():
            _write(path, before)


# ─────────────────────────── the store ───────────────────────────

class Store:
    """
    Transactions, per-actor undo and redo, and the version counter.

    One instance per process (STORE below). The history file is the shared
    truth, read and written under a cross-process lock, so several processes
    on one data dir agree on the version and on each actor's history.
    """

    def __init__(self):
        self._local = threading.local()
        self._subscribers: list = []
        self._thread_lock = threading.RLock()

    # -- files ----------------------------------------------------------------

    def _dir(self) -> str:
        return _paths.ensure_dir(_paths.subdir("state"))

    def _history_path(self) -> str:
        return os.path.join(self._dir(), HISTORY_NAME)

    def _load(self) -> dict:
        text = _read(self._history_path())
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("version", 0)
        data.setdefault("undo", {})
        data.setdefault("redo", {})
        return data

    def _save(self, data: dict) -> None:
        write_json(self._history_path(), data)

    # -- reading ----------------------------------------------------------------

    @property
    def version(self) -> int:
        return int(self._load()["version"])

    def signature(self) -> str:
        """
        Changes whenever state changes, by any route: a committed write bumps
        the version, and a board written by an older Arcana or by hand moves
        the boards folder's newest modification time.
        """
        try:
            from . import boards as _boards
        except ImportError:                           # pragma: no cover
            import boards as _boards
        newest = 0.0
        folder = _boards.boards_dir()
        for name in _paths.listdir_safe(folder):
            if name.endswith(".json"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(folder, name)))
                except OSError:
                    pass
        count = sum(1 for n in _paths.listdir_safe(folder) if n.endswith(".json"))
        return f"{self.version}:{count}:{newest:.3f}"

    def history(self, actor: str) -> dict:
        data = self._load()
        return {"undo": [e["label"] for e in data["undo"].get(actor, [])],
                "redo": [e["label"] for e in data["redo"].get(actor, [])],
                "version": data["version"]}

    # -- writing ----------------------------------------------------------------

    def current(self) -> Transaction | None:
        return getattr(self._local, "tx", None)

    def begin(self, actor: str, label: str, lazy: bool = False) -> Transaction:
        """
        Open a transaction on this thread. `lazy` defers the lock to the first
        file touched, for effects that compute for a long time before writing.
        """
        if self.current() is not None:
            raise RuntimeError("a transaction is already open on this thread")
        if not lazy:
            self._thread_lock.acquire()
        tx = Transaction(self, actor, label, locked=not lazy)
        self._local.tx = tx
        return tx

    def _end(self) -> None:
        tx = self.current()
        self._local.tx = None
        if tx is not None and tx.locked:
            self._thread_lock.release()

    def commit(self, tx: Transaction) -> int | None:
        """Record the transaction's changes as one undo step. Returns the new version."""
        if self.current() is not tx:
            raise RuntimeError("commit from the thread that began the transaction")
        try:
            changes = tx.changes()
            if not changes:
                return None
            with _FileLock(os.path.join(self._dir(), LOCK_NAME)):
                data = self._load()
                entry = {"label": tx.label, "at": time.time(),
                         "files": {p: {"before": b, "after": a} for p, (b, a) in changes.items()}}
                stack = data["undo"].setdefault(tx.actor, [])
                stack.append(entry)
                del stack[:-MAX_UNDO]
                data["redo"][tx.actor] = []
                data["version"] += 1
                self._save(data)
                version = data["version"]
            self._notify(version, [tx.label])
            return version
        finally:
            self._end()

    def rollback(self, tx: Transaction) -> None:
        try:
            tx.rollback()
        finally:
            self._end()

    def _step(self, actor: str, source: str, dest: str) -> str | None:
        """Move the newest entry of `source` to `dest`, applying it backwards or forwards."""
        undoing = source == "undo"
        with self._thread_lock, _FileLock(os.path.join(self._dir(), LOCK_NAME)):
            data = self._load()
            stack = data[source].get(actor, [])
            if not stack:
                return None
            entry = stack[-1]
            # Undo only what is still as this actor left it. If anyone has
            # changed a file since, undoing would silently discard their work.
            for path, f in entry["files"].items():
                expect = f["after"] if undoing else f["before"]
                if _read(path) != expect:
                    name = os.path.basename(path)
                    raise StateError(f"`{name}` has changed since \"{entry['label']}\", so "
                                     f"{'undoing' if undoing else 'redoing'} it would discard "
                                     f"that later change. Nothing was changed.")
            for path, f in entry["files"].items():
                _write(path, f["before"] if undoing else f["after"])
            stack.pop()
            data[dest].setdefault(actor, []).append(entry)
            data["version"] += 1
            self._save(data)
            version = data["version"]
        self._notify(version, [("undo " if undoing else "redo ") + entry["label"]])
        return entry["label"]

    def undo(self, actor: str) -> str | None:
        """Take back this actor's newest step. Returns its label, or None if there is none."""
        return self._step(actor, "undo", "redo")

    def redo(self, actor: str) -> str | None:
        return self._step(actor, "redo", "undo")

    # -- notification -----------------------------------------------------------

    def subscribe(self, fn) -> None:
        """fn(version, labels) after every commit, undo and redo in this process."""
        self._subscribers.append(fn)

    def _notify(self, version: int, labels: list[str]) -> None:
        for fn in list(self._subscribers):
            try:
                fn(version, labels)
            except Exception:
                pass


STORE = Store()
