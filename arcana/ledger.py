# ledger.py -- the derived pictures Arcana has made
#
# A colour transfer, a style transfer or a generated picture is a new file in
# output/. Before the ledger, the only record of one was the board that
# happened to reference it. The ledger gives each a handle (out:3f9c1e) that
# board add, item show and colour transfer accept like any archive picture,
# and keeps what made it: the op, its arguments and the source handles.
#
# A derived picture is always marked as one. Boards copy `kind` and
# `derived_from` into their details, and the app and the HTML board badge it,
# so nothing made here can pass for a photograph from the archive.
#
# Entries are written through a state.Transaction, so recording a result is
# undoable with the batch that made it (the file itself stays in output/).

from __future__ import annotations

import json
import os
import secrets
import time

try:
    from . import paths as _paths
    from . import state as _state
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import state as _state

LEDGER_NAME = "ledger.json"
PREFIX = "out:"
KINDS = ("derived", "generated")


def path() -> str:
    return os.path.join(_paths.subdir("output"), LEDGER_NAME)


def entries() -> list[dict]:
    try:
        with open(path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return data.get("entries", []) if isinstance(data, dict) else []


def get(ref: str) -> dict | None:
    """The entry for out:<id> (or a bare id), or None."""
    rid = ref[len(PREFIX):] if ref.startswith(PREFIX) else ref
    return next((e for e in entries() if e.get("id") == rid), None)


def for_path(p: str) -> dict | None:
    norm = os.path.normcase(os.path.abspath(p))
    return next((e for e in entries()
                 if os.path.normcase(os.path.abspath(e.get("path", ""))) == norm), None)


def record(tx, file_path: str, *, op: str, args: dict, sources: list[str],
           kind: str = "derived", derived_from: str | None = None) -> dict:
    """Add a file to the ledger inside transaction `tx`. Returns the entry."""
    if kind not in KINDS:
        raise ValueError(f"kind is one of {KINDS}")
    current = entries()
    taken = {e.get("id") for e in current}
    rid = secrets.token_hex(3)
    while rid in taken:
        rid = secrets.token_hex(3)
    entry = {"id": rid, "handle": PREFIX + rid, "path": os.path.abspath(file_path),
             "kind": kind, "op": op, "args": args, "sources": list(sources),
             "derived_from": derived_from, "created": time.time()}
    tx.touch(path())
    _paths.ensure_dir(os.path.dirname(path()))
    _state.write_json(path(), {"entries": current + [entry]})
    return entry
