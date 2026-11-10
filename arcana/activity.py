# activity.py -- what was just done, by whom, for the page to show as it happens
#
# The page learned of changes by asking every two seconds whether the state
# signature had moved, and the signature covers boards, the collection and
# settings. A search an agent ran, a pattern it rendered or a clip it
# recoloured changed nothing the signature watches, so the person saw none of
# it. This keeps a short list of the ops that ran, with who ran them and what
# came of them, and lets a waiter sleep until the list or the state moves.
# See VIDEO_ROADMAP.md, 3.7.
#
# It is memory only. The list is for watching work as it happens; what the
# work made is in the archive, the ledger and the undo history.

from __future__ import annotations

import threading
import time
from collections import deque

KEEP = 200
_items: deque = deque(maxlen=KEEP)
_cond = threading.Condition()
_seq = 0

# Ops that only look something up about Arcana itself are not worth a line.
QUIET = {"orient", "op list", "help", "history", "settings show", "job list", "job show", "job wait",
         "dataset list", "model list", "board list", "alchemy methods", "alchemy memory", "colour methods",
         "style methods", "pattern presets", "alchemy mask presets", "alchemy recipe list", "alchemy film list",
         "alchemy template list", "archive list"}


def note(actor: str, op: str, args: dict, text: str = "", result: dict | None = None) -> int:
    """Record one op that ran. Returns its number."""
    global _seq
    if op in QUIET:
        return _seq
    first = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    with _cond:
        _seq += 1
        _items.append({"seq": _seq, "t": time.time(), "actor": actor, "op": op, "args": args,
                       "summary": first[:200], "result": result or {}})
        _cond.notify_all()
        return _seq


def since(after: int = 0, actor: str | None = None) -> list[dict]:
    with _cond:
        return [dict(i) for i in _items if i["seq"] > after and (actor is None or i["actor"] == actor)]


def latest() -> int:
    return _seq


def wait(after: int, timeout: float = 25.0) -> int:
    """Sleep until an op newer than `after` has run, or the timeout. Returns the newest number."""
    with _cond:
        if _seq <= after:
            _cond.wait(timeout)
        return _seq


def poke() -> None:
    """Wake the waiters without recording anything: the state changed some other way."""
    with _cond:
        _cond.notify_all()


def clear() -> None:
    global _seq
    with _cond:
        _items.clear()
        _seq = 0
