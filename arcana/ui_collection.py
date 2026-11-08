# ui_collection.py -- the active collection, shared by every tab
#
# A collection is a board. The person picks the active one in the header, and
# that choice is a setting (`settings set --collection`), so every tab, every
# port the app is served on, the terminal and agents see the same pictures.
# "+ Collection" anywhere adds to it with `board add`; removing and clearing are
# `board remove`. Nothing about the collection lives in the browser.
#
# These functions are what the JSON API (arcana/api) calls. They hold no page
# of their own: the Svelte front end draws the rail, the picker and the viewer.

from __future__ import annotations

import time

try:
    from . import ops as _ops
except ImportError:                                   # pragma: no cover
    import ops as _ops

DEFAULT_NAME = "Collection"


def active() -> str | None:
    """The active board's id, or None when none is chosen or it has been deleted."""
    slug = _ops.archive.settings()["collection"]
    if not slug:
        return None
    try:
        _ops.call("board show", {"board": slug}, actor="person")
    except _ops.Refusal:
        return None
    return slug


def use(slug: str | None) -> None:
    _ops.call("settings set", {"collection": slug or ""}, actor="person")


def ensure_active() -> str:
    """The active board, made and chosen first when there is none."""
    slug = active()
    if slug:
        return slug
    boards = {b.slug: b for b in _ops.call("board list", actor="person").boards}
    slug = "collection"
    if slug not in boards:
        slug = _ops.call("board create", {"name": DEFAULT_NAME, "id": slug}, actor="person").board.slug
    use(slug)
    return slug


def contents(slug: str | None) -> tuple[list[str], dict, str]:
    """(paths of the pictures present, their badges, the board's name) for a board."""
    if not slug:
        return [], {}, ""
    try:
        rec = _ops.call("board show", {"board": slug}, actor="person")
    except _ops.Refusal:
        return [], {}, ""
    here = [i.path for i in rec.items if not i.missing]
    details = {i.path: {"role": i.role, "kind": i.kind, "section": i.section, "note": i.note,
                        "derived_from": i.derived_from, "handle": i.handle, "missing": i.missing}
               for i in rec.items}
    return here, details, rec.name


def add(refs: list[str]) -> str:
    """Put pictures (paths or handles) on the active collection. Returns its id."""
    slug = ensure_active()
    refs = [r for r in refs if r]
    if refs:
        _ops.call("board add", {"board": slug, "items": refs}, actor="person")
    return slug


def remove(refs: list[str]) -> None:
    slug = active()
    refs = [r for r in refs if r]
    if slug and refs:
        _ops.call("board remove", {"board": slug, "items": refs}, actor="person")


def clear() -> None:
    slug = active()
    if slug:
        remove(contents(slug)[0])


def rev() -> float:
    """A new value for a client's collection revision."""
    return time.time()


def snapshot() -> dict:
    """What the page needs about the collection in one object."""
    slug = active()
    items, details, name = contents(slug)
    return {"slug": slug, "name": name, "items": items, "details": details}
