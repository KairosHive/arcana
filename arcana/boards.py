# boards.py — named moodboards that survive the browser
#
# The collection used to live in one dcc.Store with storage_type="local": a
# single unnamed list in browser localStorage, keyed by origin. That meant one
# board at a time, no way back to yesterday's, and a Clear button that emptied
# it with no undo. "Save collection" did exist, but it COPIED the image files
# into a folder -- an export, with nothing that could be loaded again.
#
# So a board is stored here instead, as JSON in the data directory:
#
#   * it survives clearing browser data, or opening the app on another port
#   * ARCANA_DATA_DIR is already how two machines share datasets, so boards
#     travel the same way
#   * it can be backed up, inspected and diffed like anything else on disk
#
# A board records paths, not pixels. Nothing is copied, which keeps it in line
# with the rest of Arcana: the originals stay where they are.

from __future__ import annotations

import json
import os
import re
import time

try:
    from . import paths as _paths
except ImportError:
    import paths as _paths

BOARDS_DIRNAME = "boards"
SCHEMA = 1


def boards_dir() -> str:
    return _paths.ensure_dir(_paths.subdir(BOARDS_DIRNAME))


def slugify(name: str) -> str:
    """
    A filename-safe key for a board name.

    Two boards whose names differ only by punctuation would collide, which is
    why load() and delete() take the slug rather than the display name -- the
    caller always has it, because listing() hands it back.
    """
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s or "board"


def _path_for(slug: str) -> str | None:
    """The file a slug maps to, or None if it would escape the boards folder."""
    return _paths.safe_join(boards_dir(), slug + ".json")


def save(name: str, items: list[str], *, reference: str | None = None,
         transfer: str | None = None, dataset: str | None = None,
         slug: str | None = None) -> dict:
    """
    Write a board and return its record.

    Saving over an existing board keeps its created timestamp, so a board you
    keep working on does not look new every time.
    """
    name = (name or "").strip() or "Untitled board"
    slug = slug or slugify(name)
    dest = _path_for(slug)
    if dest is None:
        raise ValueError(f"unsafe board name: {name!r}")

    created = time.time()
    if os.path.exists(dest):
        try:
            with open(dest, encoding="utf-8") as fh:
                created = float(json.load(fh).get("created", created))
        except Exception:
            pass

    record = {
        "schema": SCHEMA,
        "name": name,
        "slug": slug,
        "created": created,
        "updated": time.time(),
        "dataset": dataset or "",
        "reference": reference or "",
        "transfer": transfer or "",
        "items": [p for p in (items or []) if p],
    }

    # Write beside and rename, so an interrupted save cannot leave a truncated
    # board where a good one was.
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, dest)
    return record


def load(slug: str) -> dict | None:
    """A board by slug, or None when it is absent or unreadable."""
    src = _path_for(slug)
    if not src or not os.path.exists(src):
        return None
    try:
        with open(src, encoding="utf-8") as fh:
            record = json.load(fh)
    except Exception:
        return None
    if not isinstance(record, dict):
        return None
    record.setdefault("items", [])
    record.setdefault("name", slug)
    record["slug"] = slug
    return record


def present(record: dict) -> tuple[list[str], list[str]]:
    """
    Split a board's items into those still on disk and those that are not.

    A board is a list of paths, so moving or deleting the originals leaves
    holes. Reporting them beats silently loading a shorter board than the one
    that was saved.
    """
    here, gone = [], []
    for p in record.get("items", []):
        (here if os.path.exists(p) else gone).append(p)
    return here, gone


def listing() -> list[dict]:
    """Every saved board, newest first, with enough to render a picker."""
    out = []
    try:
        names = os.listdir(boards_dir())
    except OSError:
        return out
    for fname in names:
        if not fname.endswith(".json"):
            continue
        record = load(fname[:-5])
        if not record:
            continue
        here, gone = present(record)
        out.append({
            "slug": record["slug"],
            "name": record.get("name", record["slug"]),
            "count": len(record.get("items", [])),
            "missing": len(gone),
            "updated": record.get("updated", 0),
            "dataset": record.get("dataset", ""),
        })
    out.sort(key=lambda r: r.get("updated", 0), reverse=True)
    return out


def delete(slug: str) -> bool:
    """
    Remove a board. Only ever deletes a .json inside the boards folder.

    The guard matters because the slug reaches here from the browser: without
    it a crafted value could point os.remove somewhere else entirely.
    """
    target = _path_for(slug)
    if not target or not os.path.exists(target):
        return False
    if not _paths.is_within(target, [boards_dir()]):
        return False
    try:
        os.remove(target)
        return True
    except OSError:
        return False
