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
#
# Schema 2 adds what a board built by an agent needs -- sections, a role and a
# note per picture, a palette, and where each derived picture came from --
# without changing anything a schema 1 reader depends on. `items` is still a
# plain list of paths; everything else about an item sits in `details`, keyed by
# path. Two versions of Arcana can share one data dir (the installed app beside
# a development checkout, or two machines on one ARCANA_DATA_DIR), and an older
# one reads the paths and ignores the rest. Turning items into objects would
# have crashed it on every board the agent wrote. See AGENT_PLAN.md.

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
SCHEMA = 2

# What a picture is doing on the board. One hero; the rest support it.
ROLES = ("hero", "support", "texture", "detail", "counterpoint")

# Who made the board. Absent means a person: every schema 1 board was.
AUTHORS = ("person", "agent")

# Top-level fields schema 1 did not have. save() keeps each one as it was on
# disk unless the caller passes it, which is what lets the Moodboard tab re-save
# an agent's board without knowing these fields exist.
_V2_FIELDS = ("author", "brief", "concept", "palette", "sections", "layout", "kind", "queries")

# What a board is for. A board is a moodboard, curated by hand. A set is pictures
# gathered by searches, kept with the searches that found them, for Alchemy's IP
# labels and LoRAs. Absent means a board: every board before sets was one.
KINDS = ("board", "set")


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


def _split_items(items) -> tuple[list[str], dict[str, dict]]:
    """
    Items as paths, plus whatever details came with them.

    The Moodboard tab passes plain paths. A caller building a board picture by
    picture may pass {"path": ..., "role": ..., "note": ...} instead; both end
    up in the same shape on disk.
    """
    paths, details = [], {}
    for it in items or []:
        if isinstance(it, dict):
            p = it.get("path")
            extra = {k: v for k, v in it.items() if k != "path"}
        else:
            p, extra = it, None
        if not p:
            continue
        paths.append(str(p))
        if extra:
            details.setdefault(str(p), {}).update(extra)
    return paths, details


def save(name: str, items: list, *, reference: str | None = None,
         transfer: str | None = None, dataset: str | None = None,
         slug: str | None = None, details: dict | None = None,
         author: str | None = None, brief: str | None = None,
         concept: str | None = None, palette: list | None = None,
         sections: list | None = None, layout: dict | None = None,
         fingerprints: bool = False) -> dict:
    """
    Write a board and return its record.

    Saving over an existing board keeps its created timestamp, so a board you
    keep working on does not look new every time.

    It also keeps what the caller did not mention. The `details` of every
    picture still on the board survive, merged under any new ones, and each
    schema 2 top-level field left as None keeps its value on disk. The
    Moodboard tab only knows about paths, so without this, tidying a board an
    agent built would wipe every note and section it wrote.

    `fingerprints` records a content hash for each picture that has none yet,
    so a board can find its pictures again after a folder move. It reads every
    new file whole, which is why the Moodboard tab's save leaves it off.
    """
    name = (name or "").strip() or "Untitled board"
    slug = slug or slugify(name)
    dest = _path_for(slug)
    if dest is None:
        raise ValueError(f"unsafe board name: {name!r}")
    if author is not None and author not in AUTHORS:
        raise ValueError(f"unknown author {author!r}; expected one of {AUTHORS}")

    paths, given = _split_items(items)
    for p, d in (details or {}).items():
        given.setdefault(str(p), {}).update(d or {})
    for p, d in given.items():
        role = d.get("role")
        if role is not None and role not in ROLES:
            raise ValueError(f"unknown role {role!r} for {p}; expected one of {ROLES}")

    created = time.time()
    previous: dict = {}
    if os.path.exists(dest):
        try:
            with open(dest, encoding="utf-8") as fh:
                previous = json.load(fh)
            created = float(previous.get("created", created))
        except Exception:
            pass
    if not isinstance(previous, dict):
        previous = {}

    old_details = previous.get("details")
    if not isinstance(old_details, dict):
        old_details = {}
    merged = {}
    for p in paths:
        d = dict(old_details.get(p) or {})
        d.update(given.get(p) or {})
        if fingerprints and "fp" not in d and os.path.isfile(p):
            try:
                from . import bundle as _bundle
            except ImportError:
                import bundle as _bundle
            try:
                d["fp"] = _bundle.fingerprint(p)
            except OSError:
                pass
        if d:
            merged[p] = d

    record = {
        "schema": SCHEMA,
        "name": name,
        "slug": slug,
        "created": created,
        "updated": time.time(),
        "dataset": dataset or "",
        "reference": reference or "",
        "transfer": transfer or "",
        "items": paths,
        "details": merged,
    }
    passed = {"author": author, "brief": brief, "concept": concept,
              "palette": palette, "sections": sections, "layout": layout}
    for key in _V2_FIELDS:
        value = passed.get(key) if passed.get(key) is not None else previous.get(key)
        if value is not None:
            record[key] = value

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
    details = record.get("details")
    if not isinstance(details, dict):
        details = {}
    # A hand-edited board may hold objects in `items`. Everything downstream
    # expects paths, and os.path.exists on a dict raises, so reduce them here
    # and keep what came with them as details.
    if any(isinstance(i, dict) for i in record["items"]):
        record["items"], extra = _split_items(record["items"])
        for p, d in extra.items():
            details.setdefault(p, {}).update(d)
    record["details"] = details
    return record


def item_paths(record: dict) -> list[str]:
    """The board's pictures, in order, as paths."""
    return [p for p in record.get("items", []) if isinstance(p, str) and p]


def details_for(record: dict, path: str) -> dict:
    """What the board records about one picture: role, note, section, provenance."""
    return (record.get("details") or {}).get(path, {})


def present(record: dict) -> tuple[list[str], list[str]]:
    """
    Split a board's items into those still on disk and those that are not.

    A board is a list of paths, so moving or deleting the originals leaves
    holes. Reporting them beats silently loading a shorter board than the one
    that was saved.
    """
    here, gone = [], []
    for p in item_paths(record):
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
            "count": len(item_paths(record)),
            "missing": len(gone),
            "updated": record.get("updated", 0),
            "dataset": record.get("dataset", ""),
            "author": record.get("author", "person"),
            "kind": record.get("kind", "board"),
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
