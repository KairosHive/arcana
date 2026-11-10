# catalogue.py -- everything Arcana has kept, as one tree to organise
#
# The Archive tab and the `archive` ops show what the person has made and kept, by kind:
# collections (boards), films, videos, stories, pictures made by Alchemy and the transfers,
# saves from Prompt Search, imports and exports. Each is read live from where it lives: boards
# from the boards folder, films from the films folder, the rest from output/. Nothing about them
# is copied here.
#
# What the catalogue owns (catalogue.json in the data folder) is how the person organises them:
#
#   names    a display name for a file or a folder of output, shown in place of its file name.
#            The file keeps its name on disk, so nothing that points at it breaks. A collection or
#            a film is renamed in its own record instead, since it has a name of its own.
#   folders  folders the person makes, inside a kind or inside another folder.
#   place    where a thing sits: a folder, or for a collection, another collection.
#   trash    what was deleted: moved into the trash folder, and back again on restore.
#
# A collection inside another collection is organisation only: the outer one lists the inner ones
# as it lists a folder, and its pictures stay its own.

from __future__ import annotations

import json
import os
import secrets
import shutil
import time

try:
    from . import boards as _boards
    from . import paths as _paths
    from . import state as _state
except ImportError:                                   # pragma: no cover
    import boards as _boards
    import paths as _paths
    import state as _state

KINDS = {
    "collections": "Collections",
    "films": "Films",
    "videos": "Videos",
    "stories": "Stories",
    "pictures": "Pictures made",
    "searches": "Search saves",
    "imports": "Imports",
    "exports": "Exports",
}
PICTURE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm"}
SOUND_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
# Folders of output/ that hold each kind. Old-layout folders are read too, so nothing made before the
# layout changed is missing from the tree.
PICTURE_DIRS = ("edited", "color_transfer", "transfers", "generated", "alchemy")
SEARCH_DIRS = ("search", "selections")
IMPORT_DIRS = ("imports", "_external_refs")
EXPORT_DIRS = ("collections", "moodboards")


# ─────────────────────────── where things are ───────────────────────────

def path() -> str:
    return os.path.join(_paths.data_dir(), "catalogue.json")


def trash_dir() -> str:
    return os.path.join(_paths.data_dir(), "trash")


def output_root() -> str:
    return _paths.subdir("output")


def films_dir() -> str:
    return os.path.join(_paths.data_dir(), "alchemy", "films")


def read() -> dict:
    try:
        with open(path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    for key in ("names", "folders", "place", "trash"):
        data.setdefault(key, {})
    return data


def write(tx, data: dict) -> None:
    """The catalogue written inside a transaction (`tx`), or straight away when there is none."""
    if tx is not None:
        tx.touch(path())
    _paths.ensure_dir(os.path.dirname(path()))
    _state.write_json(path(), data)


# ─────────────────────────── what there is ───────────────────────────

def _rel(p: str) -> str:
    return os.path.relpath(p, output_root()).replace(os.sep, "/")


def _count_pictures(folder: str) -> int:
    n = 0
    for _, _, files in os.walk(folder):
        # Sounds count too: a search saved from an audio dataset holds nothing else, and
        # counting pictures alone listed it as an empty folder.
        n += sum(1 for f in files if os.path.splitext(f)[1].lower() in PICTURE_EXT | SOUND_EXT)
    return n


def _stamp(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def scan() -> list[dict]:
    """Every kept thing, as {id, kind, name, path, group, count, made}. `group` is where it sits on
    disk, for the things the person has not placed: a picture's own folder, say."""
    out: list[dict] = []
    for name in sorted(os.listdir(_boards.boards_dir())) if os.path.isdir(_boards.boards_dir()) else []:
        if not name.endswith(".json"):
            continue
        slug = name[:-5]
        rec = _boards.load(slug) or {}
        p = os.path.join(_boards.boards_dir(), name)
        out.append({"id": f"collection:{slug}", "kind": "collections", "name": rec.get("name") or slug,
                    "path": p, "group": "", "count": len(rec.get("items") or []), "made": _stamp(p)})
    if os.path.isdir(films_dir()):
        for name in sorted(os.listdir(films_dir())):
            if name.endswith(".json"):
                p = os.path.join(films_dir(), name)
                try:
                    with open(p, encoding="utf-8") as fh:
                        spec = json.load(fh)
                except (OSError, ValueError):
                    continue
                out.append({"id": f"film:{name[:-5]}", "kind": "films", "name": spec.get("name") or name[:-5],
                            "path": p, "group": "", "count": len(spec.get("pictures") or []), "made": _stamp(p)})
    root = output_root()
    if not os.path.isdir(root):
        return out
    for top in PICTURE_DIRS:
        for folder, _, files in os.walk(os.path.join(root, top)):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                p = os.path.join(folder, f)
                kind = "videos" if ext in VIDEO_EXT else "pictures" if ext in PICTURE_EXT else None
                if kind:
                    out.append({"id": f"file:{_rel(p)}", "kind": kind, "name": f, "path": p,
                                "group": _rel(folder), "count": 1, "made": _stamp(p)})
    stories = os.path.join(root, "stories")
    if os.path.isdir(stories):
        for f in sorted(os.listdir(stories)):
            p = os.path.join(stories, f)
            if os.path.isdir(p):
                out.append({"id": f"dir:{_rel(p)}", "kind": "stories", "name": f, "path": p, "group": "",
                            "count": _count_pictures(p), "made": _stamp(p)})
    for kind, dirs in (("searches", SEARCH_DIRS), ("exports", EXPORT_DIRS)):
        for top in dirs:
            base = os.path.join(root, top)
            if os.path.isdir(base):
                for f in sorted(os.listdir(base)):
                    p = os.path.join(base, f)
                    if os.path.isdir(p):
                        out.append({"id": f"dir:{_rel(p)}", "kind": kind, "name": f, "path": p, "group": top,
                                    "count": _count_pictures(p), "made": _stamp(p)})
    for top in IMPORT_DIRS:
        base = os.path.join(root, top)
        if os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                p = os.path.join(base, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in PICTURE_EXT:
                    out.append({"id": f"file:{_rel(p)}", "kind": "imports", "name": f, "path": p, "group": top,
                                "count": 1, "made": _stamp(p)})
    return out


def tree(data: dict | None = None, items: list[dict] | None = None) -> dict:
    """The organised tree: {kind: {"title", "folders": [...], "items": [...]}}, each folder and each
    placed collection carrying its own "folders" and "items". Display names are laid over the scan."""
    data = data or read()
    items = scan() if items is None else items
    by_id = {it["id"]: dict(it) for it in items}
    for it in by_id.values():
        it["shown"] = data["names"].get(it["id"]) or it["name"]
    folders = {fid: {"id": fid, "kind": f["kind"], "shown": f["name"], "parent": f.get("parent"), "folders": [],
                     "items": []} for fid, f in data["folders"].items()}
    roots = {k: {"id": f"kind:{k}", "kind": k, "shown": title, "folders": [], "items": []} for k, title in KINDS.items()}
    for f in folders.values():
        host = folders.get(f["parent"]) or roots[f["kind"]]
        host["folders"].append(f)
    nodes = {**folders}
    for it in by_id.values():
        if it["kind"] == "collections":
            it.setdefault("folders", [])
            it.setdefault("items", [])
            nodes[it["id"]] = it
    for it in sorted(by_id.values(), key=lambda x: x["shown"].lower()):
        host = nodes.get(data["place"].get(it["id"])) or roots[it["kind"]]
        if host is it:
            host = roots[it["kind"]]
        host["items"].append(it)
    return roots


def find(item_id: str, items: list[dict] | None = None) -> dict:
    for it in (scan() if items is None else items):
        if it["id"] == item_id:
            return it
    raise ValueError(f"Nothing called {item_id} in the archive. `archive list` shows what is there.")


# ─────────────────────────── organising ───────────────────────────

def add_folder(tx, name: str, kind: str, parent: str | None = None) -> str:
    data = read()
    name = (name or "").strip()
    if not name:
        raise ValueError("A folder needs a name.")
    if kind not in KINDS:
        raise ValueError(f"A folder belongs to one of {', '.join(KINDS)}.")
    if parent is not None:
        p = data["folders"].get(parent)
        if p is None or p["kind"] != kind:
            raise ValueError(f"{parent} is not a folder of {KINDS[kind].lower()}.")
    fid = "folder:" + secrets.token_hex(3)
    data["folders"][fid] = {"name": name, "kind": kind, "parent": parent}
    write(tx, data)
    return fid


def _ancestors(data: dict, node: str):
    seen = set()
    while node and node not in seen:
        seen.add(node)
        yield node
        node = (data["folders"].get(node) or {}).get("parent") or data["place"].get(node)


def move(tx, item_id: str, into: str | None) -> None:
    """Put a thing (or a folder) into a folder, or a collection into another collection. `into` None
    or kind:<kind> puts it back at the top of its kind."""
    data = read()
    kind = (data["folders"].get(item_id) or {}).get("kind") if item_id.startswith("folder:") else find(item_id)["kind"]
    if kind is None:
        raise ValueError(f"No folder {item_id}.")
    if into in (None, "", f"kind:{kind}"):
        target = None
    elif into.startswith("folder:"):
        f = data["folders"].get(into)
        if f is None or f["kind"] != kind:
            raise ValueError(f"{into} is not a folder of {KINDS[kind].lower()}.")
        target = into
    elif into.startswith("collection:") and kind == "collections":
        find(into)
        target = into
    else:
        raise ValueError(f"A {kind[:-1]} goes into a folder of {KINDS[kind].lower()}"
                         + (" or into another collection." if kind == "collections" else "."))
    if target is not None and item_id in set(_ancestors(data, target)):
        raise ValueError("A folder cannot go inside itself or anything inside it.")
    if item_id.startswith("folder:"):
        data["folders"][item_id]["parent"] = target
    elif target is None:
        data["place"].pop(item_id, None)
    else:
        data["place"][item_id] = target
    write(tx, data)


def rename_display(tx, item_id: str, name: str) -> None:
    """A display name for a file or a folder of output, or a folder of the archive's own."""
    data = read()
    name = (name or "").strip()
    if not name:
        raise ValueError("A name cannot be empty.")
    if item_id.startswith("folder:"):
        if item_id not in data["folders"]:
            raise ValueError(f"No folder {item_id}.")
        data["folders"][item_id]["name"] = name
    else:
        find(item_id)
        data["names"][item_id] = name
    write(tx, data)


def merge_folders(tx, source: str, into: str) -> None:
    """Everything in folder `source` moves into folder `into`, and `source` goes."""
    data = read()
    a, b = data["folders"].get(source), data["folders"].get(into)
    if a is None or b is None or source == into:
        raise ValueError("Merge takes two different folders.")
    if a["kind"] != b["kind"]:
        raise ValueError("Folders of different kinds do not merge.")
    if source in set(_ancestors(data, into)):
        raise ValueError("A folder cannot merge into one inside it.")
    for fid, f in data["folders"].items():
        if f.get("parent") == source:
            f["parent"] = into
    for item, where in list(data["place"].items()):
        if where == source:
            data["place"][item] = into
    del data["folders"][source]
    write(tx, data)


# ─────────────────────────── the trash ───────────────────────────

def trash(item_id: str) -> dict:
    """Move a thing into the trash folder, outside undo: restore brings it back. A folder of the
    archive goes only if it is empty."""
    data = read()
    if item_id.startswith("folder:"):
        if item_id not in data["folders"]:
            raise ValueError(f"No folder {item_id}.")
        if any(f.get("parent") == item_id for f in data["folders"].values()) or item_id in data["place"].values():
            raise ValueError("Only an empty folder can be deleted. Move what is in it first.")
        gone = data["folders"].pop(item_id)
        write(None, data)
        return {"id": item_id, "name": gone["name"]}
    it = find(item_id)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_dir = _paths.ensure_dir(os.path.join(trash_dir(), stamp + "-" + secrets.token_hex(2)))
    moved = []
    for p in [it["path"]] + ([os.path.splitext(it["path"])[0] + ".json"] if it["id"].startswith("file:") else []):
        if os.path.exists(p):
            dest = os.path.join(dest_dir, os.path.basename(p))
            shutil.move(p, dest)
            moved.append([p, dest])
    entry = {"name": data["names"].get(item_id) or it["name"], "kind": it["kind"], "moved": moved,
             "place": data["place"].pop(item_id, None), "at": time.time()}
    data["trash"][item_id] = entry
    for other, where in list(data["place"].items()):
        if where == item_id:                      # collections that sat in this one go back to the top
            data["place"].pop(other)
    write(None, data)
    return {"id": item_id, **entry}


def restore(item_id: str) -> dict:
    data = read()
    entry = data["trash"].get(item_id)
    if entry is None:
        raise ValueError(f"{item_id} is not in the trash.")
    for orig, dest in entry["moved"]:
        if os.path.exists(orig):
            raise ValueError(f"Something new is already at {orig}. Rename or move it first.")
    for orig, dest in entry["moved"]:
        _paths.ensure_dir(os.path.dirname(orig))
        shutil.move(dest, orig)
    if entry.get("place"):
        data["place"][item_id] = entry["place"]
    del data["trash"][item_id]
    write(None, data)
    return {"id": item_id, **entry}


def empty_trash() -> int:
    """Delete everything in the trash for good. Returns how many things went."""
    data = read()
    n = 0
    for item_id, entry in list(data["trash"].items()):
        for _, dest in entry["moved"]:
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            elif os.path.exists(dest):
                os.remove(dest)
        n += 1
        del data["trash"][item_id]
    if os.path.isdir(trash_dir()):
        for d in os.listdir(trash_dir()):
            full = os.path.join(trash_dir(), d)
            if os.path.isdir(full) and not os.listdir(full):
                os.rmdir(full)
    write(None, data)
    return n
