# catalogue.py -- the archive: everything kept, organised (arcana/catalogue.py)
#
# One tree of the person's collections, films, videos, stories, pictures made, search saves,
# imports and exports. Renaming, moving and making folders are writes that undo takes back.
# Deleting moves a thing into the trash, outside undo, and restore brings it back; emptying the
# trash is for good.

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .registry import Args, Context, Result, TextResult, op, positional

try:
    from .. import catalogue as _cat
except ImportError:                                   # pragma: no cover
    import catalogue as _cat


def _id_arg(index=0, what="A thing in the archive, as `archive list` names it: collection:<id>, film:<id>, "
                          "file:<path>, dir:<path> or folder:<id>."):
    return positional(index, description=what, json_schema_extra={"metavar": "ID"})


class Node(BaseModel):
    id: str
    kind: str
    shown: str
    count: int = 0
    path: str = ""
    group: str = ""
    author: str = ""
    folders: list["Node"] = Field(default_factory=list)
    items: list["Node"] = Field(default_factory=list)

    def lines(self, depth: int = 0) -> list[str]:
        pad = "  " * depth
        head = f"{pad}{self.shown}  [{self.id}]" + (f"  {self.count}" if self.count else "") + \
            ("  (by an agent)" if self.author == "agent" else "")
        out = [head]
        for f in self.folders:
            out += f.lines(depth + 1)
        for it in self.items:
            out += it.lines(depth + 1)
        return out


class TrashEntry(BaseModel):
    id: str
    name: str
    kind: str
    at: float


class ArchiveListResult(Result):
    kinds: list[Node]
    trash: list[TrashEntry] = Field(default_factory=list)

    def text(self) -> str:
        lines = []
        for k in self.kinds:
            lines += k.lines()
        if self.trash:
            lines.append("Trash")
            lines += [f"  {t.name}  [{t.id}]" for t in self.trash]
        return "\n".join(lines) + "\n"


def _node(d: dict) -> Node:
    return Node(id=d["id"], kind=d["kind"], shown=d["shown"], count=int(d.get("count") or 0), path=d.get("path", ""),
                group=d.get("group", ""), author=d.get("author", "") or "", folders=[_node(f) for f in d.get("folders") or []],
                items=[_node(i) for i in d.get("items") or []])


class ArchiveListArgs(Args):
    kind: str | None = Field(None, description="Only this kind: " + ", ".join(_cat.KINDS) + ".")


@op("archive list", "read", ArchiveListArgs, ArchiveListResult,
    doc="""
    Everything kept, as one tree by kind.

    The kinds are collections, sets, IP labels, LoRAs, board pages, films,
    videos, stories, pictures made, contact sheets, files copied from search,
    imports and exports. Boards an agent made are marked. It lists the person's folders inside each kind
    and what the trash holds. Each thing comes with the id the other archive
    ops take.
    """,
    examples=("archive list", "archive list --kind collections"))
def _archive_list(a: ArchiveListArgs, ctx: Context) -> ArchiveListResult:
    if a.kind and a.kind not in _cat.KINDS:
        raise ValueError(f"A kind is one of {', '.join(_cat.KINDS)}.")
    data = _cat.read()
    roots = _cat.tree(data)
    kinds = [_node(roots[k]) for k in _cat.KINDS if a.kind in (None, k)]
    trash = [TrashEntry(id=i, name=e["name"], kind=e["kind"], at=e["at"]) for i, e in data["trash"].items()]
    return ArchiveListResult(kinds=kinds, trash=trash)


class ArchiveShowResult(Result):
    id: str
    kind: str
    name: str
    path: str
    contents: list[str] = Field(default_factory=list)
    detail: dict = Field(default_factory=dict)

    def text(self) -> str:
        out = f"{self.name}  [{self.id}]  {self.kind}\n  {self.path}\n"
        return out + "".join(f"  {c}\n" for c in self.contents[:40]) + \
            (f"  ... and {len(self.contents) - 40} more\n" if len(self.contents) > 40 else "")


class ArchiveShowArgs(Args):
    id: str = _id_arg()


def contents_of(it: dict) -> list[str]:
    """The pictures a thing holds, in order: a collection's, a story's or a saved search's frames, or
    the file itself."""
    if it["kind"] == "pages":
        return [p for p in it.get("files") or [] if os.path.splitext(p)[1].lower() in _cat.PICTURE_EXT]
    if it["kind"] == "stories":
        return [s.get("poetry") or s.get("original") for s in story_scenes(it["path"]) if s.get("poetry") or s.get("original")]
    if it["kind"] == "labels":
        try:
            from .. import ip_labels as _labels
        except ImportError:                           # pragma: no cover
            import ip_labels as _labels
        return _labels.load(it["id"].split(":", 1)[1]).nearest(48)
    if it["kind"] == "loras":
        try:
            from .. import lora_train as _train
        except ImportError:                           # pragma: no cover
            import lora_train as _train
        run = next((r for r in _train.runs() if r["name"] == it["id"].split(":", 1)[1]), None)
        return [c["sheet"] for c in (run or {}).get("checkpoints", []) if c.get("sheet")]
    if it["kind"] in _cat.BOARD_KINDS:
        try:
            from .. import boards as _boards
        except ImportError:                           # pragma: no cover
            import boards as _boards
        rec = _boards.load(it["id"].split(":", 1)[1]) or {}
        return [p for p in rec.get("items") or [] if isinstance(p, str)]
    if it["kind"] == "films":
        import json
        with open(it["path"], encoding="utf-8") as fh:
            spec = json.load(fh)
        return [str(p.get("ref")) for p in sorted(spec.get("pictures") or [], key=lambda p: float(p["t"]))]
    if it["id"].startswith("dir:"):
        found = []
        for d, _, files in os.walk(it["path"]):
            found += [os.path.join(d, f) for f in sorted(files)
                      if os.path.splitext(f)[1].lower() in _cat.PICTURE_EXT]
        return found
    return [it["path"]]


def story_scenes(folder: str) -> list[dict]:
    """A saved story as its scenes: each sentence of story.txt with the picture found for it and, when
    poetry was injected, the picture drawn from it. Pictures are matched by their two-digit number."""
    import re

    lines = []
    try:
        with open(os.path.join(folder, "story.txt"), encoding="utf-8") as fh:
            lines = [re.sub(r"^\s*\d+\.\s*", "", ln).strip() for ln in fh if ln.strip()]
    except OSError:
        pass

    def numbered(sub):
        # The picture saved with the story is NN_original / NN_poetry; a folder that also holds an
        # earlier poetry pass (NN_<file>_<sentence>_poetry.png) keeps those as leftovers, so the
        # exact name wins and the rest only fill numbers it lacks.
        d = os.path.join(folder, sub)
        out, exact = {}, {}
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                m = re.match(r"(\d+)_(original|poetry)?", f)
                if m and os.path.splitext(f)[1].lower() in _cat.PICTURE_EXT:
                    n = int(m.group(1))
                    out.setdefault(n, os.path.join(d, f))
                    if m.group(2) and re.fullmatch(r"\d+_(original|poetry)\.\w+", f):
                        exact[n] = os.path.join(d, f)
        return {**out, **exact}

    originals = numbered("original")
    poetry = {**numbered("poetry_injected"), **numbered("poetry")}
    n = max([len(lines)] + [k + 1 for k in (*originals, *poetry)])
    return [{"n": i + 1, "text": lines[i] if i < len(lines) else "", "original": originals.get(i),
             "poetry": poetry.get(i)} for i in range(n)]


def detail_of(it: dict) -> dict:
    """What a thing is beyond its pictures: a story's sentences, the searches behind a set or a
    copied search, a label's origin, a LoRA's checkpoints, a board page's files."""
    import json

    kind = it["kind"]
    if kind == "stories":
        return {"scenes": story_scenes(it["path"])}
    if kind in _cat.BOARD_KINDS:
        try:
            from .. import boards as _boards
        except ImportError:                           # pragma: no cover
            import boards as _boards
        rec = _boards.load(it["id"].split(":", 1)[1]) or {}
        return {k: rec.get(k) for k in ("author", "brief", "concept", "dataset", "queries") if rec.get(k)}
    if kind == "searches":
        try:
            with open(os.path.join(it["path"], "search.json"), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            name = os.path.basename(it["path"])
            date, _, words = name.partition(" ")
            if len(date) == 10 and date[4] == "-" and words:
                return {"queries": [{"text": words, "dataset": "", "k": 0}], "date": date, "guessed": True}
            return {"queries": [{"text": name, "dataset": "", "k": 0}], "guessed": True}
    if kind == "labels":
        try:
            from .. import ip_labels as _labels
        except ImportError:                           # pragma: no cover
            import ip_labels as _labels
        row = _labels.load(it["id"].split(":", 1)[1]).public()
        return {"origin": row["origin"], "tightness": row["tightness"], "members": row["members"]}
    if kind == "loras":
        try:
            from .. import lora_train as _train
            from .. import loras as _loras
        except ImportError:                           # pragma: no cover
            import lora_train as _train
            import loras as _loras
        name = it["id"].split(":", 1)[1]
        run = next((r for r in _train.runs() if r["name"] == name), {"checkpoints": []})
        cps = run["checkpoints"]
        last = cps[-1]["name"] if cps else None
        return {"checkpoints": [{k: c.get(k) for k in ("name", "step", "sheet")} for c in cps],
                "trigger": cps[-1].get("trigger") if cps else None, "prompt": _loras.prompt(last) if last else None}
    if kind == "pages":
        return {"slug": it["id"].split(":", 1)[1], "files": it.get("files") or [], "author": it.get("author") or ""}
    return {}


@op("archive show", "read", ArchiveShowArgs, ArchiveShowResult,
    doc="""
    One thing in the archive. It comes with its name, where it lives on disk
    and the pictures it holds, in order.
    """,
    examples=("archive show collection:monsoon",))
def _archive_show(a: ArchiveShowArgs, ctx: Context) -> ArchiveShowResult:
    data = _cat.read()
    it = _cat.find(a.id)
    return ArchiveShowResult(id=it["id"], kind=it["kind"], name=data["names"].get(it["id"]) or it["name"],
                             path=it["path"], contents=contents_of(it), detail=detail_of(it))


class ArchiveRenameArgs(Args):
    id: str = _id_arg()
    name: str = positional(1, description="The new name.", json_schema_extra={"metavar": "NAME"})


@op("archive rename", "write", ArchiveRenameArgs, TextResult,
    doc="""
    Rename a thing. A collection or a film takes the new name in its own
    record. A file or a folder of output keeps its name on disk and shows the
    new one in the archive, so nothing that points at it breaks.
    """,
    examples=('archive rename collection:monsoon "Monsoon rains"',))
def _archive_rename(a: ArchiveRenameArgs, ctx: Context) -> TextResult:
    from .registry import call
    name = a.name.strip()
    if not name:
        raise ValueError("A name cannot be empty.")
    if a.id.startswith("collection:"):
        call("board set", {"board": a.id.split(":", 1)[1], "name": name}, ctx=ctx)
    elif a.id.startswith("film:"):
        import json
        try:
            from ..alchemy import film as _film
        except ImportError:                           # pragma: no cover
            from alchemy import film as _film
        old = _cat.find(a.id)["path"]
        with open(old, encoding="utf-8") as fh:
            spec = json.load(fh)
        new = _film.film_path(name)
        if os.path.abspath(new) != os.path.abspath(old) and os.path.exists(new):
            raise ValueError(f"A film called {name!r} is already kept.")
        ctx.tx.touch(old)
        ctx.tx.touch(new)
        spec["name"] = name
        os.remove(old)
        _film.save(spec)
    else:
        _cat.rename_display(ctx.tx, a.id, name)
    return TextResult(body=f"Renamed {a.id} to {name}.\n")


class ArchiveMoveArgs(Args):
    id: str = _id_arg()
    into: str | None = Field(None, description="A folder (folder:<id>), another collection for a collection, "
                                               "or kind:<kind> for the top of its kind.")


@op("archive move", "write", ArchiveMoveArgs, TextResult,
    doc="""
    Put a thing or a folder into a folder, or a collection into a collection.

    A thing goes into a folder of its own kind. A collection inside another is
    organisation only, and
    its pictures stay its own. Without --into it goes back to the top of its
    kind.
    """,
    examples=("archive move collection:monsoon --into kind:collections",))
def _archive_move(a: ArchiveMoveArgs, ctx: Context) -> TextResult:
    _cat.move(ctx.tx, a.id, a.into)
    return TextResult(body=f"Moved {a.id} into {a.into or 'the top of its kind'}.\n")


class ArchiveFolderArgs(Args):
    name: str = positional(0, description="The folder's name.", json_schema_extra={"metavar": "NAME"})
    kind: str = Field(description="The kind it holds: " + ", ".join(_cat.KINDS) + ".")
    into: str | None = Field(None, description="A folder to make it inside.")


class FolderResult(Result):
    id: str
    name: str

    def text(self) -> str:
        return f"Made the folder {self.name}  [{self.id}].\n"


@op("archive folder add", "write", ArchiveFolderArgs, FolderResult,
    doc="""
    Make a folder inside a kind, or inside another folder.
    """,
    examples=("archive folder add Rituals --kind collections",))
def _archive_folder_add(a: ArchiveFolderArgs, ctx: Context) -> FolderResult:
    fid = _cat.add_folder(ctx.tx, a.name, a.kind, a.into)
    return FolderResult(id=fid, name=a.name.strip())


class ArchiveMergeArgs(Args):
    id: str = _id_arg(0, "The collection or folder that goes.")
    into: str = Field(description="The collection or folder that takes everything.")


@op("archive merge", "write", ArchiveMergeArgs, TextResult,
    doc="""
    Merge two collections, or two folders. The first one's pictures, or its
    contents, go into the second, and the first goes. Undo takes the merge
    back.
    """,
    examples=("archive merge collection:monsoon --into collection:rains",), actors=("person", "cli"))
def _archive_merge(a: ArchiveMergeArgs, ctx: Context) -> TextResult:
    from .registry import call
    if a.id.startswith("collection:") and a.into.startswith("collection:"):
        if a.id == a.into:
            raise ValueError("Merge takes two different collections.")
        src = _cat.find(a.id)
        _cat.find(a.into)
        pictures = contents_of(src)
        if pictures:
            call("board add", {"board": a.into.split(":", 1)[1], "items": pictures}, ctx=ctx)
        data = _cat.read()
        for other, where in list(data["place"].items()):
            if where == a.id:
                data["place"][other] = a.into
        data["place"].pop(a.id, None)
        _cat.write(ctx.tx, data)
        call("board delete", {"board": a.id.split(":", 1)[1]}, ctx=ctx)
        return TextResult(body=f"Merged {len(pictures)} picture(s) of {a.id} into {a.into}, and {a.id} went.\n")
    if a.id.startswith("folder:") and a.into.startswith("folder:"):
        _cat.merge_folders(ctx.tx, a.id, a.into)
        return TextResult(body=f"Merged the folder {a.id} into {a.into}.\n")
    raise ValueError("Merge takes two collections or two folders.")


class ArchiveDeleteArgs(Args):
    id: str = _id_arg()


@op("archive delete", "write", ArchiveDeleteArgs, TextResult,
    doc="""
    Move a thing into the trash. Its files leave their folder, and archive
    restore brings them back. A folder of the archive goes only when empty.
    """,
    examples=("archive delete collection:monsoon",), actors=("person", "cli"), journal=False)
def _archive_delete(a: ArchiveDeleteArgs, ctx: Context) -> TextResult:
    gone = _cat.trash(a.id)
    if a.id.startswith("folder:"):                    # an empty folder holds no files, so it simply goes
        return TextResult(body=f"The folder {gone['name']} went.\n")
    return TextResult(body=f"{gone['name']} is in the trash. `archive restore {a.id}` brings it back.\n")


class ArchiveRestoreArgs(Args):
    id: str = _id_arg()


@op("archive restore", "write", ArchiveRestoreArgs, TextResult,
    doc="""
    Bring a thing back from the trash, to where it was.
    """,
    examples=("archive restore collection:monsoon",), actors=("person", "cli"), journal=False)
def _archive_restore(a: ArchiveRestoreArgs, ctx: Context) -> TextResult:
    back = _cat.restore(a.id)
    return TextResult(body=f"{back['name']} is back.\n")


class ArchiveEmptyArgs(Args):
    yes: bool = Field(False, description="Confirm: everything in the trash is deleted for good.")


@op("archive empty", "write", ArchiveEmptyArgs, TextResult,
    doc="""
    Delete everything in the trash for good. It takes --yes, since nothing
    brings it back.
    """,
    examples=("archive empty --yes",), actors=("person", "cli"), journal=False)
def _archive_empty(a: ArchiveEmptyArgs, ctx: Context) -> TextResult:
    if not a.yes:
        raise ValueError("Emptying the trash deletes for good. Run it again with --yes.")
    n = _cat.empty_trash()
    return TextResult(body=f"Deleted {n} thing{'s' if n != 1 else ''} for good.\n")
