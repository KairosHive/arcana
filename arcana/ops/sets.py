# sets.py -- pictures gathered by searches, for Alchemy's IP labels and LoRAs
#
# A set is a board (boards.py) whose kind is "set". It holds paths like any
# board, plus the searches that filled it: the words, the dataset, how many
# results were asked for and when. A board is curated by hand and shown in the
# Moodboard tab; a set is gathered in bulk, often hundreds of pictures, and is
# what an IP label is measured on and what a LoRA trains on.
#
# Saving to a set adds and skips twins, so one set can grow over several
# searches. Exporting copies its files into output/search/<folder>, for work
# outside Arcana (goofi, latent-lab); the set itself copies nothing.

from __future__ import annotations

import hashlib
import os
import shutil
import time

from pydantic import BaseModel, Field

from . import archive
from .board import BoardRow, add_item, author_for, load, new_record, put, row
from .registry import Args, Context, Result, op, positional

try:
    from .. import boards as _boards
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import boards as _boards
    import paths as _paths

MAX_ITEMS = 2000


def set_arg(index: int = 0):
    return positional(index, description="The set's id (its slug), as `set list` shows it.",
                      json_schema_extra={"complete": "board", "metavar": "SET"})


def load_set(slug: str) -> dict:
    rec = load(slug)
    if (rec.get("kind") or "board") != "set":
        raise ValueError(f"`{slug}` is a board, not a set. `board set {slug} --kind set` makes it one.")
    return rec


class SetListResult(Result):
    sets: list[BoardRow]

    def text(self) -> str:
        if not self.sets:
            return "No sets yet. `set save \"ice\" japan:1 japan:2 ...` starts one, as Save does in Search.\n"
        width = max(len(s.slug) for s in self.sets) + 2
        return "".join(f"{s.slug:<{width}}{s.name}  ·  {s.count} picture{'s' if s.count != 1 else ''}"
                       f"{f'  ·  {s.missing} missing' if s.missing else ''}\n" for s in self.sets)


class SetQuery(BaseModel):
    text: str = ""
    dataset: str = ""
    k: int = 0
    added: int = 0
    when: float = 0


class SetShowResult(Result):
    slug: str
    name: str
    items: list[str]
    missing: list[str] = Field(default_factory=list)
    queries: list[SetQuery] = Field(default_factory=list)

    def text(self) -> str:
        lines = [f"{self.name} ({self.slug}): {len(self.items)} pictures"
                 + (f", {len(self.missing)} missing" if self.missing else "")]
        for q in self.queries:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(q.when)) if q.when else ""
            lines.append(f"  searched \"{q.text}\" in {q.dataset or '?'}, added {q.added}  ·  {when}")
        return "\n".join(lines) + "\n"


class SetWriteResult(Result):
    set: BoardRow
    added: int = 0
    message: str

    def text(self) -> str:
        return self.message + "\n"


class SetListArgs(Args):
    pass


@op("set list", "read", SetListArgs, SetListResult,
    doc="""
    Every set, newest first. A set holds pictures gathered by searches, which
    Alchemy's IP labels and LoRAs read. Boards are listed by `board list`.
    """,
    examples=("set list",))
def _set_list(a: SetListArgs, ctx: Context) -> SetListResult:
    rows = []
    for b in _boards.listing():
        if b.get("kind") == "set":
            rec = _boards.load(b["slug"])
            if rec:
                rows.append(row(rec))
    return SetListResult(sets=rows)


class SetShowArgs(Args):
    set: str = set_arg()


@op("set show", "read", SetShowArgs, SetShowResult,
    doc="""
    A set's pictures, in the order they were added, and the searches that
    gathered them.
    """,
    examples=("set show ice",))
def _set_show(a: SetShowArgs, ctx: Context) -> SetShowResult:
    rec = load_set(a.set)
    here, gone = _boards.present(rec)
    return SetShowResult(slug=rec["slug"], name=rec.get("name", rec["slug"]), items=here, missing=gone,
                         queries=[SetQuery(**q) for q in rec.get("queries") or [] if isinstance(q, dict)])


class SetSaveArgs(Args):
    name: str = positional(0, description="The set's name, e.g. \"ice\".")
    items: list[str] = positional(1, min_length=1, max_length=MAX_ITEMS,
                                  description="Pictures: handles (japan:1423) or, for the person, paths.",
                                  json_schema_extra={"metavar": "ITEM"})
    id: str | None = Field(None, description="The set's id. Defaults to the name as a slug.")
    replace: bool = Field(False, description="Hold exactly these pictures. By default they are added, "
                          "and pictures already in the set are skipped.")
    query: str | None = Field(None, description="The words searched, kept with the set.")
    dataset: str | None = Field(None, description="The dataset searched.")
    k: int = Field(0, ge=0, description="How many results the search asked for.")


@op("set save", "write", SetSaveArgs, SetWriteResult,
    doc="""
    Put pictures in a set, making it when it is new. Pictures already there
    are skipped, so a set grows over several searches; --replace makes it hold
    exactly these. The search that found them (--query, --dataset) is kept with
    the set.
    """,
    examples=('set save ice japan:12 japan:7 japan:3 --query "ice" --dataset japan',))
def _set_save(a: SetSaveArgs, ctx: Context) -> SetWriteResult:
    slug = a.id or _boards.slugify(a.name)
    existing = _boards.load(slug)
    if existing is not None and (existing.get("kind") or "board") != "set":
        raise ValueError(f"`{slug}` is a board. Give the set another name, or `board set {slug} --kind set` "
                         f"to make that board a set.")
    rec = existing or new_record(a.name, slug, author_for(ctx.actor), kind="set", dataset=a.dataset)
    rec.setdefault("details", {})
    rec["kind"] = "set"
    if a.replace:
        rec["items"], rec["details"] = [], {}
    added = 0
    for ref in a.items:
        added += add_item(rec, archive.resolve_item(ref, ctx.actor))
    if a.query or a.dataset:
        rec.setdefault("queries", []).append({"text": a.query or "", "dataset": a.dataset or "", "k": a.k,
                                              "added": added, "when": time.time()})
    put(ctx, rec, creating=existing is None)
    skipped = len(a.items) - added
    msg = f"{added} picture{'s' if added != 1 else ''} added to the set \"{rec['name']}\""
    msg += f" ({skipped} already there)" if skipped else ""
    return SetWriteResult(set=row(rec), added=added, message=msg + f". It holds {len(rec['items'])}.")


class SetExportArgs(Args):
    set: str = set_arg()
    folder: str | None = Field(None, description="The folder under output/search. Defaults to the set's id.")


class SetExportResult(Result):
    folder: str
    copied: int

    def text(self) -> str:
        return f"{self.copied} files copied to {self.folder}.\n"


@op("set export", "effect", SetExportArgs, SetExportResult,
    doc="""
    Copy a set's files into output/search/<folder>, for work outside Arcana.
    Each file is named by a hash of its path, so twins from different folders
    stay apart. The set itself is unchanged.
    """,
    examples=("set export ice", "set export ice --folder ice-for-goofi"))
def _set_export(a: SetExportArgs, ctx: Context) -> SetExportResult:
    rec = load_set(a.set)
    folder = (a.folder or rec["slug"]).strip()
    target = _paths.safe_join(_paths.subdir("output"), "search", *[p for p in folder.replace("\\", "/").split("/") if p])
    if target is None or ".." in folder:
        raise ValueError("That folder name is not allowed. Use a plain name without drive letters or '..'.")
    _paths.ensure_dir(target)
    here, _ = _boards.present(rec)
    import json
    with open(os.path.join(target, "search.json"), "w", encoding="utf-8") as fh:
        json.dump({"set": rec["slug"], "name": rec.get("name", rec["slug"]), "queries": rec.get("queries") or [],
                   "exported": time.time()}, fh, indent=1)
    n = 0
    for path in here:
        name = f"{hashlib.md5(path.encode('utf-8')).hexdigest()[:8]}_{os.path.basename(path)}"
        dest = os.path.join(target, name)
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
        n += 1
    return SetExportResult(folder=target, copied=n)
