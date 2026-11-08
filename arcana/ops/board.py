# board.py -- moodboards as ops
#
# A board is a JSON file in boards/ (boards.py, schema 2): a list of picture
# paths, details per path (handle, section, role, note, provenance), sections
# with a title, a rationale and the queries behind them, a palette, a brief and
# a concept. Every op here that changes one runs in a state.Transaction, so a
# batch of them is one undo step, and an agent can change only boards an agent
# made (A7, enforced in state.py).
#
# Pictures are named the way the rest of the ops name them: a handle
# (japan:1423), a derived picture's ledger handle (out:3f9c1e), or, inside a
# board, its number from `board show` (#3). The person's app may also pass
# paths, since its collection is a list of paths.

from __future__ import annotations

import os
import shutil
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import archive
from .registry import Args, Context, Result, op, positional

try:
    from .. import boards as _boards
    from .. import core
    from .. import paths as _paths
    from .. import state as _state
except ImportError:                                   # pragma: no cover
    import boards as _boards
    import core
    import paths as _paths
    import state as _state

Role = Literal["hero", "support", "texture", "detail", "counterpoint"]
NO_SECTION = "none"
MAX_DERIVED_SHARE = 1 / 3
MAX_GENERATED = 2


def slug_arg(index: int = 0):
    return positional(index, description="The board's id (its slug), as `board list` shows it.",
                      json_schema_extra={"complete": "board", "metavar": "BOARD"})


def items_arg(index: int, lo: int = 1, hi: int = 96):
    return positional(index, min_length=lo, max_length=hi,
                      description="Pictures: handles (japan:1423), derived pictures (out:3f9c1e), "
                                  "or numbers from `board show` (#3).",
                      json_schema_extra={"metavar": "ITEM"})


# ─────────────────────────── records ───────────────────────────

def _known() -> str:
    names = [b["slug"] for b in _boards.listing()]
    return ", ".join(names) if names else "none yet"


def load(slug: str) -> dict:
    rec = _boards.load(slug)
    if rec is None:
        raise ValueError(f"No board called `{slug}`. Boards: {_known()}.")
    rec.setdefault("details", {})
    rec.setdefault("sections", [])
    return rec


def put(ctx: Context, rec: dict, *, creating: bool = False, tx=None) -> dict:
    """Write a board inside the open transaction (or `tx`, for a job recording its result)."""
    path = (tx or ctx.tx).touch_board(rec["slug"], creating=creating)
    rec["schema"] = _boards.SCHEMA
    rec["updated"] = time.time()
    _state.write_json(path, rec)
    return rec


def new_record(name: str, slug: str, author: str, **fields) -> dict:
    now = time.time()
    rec = {"schema": _boards.SCHEMA, "name": name, "slug": slug, "created": now, "updated": now,
           "dataset": fields.pop("dataset", None) or "", "reference": "", "transfer": "",
           "items": [], "details": {}, "author": author, "sections": []}
    rec.update({k: v for k, v in fields.items() if v is not None})
    return rec


def author_for(actor: str) -> str:
    return "agent" if actor == "agent" else "person"


def find(rec: dict, ref: str, actor: str) -> str:
    """The path of a picture already on the board, by number, handle or path."""
    items = _boards.item_paths(rec)
    ref = ref.strip()
    if ref.startswith("#") and ref[1:].isdigit():
        n = int(ref[1:])
        if 1 <= n <= len(items):
            return items[n - 1]
        raise ValueError(f"`{rec['slug']}` has {len(items)} pictures, so there is no {ref}.")
    for p in items:
        if (rec["details"].get(p) or {}).get("handle") == ref:
            return p
    if archive.is_path(ref) and actor != "agent" and ref in items:
        return ref
    try:
        item = archive.resolve_item(ref, actor)
    except ValueError:
        item = None
    if item is not None:
        norm = os.path.normcase(os.path.abspath(item.path))
        for p in items:
            if os.path.normcase(os.path.abspath(p)) == norm:
                return p
    raise ValueError(f"{ref} is not on `{rec['slug']}`. `board show {rec['slug']}` numbers "
                     f"its pictures.")


def section_ids(rec: dict) -> list[str]:
    return [s.get("id") for s in rec.get("sections") or []]


def check_section(rec: dict, sid: str | None) -> None:
    if sid and sid != NO_SECTION and sid not in section_ids(rec):
        have = ", ".join(section_ids(rec)) or "none yet"
        raise ValueError(f"`{rec['slug']}` has no section `{sid}`. Sections: {have}. "
                         f"Add one with `board section add {rec['slug']} --id {sid} --title ...`.")


def add_item(rec: dict, item: archive.Item, **details) -> bool:
    """Put a resolved picture on a record. False when it was already there."""
    there = item.path in rec["items"]
    if not there:
        rec["items"].append(item.path)
    d = rec["details"].setdefault(item.path, {})
    if item.handle:
        d["handle"] = item.handle
    if item.kind in ("derived", "generated"):
        d["kind"] = item.kind
        if item.derived_from:
            d["derived_from"] = item.derived_from
    for k, v in details.items():
        if v is None:
            continue
        if k == "section" and v == NO_SECTION:
            d.pop("section", None)
        else:
            d[k] = v
    return not there


# ─────────────────────────── results ───────────────────────────

class BoardRow(BaseModel):
    slug: str
    name: str
    author: str
    count: int
    missing: int = 0
    sections: int = 0
    updated: float = 0
    dataset: str = ""


class BoardListResult(Result):
    boards: list[BoardRow]

    def text(self) -> str:
        if not self.boards:
            return "No boards yet. `board create \"A name\"` starts one.\n"
        width = max(len(b.slug) for b in self.boards) + 2
        lines = []
        for b in self.boards:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(b.updated)) if b.updated else ""
            bits = [f"{b.count} picture{'s' if b.count != 1 else ''}"]
            if b.sections:
                bits.append(f"{b.sections} section{'s' if b.sections != 1 else ''}")
            if b.missing:
                bits.append(f"{b.missing} missing")
            lines.append(f"{b.slug:<{width}}{b.name}  ·  by {b.author}  ·  {', '.join(bits)}  ·  {when}")
        return "\n".join(lines) + "\n"


def row(rec: dict) -> BoardRow:
    here, gone = _boards.present(rec)
    return BoardRow(slug=rec["slug"], name=rec.get("name", rec["slug"]),
                    author=rec.get("author", "person"), count=len(_boards.item_paths(rec)),
                    missing=len(gone), sections=len(rec.get("sections") or []),
                    updated=float(rec.get("updated") or 0), dataset=rec.get("dataset") or "")


class BoardWriteResult(Result):
    board: BoardRow
    message: str
    handles: list[str] = Field(default_factory=list)
    deleted: bool = False

    def text(self) -> str:
        if self.deleted:
            return self.message + " `undo` brings it back.\n"
        b = self.board
        return (f"{self.message}\n`{b.slug}` now holds {b.count} picture{'s' if b.count != 1 else ''}"
                f" in {b.sections} section{'s' if b.sections != 1 else ''}.\n")


class BoardItem(BaseModel):
    num: int
    path: str
    file: str
    handle: str | None = None
    section: str | None = None
    role: str | None = None
    note: str | None = None
    kind: str = "archive"
    derived_from: str | None = None
    missing: bool = False

    def line(self) -> str:
        bits = [f"#{self.num}", self.handle or "(no handle)"]
        if self.role:
            bits.append(self.role)
        if self.kind in ("derived", "generated"):
            bits.append(self.kind.upper() + (f" from {self.derived_from}" if self.derived_from else ""))
        bits.append(self.file)
        if self.missing:
            bits.append("(file missing)")
        out = "  ".join(bits)
        if self.note:
            out += f"\n      {self.note}"
        return out


class SectionView(BaseModel):
    id: str
    title: str = ""
    rationale: str = ""
    queries: list[str] = Field(default_factory=list)
    items: list[BoardItem] = Field(default_factory=list)


class Metrics(BaseModel):
    count: int = 0
    heroes: list[int] = Field(default_factory=list)
    near_duplicates: list[tuple[int, int]] = Field(default_factory=list)
    diversity: float | None = None
    palette_spread: float | None = None
    derived: int = 0
    generated: int = 0
    missing: int = 0
    section_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def line(self) -> str:
        bits = []
        if self.diversity is not None:
            bits.append(f"diversity {self.diversity:.2f}")
        bits.append("near-duplicate pairs " + (", ".join(f"#{a}/#{b}" for a, b in self.near_duplicates)
                                               if self.near_duplicates else "none"))
        if self.palette_spread is not None:
            bits.append(f"palette spread {self.palette_spread:.0f}")
        if self.count:
            share = (self.derived + self.generated) / self.count
            bits.append(f"derived {self.derived + self.generated} of {self.count} ({share:.0%})")
        bits.append(f"heroes {len(self.heroes)}")
        return " · ".join(bits)


class PaletteEntry(BaseModel):
    hex: str
    weight: float | None = None
    role: str | None = None


class BoardShowResult(Result):
    slug: str
    name: str
    author: str
    brief: str = ""
    concept: str = ""
    dataset: str = ""
    palette: list[PaletteEntry] = Field(default_factory=list)
    sections: list[SectionView] = Field(default_factory=list)
    unsectioned: list[BoardItem] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    reference: str = ""
    transfer: str = ""
    page: str | None = None            # the board's own designed page, when it has one

    @property
    def items(self) -> list[BoardItem]:
        out = [i for s in self.sections for i in s.items] + list(self.unsectioned)
        return sorted(out, key=lambda i: i.num)

    def text(self) -> str:
        m = self.metrics
        head = (f"{self.slug} · \"{self.name}\" · by {self.author} · {m.count} picture"
                f"{'s' if m.count != 1 else ''} · {len(self.sections)} section"
                f"{'s' if len(self.sections) != 1 else ''}")
        lines = [head]
        if self.brief:
            lines.append(f"Brief: {self.brief}")
        if self.concept:
            lines.append(f"Concept: {self.concept}")
        if self.palette:
            lines.append("Palette: " + " · ".join(
                p.hex + (f" {p.weight:.0%}" if p.weight is not None else "")
                + (f" {p.role}" if p.role else "") for p in self.palette))
        for s in self.sections:
            lines.append(f"\n## {s.id} {s.title} ({len(s.items)})")
            if s.rationale:
                lines.append(f"Why: {s.rationale}")
            if s.queries:
                lines.append("Queries: " + "; ".join(s.queries))
            lines += [i.line() for i in s.items]
        if self.unsectioned:
            lines.append(f"\n## Unsectioned ({len(self.unsectioned)})" if self.sections else "")
            lines += [i.line() for i in self.unsectioned]
        if self.page:
            lines.append(f"\nPage: {self.page} (its own designed page; `board pdf` prints it)")
        lines.append("\nMetrics: " + m.line())
        if m.warnings:
            lines.append("Look at: " + " ".join(m.warnings))
        return "\n".join(l for l in lines if l is not None).replace("\n\n\n", "\n\n") + "\n"


# ─────────────────────────── metrics ───────────────────────────

def _hex_to_lab(hx: str) -> np.ndarray:
    h = hx.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def metrics(items: list[BoardItem], sections: list[SectionView]) -> Metrics:
    """
    What the board-design skill reads a board by: one hero, no near-duplicates,
    variety across the pictures, how far the colours spread, and how much of
    the board was made rather than found.
    """
    m = Metrics(count=len(items))
    m.heroes = [i.num for i in items if i.role == "hero"]
    m.derived = sum(1 for i in items if i.kind == "derived")
    m.generated = sum(1 for i in items if i.kind == "generated")
    m.missing = sum(1 for i in items if i.missing)
    m.section_counts = {s.id: len(s.items) for s in sections}

    by_ds: dict = {}
    for i in items:
        if i.handle and not i.handle.startswith("out:"):
            ds_id, _, key = i.handle.rpartition(":")
            if key.isdigit():
                by_ds.setdefault(ds_id, []).append((i.num, int(key)))
    dists, tops = [], []
    for ds_id, pairs in by_ds.items():
        try:
            ds = archive.open_dataset(ds_id)
        except ValueError:
            continue
        pairs = [(n, k) for n, k in pairs if k in ds.idx2path]
        if not pairs:
            continue
        vecs = core.unit_rows(np.stack([ds.index.get(k) for _, k in pairs]))
        sims = vecs @ vecs.T
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                dists.append(1.0 - float(sims[a, b]))
                if sims[a, b] >= core.NEAR_DUPLICATE:
                    m.near_duplicates.append((pairs[a][0], pairs[b][0]))
        pal = archive.palettes(ds, [k for _, k in pairs], n=1)
        tops += [_hex_to_lab(pal[k][0]) for _, k in pairs if pal.get(k)]
    if dists:
        m.diversity = float(np.mean(dists))
    if len(tops) > 1:
        t = np.stack(tops)
        d = np.sqrt(((t[:, None, :] - t[None, :, :]) ** 2).sum(-1))
        m.palette_spread = float(d[np.triu_indices(len(t), 1)].mean())

    w = m.warnings
    if items and not m.heroes:
        w.append("No hero yet: give one picture --role hero.")
    elif len(m.heroes) > 1:
        w.append(f"{len(m.heroes)} heroes ({', '.join(f'#{n}' for n in m.heroes)}), so none leads.")
    for a, b in m.near_duplicates:
        w.append(f"#{a} and #{b} are near-duplicates. Keep one.")
    if items and (m.derived + m.generated) / len(items) > MAX_DERIVED_SHARE:
        w.append(f"{m.derived + m.generated} of {len(items)} pictures are derived, more than a third.")
    if m.generated > MAX_GENERATED:
        w.append(f"{m.generated} generated pictures, more than two.")
    for sid, n in m.section_counts.items():
        if n == 0:
            w.append(f"Section {sid} is empty.")
    if m.missing:
        w.append(f"{m.missing} file{'s are' if m.missing != 1 else ' is'} missing.")
    return m


def view(rec: dict) -> BoardShowResult:
    items = _boards.item_paths(rec)
    details = rec.get("details") or {}
    built = []
    for n, p in enumerate(items, 1):
        d = details.get(p) or {}
        h = d.get("handle") or archive.handle_for_path(p, prefer=rec.get("dataset") or None)
        kind = d.get("kind") or ("derived" if d.get("derived_from") else "archive")
        if kind == "archive" and h is None:
            kind = "file"
        built.append(BoardItem(num=n, path=p, file=os.path.basename(p), handle=h,
                               section=d.get("section"), role=d.get("role"), note=d.get("note"),
                               kind=kind, derived_from=d.get("derived_from"),
                               missing=not os.path.exists(p)))
    sections = []
    for s in rec.get("sections") or []:
        sections.append(SectionView(id=s.get("id", ""), title=s.get("title", ""),
                                    rationale=s.get("rationale", ""),
                                    queries=list(s.get("queries") or []),
                                    items=[i for i in built if i.section == s.get("id")]))
    known = {s.id for s in sections}
    loose = [i for i in built if i.section not in known]
    pal = []
    for e in rec.get("palette") or []:
        if isinstance(e, dict) and e.get("hex"):
            pal.append(PaletteEntry(hex=e["hex"], weight=e.get("weight"), role=e.get("role")))
        elif isinstance(e, str):
            pal.append(PaletteEntry(hex=e))
    return BoardShowResult(slug=rec["slug"], name=rec.get("name", rec["slug"]),
                           author=rec.get("author", "person"), brief=rec.get("brief") or "",
                           concept=rec.get("concept") or "", dataset=rec.get("dataset") or "",
                           palette=pal, sections=sections, unsectioned=loose,
                           metrics=metrics(built, sections), reference=rec.get("reference") or "",
                           transfer=rec.get("transfer") or "",
                           page=page_path(rec["slug"]) if rec.get("page") else None)


def write_result(rec: dict, message: str, handles=()) -> BoardWriteResult:
    return BoardWriteResult(board=row(rec), message=message, handles=list(handles))


# ─────────────────────────── reads ───────────────────────────

class BoardListArgs(Args):
    pass


@op("board list", "read", BoardListArgs, BoardListResult,
    doc="""
    Every board, newest first. Each line gives the id, the name, who made it
    (person or agent) and how many pictures it holds.
    """,
    examples=("board list",))
def _board_list(a: BoardListArgs, ctx: Context) -> BoardListResult:
    rows = []
    for b in _boards.listing():
        rec = _boards.load(b["slug"])
        if rec:
            rows.append(row(rec))
    return BoardListResult(boards=rows)


class BoardShowArgs(Args):
    board: str = slug_arg()
    sheet: bool = Field(False, description="Add a numbered contact sheet of the board's pictures.")


@op("board show", "read", BoardShowArgs, BoardShowResult,
    doc="""
    A board in full, grouped by section, with metrics. Every picture is
    numbered, and other board ops accept those numbers (#3). The metrics are
    the ones the board-design skill reads: one hero, no near-duplicate pairs,
    diversity across the pictures, palette spread, and the share of derived
    pictures, with a line on anything to look at.
    """,
    examples=("board show monsoon", "board show monsoon --sheet"))
def _board_show(a: BoardShowArgs, ctx: Context) -> BoardShowResult:
    out = view(load(a.board))
    if a.sheet and ctx.pixels:
        items = out.items
        if items:
            out.attachments.append(archive.sheet([i.path for i in items], [i.num for i in items],
                                                 caption=f"board {a.board}", title=out.name))
    return out


# ─────────────────────────── making boards ───────────────────────────

class BoardCreateArgs(Args):
    name: str = positional(0, description="What the board is called, e.g. \"Monsoon melancholy\".")
    id: str | None = Field(None, description="The board's id. Defaults to the name as a slug.")
    brief: str | None = Field(None, description="The words that were asked for.")
    concept: str | None = Field(None, description="One paragraph on what the board argues.")
    dataset: str | None = Field(None, description="The dataset it draws on, e.g. japan.")


@op("board create", "write", BoardCreateArgs, BoardWriteResult,
    doc="""
    Start an empty board. An agent's board is marked as made by an agent, and
    only boards made by an agent can be changed by one.
    """,
    examples=('board create "Monsoon melancholy" --id monsoon --brief "rain and waiting"',))
def _board_create(a: BoardCreateArgs, ctx: Context) -> BoardWriteResult:
    slug = a.id or _boards.slugify(a.name)
    if slug != _boards.slugify(slug):
        raise ValueError(f"`{slug}` cannot be a board id. Try `{_boards.slugify(slug)}`.")
    rec = new_record(a.name, slug, author_for(ctx.actor), brief=a.brief, concept=a.concept,
                     dataset=a.dataset)
    put(ctx, rec, creating=True)
    return write_result(rec, f"Created `{slug}`, \"{a.name}\".")


class BoardCopyArgs(Args):
    board: str = slug_arg()
    id: str = Field(description="The copy's id.")
    name: str | None = Field(None, description="The copy's name. Defaults to the original's.")


@op("board copy", "write", BoardCopyArgs, BoardWriteResult,
    doc="""
    Copy a board under a new id, made by whoever copies it. This is how an
    agent works on a board the person made: it changes the copy and leaves the
    original as the person left it.
    """,
    examples=("board copy monsoon --id monsoon-agent",))
def _board_copy(a: BoardCopyArgs, ctx: Context) -> BoardWriteResult:
    src = load(a.board)
    rec = dict(src)
    rec.update(slug=a.id, name=a.name or src.get("name", a.board), author=author_for(ctx.actor),
               created=time.time(), details={p: dict(d) for p, d in (src.get("details") or {}).items()},
               items=list(src.get("items") or []), copied_from=a.board)
    put(ctx, rec, creating=True)
    return write_result(rec, f"Copied `{a.board}` to `{a.id}`.")


class BoardSaveArgs(Args):
    name: str = positional(0, description="The board's name.")
    items: list[str] = items_arg(1, 0)
    id: str | None = Field(None, description="The board's id. Defaults to the name as a slug.")
    reference: str | None = Field(None, description="The picture marked [R] for colour transfer.")
    transfer: str | None = Field(None, description="The picture marked [T].")
    dataset: str | None = Field(None, description="The dataset the pictures came from.")


@op("board save", "write", BoardSaveArgs, BoardWriteResult,
    doc="""
    Make a board hold exactly these pictures. They go in the order given, and
    the board is created if needed. It is how the Moodboard tab saves its
    collection. Notes, roles and
    sections of pictures that stay on the board are kept, and so is every
    field the caller leaves out.
    """,
    examples=('board save "Monsoon melancholy" japan:12 japan:7 japan:3 --id monsoon',))
def _board_save(a: BoardSaveArgs, ctx: Context) -> BoardWriteResult:
    slug = a.id or _boards.slugify(a.name)
    existing = _boards.load(slug)
    rec = existing or new_record(a.name, slug, author_for(ctx.actor))
    rec.setdefault("details", {})
    rec["name"] = a.name
    resolved = [archive.resolve_item(r, ctx.actor) for r in a.items]
    old = rec["details"]
    rec["items"], rec["details"] = [], {}
    for it in resolved:
        rec["details"][it.path] = dict(old.get(it.path) or {})
        add_item(rec, it)
    for field_name in ("reference", "transfer", "dataset"):
        v = getattr(a, field_name)
        if v is not None:
            rec[field_name] = archive.resolve_item(v, ctx.actor).path if v and field_name != "dataset" else v
    put(ctx, rec, creating=existing is None)
    return write_result(rec, f"Saved `{slug}`, \"{a.name}\".", [i.handle or i.path for i in resolved])


class BoardAddArgs(Args):
    board: str = slug_arg()
    items: list[str] = items_arg(1)
    section: str | None = Field(None, description="The section to place them in, by id.")
    role: Role | None = Field(None, description="hero, support, texture, detail or counterpoint.")
    note: str | None = Field(None, description="Why they are here.")


@op("board add", "write", BoardAddArgs, BoardWriteResult,
    doc="""
    Put pictures on a board, with a section, a role and a note. A picture
    already there keeps its place and takes the new section, role or
    note. Derived pictures (out:) are marked as derived on the board.
    """,
    examples=("board add monsoon japan:12 japan:7 --role support",
              'board add monsoon japan:3 --section s1 --role hero --note "the one the rest answer"'))
def _board_add(a: BoardAddArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    check_section(rec, a.section)
    resolved = [archive.resolve_item(r, ctx.actor) for r in a.items]
    added = [it for it in resolved
             if add_item(rec, it, section=a.section, role=a.role, note=a.note)]
    put(ctx, rec)
    names = ", ".join(i.handle or i.ref for i in resolved)
    where = f" to {a.section}" if a.section and a.section != NO_SECTION else ""
    msg = f"Added {names}{where}." if len(added) == len(resolved) else \
        f"Added {len(added)} new of {len(resolved)} ({names}){where}. The rest were already there."
    return write_result(rec, msg, [i.handle or i.ref for i in resolved])


class BoardRemoveArgs(Args):
    board: str = slug_arg()
    items: list[str] = items_arg(1)


@op("board remove", "write", BoardRemoveArgs, BoardWriteResult,
    doc="""
    Take pictures off a board, with whatever the board said about them. The
    files are untouched.
    """,
    examples=("board remove monsoon japan:7", "board remove monsoon #2 #5"))
def _board_remove(a: BoardRemoveArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    gone = [find(rec, r, ctx.actor) for r in a.items]
    for p in gone:
        if p in rec["items"]:
            rec["items"].remove(p)
        rec["details"].pop(p, None)
    put(ctx, rec)
    return write_result(rec, f"Removed {len(set(gone))} picture{'s' if len(set(gone)) != 1 else ''}.")


class BoardMoveArgs(Args):
    board: str = slug_arg()
    item: str = positional(1, description="The picture: handle, out: handle or #number.",
                           json_schema_extra={"metavar": "ITEM"})
    section: str | None = Field(None, description="The section to move it to, or none.")
    before: str | None = Field(None, description="Place it before this picture.",
                               json_schema_extra={"metavar": "ITEM"})
    to_end: bool = Field(False, description="Place it last.")


@op("board move", "write", BoardMoveArgs, BoardWriteResult,
    doc="""
    Move a picture to another section or another place in the order. --section none takes it out of every section.
    """,
    examples=("board move monsoon japan:7 --section s2", "board move monsoon #4 --before #1"))
def _board_move(a: BoardMoveArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    p = find(rec, a.item, ctx.actor)
    if a.section is not None:
        check_section(rec, a.section)
        d = rec["details"].setdefault(p, {})
        if a.section == NO_SECTION:
            d.pop("section", None)
        else:
            d["section"] = a.section
    if a.before is not None or a.to_end:
        target = find(rec, a.before, ctx.actor) if a.before else None
        rec["items"].remove(p)
        rec["items"].insert(rec["items"].index(target) if target else len(rec["items"]), p)
    put(ctx, rec)
    return write_result(rec, f"Moved {a.item}.")


class BoardNoteArgs(Args):
    board: str = slug_arg()
    item: str = positional(1, description="The picture: handle, out: handle or #number.",
                           json_schema_extra={"metavar": "ITEM"})
    text: str | None = Field(None, description="Why it is here. An empty string clears it.")
    role: Role | None = Field(None, description="hero, support, texture, detail or counterpoint.")


@op("board note", "write", BoardNoteArgs, BoardWriteResult,
    doc="""
    Say why a picture is on the board, or what it does there. Sets its note,
    its role, or both.
    """,
    examples=('board note monsoon japan:3 --text "the weight of water, seen from under it"',
              "board note monsoon #2 --role texture"))
def _board_note(a: BoardNoteArgs, ctx: Context) -> BoardWriteResult:
    if a.text is None and a.role is None:
        raise ValueError("Give --text, --role, or both.")
    rec = load(a.board)
    p = find(rec, a.item, ctx.actor)
    d = rec["details"].setdefault(p, {})
    if a.text is not None:
        if a.text:
            d["note"] = a.text
        else:
            d.pop("note", None)
    if a.role is not None:
        d["role"] = a.role
    put(ctx, rec)
    return write_result(rec, f"Noted {a.item}.")


class BoardSetArgs(Args):
    board: str = slug_arg()
    name: str | None = Field(None, description="A new name.")
    brief: str | None = Field(None, description="The words that were asked for.")
    concept: str | None = Field(None, description="One paragraph on what the board argues.")
    dataset: str | None = Field(None, description="The dataset it draws on.")


@op("board set", "write", BoardSetArgs, BoardWriteResult,
    doc="""
    Change a board's own fields. They are its name, the brief it answers, its
    concept, and the dataset it draws on.
    """,
    examples=('board set monsoon --concept "Rain as a weight the city carries"',))
def _board_set(a: BoardSetArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    changed = [f for f in ("name", "brief", "concept", "dataset") if getattr(a, f) is not None]
    if not changed:
        raise ValueError("Give at least one of --name, --brief, --concept, --dataset.")
    for f in changed:
        rec[f] = getattr(a, f)
    put(ctx, rec)
    return write_result(rec, f"Set {', '.join(changed)} on `{a.board}`.")


# ─────────────────────────── sections ───────────────────────────

class SectionAddArgs(Args):
    board: str = slug_arg()
    title: str = Field(description="What the section is about, e.g. \"Water as weight\".")
    id: str | None = Field(None, description="The section's id. Defaults to s1, s2, ...")
    rationale: str | None = Field(None, description="Why the board needs this angle.")
    query: list[str] = Field(default_factory=list,
                             description="A search phrase behind the section. Repeat for several.")


@op("board section add", "write", SectionAddArgs, BoardWriteResult,
    doc="""
    Add a section, one angle on the theme. It has a title, the reason it is
    there and the queries that found its pictures.
    """,
    examples=('board section add monsoon --id s3 --title "Water as weight" '
              '--rationale "rain as something carried" --query "heavy rain on a dark street"',))
def _section_add(a: SectionAddArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    ids = section_ids(rec)
    sid = a.id or next(f"s{n}" for n in range(1, 1000) if f"s{n}" not in ids)
    if sid in ids:
        raise ValueError(f"`{a.board}` already has a section `{sid}`. "
                         f"Change it with `board section set {a.board} {sid} ...`.")
    if sid == NO_SECTION:
        raise ValueError(f"`{NO_SECTION}` is reserved. Pick another id.")
    rec["sections"].append({"id": sid, "title": a.title, "rationale": a.rationale or "",
                            "queries": list(a.query)})
    put(ctx, rec)
    return write_result(rec, f"Added section {sid}, \"{a.title}\".")


class SectionSetArgs(Args):
    board: str = slug_arg()
    section: str = positional(1, description="The section's id.")
    title: str | None = Field(None, description="A new title.")
    rationale: str | None = Field(None, description="A new rationale.")
    query: list[str] | None = Field(None, description="Replace the queries. Repeat for several.")


@op("board section set", "write", SectionSetArgs, BoardWriteResult,
    doc="Change a section's title, rationale or queries.",
    examples=('board section set monsoon s1 --title "Water, carried"',))
def _section_set(a: SectionSetArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    check_section(rec, a.section)
    s = next(s for s in rec["sections"] if s.get("id") == a.section)
    if a.title is not None:
        s["title"] = a.title
    if a.rationale is not None:
        s["rationale"] = a.rationale
    if a.query is not None:
        s["queries"] = list(a.query)
    put(ctx, rec)
    return write_result(rec, f"Updated section {a.section}.")


class SectionRemoveArgs(Args):
    board: str = slug_arg()
    section: str = positional(1, description="The section's id.")


@op("board section remove", "write", SectionRemoveArgs, BoardWriteResult,
    doc="""
    Remove a section. Its pictures stay on the board, outside any section.
    """,
    examples=("board section remove monsoon s2",))
def _section_remove(a: SectionRemoveArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    check_section(rec, a.section)
    rec["sections"] = [s for s in rec["sections"] if s.get("id") != a.section]
    for d in rec["details"].values():
        if d.get("section") == a.section:
            d.pop("section", None)
    put(ctx, rec)
    return write_result(rec, f"Removed section {a.section}. Its pictures stay on the board.")


# ─────────────────────────── palette ───────────────────────────

def _palette_entry(raw: str) -> dict:
    parts = raw.strip().split(":")
    h = parts[0].strip().lstrip("#").lower()
    if len(h) != 6 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"{raw!r} is not a colour. Write #1f2a36, #1f2a36:0.3 or #1f2a36:0.3:ground.")
    entry = {"hex": "#" + h}
    if len(parts) > 1 and parts[1]:
        try:
            entry["weight"] = float(parts[1])
        except ValueError:
            raise ValueError(f"{raw!r} has a weight that is not a number.") from None
    if len(parts) > 2 and parts[2]:
        entry["role"] = parts[2]
    return entry


class PaletteSetArgs(Args):
    board: str = slug_arg()
    colour: list[str] = Field(min_length=1, max_length=12,
                              description="A colour as #hex, with an optional weight and role: "
                                          "#1f2a36:0.3:ground. Repeat for each colour.",
                              json_schema_extra={"metavar": "HEX"})


@op("board palette set", "write", PaletteSetArgs, BoardWriteResult,
    doc="""
    Give a board a palette you choose. Roles such as ground, secondary and
    accent say how much of the board each colour should hold.
    """,
    examples=("board palette set monsoon --colour '#1f2a36:0.5:ground' --colour '#b5651d:0.1:accent'",))
def _palette_set(a: PaletteSetArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    rec["palette"] = [_palette_entry(c) for c in a.colour]
    put(ctx, rec)
    return write_result(rec, "Palette: " + " ".join(e["hex"] for e in rec["palette"]) + ".")


def extract_palette(paths: list[str], n: int = 5) -> list[dict]:
    """
    A board's palette from its pictures: each picture's own strip, merged.

    Colours closer than 12 in LAB are one colour, weighted by how much of each
    picture they cover, so a colour every picture shares outweighs one that
    fills a single frame.
    """
    try:
        from .. import refselect as _refselect
    except ImportError:                               # pragma: no cover
        import refselect as _refselect
    from PIL import Image
    pool: list[list] = []        # [lab, weight, hex]
    used = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                im.thumbnail((256, 256))
                strip = _refselect.palette_strip(im.convert("RGB"), n=6)
        except Exception:
            continue
        used += 1
        for hx, share in strip:
            lab = _hex_to_lab(hx)
            for c in pool:
                if np.linalg.norm(c[0] - lab) < 12:
                    w = c[1] + share
                    c[0] = (c[0] * c[1] + lab * share) / w
                    c[1] = w
                    break
            else:
                pool.append([lab, float(share), hx])
    if not used:
        raise ValueError("None of the board's pictures could be read, so there is no palette to take.")
    pool.sort(key=lambda c: -c[1])
    top = pool[:n]
    total = sum(c[1] for c in top) or 1.0
    roles = ["ground", "secondary"] + ["accent"] * n
    return [{"hex": c[2], "weight": round(c[1] / total, 3), "role": roles[i]}
            for i, c in enumerate(top)]


class PaletteExtractArgs(Args):
    board: str = slug_arg()
    n: int = Field(5, ge=2, le=12, description="Colours to keep.")


@op("board palette extract", "write", PaletteExtractArgs, BoardWriteResult,
    doc="""
    Take the board's palette from its own pictures. Each picture's dominant
    colours are merged across the board and weighted by how much of each
    picture they cover.
    """,
    examples=("board palette extract monsoon --n 5",))
def _palette_extract(a: PaletteExtractArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    here, _ = _boards.present(rec)
    if not here:
        raise ValueError(f"`{a.board}` has no pictures on disk to take a palette from.")
    rec["palette"] = extract_palette(here, a.n)
    put(ctx, rec)
    return write_result(rec, "Palette: " + " ".join(
        f"{e['hex']} {e['weight']:.0%}" for e in rec["palette"]) + ".")


# ─────────────────────────── export and delete ───────────────────────────

class ExportResult(Result):
    folder: str
    copied: int
    missing: list[str] = Field(default_factory=list)

    def text(self) -> str:
        out = f"Copied {self.copied} picture{'s' if self.copied != 1 else ''} to {self.folder}."
        if self.missing:
            out += f" {len(self.missing)} could not be found."
        return out + "\n"


def export_files(paths: list[str], kind: str, folder: str) -> ExportResult:
    """Copy pictures into output/<kind>/<folder>. The originals stay where they are."""
    base = _paths.output_dir(kind)
    dest = _paths.safe_join(base, folder.strip())
    if not folder.strip() or dest is None:
        raise ValueError(f"{folder!r} cannot be a folder name here. Use a plain name.")
    os.makedirs(dest, exist_ok=True)
    copied, missing = 0, []
    for p in paths:
        if not os.path.isfile(p):
            missing.append(p)
            continue
        shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
        copied += 1
    return ExportResult(folder=dest, copied=copied, missing=missing)


class BoardExportArgs(Args):
    board: str = slug_arg()
    folder: str | None = Field(None, description="The folder's name under output/collections. "
                                                 "Defaults to the board's id.")


@op("board export", "effect", BoardExportArgs, ExportResult,
    doc="""
    Copy a board's pictures into a folder under output/collections, to hand
    on or print. The originals are not moved.
    """,
    examples=("board export monsoon",))
def _board_export(a: BoardExportArgs, ctx: Context) -> ExportResult:
    rec = load(a.board)
    return export_files(_boards.item_paths(rec), "collections", a.folder or a.board)


class ItemExportArgs(Args):
    items: list[str] = items_arg(0, 1, 2000)
    folder: str = Field(description="The folder's name under output/<kind>.")
    kind: Literal["collections", "search"] = Field("search",
                                                   description="Which output folder to use: search for "
                                                               "pictures picked from results, collections "
                                                               "for a collection's copies.")


@op("item export", "effect", ItemExportArgs, ExportResult,
    doc="""
    Copy pictures into a folder under output/search (or output/collections).
    The originals are not moved.
    """,
    examples=("item export japan:12 japan:7 --folder picks",))
def _item_export(a: ItemExportArgs, ctx: Context) -> ExportResult:
    paths = [archive.resolve_item(r, ctx.actor).path for r in a.items]
    return export_files(paths, a.kind, a.folder)


class BoardDeleteArgs(Args):
    board: str = slug_arg()


@op("board delete", "write", BoardDeleteArgs, BoardWriteResult,
    doc="""
    Delete a board. The pictures on it are untouched, and `undo` brings the
    board back. Only the person deletes boards.
    """,
    examples=("board delete monsoon",), actors=("person", "cli"))
def _board_delete(a: BoardDeleteArgs, ctx: Context) -> BoardWriteResult:
    rec = load(a.board)
    path = ctx.tx.touch_board(a.board)
    os.remove(path)
    summary = row(rec)
    return BoardWriteResult(board=summary, deleted=True,
                            message=f"Deleted `{a.board}`, \"{rec.get('name', a.board)}\".")


# ─────────────────────────── render and open ───────────────────────────

class RenderResult(Result):
    board: str
    path: str
    format: str
    layout: str
    count: int
    warnings: list[str] = Field(default_factory=list)

    def text(self) -> str:
        out = f"Rendered `{self.board}` ({self.count} pictures, {self.layout}) to {self.path}\n"
        if self.warnings:
            out += "Look at: " + " ".join(self.warnings) + "\n"
        return out


class BoardRenderArgs(Args):
    board: str = slug_arg()
    format: Literal["html", "png"] = Field("html", description="html for people, png for "
                                                               "looking at the layout yourself.")
    layout: Literal["editorial", "masonry", "grid"] | None = Field(
        None, description="editorial (one hero, sections by role), masonry or grid. Defaults "
                          "to the board's own, else editorial.")
    width: int = Field(1568, ge=800, le=3000, description="png only: the width in pixels.")


def render_board(slug: str, fmt: str = "html", layout: str | None = None,
                 width: int = 1568) -> tuple[str, "BoardShowResult", bytes | None, str]:
    """Write a board to output/boards. Returns (path, the board, png bytes or None, layout)."""
    try:
        from .. import compose as _compose
    except ImportError:                               # pragma: no cover
        import compose as _compose
    rec = load(slug)
    shown = view(rec)
    layout = layout or ((rec.get("layout") or {}).get("kind") if isinstance(rec.get("layout"), dict)
                        else None) or "editorial"
    if layout not in _compose.LAYOUTS:
        layout = "editorial"
    out_dir = _paths.ensure_dir(os.path.join(_paths.subdir("output"), "boards"))
    if fmt == "html":
        path = os.path.join(out_dir, f"{slug}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_compose.to_html(shown, layout))
        return path, shown, None, layout
    img = _compose.to_png(shown, width)
    data = _compose.png_bytes(img)
    path = os.path.join(out_dir, f"{slug}.png")
    with open(path, "wb") as fh:
        fh.write(data)
    return path, shown, data, layout


@op("board render", "effect", BoardRenderArgs, RenderResult,
    doc="""
    Lay a board out as one self-contained HTML page, or as one picture. The
    page carries the concept, the palette strip, each section with its reason,
    every picture with its role and note, and a mark on anything derived or
    generated. It opens anywhere, since the pictures are embedded. The png is
    for looking at your own layout and critiquing it. Written to
    output/boards.
    """,
    examples=("board render monsoon", "board render monsoon --format png --width 1200"))
def _board_render(a: BoardRenderArgs, ctx: Context) -> RenderResult:
    path, shown, data, layout = render_board(a.board, a.format, a.layout, a.width)
    res = RenderResult(board=a.board, path=path, format=a.format, layout=layout,
                       count=shown.metrics.count, warnings=shown.metrics.warnings)
    if data is not None and ctx.pixels:
        from .registry import Attachment
        res.attachments.append(Attachment(caption=f"board {a.board}", mime="image/png", data=data))
    return res


class BoardOpenArgs(Args):
    board: str = slug_arg()
    layout: Literal["editorial", "masonry", "grid"] | None = Field(
        None, description="The layout to render it in first.")


@op("board open", "effect", BoardOpenArgs, RenderResult,
    doc="Render a board as HTML and open it in the browser.",
    examples=("board open monsoon",), actors=("person", "cli"))
def _board_open(a: BoardOpenArgs, ctx: Context) -> RenderResult:
    import webbrowser
    path, shown, _, layout = render_board(a.board, "html", a.layout)
    if not os.environ.get("ARCANA_NO_BROWSER"):
        webbrowser.open("file:///" + path.replace("\\", "/"))
    return RenderResult(board=a.board, path=path, format="html", layout=layout,
                        count=shown.metrics.count, warnings=shown.metrics.warnings)


# ─────────────────────────── designed pages and PDF ───────────────────────────
# `board render` lays a board out from a fixed template. A board can also carry
# its own designed page: one self-contained HTML file an agent or a person made
# for it (images embedded, no network). It lives beside the rendered boards as
# output/boards/<slug>.page.html, so `board render` (which writes <slug>.html)
# never overwrites it, and the board records that it has one. The app shows it
# in a sandboxed frame, and `board pdf` prints it with a headless Chromium,
# honouring the page's own @page CSS.

PAGE_SUFFIX = ".page.html"
PAGE_LIMIT_MB = 300


def page_path(slug: str) -> str:
    return os.path.join(_paths.subdir("output"), "boards", f"{slug}{PAGE_SUFFIX}")


def page_file(slug: str) -> str | None:
    """The board's designed page on disk, or None when it has none."""
    rec = _boards.load(slug)
    if not rec or not rec.get("page"):
        return None
    p = page_path(rec["slug"])
    return p if os.path.isfile(p) else None


def _allowed_source(path: str, actor: str) -> bool:
    """An agent hands in pages from its workspace or Arcana's output only."""
    if actor != "agent":
        return True
    try:
        from .. import workspace as _ws
    except ImportError:                               # pragma: no cover
        import workspace as _ws
    return _paths.is_within(path, [_ws.workspace_dir(), _paths.subdir("output")])


def _resolve_page_source(given: str) -> str:
    """A path as given, or relative to the folder that holds output/ and workspace/."""
    if os.path.isabs(given):
        return os.path.abspath(given)
    here = os.path.abspath(given)
    if os.path.exists(here):
        return here
    base = os.path.dirname(os.path.abspath(_paths.subdir("output")))
    return os.path.abspath(os.path.join(base, given))


class PageResult(Result):
    board: str
    path: str | None = None
    size_mb: float = 0.0
    message: str

    def text(self) -> str:
        return self.message + "\n"


class PageSetArgs(Args):
    board: str = slug_arg()
    file: str = positional(1, description="A self-contained HTML page for the board (images "
                                          "embedded, no network). A relative path is read from "
                                          "the folder that holds output/ and workspace/.",
                           json_schema_extra={"metavar": "FILE"})


@op("board page set", "write", PageSetArgs, PageResult,
    doc="""
    Give a board its own designed page. The file is copied to
    output/boards/<board>.page.html, which `board render` never overwrites,
    and the app's Moodboard tab shows it, sandboxed, with Save as PDF. An agent
    hands in pages from its workspace or from Arcana's output folder.
    """,
    examples=("board page set monsoon output/boards/monsoon.page.html",))
def _page_set(a: PageSetArgs, ctx: Context) -> PageResult:
    src = _resolve_page_source(a.file)
    if not os.path.isfile(src):
        raise ValueError(f"No file at {a.file}.")
    if not src.lower().endswith((".html", ".htm")):
        raise ValueError("A board's page is an .html file.")
    size = os.path.getsize(src) / 1e6
    if size > PAGE_LIMIT_MB:
        raise ValueError(f"The page is {size:,.0f} MB, over the {PAGE_LIMIT_MB} MB limit.")
    if not _allowed_source(src, ctx.actor):
        raise ValueError("An agent hands in pages from its workspace or from Arcana's output "
                         "folder. Write the page there first.")
    rec = load(a.board)
    dest = page_path(rec["slug"])
    _paths.ensure_dir(os.path.dirname(dest))
    rec["page"] = {"file": os.path.basename(dest), "source": os.path.basename(src),
                   "set": time.time()}
    put(ctx, rec)                     # the A7 check runs here, before anything is copied
    if os.path.normcase(src) != os.path.normcase(os.path.abspath(dest)):
        shutil.copyfile(src, dest)
    return PageResult(board=a.board, path=dest, size_mb=round(size, 2),
                      message=f"`{a.board}` now carries its own page, {size:,.1f} MB, at {dest}. "
                              f"The Moodboard tab shows it under View page.")


class PageClearArgs(Args):
    board: str = slug_arg()


@op("board page clear", "write", PageClearArgs, PageResult,
    doc="""
    Detach a board's designed page. The board goes back to its rendered
    layout, and the page file stays in output/boards.
    """,
    examples=("board page clear monsoon",))
def _page_clear(a: PageClearArgs, ctx: Context) -> PageResult:
    rec = load(a.board)
    if not rec.get("page"):
        return PageResult(board=a.board, message=f"`{a.board}` has no page to clear.")
    rec.pop("page", None)
    put(ctx, rec)
    return PageResult(board=a.board, path=page_path(rec["slug"]),
                      message=f"Detached the page from `{a.board}`. The file stays at "
                              f"{page_path(rec['slug'])}.")


def _program_files(var: str, default: str) -> str:
    return os.environ.get(var) or default


BROWSERS = (
    os.path.join(_program_files("PROGRAMFILES", r"C:\Program Files"),
                 "Google", "Chrome", "Application", "chrome.exe"),
    os.path.join(_program_files("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(_program_files("PROGRAMFILES", r"C:\Program Files"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def chromium() -> str | None:
    """A Chromium to print with: ARCANA_BROWSER, then Chrome or Edge, then one on PATH."""
    given = os.environ.get("ARCANA_BROWSER")
    if given and os.path.isfile(given):
        return given
    for p in BROWSERS:
        if p and os.path.isfile(p):
            return p
    for name in ("chrome", "msedge", "chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


class PdfResult(Result):
    board: str
    path: str
    pages: int
    size_mb: float
    source: str

    def text(self) -> str:
        return (f"Saved `{self.board}` as a PDF of {self.pages} page{'s' if self.pages != 1 else ''}, "
                f"{self.size_mb:,.1f} MB, from its {self.source}:\n  {self.path}\n")


class PdfArgs(Args):
    board: str = slug_arg()
    layout: Literal["editorial", "masonry", "grid"] | None = Field(
        None, description="For a board without its own page, the layout to render first.")
    wait: float = Field(4.0, ge=0, le=30,
                        description="Seconds the page may run its scripts before printing.")


@op("board pdf", "effect", PdfArgs, PdfResult,
    doc="""
    Save a board as a PDF in output/boards. A board with its own designed page
    prints that page, following its @page size and print styles. Otherwise the
    board is rendered as HTML first and that is printed. It uses Chrome or Edge
    without a window.
    """,
    examples=("board pdf monsoon",))
def _board_pdf(a: PdfArgs, ctx: Context) -> PdfResult:
    import subprocess
    import tempfile
    rec = load(a.board)
    browser = chromium()
    if browser is None:
        raise ValueError("No Chrome or Edge found to print with. Set ARCANA_BROWSER to a "
                         "Chromium, or print the page from the Moodboard tab.")
    src = page_file(rec["slug"])
    source = "designed page"
    if src is None:
        src, _, _, layout = render_board(rec["slug"], "html", a.layout)
        source = f"{layout} layout"
    out = os.path.join(os.path.dirname(page_path(rec["slug"])), f"{rec['slug']}.pdf")
    if os.path.exists(out):
        os.remove(out)
    url = "file:///" + os.path.abspath(src).replace(os.sep, "/")
    with tempfile.TemporaryDirectory(prefix="arcana-pdf-") as profile:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", f"--user-data-dir={profile}",
               "--no-pdf-header-footer", "--print-to-pdf-no-header",
               "--run-all-compositor-stages-before-draw",
               f"--virtual-time-budget={int(a.wait * 1000)}",
               f"--print-to-pdf={out}", url]
        try:
            subprocess.run(cmd, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise ValueError("The browser took over three minutes to print the page.") from None
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        raise ValueError(f"{os.path.basename(browser)} did not write a PDF.")
    with open(out, "rb") as fh:
        data = fh.read()
    pages = max(1, data.count(b"/Type /Page") - data.count(b"/Type /Pages"))
    return PdfResult(board=a.board, path=out, pages=pages, size_mb=round(len(data) / 1e6, 2),
                     source=source)
