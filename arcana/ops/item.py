# item.py -- looking closely at particular pictures
#
# Every op here takes handles. They read the picture files, so a dataset whose
# files have moved answers with what the index knows and says the file is gone.

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from . import archive
from .archive import Hit
from .registry import Args, Context, Result, op, positional

try:
    from .. import core
except ImportError:                                   # pragma: no cover
    import core


def handles_arg(lo: int, hi: int):
    return positional(0, min_length=lo, max_length=hi,
                      description="Pictures, as handles: japan:1423, or out:3f9c1e for a "
                                  "derived one.",
                      json_schema_extra={"metavar": "HANDLE"})


def handle_arg():
    return positional(0, description="A picture, as dataset:key.",
                      json_schema_extra={"metavar": "HANDLE"})


def _open_rgb(path: str):
    from PIL import Image
    im = Image.open(path)
    im.load()
    return im


def _describe(path: str, ds=None) -> str:
    """CLIP's words for a picture. Any CLIP will do; the dataset's own is preferred."""
    try:
        from .. import style_transfer as _st
    except ImportError:                               # pragma: no cover
        import style_transfer as _st
    try:
        model_id = core.text_model_for_dim(int(ds.index.ndim)) if ds is not None else None
    except RuntimeError:
        model_id = None
    return _st.describe(path, model_id=model_id) or ""


def _lookup(ref: str, actor: str):
    """(item, dataset or None, key or None). Archive handles bring the index's facts."""
    it = archive.resolve_item(ref, actor)
    if it.dataset and it.handle:
        ds = archive.open_dataset(it.dataset)
        return it, ds, archive.parse_handle(it.handle)[1]
    return it, None, None


def _hit(num: int, it, ds, key) -> Hit:
    if ds is not None:
        return archive.make_hit(num, it.dataset, ds, key, it.path)
    return Hit(num=num, handle=it.handle or it.ref, file=os.path.basename(it.path), path=it.path,
               metric=it.kind if it.kind != "archive" else None,
               missing=not os.path.exists(it.path))


# ─────────────────────────── item show ───────────────────────────

class Swatch(BaseModel):
    hex: str
    share: float


class ItemInfo(BaseModel):
    handle: str
    file: str = ""
    path: str = ""
    facts: dict = Field(default_factory=dict)
    width: int | None = None
    height: int | None = None
    palette: list[Swatch] = Field(default_factory=list)
    looks_like: str | None = None
    error: str | None = None

    def text(self) -> str:
        if not self.path:
            return f"{self.handle}: {self.error}"
        lines = [f"{self.handle}  {self.file}"]
        lines += [f"  {name}: {val}" for name, val in self.facts.items()]
        if self.error:
            lines.append(f"  could not be read: {self.error}")
            return "\n".join(lines)
        lines.append(f"  size: {self.width} x {self.height}")
        lines.append("  palette: " + " ".join(f"{s.hex} {s.share:.0%}" for s in self.palette))
        if self.looks_like is not None:
            lines.append(f"  looks like: {self.looks_like}")
        return "\n".join(lines)


class ItemShowResult(Result):
    items: list[ItemInfo]

    def text(self) -> str:
        return "\n\n".join(i.text() for i in self.items) + "\n"


class ItemShowArgs(Args):
    handles: list[str] = handles_arg(1, 4)
    side: int = Field(archive.INSPECT_SIDE, ge=256, le=2048,
                      description="Longest side of the picture sent, in pixels.")
    describe: bool | None = Field(None, description="Add CLIP's description of the picture. "
                                                    "Defaults to on when the picture itself is "
                                                    "not sent.")


@op("item show", "read", ItemShowArgs, ItemShowResult,
    modalities=("image",),
    doc="""
    A close look at up to four pictures. Each comes with its file, size,
    palette, stars, colour tag, capture date and cluster. The picture itself
    comes too when it can reach you. Otherwise a CLIP description stands in
    for it, which loads an image encoder the first time.
    """,
    examples=("item show japan:12", "item show japan:12 japan:7 --no-describe"))
def _item_show(a: ItemShowArgs, ctx: Context) -> ItemShowResult:
    try:
        from .. import refselect as _refselect
    except ImportError:                               # pragma: no cover
        import refselect as _refselect
    describe = (not ctx.sheets) if a.describe is None else a.describe
    out = ItemShowResult(items=[])
    for h in a.handles:
        try:
            it, ds, key = _lookup(h, ctx.actor)
        except ValueError as e:
            out.items.append(ItemInfo(handle=h, error=str(e)))
            continue
        path = it.path
        facts = archive.facts(ds, key) if ds is not None else {}
        if it.kind in ("derived", "generated"):
            facts = {"kind": it.kind, **({"derived from": it.derived_from} if it.derived_from else {})}
        info = ItemInfo(handle=it.handle or h, file=os.path.basename(path), path=path, facts=facts)
        try:
            with _open_rgb(path) as im:
                info.width, info.height = im.width, im.height
                strip = _refselect.palette_strip(im.convert("RGB"), n=6)
            info.palette = [Swatch(hex=hx, share=float(share)) for hx, share in strip]
        except Exception as e:
            info.error = str(e)
            out.items.append(info)
            continue
        if describe:
            # When this is the reader's only view of the picture, a failure is
            # reported rather than swallowed.
            try:
                words = _describe(path, ds)
                info.looks_like = words or "(no description available)"
            except Exception as e:
                info.looks_like = f"(description failed: {type(e).__name__}: {e})"
        out.items.append(info)
        if ctx.sheets:
            try:
                from .. import sheets as _sheets
            except ImportError:                       # pragma: no cover
                import sheets as _sheets
            jpeg = _sheets.fitted_jpeg(path, a.side)
            if jpeg:
                from .registry import Attachment
                out.attachments.append(Attachment(caption=h, data=jpeg))
    return out


# ─────────────────────────── item sheet ───────────────────────────

class SheetResult(Result):
    hits: list[Hit]
    withheld: bool = False

    def text(self) -> str:
        lines = [h.line() for h in self.hits]
        if self.withheld:
            lines.append("The sheet stays on this machine: the person has not allowed pictures "
                         "to reach agents.")
        return "\n".join(lines) + "\n"


class ItemSheetArgs(Args):
    handles: list[str] = handles_arg(1, 48)
    tile: int = Field(archive.SCAN_TILE, ge=64, le=320, description="Tile size in pixels.")
    cols: int = Field(archive.SHEET_COLS, ge=1, le=12, description="Tiles per row.")
    title: str | None = Field(None, description="A title drawn above the tiles.")


@op("item sheet", "read", ItemSheetArgs, SheetResult,
    modalities=("image",),
    doc="""
    A numbered contact sheet of any pictures. The tiles follow the order given,
    for judging a selection side by side. From the terminal the sheet is
    written to output/sheets. An agent receives it only when the person allows
    pixels.
    """,
    examples=("item sheet japan:12 japan:7 japan:3 --cols 3",))
def _item_sheet(a: ItemSheetArgs, ctx: Context) -> SheetResult:
    hits = []
    for i, h in enumerate(a.handles, 1):
        it, ds, key = _lookup(h, ctx.actor)
        hits.append(_hit(i, it, ds, key))
    out = SheetResult(hits=hits, withheld=not ctx.pixels)
    if ctx.pixels:
        out.attachments.append(archive.sheet([h.path for h in hits], [h.num for h in hits],
                                             tile=a.tile, cols=a.cols, title=a.title,
                                             caption=a.title or "contact sheet"))
    return out


# ─────────────────────────── item palette ───────────────────────────

class PaletteResult(Result):
    handle: str
    file: str
    colours: list[Swatch]
    filtered: str = ""

    def text(self) -> str:
        head = f"{self.handle}  {self.file}  palette, darkest first"
        if self.filtered:
            head += f" ({self.filtered})"
        return head + "\n" + "\n".join(f"  {s.hex}  {s.share:.0%}" for s in self.colours) + "\n"


class ItemPaletteArgs(Args):
    handle: str = handle_arg()
    n: int = Field(8, ge=2, le=16, description="Colours to extract.")
    l_min: float = Field(0.0, ge=0, le=100, description="Ignore pixels darker than this LAB lightness.")
    l_max: float = Field(100.0, ge=0, le=100, description="Ignore pixels lighter than this.")
    s_min: float = Field(0.0, ge=0, le=100, description="Ignore pixels less saturated than this (percent).")


@op("item palette", "read", ItemPaletteArgs, PaletteResult,
    modalities=("image",),
    doc="""
    A picture's dominant colours, darkest first, with their shares. It is the
    same k-means in LAB that the app draws under its Reference and Target. The
    lightness and saturation filters keep shadows, skies or grey walls from
    dominating, which is how a reference palette is chosen for a colour
    transfer.
    """,
    examples=("item palette japan:12", "item palette japan:12 --n 5 --l-min 15 --s-min 10"))
def _item_palette(a: ItemPaletteArgs, ctx: Context) -> PaletteResult:
    try:
        from .. import refselect as _refselect
    except ImportError:                               # pragma: no cover
        import refselect as _refselect
    it, ds, key = _lookup(a.handle, ctx.actor)
    path = it.path
    if not os.path.exists(path):
        raise ValueError(f"{a.handle}'s file is not where the index says ({path}).")
    with _open_rgb(path) as im:
        strip = _refselect.palette_strip(im.convert("RGB"), n=a.n, l_min=a.l_min,
                                         l_max=a.l_max, s_min=a.s_min)
    bits = []
    if a.l_min > 0 or a.l_max < 100:
        bits.append(f"lightness {a.l_min:g} to {a.l_max:g}")
    if a.s_min > 0:
        bits.append(f"saturation from {a.s_min:g}")
    return PaletteResult(handle=a.handle, file=os.path.basename(path),
                         colours=[Swatch(hex=h, share=float(s)) for h, s in strip],
                         filtered=", ".join(bits))


# ─────────────────────────── item describe ───────────────────────────

class DescribeResult(Result):
    handle: str
    words: str

    def text(self) -> str:
        return f"{self.handle}  looks like: {self.words or '(no description available)'}\n"


class ItemDescribeArgs(Args):
    handle: str = handle_arg()


@op("item describe", "read", ItemDescribeArgs, DescribeResult,
    modalities=("image",),
    doc="""
    CLIP's words for a picture. They are the nearest labels from Arcana's own
    vocabulary, the description img2img is guided by, and a blind agent's way
    to read a picture. It loads an image encoder the first time.
    """,
    examples=("item describe japan:12",))
def _item_describe(a: ItemDescribeArgs, ctx: Context) -> DescribeResult:
    it, ds, key = _lookup(a.handle, ctx.actor)
    if not os.path.exists(it.path):
        raise ValueError(f"{a.handle}'s file is not where the index says ({it.path}).")
    return DescribeResult(handle=a.handle, words=_describe(it.path, ds))
