# colour.py -- recolouring pictures, and harmonizing a board
#
# Every method lives in transfer_methods.py, so the app's dropdown, these ops
# and an agent choose from one table. A result is a new file in
# output/color_transfer, recorded in the ledger as derived from its target, so
# it has an out: handle and is badged as derived wherever it goes. The original
# is never written to.

from __future__ import annotations

import io
import os
import sys
import time
from typing import Literal

from PIL import Image, PngImagePlugin
from pydantic import BaseModel, Field, field_validator

from . import archive
from .effects import run_effect
from .registry import Args, Attachment, Context, Result, op, positional

try:
    from .. import ledger as _ledger
    from .. import paths as _paths
    from .. import transfer_methods as _tm
except ImportError:                                   # pragma: no cover
    import ledger as _ledger
    import paths as _paths
    import transfer_methods as _tm

Method = Literal["lab", "mkl", "idt", "modflows"]
Quality = Literal["quick", "balanced", "best"]
PREVIEW_SIDE = 1024


def item_arg(index: int, what: str):
    return positional(index, description=what, json_schema_extra={"metavar": "ITEM"})


def preview(img: Image.Image, side: int = PREVIEW_SIDE, caption: str = "") -> Attachment:
    small = img.convert("RGB").copy()
    small.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=88)
    return Attachment(caption=caption, data=buf.getvalue())


def palette_image(entries: list[str]) -> Image.Image:
    """A palette as a picture, each colour as wide as its weight."""
    from .search import parse_colour, swatch
    parsed = [parse_colour(e) for e in entries]
    bgr = swatch([h for h, _ in parsed], [w for _, w in parsed])
    return Image.fromarray(bgr[..., ::-1].copy())


# ─────────────────────────── colour methods ───────────────────────────

class MethodRow(BaseModel):
    id: str
    label: str
    quality: str
    timing: str
    params: list[str]
    available: bool
    why_not: str = ""
    licence: str
    blurb: str


class QualityRow(BaseModel):
    id: str
    label: str
    note: str


class MethodsResult(Result):
    methods: list[MethodRow]
    qualities: list[QualityRow] = Field(default_factory=list)   # ModFlows' presets

    def text(self) -> str:
        lines = []
        for m in self.methods:
            state = "ready" if m.available else m.why_not
            lines.append(f"{m.id:<9} {m.label}  ·  {m.quality}  ·  {m.timing}  ·  {state}")
            lines.append(f"          {m.blurb}")
        return "\n".join(lines) + "\n"


class NoArgs(Args):
    pass


@op("colour methods", "read", NoArgs, MethodsResult,
    doc="""
    The colour transfer methods, cheapest first. Each comes with what it is
    good for, how long it takes here at 1024 px, and whether it can run now.
    """,
    examples=("colour methods",))
def _colour_methods(a: NoArgs, ctx: Context) -> MethodsResult:
    # Asking whether the card is usable imports PyTorch. Only do it when
    # something has already paid for that import.
    gpu = ctx.has("gpu") if "torch" in sys.modules else False
    rows = []
    for m in _tm.METHODS:
        ok, why = _tm.available(m.id)
        rows.append(MethodRow(id=m.id, label=m.label, quality=m.quality, timing=m.timing(gpu),
                              params=list(m.params), available=ok, why_not=why,
                              licence=m.licence, blurb=m.blurb))
    try:
        from .. import color_transfer as _ct
    except ImportError:                               # pragma: no cover
        import color_transfer as _ct
    qualities = [QualityRow(id=k, label=v["label"], note=v["note"])
                 for k, v in _ct.QUALITY_PRESETS.items()]
    return MethodsResult(methods=rows, qualities=qualities)


# ─────────────────────────── colour transfer ───────────────────────────

class TransferResult(Result):
    handle: str
    path: str
    file: str
    method: str
    label: str
    device: str
    elapsed: float
    width: int
    height: int
    target: str
    reference: str
    kept: float = 1.0                 # share of the reference's pixels the filter kept

    def text(self) -> str:
        note = ""
        if self.kept == 0.0:
            note = " The colour range matched nothing, so the whole reference was used."
        elif self.kept < 1.0:
            note = f" From {self.kept:.1%} of the reference's pixels."
        return (f"{self.handle}  {self.label}, {self.target} recoloured from {self.reference}.{note}\n"
                f"  {self.width} x {self.height} · {self.device} · {self.elapsed:.1f} s · {self.file}\n"
                f"  Derived, so it is marked as such on any board. The original is untouched.\n")


class ReferenceArgs(Args):
    reference: str | None = Field(None, description="The picture whose colours to take.",
                                  json_schema_extra={"metavar": "ITEM"})
    palette: list[str] = Field(default_factory=list, max_length=8,
                               description="Or a palette instead of a picture: #hex, with an "
                                           "optional weight, #1f2a36:0.4. Repeat per colour.",
                               json_schema_extra={"metavar": "HEX"})
    method: Method | None = Field(None, description="lab, mkl, idt or modflows. Defaults to "
                                                    "modflows when it can run here, mkl otherwise.")
    strength: float = Field(1.0, ge=0.0, le=1.0, description="0 leaves the target as it was.")
    quality: Quality = Field("balanced", description="ModFlows only: quick, balanced or best.")
    l_min: float = Field(0, ge=0, le=100, description="Use only reference pixels at least this light.")
    l_max: float = Field(100, ge=0, le=100, description="Use only reference pixels at most this light.")
    s_min: float = Field(0, ge=0, le=100, description="Use only reference pixels at least this saturated.")
    s_max: float = Field(100, ge=0, le=100, description="Use only reference pixels at most this saturated.")

    @field_validator("palette")
    @classmethod
    def _palette_parses(cls, v):
        from .search import parse_colour
        for e in v:
            parse_colour(e)
        return v


def reference_image(a: ReferenceArgs, actor: str) -> tuple[Image.Image, str]:
    if bool(a.reference) == bool(a.palette):
        raise ValueError("Give either --reference (a picture) or --palette (colours), not both.")
    if a.palette:
        return palette_image(a.palette), "palette " + " ".join(a.palette)
    ref = archive.resolve_item(a.reference, actor)
    if not os.path.exists(ref.path):
        raise ValueError(f"{a.reference}'s file is not where the index says ({ref.path}).")
    return Image.open(ref.path).convert("RGB"), ref.handle or a.reference


def recolour(target: archive.Item, ref_img: Image.Image, ref_name: str, a: ReferenceArgs,
             progress=None) -> dict:
    """The transfer itself, written to output/color_transfer. Returns what record() needs."""
    try:
        from .. import refselect as _refselect
    except ImportError:                               # pragma: no cover
        import refselect as _refselect
    method = a.method or _tm.default_for(False).id
    ok, why = _tm.available(method)
    if not ok:
        raise ValueError(why)
    if not os.path.exists(target.path):
        raise ValueError(f"{target.ref}'s file is not where the index says ({target.path}).")
    if progress:
        progress(0.05, f"{_tm.BY_ID[method].label} on {target.handle or target.ref}")
    content = Image.open(target.path).convert("RGB")
    proxy, kept = _refselect.filter_reference(ref_img, a.l_min, a.l_max, a.s_min, a.s_max)
    out, info = _tm.transfer(method, content, proxy, a.strength, a.quality, content_path=target.path)

    out_dir = _paths.ensure_dir(os.path.join(_paths.subdir("output"), "color_transfer"))
    tname = os.path.splitext(os.path.basename(target.path))[0]
    rname = ref_name.replace(":", "-").replace("#", "").replace(" ", "")[:40]
    name = _paths.fit_filename(out_dir, tname, rname,
                               f"{method}_{time.strftime('%Y%m%d_%H%M%S')}", ".png")
    path = os.path.join(out_dir, name)
    # The filename is truncated to fit the filesystem, so the provenance goes
    # inside the PNG, where it survives the file being copied anywhere.
    meta = PngImagePlugin.PngInfo()
    meta.add_text("arcana:target", target.path)
    meta.add_text("arcana:reference", ref_name)
    meta.add_text("arcana:method", info["label"])
    meta.add_text("arcana:strength", str(a.strength))
    meta.add_text("arcana:device", info["device"])
    out.save(path, "PNG", pnginfo=meta)
    if progress:
        progress(1.0, "done")
    return {"path": path, "method": method, "info": info, "kept": kept, "size": out.size,
            "target": target, "ref_name": ref_name}


def record_transfer(tx, payload: dict, a: ReferenceArgs, op_name: str = "colour transfer"):
    t = payload["target"]
    entry = _ledger.record(tx, payload["path"], op=op_name,
                           args={"method": payload["method"], "strength": a.strength,
                                 "quality": a.quality, "reference": payload["ref_name"]},
                           sources=[s for s in (t.handle, payload["ref_name"]) if s],
                           derived_from=t.handle)
    w, h = payload["size"]
    return TransferResult(handle=entry["handle"], path=payload["path"],
                          file=os.path.basename(payload["path"]), method=payload["method"],
                          label=payload["info"]["label"], device=payload["info"]["device"],
                          elapsed=payload["info"]["elapsed"], width=w, height=h,
                          target=t.handle or t.ref, reference=payload["ref_name"],
                          kept=payload["kept"])


class ColourTransferArgs(ReferenceArgs):
    target: str = item_arg(0, "The picture to recolour: a handle, or out: for a derived one.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("colour transfer", "effect", ColourTransferArgs, TransferResult,
    doc="""
    Recolour a picture toward another picture's colours, or toward a palette.
    The result is a new file with an out: handle, marked as derived. The
    original is untouched. `colour methods` lists the methods with their cost.
    The lightness and saturation filters narrow which reference pixels count,
    so a sky or a shadow does not dominate.
    """,
    examples=("colour transfer japan:12 --reference japan:7 --method mkl",
              "colour transfer japan:12 --palette '#1f2a36:0.6' --palette '#b5651d:0.4' "
              "--method lab --strength 0.5"))
def _colour_transfer(a: ColourTransferArgs, ctx: Context) -> Result:
    target = archive.resolve_item(a.target, ctx.actor)
    ref_img, ref_name = reference_image(a, ctx.actor)

    def compute(progress):
        return recolour(target, ref_img, ref_name, a, progress)

    def record(tx, payload):
        res = record_transfer(tx, payload, a)
        if ctx.sheets:
            res.attachments.append(preview(Image.open(payload["path"]), caption=res.handle))
        return res

    return run_effect(ctx, f"colour transfer {a.target}", compute, record, a.background)


# ─────────────────────────── colour lut ───────────────────────────

class LutResult(Result):
    path: str
    size: int
    target: str
    reference: str

    def text(self) -> str:
        return (f"A {self.size}^3 LUT taking {self.target} toward {self.reference}:\n"
                f"  {self.path}\nIt loads in Resolve, Premiere, Photoshop and most graders "
                f"as a .cube file.\n")


class ColourLutArgs(Args):
    target: str = item_arg(0, "The picture the LUT starts from.")
    reference: str = Field(description="The picture whose colours the LUT moves toward.",
                           json_schema_extra={"metavar": "ITEM"})
    n: int = Field(33, ge=9, le=65, description="Lattice size. 33 is the usual grading size.")
    strength: float = Field(1.0, ge=0.0, le=1.0, description="How far the LUT moves the colours.")
    steps: int = Field(8, ge=2, le=64, description="Flow steps. More is smoother and slower.")
    out: str | None = Field(None, description="Where to write the .cube file. Defaults to "
                                              "output/luts. For the person and the terminal.",
                            json_schema_extra={"metavar": "PATH"})


@op("colour lut", "effect", ColourLutArgs, LutResult,
    doc="""
    Bake a colour transfer into a .cube LUT. It takes the ModFlows transfer
    between two pictures, to grade other footage or photographs the same way in
    another program, and writes it to output/luts.
    """,
    examples=("colour lut japan:12 --reference japan:7 --n 33",), needs=("modflows",))
def _colour_lut(a: ColourLutArgs, ctx: Context) -> LutResult:
    try:
        from .. import lut as _lut
    except ImportError:                               # pragma: no cover
        import lut as _lut
    ok, why = _tm.available("modflows")
    if not ok:
        raise ValueError(why)
    if a.out and ctx.actor == "agent":
        raise ValueError("An agent writes inside Arcana's data folder only. Leave out --out "
                         "and the LUT goes to output/luts.")
    t = archive.resolve_item(a.target, ctx.actor)
    r = archive.resolve_item(a.reference, ctx.actor)
    table = _lut.bake_modflows(t.path, r.path, n=a.n, steps=a.steps, strength=a.strength)
    if a.out:
        dest = os.path.abspath(a.out)
    else:
        out_dir = _paths.ensure_dir(os.path.join(_paths.subdir("output"), "luts"))
        dest = os.path.join(out_dir, _paths.fit_filename(
            out_dir, os.path.splitext(os.path.basename(t.path))[0],
            os.path.splitext(os.path.basename(r.path))[0], f"{a.n}", ".cube"))
    path = str(_lut.write_cube(dest, table, a.n,
                               title=f"arcana {t.handle or a.target} to {r.handle or a.reference}"))
    return LutResult(path=path, size=a.n, target=t.handle or a.target, reference=r.handle or a.reference)


# ─────────────────────────── colour harmonize ───────────────────────────

class HarmonizeResult(Result):
    board: str
    made: list[TransferResult]

    def text(self) -> str:
        lines = [f"Harmonized {len(self.made)} picture{'s' if len(self.made) != 1 else ''} on "
                 f"`{self.board}`. Each result sits beside its original as a derived support:"]
        lines += [f"  {m.handle} from {m.target}  ({m.label}, {m.elapsed:.1f} s)" for m in self.made]
        return "\n".join(lines) + "\n"


class HarmonizeArgs(ReferenceArgs):
    board: str = positional(0, description="The board, by id.",
                            json_schema_extra={"complete": "board", "metavar": "BOARD"})
    items: list[str] = positional(1, min_length=1, max_length=12,
                                  description="The outliers to bring in: handles or #numbers "
                                              "from `board show`.",
                                  json_schema_extra={"metavar": "ITEM"})
    strength: float = Field(0.45, ge=0.0, le=1.0,
                            description="0.3 to 0.6 keeps each picture itself.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("colour harmonize", "effect", HarmonizeArgs, HarmonizeResult,
    doc="""
    Bring a board's outliers toward its palette or toward one picture. Each
    result is added beside its original as a derived support, with a note
    saying how it was made, and the original stays on the board. The hero is
    never harmonized, and derived pictures may make up at most a third of the
    board.
    """,
    examples=("colour harmonize monsoon japan:3 --reference japan:12 --method mkl",))
def _colour_harmonize(a: HarmonizeArgs, ctx: Context) -> Result:
    from . import board as _board
    rec = _board.load(a.board)
    paths = [_board.find(rec, r, ctx.actor) for r in a.items]
    for r, p in zip(a.items, paths):
        if (rec["details"].get(p) or {}).get("role") == "hero":
            raise ValueError(f"{r} is the board's hero. The hero sets the tone, so it is "
                             f"never harmonized.")
    shown = _board.view(rec)
    derived = shown.metrics.derived + shown.metrics.generated
    after = len(shown.items) + len(paths)
    if (derived + len(paths)) / after > _board.MAX_DERIVED_SHARE:
        raise ValueError(f"That would make {derived + len(paths)} of {after} pictures derived, "
                         f"more than a third. Harmonize fewer, or add more found pictures first.")
    if not a.palette and not a.reference and rec.get("palette"):
        a = a.model_copy(update={"palette": [f"{e['hex']}:{e.get('weight') or 1}"
                                             for e in rec["palette"] if isinstance(e, dict)]})
    ref_img, ref_name = reference_image(a, ctx.actor)
    targets = [archive.resolve_item(p, "cli") for p in paths]

    def compute(progress):
        out = []
        for i, t in enumerate(targets):
            sub = (lambda f, m, i=i: progress((i + (f or 0)) / len(targets), m)) if progress else None
            out.append(recolour(t, ref_img, ref_name, a, sub))
        return out

    def record(tx, payloads):
        rec2 = _board.load(a.board)
        made = []
        for orig_path, payload in zip(paths, payloads):
            res = record_transfer(tx, payload, a, "colour harmonize")
            made.append(res)
            d = rec2["details"].get(orig_path) or {}
            item = archive.Item(ref=res.handle, path=res.path, handle=res.handle, kind="derived",
                                derived_from=res.target)
            _board.add_item(rec2, item, section=d.get("section"), role="support",
                            note=f"{res.target} harmonized toward {ref_name} "
                                 f"({res.label}, strength {a.strength})")
            # Beside its original, not at the end of the board.
            rec2["items"].remove(res.path)
            rec2["items"].insert(rec2["items"].index(orig_path) + 1, res.path)
        _board.put(ctx, rec2, tx=tx)
        return HarmonizeResult(board=a.board, made=made)

    return run_effect(ctx, f"colour harmonize {a.board}", compute, record, a.background)
