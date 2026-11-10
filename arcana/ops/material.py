# material.py -- new material drawn from places in your archive (material_draw.py), as ops
#
#   material draw --source set:ice=0.7 --source label:lichen=0.3 --n 4
#   material draw --source set:water-reflections --engine pictures --words "at dusk"
#   material walk --a set:ice --b label:lichen --frames 6
#   material keep <draft path>
#
# A source is kind:id=weight. The kinds are set, label and picture (a handle or, for the person, a path).

from __future__ import annotations

from pydantic import BaseModel, Field

from .registry import Args, Context, Result, op, positional

try:
    from .. import material_draw as _md
except ImportError:                                   # pragma: no cover
    import material_draw as _md

ENGINE = {"place": _md.ENGINES[0], "pictures": _md.ENGINES[1]}


def parse_sources(items: list[str], actor: str) -> list[dict]:
    from . import archive

    out = []
    for raw in items or []:
        spec, _, w = raw.partition("=")
        kind, _, ident = spec.partition(":")
        kind, ident = kind.strip(), ident.strip()
        if kind not in ("set", "label", "picture") or not ident:
            raise ValueError(f"A source is set:<id>, label:<id> or picture:<handle>, with =weight after it. Got {raw!r}.")
        if kind == "picture":
            ident = archive.resolve_item(ident, actor).path
        out.append({"kind": kind, "id": ident, "weight": float(w) if w.strip() else 1.0})
    return out


class Drawn(BaseModel):
    path: str
    seed: int
    t: float | None = None


class MaterialDrawResult(Result):
    drafts: list[Drawn]

    def text(self) -> str:
        return "".join(f"{d.path}  (seed {d.seed}{'' if d.t is None else f', t {d.t}'})\n" for d in self.drafts) + \
            "Drafts until kept: `material keep <path>` writes one to output/generated/materials.\n"


class MaterialDrawArgs(Args):
    source: list[str] = Field(default_factory=list, description="A place to draw from: set:<id>, label:<id> or "
                              "picture:<handle>, with =weight after it. Several are blended by weight.")
    words: str = Field("", description="The material in words, alone or added to the sources' place.")
    engine: str = Field("place", description="place: Stable Diffusion at the sources' blended position, which mixes "
                        "evenly, pushes and tiles. pictures: FLUX.2 klein looking at the sources' own photographs, "
                        "faithful to them.")
    n: int = Field(4, ge=1, le=16, description="How many to draw, one per seed.")
    seed: int = Field(0, ge=0, description="The first seed.")
    size: int = Field(512, ge=256, le=1024, description="Pixels on a side.")
    tileable: bool = Field(False, description="Draw so the edges wrap round (the place engine only).")
    push: float = Field(2.0, ge=0.5, le=4.0, description="How far the place sits from the average picture.")
    spread: float = Field(0.25, ge=0.0, le=1.0, description="How far each seed slides toward one member's own look.")
    axis: str | None = Field(None, description="A concept axis to move the place along (the place engine).")
    amount: float = Field(0.0, ge=-2.0, le=2.0, description="How far along the axis: 1 is the distance between its ends.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("material draw", "effect", MaterialDrawArgs, MaterialDrawResult,
    doc="""
    Draw new material from places in your archive. Each source (a set, an IP
    label or a picture) has a weight, and words can be added or used alone.
    Results are drafts until kept.
    """,
    examples=("material draw --source set:ice=0.7 --source label:lichen=0.3 --n 4 --background",
              'material draw --words "molten copper" --n 2 --background'),
    needs=("diffusers",))
def _material_draw(a: MaterialDrawArgs, ctx: Context) -> Result:
    from .effects import run_effect

    if a.engine not in ENGINE:
        raise ValueError("--engine is place or pictures.")
    sources = parse_sources(a.source, ctx.actor)
    if not sources and not a.words.strip():
        raise ValueError("Give at least one --source or --words.")

    def compute(progress):
        cancelled = getattr(progress, "cancelled", None)
        return _md.draw(sources, a.words, ENGINE[a.engine], a.n, a.seed, a.size, a.tileable, a.push, a.spread,
                        progress, cancelled, axis=a.axis, amount=a.amount)

    def record(tx, payload):
        return MaterialDrawResult(drafts=[Drawn(path=d["path"], seed=d["seed"]) for d in payload])

    return run_effect(ctx, "material draw", compute, record, a.background, kind="material")


class MaterialWalkArgs(Args):
    a: list[str] = Field(description="Where the walk starts: sources as for `material draw`.")
    b: list[str] = Field(description="Where it ends.")
    words: str = Field("", description="Words added to every frame's place.")
    frames: int = Field(6, ge=2, le=24, description="How many points along the walk.")
    seed: int = Field(0, ge=0, description="The seed, held along the walk.")
    size: int = Field(512, ge=256, le=1024, description="Pixels on a side.")
    tileable: bool = Field(False, description="Draw so the edges wrap round.")
    push: float = Field(2.0, ge=0.5, le=4.0, description="How far both ends sit from the average picture.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("material walk", "effect", MaterialWalkArgs, MaterialDrawResult,
    doc="""
    Walk from one place to another, one material turning into the other. The
    seed is held and each frame is drawn by Stable Diffusion at its position
    on the great circle between the two places.
    """,
    examples=("material walk --a set:ice --b label:lichen --frames 6 --background",),
    needs=("diffusers",))
def _material_walk(a: MaterialWalkArgs, ctx: Context) -> Result:
    from .effects import run_effect

    A, B = parse_sources(a.a, ctx.actor), parse_sources(a.b, ctx.actor)
    if not A or not B:
        raise ValueError("Give both ends: --a and --b.")

    def compute(progress):
        cancelled = getattr(progress, "cancelled", None)
        return _md.walk(A, B, a.words, a.frames, a.seed, a.size, a.tileable, a.push, progress, cancelled)

    def record(tx, payload):
        return MaterialDrawResult(drafts=[Drawn(path=d["path"], seed=d["seed"], t=d.get("t")) for d in payload])

    return run_effect(ctx, "material walk", compute, record, a.background, kind="material")


class MaterialKeepArgs(Args):
    draft: str = positional(0, description="A drawn material's path, as `material draw` lists it.",
                            json_schema_extra={"metavar": "DRAFT"})
    collect: bool = Field(False, description="Also put it on the active collection.")


class MaterialKeepResult(Result):
    handle: str
    path: str

    def text(self) -> str:
        return f"Kept as {self.handle}: {self.path}\n"


@op("material keep", "write", MaterialKeepArgs, MaterialKeepResult,
    doc="""
    Keep a drawn material. It is written to output/generated/materials and
    recorded as generated, with the places and words it was drawn from.
    """,
    examples=("material keep C:/data/cache/generate/20261109-120000-abc123.png",), actors=("person", "cli"))
def _material_keep(a: MaterialKeepArgs, ctx: Context) -> MaterialKeepResult:
    try:
        from .. import ledger as _ledger
        from .. import ui_collection as _collection
    except ImportError:                               # pragma: no cover
        import ledger as _ledger
        import ui_collection as _collection
    meta = _md.read_meta(a.draft) if a.draft else {}
    path = _md.keep(a.draft)
    entry = _ledger.record(ctx.tx, path, op="material draw", kind="generated", args=meta,
                           sources=[str(s.get("id")) for s in meta.get("sources") or []])
    if a.collect:
        _collection.add([path])
    return MaterialKeepResult(handle=entry["handle"], path=path)
