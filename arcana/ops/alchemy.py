# alchemy.py -- mixing pictures: the Alchemy tab's ops
#
# One step takes a picture [T], up to three materials [1] [2] [3] with weights,
# a method from one of four families (pixels, tokens, colour, generative), a
# mask any method can be laid through, and a finish. A recipe fills any subset
# of those settings. The engine is arcana/alchemy; these ops are its one door,
# for the tab, the CLI and agents alike.
#
# `alchemy preview` is a read: it draws the step at a small size and keeps
# nothing. `alchemy make` is an effect: it draws at full size and writes the
# picture to output/alchemy with a sidecar naming the op, the arguments and the
# sources, under an `out:` handle any step can take as an input.

from __future__ import annotations

import io
import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .registry import Args, Attachment, Context, Result, TextResult, op, positional

COMBINES = ("and", "or", "but not")
MASK_KEYS = ("invert", "grow", "follow", "ragged", "feather", "amount", "seed")


def _mods():
    """The engine's modules, imported on first use so `op list` never loads torch."""
    try:
        from ..alchemy import finish, inputs, mask, methods, pipeline, recipes
    except ImportError:                               # pragma: no cover
        from alchemy import finish, inputs, mask, methods, pipeline, recipes
    return finish, inputs, mask, methods, pipeline, recipes


def _json_dict(v, what):
    if v is None or isinstance(v, dict):
        return v or {}
    if isinstance(v, str):
        try:
            got = json.loads(v)
        except ValueError:
            raise ValueError(f"{what} is a JSON object, such as '{{\"exposure\": 0.3}}'.") from None
        if not isinstance(got, dict):
            raise ValueError(f"{what} is a JSON object.")
        return got
    raise ValueError(f"{what} is a JSON object.")


def _value(raw: str):
    """A --set value: a number, true or false, or the text as given."""
    low = raw.strip().lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    try:
        return int(raw) if low.lstrip("-").isdigit() else float(raw)
    except ValueError:
        return raw


# ─────────────────────────── the settings of one step ───────────────────────────

class StepSettings(Args):
    method: str | None = Field(None, description="The method, as family.name (`alchemy methods` lists "
                                                 "them). Defaults to the recipe's, or tokens.mix.")
    recipe: str | None = Field(None, description="A recipe whose settings fill what is not given "
                                                 "(`alchemy recipe list`).")
    knobs: dict = Field(default_factory=dict, description="The method's knobs as a JSON object.")
    set: list[str] = Field(default_factory=list, description="One setting as key=value, repeated. A bare "
                           "key is a knob of the method, `finish.<key>` a finish setting, `mask.<key>` a "
                           "mask refinement (invert, grow, follow, ragged, feather, amount, seed).",
                           json_schema_extra={"metavar": "KEY=VALUE"})
    lead: Literal["picture", "material"] | None = Field(None, description="Which leads: the picture with "
                                                        "the materials laid in (default), or the first "
                                                        "material with the picture hidden in it.")
    mask: list[str] = Field(default_factory=list, max_length=6, description="A mask layer, repeated: a "
                            "preset (`alchemy mask presets`), `prompt:<words>`, `thing:<words>`, "
                            "`stuff:<words>`, `colour:#hex`, `spot:x,y,r`, `gradient:<side>`, "
                            "`band:y,h`, `stripes:n,angle`, `noise:<kind>,<size>`, `picture:<handle>`.",
                            json_schema_extra={"metavar": "LAYER"})
    combine: list[Literal["and", "or", "but not"]] = Field(default_factory=list, description="How each "
                                                           "layer after the first meets those before it, "
                                                           "repeated. Defaults to and.")
    coverage: list[float] = Field(default_factory=list, description="Per layer, the share of the picture "
                                  "a field layer covers (0 to 1), repeated. Defaults to the preset's.")
    finish: dict = Field(default_factory=dict, description="Finish settings as a JSON object "
                         "(exposure, contrast, highlights, shadows, temperature, scope and the rest).")
    zoom: float | None = Field(None, ge=0.25, le=4.0, description="How large the materials are drawn: "
                               "1 fits each to the frame, 2 shows half of it twice as large.")
    crop: int | None = Field(None, ge=0, le=99999, description="Which crop of a zoomed material, and the "
                             "seed of anything random in the step.")
    @field_validator("knobs", "finish", mode="before")
    @classmethod
    def _dicts(cls, v, info):
        return _json_dict(v, f"--{info.field_name}")


class StepInputs(StepSettings):
    picture: str | None = positional(0, None, description="The picture [T]: dataset:key, out:<id>, or a file.",
                                     json_schema_extra={"metavar": "PICTURE"})
    material: list[str] = Field(default_factory=list, max_length=3, description="A material [1], [2], [3], "
                                "repeated: dataset:key, out:<id>, or a file.",
                                json_schema_extra={"metavar": "MATERIAL"})
    weight: list[float] = Field(default_factory=list, max_length=3, description="Each material's weight, "
                                "repeated in the materials' order.")


def settings_of(a: StepSettings) -> dict:
    """The spec arcana.alchemy.pipeline reads, from an op's arguments. Only what was given is in it,
    so a recipe fills the rest."""
    finish_m, _, mask_m, methods, pipeline, recipes = _mods()
    knobs, finish, mask = dict(a.knobs), dict(a.finish), {}
    for line in a.set:
        key, eq, raw = line.partition("=")
        key = key.strip().replace("-", "_")
        if not eq or not key:
            raise ValueError(f"--set takes key=value, got {line!r}.")
        if key.startswith("finish."):
            finish[key[7:]] = _value(raw)
        elif key.startswith("mask."):
            if key[5:] not in MASK_KEYS:
                raise ValueError(f"mask.{key[5:]} is no mask refinement. They are {', '.join(MASK_KEYS)}.")
            mask[key[5:]] = _value(raw)
        else:
            knobs[key] = _value(raw)
    if a.mask:
        mask["layers"] = list(a.mask)
    if a.combine:
        mask["combine"] = list(a.combine)
    if a.coverage:
        mask["coverage"] = list(a.coverage)
    known = {f for f in finish_m.Finish.__dataclass_fields__}
    wrong = [k for k in finish if k not in known]
    if wrong:
        raise ValueError(f"No finish setting called {wrong[0]!r}. They are {', '.join(sorted(known))}.")
    spec = {"method": a.method, "recipe": a.recipe, "lead": a.lead, "zoom": a.zoom, "crop": a.crop,
            "knobs": knobs or None, "mask": mask or None, "finish": finish or None,
            "weights": list(getattr(a, "weight", [])) or None}
    spec = {k: v for k, v in spec.items() if v is not None}
    # Unknown knobs are refused here, against the method the recipe settles on, rather than
    # dropped by the engine.
    method = methods.get(pipeline.resolved(spec)["method"])
    names = {k.key for k in method.knobs}
    stray = [k for k in knobs if k not in names]
    if stray:
        raise ValueError(f"{method.id} has no knob called {stray[0]!r}. Its knobs are "
                         f"{', '.join(sorted(names))}. Finish and mask settings take a finish. or mask. prefix.")
    return spec


def _with_pictures(a: StepInputs, spec: dict, ctx: Context) -> dict:
    """The spec with its pictures, each checked against the caller's rules by the one resolver every
    op uses: an agent names pictures by handle, the person and the terminal may also give paths."""
    from . import archive

    for line in (spec.get("mask") or {}).get("layers", []):
        if str(line).startswith("picture:"):
            archive.resolve_item(str(line)[8:], ctx.actor)
    picture = archive.resolve_item(a.picture, ctx.actor).ref if a.picture else None
    materials = [archive.resolve_item(m, ctx.actor).ref for m in a.material]
    return {**spec, "picture": picture, "materials": materials}


def _jpeg(image, side=None, quality=90) -> bytes:
    im = image.convert("RGB")
    if side and max(im.size) > side:
        im = im.copy()
        im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _run(spec, long_side, view):
    """(image, the picture as it was or None, info). `view` compare draws both in one pass."""
    _, _, _, _, pipeline, _ = _mods()
    try:
        if view == "compare":
            return pipeline.run_compare(spec, long_side=long_side)
        image, info = pipeline.run(spec, long_side=long_side, view=view)
        return image, None, info
    except MemoryError as e:
        # call() turns a ValueError into a refusal the caller reads; running short of memory is one.
        raise ValueError(str(e) or "Not enough memory for this step. Close other programs and try again.") from None


# ─────────────────────────── alchemy methods ───────────────────────────

class KnobInfo(BaseModel):
    key: str
    kind: str
    default: Any = None
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    label: str = ""
    doc: str = ""
    options: list = Field(default_factory=list)
    when: list = Field(default_factory=list)


class MethodInfo(BaseModel):
    id: str
    family: str
    label: str
    doc: str
    cost: str
    materials: list[int]
    picture: bool
    needs: list[str] = Field(default_factory=list)
    knobs: list[KnobInfo] = Field(default_factory=list)

    def line(self) -> str:
        lo, hi = self.materials
        mats = f"{lo}" if lo == hi else f"{lo}-{hi}"
        need = f"  needs {', '.join(self.needs)}" if self.needs else ""
        return f"  {self.id:<22} {self.label}: {self.doc} ({mats} material{'s' if hi != 1 else ''}, {self.cost}){need}"


class FinishInfo(BaseModel):
    key: str
    group: str
    label: str
    lo: float
    hi: float
    step: float
    doc: str


class MethodsResult(Result):
    families: dict[str, str]
    methods: list[MethodInfo]
    finish: list[FinishInfo] = Field(default_factory=list)
    detail: bool = False

    def text(self) -> str:
        lines = []
        for fam, doc in self.families.items():
            these = [m for m in self.methods if m.family == fam]
            if not these:
                continue
            lines.append(f"{fam}: {doc}")
            lines += [m.line() for m in these]
            if self.detail:
                for m in these:
                    for k in m.knobs:
                        span = f" {k.lo:g} to {k.hi:g}" if k.lo is not None and k.hi is not None else ""
                        opts = f" ({' | '.join(str(o) for o in k.options)})" if k.options else ""
                        cond = f"  when {' and '.join(f'{c[0]} {c[1]} {c[2]}' for c in k.when)}" if k.when else ""
                        lines.append(f"      {k.key} = {k.default!r}{span}{opts}: {k.doc}{cond}")
            lines.append("")
        if self.finish:
            lines.append("finish (finish.<key>=value; scope whole | mask):")
            for group in dict.fromkeys(f.group for f in self.finish):
                lines.append(f"  {group}: " + ", ".join(f"{f.key} {f.lo:g}..{f.hi:g}" for f in self.finish
                                                         if f.group == group))
        return "\n".join(lines).rstrip() + "\n"


class AlchemyMethodsArgs(Args):
    method: str | None = positional(0, None, description="One method, shown with its knobs.",
                                    json_schema_extra={"metavar": "METHOD"})
    family: Literal["pixels", "tokens", "colour", "generative"] | None = Field(
        None, description="Only this family, shown with its knobs.")


@op("alchemy methods", "read", AlchemyMethodsArgs, MethodsResult,
    doc="""
    Every way Alchemy mixes a picture with its materials. They come in four
    families, pixels, tokens, colour and generative. Each method comes with its knobs,
    their ranges and when each applies. Naming a method or a family shows the
    knobs, and the finish settings follow the list.
    """,
    examples=("alchemy methods", "alchemy methods tokens.mix", "alchemy methods --family pixels"))
def _alchemy_methods(a: AlchemyMethodsArgs, ctx: Context) -> MethodsResult:
    finish, _, _, methods, _, _ = _mods()
    if a.method:
        chosen = [methods.get(a.method)]
    else:
        chosen = [m for m in methods.METHODS.values() if a.family in (None, m.family)]
    return MethodsResult(
        families=dict(methods.FAMILIES),
        methods=[MethodInfo(**m.public()) for m in chosen],
        finish=[FinishInfo(key=k, group=g, label=lab, lo=lo, hi=hi, step=st, doc=d)
                for k, g, lab, lo, hi, st, d in finish.SPEC] if not a.method else [],
        detail=bool(a.method or a.family))


# ─────────────────────────── alchemy memory ───────────────────────────

class HeldModel(BaseModel):
    model: str
    ram_gb: float
    gpu_gb: float
    idle_s: float
    uses: int


class MemoryResult(Result):
    models: list[HeldModel]
    process_gb: float
    committed_gb: float
    free_commit_gb: float
    free_gpu_gb: float | None = None

    def text(self) -> str:
        lines = [f"This process: {self.process_gb:.1f} GB. System: {self.committed_gb:.1f} GB committed, "
                 f"{self.free_commit_gb:.1f} GB free"
                 + (f", {self.free_gpu_gb:.1f} GB free on the card." if self.free_gpu_gb is not None else ".")]
        if not self.models:
            lines.append("Alchemy holds no models.")
        for m in self.models:
            lines.append(f"  {m.model:<26} {m.ram_gb:>5.1f} GB RAM  {m.gpu_gb:>5.1f} GB card  "
                         f"idle {m.idle_s:.0f} s, {m.uses} uses")
        return "\n".join(lines) + "\n"


class AlchemyMemoryArgs(Args):
    pass


@op("alchemy memory", "read", AlchemyMemoryArgs, MemoryResult,
    doc="""
    What Alchemy holds in memory, model by model, and how much the system has
    left. A model unused for a while is let go when memory runs short, and
    loaded again on its next use.
    """,
    examples=("alchemy memory",))
def _alchemy_memory(a: AlchemyMemoryArgs, ctx: Context) -> MemoryResult:
    try:
        from ..alchemy import memory
    except ImportError:                               # pragma: no cover
        from alchemy import memory
    r = memory.MODELS.report()
    return MemoryResult(models=[HeldModel(**m) for m in r["models"]], process_gb=r["process_gb"],
                        committed_gb=round(memory.committed() / memory.GB, 1),
                        free_commit_gb=r["free_commit_gb"], free_gpu_gb=r["free_gpu_gb"])


# ─────────────────────────── alchemy preview, alchemy make ───────────────────────────

class StepResult(Result):
    method: str
    family: str
    size: list[int]
    ms: int
    mask_share: float | None = None
    recipe: str | None = None
    view: str = "result"
    drawn: list[int] | None = None
    id: str | None = None
    path: str | None = None
    withheld: bool = False

    def text(self) -> str:
        bits = [f"{self.method}", f"{self.size[0]} x {self.size[1]}", f"{self.ms} ms"]
        if self.drawn:
            bits.insert(2, f"drawn at {self.drawn[0]} x {self.drawn[1]} and enlarged by Real-ESRGAN")
        if self.recipe:
            bits.insert(0, f"recipe {self.recipe}")
        if self.mask_share is not None:
            bits.append(f"mask lets in {self.mask_share:.0%}")
        head = ", ".join(bits)
        if self.id:
            head = f"{self.id}  {self.path}\n  {head}"
        elif self.view != "result":
            head = f"{self.view}: {head}"
        if self.withheld:
            head += "\nThe picture stays on this machine: the person has not allowed pictures to reach agents."
        return head + "\n"


class AlchemyPreviewArgs(StepInputs):
    view: Literal["result", "mask", "before", "compare"] = Field(
        "result", description="The step's result, its mask over the picture, the picture as it was, or "
                              "compare, which sends the result and then the picture as it was.")
    size: int = Field(640, ge=256, le=1024, description="Longest side of the preview, in pixels.")


@op("alchemy preview", "read", AlchemyPreviewArgs, StepResult,
    doc="""
    One Alchemy step, drawn small and kept nowhere. The picture [T] is mixed
    with up to three materials by a method, through a mask, with a finish. Live
    methods answer in under a second once their model is loaded. The first
    use of a model loads it, which takes a few seconds. Diffusion methods take
    several seconds every time.
    """,
    examples=("alchemy preview japan:12 --material japan:7 --method pixels.displace",
              "alchemy preview japan:12 --material japan:7 --recipe \"hidden: pushed\"",
              "alchemy preview japan:12 --material japan:7 --method colour.lab --mask highlights "
              "--set finish.exposure=0.3",
              "alchemy preview japan:12 --mask \"from the top\" --view mask"))
def _alchemy_preview(a: AlchemyPreviewArgs, ctx: Context) -> StepResult:
    spec = _with_pictures(a, settings_of(a), ctx)
    image, before, info = _run(spec, a.size, a.view)
    out = StepResult(**info, view=a.view, withheld=not ctx.pixels)
    if ctx.pixels:
        out.attachments.append(Attachment(caption=f"{info['method']} {a.view}", data=_jpeg(image)))
        if before is not None:
            out.attachments.append(Attachment(caption="before", data=_jpeg(before)))
    return out


class AlchemyMakeArgs(StepInputs):
    size: int = Field(0, ge=0, le=8192, description="Longest side of the picture kept, in pixels. 0 keeps "
                      "the leading picture's own size. Past the largest size a method draws, Real-ESRGAN "
                      "enlarges the result.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("alchemy make", "effect", AlchemyMakeArgs, StepResult,
    doc="""
    One Alchemy step, drawn at full size and kept. By default the picture keeps
    the size of the photograph it came from. It is written to
    output/edited/<family>/<date>/ and recorded in the ledger, which gives it
    an out: handle that any step, a board or a transfer takes as an input.
    """,
    examples=("alchemy make japan:12 --material japan:7 --recipe \"water light\"",
              "alchemy make japan:12 --material japan:7 --material japan:3 "
              "--weight 0.7 --weight 0.3 --method tokens.weave --size 4096 --background"))
def _alchemy_make(a: AlchemyMakeArgs, ctx: Context) -> Result:
    from . import archive
    from .effects import run_effect

    try:
        from .. import ledger as _ledger
    except ImportError:                               # pragma: no cover
        import ledger as _ledger
    _, inputs, _, methods, pipeline, _ = _mods()
    settings = settings_of(a)
    spec = _with_pictures(a, settings, ctx)
    method = methods.get(pipeline.resolved(spec)["method"])
    sources = [archive.resolve_item(r, ctx.actor) for r in [a.picture, *a.material] if r]

    def compute(progress):
        if progress:
            progress(0.05, f"drawing {method.label}")
        image, _, info = _run(spec, a.size or None, "result")
        lead = os.path.splitext(os.path.basename(sources[0].path))[0] if sources else "kept"
        path = inputs.write(image, method.family, f"{lead}__{method.id.split('.')[-1]}",
                            {"op": "alchemy make", "method": method.id, "sources": [s.ref for s in sources],
                             "settings": settings})
        if progress:
            progress(1.0, "kept")
        return {"path": path, "info": info, "image": image}

    def record(tx, payload):
        entry = _ledger.record(tx, payload["path"], op="alchemy make", args={**settings, "size": a.size},
                               sources=[s.handle or s.ref for s in sources],
                               derived_from=(sources[0].handle or sources[0].ref) if sources else None)
        out = StepResult(**payload["info"], id=entry["handle"], path=payload["path"],
                         withheld=ctx.actor == "agent" and not ctx.pixels)
        if ctx.actor == "agent" and ctx.pixels:
            out.attachments.append(Attachment(caption=entry["handle"], data=_jpeg(payload["image"], side=1024)))
        return out

    return run_effect(ctx, f"alchemy make {method.id}", compute, record, a.background)


# ─────────────────────────── derivatives ───────────────────────────

VARY = {"seed": "the same step drawn again with new seeds: similar textures",
        "material": "the step once with each material: different textures",
        "recipe": "each recipe in turn on the picture: different looks",
        "knob": "one knob stepped from a start to an end: the look growing"}


class DeriveResult(Result):
    picture: str
    vary: str
    ids: list[str]
    paths: list[str]
    labels: list[str]

    def text(self) -> str:
        lines = [f"{len(self.ids)} derivatives of {self.picture}, by {self.vary}:"]
        lines += [f"  {i}  {lab}  {p}" for i, lab, p in zip(self.ids, self.labels, self.paths)]
        return "\n".join(lines) + "\n"


class AlchemyDeriveArgs(StepInputs):
    vary: Literal["seed", "material", "recipe", "knob"] = Field("seed", description="What changes from one "
                                                                 "derivative to the next: " + "; ".join(
                                                                     f"{k}, {v}" for k, v in VARY.items()) + ".")
    count: int = Field(4, ge=2, le=12, description="How many derivatives, with --vary seed or knob.")
    each: list[str] = Field(default_factory=list, max_length=12, description="With --vary material, one "
                            "material per derivative; with --vary recipe, one recipe per derivative. Repeated.",
                            json_schema_extra={"metavar": "EACH"})
    knob: str | None = Field(None, description="With --vary knob, the knob stepped.")
    start: float | None = Field(None, description="With --vary knob, its first value. Defaults to its lowest.")
    end: float | None = Field(None, description="With --vary knob, its last value. Defaults to its highest.")
    size: int = Field(1024, ge=256, le=4096, description="Longest side of each derivative, in pixels.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


def _variants(a: AlchemyDeriveArgs, spec: dict, ctx: Context) -> list[tuple[str, dict]]:
    """(label, spec) for each derivative."""
    from . import archive

    _, _, _, methods, pipeline, recipes = _mods()
    method = methods.get(pipeline.resolved(spec)["method"])
    knobs = dict(spec.get("knobs") or {})
    if a.vary == "seed":
        if not any(k.key == "seed" for k in method.knobs) and method.family != "tokens":
            raise ValueError(f"{method.label} draws the same picture every time. Vary the materials, the "
                             f"recipes or a knob instead.")
        first = int(knobs.get("seed", spec.get("crop", 0)) or 0)
        out = []
        for i in range(a.count):
            seed = first + i
            k = {**knobs, "seed": seed} if any(x.key == "seed" for x in method.knobs) else knobs
            out.append((f"seed {seed}", {**spec, "knobs": k, "crop": seed}))
        return out
    if a.vary == "material":
        if len(a.each) < 2:
            raise ValueError("Give two or more materials with --each, one for each derivative.")
        if method.materials[1] == 0:
            raise ValueError(f"{method.label} takes no material. Choose a method that lays one in.")
        out = []
        for m in a.each:
            item = archive.resolve_item(m, ctx.actor)
            rest = list(spec.get("materials") or [])[1:]
            out.append((os.path.basename(str(item.path)), {**spec, "materials": [item.ref, *rest]}))
        return out
    if a.vary == "recipe":
        if len(a.each) < 2:
            raise ValueError("Give two or more recipes with --each, one for each derivative.")
        out = []
        for r in a.each:
            recipes.get(r)
            base = {k: v for k, v in spec.items() if k in ("picture", "materials", "weights")}
            out.append((r, {**base, "recipe": r}))
        return out
    k = next((x for x in method.knobs if x.key == a.knob), None)
    if k is None or k.kind not in ("float", "int", "angle"):
        nums = [x.key for x in method.knobs if x.kind in ("float", "int", "angle")]
        raise ValueError(f"Name a number knob of {method.label} with --knob: {', '.join(nums) or 'it has none'}.")
    lo = a.start if a.start is not None else k.lo
    hi = a.end if a.end is not None else k.hi
    out = []
    for i in range(a.count):
        v = lo + (hi - lo) * i / (a.count - 1)
        v = int(round(v)) if k.kind == "int" else round(float(v), 4)
        out.append((f"{k.label} {v:g}", {**spec, "knobs": {**knobs, k.key: v}}))
    return out


@op("alchemy derive", "effect", AlchemyDeriveArgs, DeriveResult,
    doc="""
    Versions of one picture, each kept like alchemy make. The step is drawn
    with new seeds, with each material in turn, with each recipe in turn, or
    with one knob stepped. Laid on a film's picture lane, they
    morph into one another, so the film passes through versions of one
    picture instead of from one photograph to the next.
    """,
    examples=("alchemy derive japan:12 --material japan:7 --method generative.controlnet --vary seed --count 5",
              "alchemy derive japan:12 --method tokens.mix --vary material --each japan:7 --each japan:3 "
              "--each japan:40",
              "alchemy derive japan:12 --material japan:7 --vary recipe --each \"water light\" --each \"lit relief\" "
              "--each moody",
              "alchemy derive japan:12 --material japan:7 --method generative.controlnet --vary knob "
              "--knob strength --start 0.2 --end 0.9 --count 6 --background"))
def _alchemy_derive(a: AlchemyDeriveArgs, ctx: Context) -> Result:
    from . import archive
    from .effects import run_effect

    try:
        from .. import ledger as _ledger
    except ImportError:                               # pragma: no cover
        import ledger as _ledger
    if not a.picture:
        raise ValueError("Name the picture to derive from.")
    _, inputs, _, methods, pipeline, _ = _mods()
    settings = settings_of(a)
    spec = _with_pictures(a, settings, ctx)
    variants = _variants(a, spec, ctx)
    source = archive.resolve_item(a.picture, ctx.actor)
    lead = os.path.splitext(os.path.basename(source.path))[0]

    def compute(progress):
        made = []
        for i, (label, sp) in enumerate(variants):
            if progress is not None and getattr(progress, "cancelled", lambda: False)():
                break
            if progress:
                progress(i / len(variants), f"derivative {i + 1} of {len(variants)}: {label}")
            method = methods.get(pipeline.resolved(sp)["method"])
            image, _, _ = _run(sp, a.size, "result")
            path = inputs.write(image, method.family, f"{lead}__derive-{i + 1:02d}",
                                {"op": "alchemy derive", "vary": a.vary, "label": label, "method": method.id,
                                 "sources": [source.ref], "settings": {k: v for k, v in sp.items()
                                                                       if k not in ("picture",)}})
            made.append((label, path, sp))
        if not made:
            raise ValueError("The derivatives stopped before the first.")
        if progress:
            progress(1.0, f"{len(made)} derivatives kept")
        return made

    def record(tx, made):
        ids, paths, labels = [], [], []
        for label, path, sp in made:
            entry = _ledger.record(tx, path, op="alchemy derive", args={"vary": a.vary, "label": label},
                                   sources=[source.handle or source.ref], derived_from=source.handle or source.ref)
            ids.append(entry["handle"])
            paths.append(path)
            labels.append(label)
        return DeriveResult(picture=source.handle or source.ref, vary=a.vary, ids=ids, paths=paths, labels=labels)

    return run_effect(ctx, f"alchemy derive {a.vary}", compute, record, a.background, kind="derive")


# ─────────────────────────── masks ───────────────────────────

class PresetInfo(BaseModel):
    name: str
    group: str
    doc: str
    source: str
    coverage: float | None = None


class MaskPresetsResult(Result):
    presets: list[PresetInfo]
    sources: str

    def text(self) -> str:
        lines = []
        for group in dict.fromkeys(p.group for p in self.presets):
            lines.append(f"{group}:")
            lines += [f"  {p.name:<18} {p.doc}" for p in self.presets if p.group == group]
        return "\n".join(lines) + "\n\n" + self.sources + "\n"


SOURCES_DOC = """A layer is a preset above, or one of:
  prompt:<words>      what the words name: things cut out by Grounding DINO and SAM 2, else CLIPSeg
  thing:<words>       named things only, each cut out
  stuff:<words>       surfaces and qualities (sky, wet, rust), soft
  subject             the main subject's matte (BiRefNet); background is its opposite
  colour:#hex         places near a colour
  spot:x,y,r          a soft disc, in shares of the frame
  gradient:<side>     top, bottom, left, right, or an angle in degrees
  band:y,h            a horizontal band
  stripes:n,angle     n stripes at an angle
  depth band:lo,hi    places between two depths, near is 1
  noise:<kind>,<size> growth, cells, cracks or flow at a scale
  picture:<handle>    another picture's lightness
Layers meet by and, or, but not. Refinements: invert, grow, follow (snap to contours), ragged, feather, amount."""


class AlchemyMaskPresetsArgs(Args):
    group: str | None = Field(None, description="Only this group of presets.")


@op("alchemy mask presets", "read", AlchemyMaskPresetsArgs, MaskPresetsResult,
    doc="""
    The masks a step can be laid through. The presets cover light and colour,
    depth and focus, things and stuff, geometry, procedural growth and the
    materials. Any layer can also be asked for in words. The list ends with
    the syntax for every kind of layer.
    """,
    examples=("alchemy mask presets", "alchemy mask presets --group \"things and stuff\""))
def _alchemy_mask_presets(a: AlchemyMaskPresetsArgs, ctx: Context) -> MaskPresetsResult:
    _, _, mask, _, _, _ = _mods()
    if a.group and a.group not in mask.GROUPS:
        raise ValueError(f"No preset group called {a.group!r}. Groups: {', '.join(mask.GROUPS)}.")
    return MaskPresetsResult(presets=[PresetInfo(name=p.name, group=p.group, doc=p.doc, source=p.source,
                                                 coverage=p.coverage)
                                      for p in mask.PRESETS if a.group in (None, p.group)],
                             sources=SOURCES_DOC)


class MaskPreviewResult(Result):
    share: float
    size: list[int]
    withheld: bool = False

    def text(self) -> str:
        t = f"The mask lets in {self.share:.0%} of the picture ({self.size[0]} x {self.size[1]})."
        if self.withheld:
            t += "\nThe picture stays on this machine: the person has not allowed pictures to reach agents."
        return t + "\n"


class AlchemyMaskPreviewArgs(Args):
    picture: str = positional(0, description="The picture the mask is cut from.",
                              json_schema_extra={"metavar": "PICTURE"})
    mask: list[str] = Field(..., min_length=1, max_length=6, description="A layer, repeated "
                            "(`alchemy mask presets` lists them).", json_schema_extra={"metavar": "LAYER"})
    combine: list[Literal["and", "or", "but not"]] = Field(default_factory=list,
                                                           description="How each later layer meets those before.")
    coverage: list[float] = Field(default_factory=list, description="Per layer, the share a field layer covers.")
    set: list[str] = Field(default_factory=list, description="A refinement as key=value: invert, grow, "
                           "follow, ragged, feather, amount, seed.", json_schema_extra={"metavar": "KEY=VALUE"})
    material: str | None = Field(None, description="The material that `material light` and "
                                 "`material dark` read.")
    size: int = Field(640, ge=256, le=1024, description="Longest side, in pixels.")


@op("alchemy mask preview", "read", AlchemyMaskPreviewArgs, MaskPreviewResult,
    doc="""
    A mask on its own, drawn over its picture in gold. It comes with the share
    of the picture it lets in, to judge a mask before a step runs through it.
    """,
    examples=("alchemy mask preview japan:12 --mask highlights",
              "alchemy mask preview japan:12 --mask \"from the top\" --mask shadows --combine \"but not\" "
              "--set feather=12"))
def _alchemy_mask_preview(a: AlchemyMaskPreviewArgs, ctx: Context) -> MaskPreviewResult:
    _, _, _, _, pipeline, _ = _mods()
    spec = {"layers": list(a.mask), "combine": list(a.combine), "coverage": list(a.coverage)}
    for line in a.set:
        key, eq, raw = line.partition("=")
        key = key.strip().removeprefix("mask.")
        if not eq or key not in MASK_KEYS:
            raise ValueError(f"--set takes one of {', '.join(MASK_KEYS)} as key=value, got {line!r}.")
        spec[key] = _value(raw)
    try:
        image, share = pipeline.mask_view(a.picture, spec, a.material, a.size)
    except MemoryError as e:
        raise ValueError(str(e)) from None
    out = MaskPreviewResult(share=share, size=list(image.size), withheld=not ctx.pixels)
    if ctx.pixels:
        out.attachments.append(Attachment(caption="mask", data=_jpeg(image)))
    return out


# ─────────────────────────── recipes ───────────────────────────

class RecipeInfo(BaseModel):
    name: str
    group: str
    doc: str
    settings: dict = Field(default_factory=dict)
    own: bool = False


class RecipeListResult(Result):
    recipes: list[RecipeInfo]

    def text(self) -> str:
        lines = []
        for group in dict.fromkeys(r.group for r in self.recipes):
            lines.append(f"{group}:")
            lines += [f"  {r.name:<24} {r.doc}" for r in self.recipes if r.group == group]
        return "\n".join(lines) + "\n"


class AlchemyRecipeListArgs(Args):
    group: str | None = Field(None, description="Only this group: material on a thing, picture in a material, "
                                                "materials blended, hidden image, glitch, relief, where, look, "
                                                "or mine.")


@op("alchemy recipe list", "read", AlchemyRecipeListArgs, RecipeListResult,
    doc="""
    The recipes, named settings for Alchemy's flagship setups and the person's
    own. A recipe sets only what it holds, so a mask recipe and
    a finish recipe stack, and the pictures chosen stay.
    """,
    examples=("alchemy recipe list", 'alchemy recipe list --group glitch'))
def _alchemy_recipe_list(a: AlchemyRecipeListArgs, ctx: Context) -> RecipeListResult:
    *_, recipes = _mods()
    return RecipeListResult(recipes=[RecipeInfo(**r.public()) for r in recipes.menu()
                                     if a.group in (None, r.group)])


class RecipeShowResult(Result):
    recipe: RecipeInfo

    def text(self) -> str:
        r = self.recipe
        tag = f"{r.group}, mine" if r.own and r.group != "mine" else r.group
        return (f"{r.name}  ({tag})\n  {r.doc}\n"
                + "\n".join(f"  {k}: {json.dumps(v)}" for k, v in r.settings.items()) + "\n")


class AlchemyRecipeShowArgs(Args):
    name: str = positional(0, description="The recipe's name.", json_schema_extra={"metavar": "NAME"})


@op("alchemy recipe show", "read", AlchemyRecipeShowArgs, RecipeShowResult,
    doc="""
    One recipe's settings. They are its method and knobs, mask, finish, lead
    and zoom, as `alchemy preview` takes them.
    """,
    examples=("alchemy recipe show \"water light\"",))
def _alchemy_recipe_show(a: AlchemyRecipeShowArgs, ctx: Context) -> RecipeShowResult:
    *_, recipes = _mods()
    return RecipeShowResult(recipe=RecipeInfo(**recipes.get(a.name).public()))


class AlchemyRecipeSaveArgs(StepSettings):
    name: str = positional(0, description="The recipe's name. One of the person's own with this name is "
                           "replaced.", json_schema_extra={"metavar": "NAME"})
    doc: str = Field("", max_length=300, description="One line on what the recipe does.")
    group: str = Field("mine", description="The group it is listed under.")


@op("alchemy recipe save", "write", AlchemyRecipeSaveArgs, RecipeShowResult,
    doc="""
    Keep a recipe of the person's own. It holds any subset of a step's settings
    under a name, listed with the built-in recipes. Settings come as for `alchemy
    preview`, and a recipe given with --recipe is folded into the new one.
    """,
    examples=("alchemy recipe save \"warm sky\" --method colour.lab --mask sky --set finish.temperature=20",),
    actors=("person", "cli"))
def _alchemy_recipe_save(a: AlchemyRecipeSaveArgs, ctx: Context) -> RecipeShowResult:
    *_, pipeline, recipes = _mods()
    spec = settings_of(a)
    if spec.get("recipe"):
        spec = pipeline.resolved(spec)
    spec.pop("recipe", None)
    if not {k: v for k, v in spec.items() if k in recipes.SETTING_KEYS}:
        raise ValueError("A recipe needs at least one setting: a method, knobs, a mask or a finish.")
    ctx.tx.touch(recipes.store_path())          # undoable like any other write
    return RecipeShowResult(recipe=RecipeInfo(**recipes.save(a.name, spec, a.doc, a.group).public()))


# ─────────────────────────── films ───────────────────────────
# A film is pictures morphed through a token space on a timeline, with any number of a step keyed in
# time (arcana/alchemy/film.py holds the format). Films are kept by name in the data directory.

def _film_mod():
    try:
        from ..alchemy import film
    except ImportError:                               # pragma: no cover
        from alchemy import film
    return film


def _film_of(name: str | None, spec: dict | None, ctx: Context, depth: int = 0) -> dict:
    """The film an op names or carries, its pictures checked against the caller's rules: an agent
    names pictures by handle. A film held on the picture lane is checked the same way."""
    from . import archive

    film = _film_mod()
    spec = spec if spec else film.load(name) if name else None
    if spec is None:
        raise ValueError("Name a kept film, or give one with --spec.")
    for p in spec.get("pictures") or []:
        ref = str(p.get("ref", ""))
        if film.is_film(ref):
            if depth < film.DEPTH:
                _film_of(ref[len(film.FILM_REF):], None, ctx, depth + 1)
            continue
        archive.resolve_item(ref, ctx.actor)
    for k in spec.get("looks") or []:
        for m in (k.get("settings") or {}).get("materials") or []:
            if m:
                archive.resolve_item(str(m), ctx.actor)
    return film.normalised(spec)


def _film_arg():
    return positional(0, None, description="A kept film, by name.", json_schema_extra={"metavar": "FILM"})


def _spec_field():
    return Field(None, description="A film as a JSON object, in place of a kept one.")


class FilmInfo(BaseModel):
    name: str
    length: float
    fps: int
    size: int
    space: str
    loop: bool
    pictures: list[str]
    passes: list[float]
    looks: int
    lanes: list[str]

    def text(self) -> str:
        film = _film_mod()
        lines = [f"{self.name}  {film.length_text(self.length)} at {self.fps} fps, {self.size} px, "
                 f"{self.space}{', looping' if self.loop else ''}"]
        lines += [f"  {t:7.2f} s  {ref}" for t, ref in zip(self.passes, self.pictures)]
        lines.append(f"  {self.looks} look{'s' if self.looks != 1 else ''}; lanes: {', '.join(self.lanes) or 'none'}")
        return "\n".join(lines)


def _info_of(f: dict) -> FilmInfo:
    return FilmInfo(name=f["name"], length=f["length"], fps=f["fps"], size=f["size"], space=f["space"],
                    loop=f["loop"], pictures=[p["ref"] for p in f["pictures"]],
                    passes=[p["t"] for p in f["pictures"]], looks=len(f["looks"]), lanes=list(f["lanes"]))


class FilmListResult(Result):
    films: list[FilmInfo]

    def text(self) -> str:
        return ("\n\n".join(f.text() for f in self.films) or "No films kept yet.") + "\n"


class AlchemyFilmListArgs(Args):
    pass


@op("alchemy film list", "read", AlchemyFilmListArgs, FilmListResult,
    doc="""
    The films kept so far. Each comes with its length, its pictures and when
    the film passes them, its looks and its lanes.
    """,
    examples=("alchemy film list",))
def _alchemy_film_list(a: AlchemyFilmListArgs, ctx: Context) -> FilmListResult:
    film = _film_mod()
    out = []
    for name in film.listing():
        try:
            out.append(_info_of(film.normalised(film.load(name))))
        except ValueError:
            continue
    return FilmListResult(films=out)


class FilmShowResult(Result):
    film: FilmInfo
    spec: dict

    def text(self) -> str:
        return self.film.text() + "\n"


class FilmSpecArgs(Args):
    film: str | None = _film_arg()
    spec: dict | None = _spec_field()

    @field_validator("spec", mode="before")
    @classmethod
    def _spec(cls, v):
        return _json_dict(v, "--spec") if v is not None else None


class AlchemyFilmShowArgs(FilmSpecArgs):
    pass


EXAMPLE_SPEC = ('\'{"name": "rites", "pictures": [{"t": 0, "ref": "japan:12"}, {"t": 4, "ref": "japan:7"}], '
                '"lanes": {"finish.saturation": [{"t": 0, "v": 0}, {"t": 6, "v": -100}]}}\'')


@op("alchemy film show", "read", AlchemyFilmShowArgs, FilmShowResult,
    doc="""
    One film in full, every default filled in. It lists its pictures and their
    times, its looks and its lanes, and the spec it returns is what alchemy
    film save takes.
    """,
    examples=("alchemy film show --spec " + EXAMPLE_SPEC,))
def _alchemy_film_show(a: AlchemyFilmShowArgs, ctx: Context) -> FilmShowResult:
    f = _film_of(a.film, a.spec, ctx)
    return FilmShowResult(film=_info_of(f), spec=f)


class AlchemyFilmSaveArgs(Args):
    spec: dict = Field(description="The film as a JSON object. Its name is the name it is kept under.")

    @field_validator("spec", mode="before")
    @classmethod
    def _spec(cls, v):
        return _json_dict(v, "--spec")


@op("alchemy film save", "write", AlchemyFilmSaveArgs, FilmShowResult,
    doc="""
    Keep a film under its name, replacing one of the same name. Undo takes it
    back like any other write.
    """,
    examples=("alchemy film save --spec " + EXAMPLE_SPEC,))
def _alchemy_film_save(a: AlchemyFilmSaveArgs, ctx: Context) -> FilmShowResult:
    film = _film_mod()
    f = _film_of(None, a.spec, ctx)
    ctx.tx.touch(film.film_path(f["name"]))
    f = film.save(f)
    return FilmShowResult(film=_info_of(f), spec=f)


class AlchemyFilmSpaceArgs(FilmSpecArgs):
    shape: Literal["even", "faster", "slower", "smooth"] = Field("even", description="even (one rate), faster "
                                                                 "(speeding up), slower (slowing down) or "
                                                                 "smooth (slow at both ends).")
    by: Literal["change", "time"] = Field("change", description="change: each passage as long as the picture "
                                          "changes along it, so the morph keeps one rate; time: every passage "
                                          "as long.")
    start: float | None = Field(None, ge=0, description="When the first picture comes. Defaults to where it is.")
    end: float | None = Field(None, ge=0, description="When the last picture comes. Defaults to where it is.")


@op("alchemy film space", "write", AlchemyFilmSpaceArgs, FilmShowResult,
    doc="""
    Re-time a film's pictures between the first and the last. By change, each
    passage takes as long as the picture changes along it, so the morph runs
    at one rate, and the shape bends that rate. A held film keeps its length.
    A kept film is saved re-timed; a film given with --spec is returned.
    """,
    examples=("alchemy film space --spec " + EXAMPLE_SPEC + " --by time --shape faster",))
def _alchemy_film_space(a: AlchemyFilmSpaceArgs, ctx: Context) -> FilmShowResult:
    film = _film_mod()
    f = _film_of(a.film, a.spec, ctx)
    try:
        f = film.normalised(film.spaced(f, a.shape, a.start, a.end, by=a.by))
    except MemoryError as e:
        raise ValueError(str(e)) from None
    if a.film and not a.spec:
        ctx.tx.touch(film.film_path(f["name"]))
        f = film.save(f)
    return FilmShowResult(film=_info_of(f), spec=f)


class AlchemyFilmDeleteArgs(Args):
    film: str = positional(0, description="The kept film to delete, by name.", json_schema_extra={"metavar": "FILM"})


@op("alchemy film delete", "write", AlchemyFilmDeleteArgs, TextResult,
    doc="""
    Delete a kept film. Its rendered videos stay in output, and undo brings
    the film back.
    """,
    examples=("alchemy film delete rites",), actors=("person", "cli"))
def _alchemy_film_delete(a: AlchemyFilmDeleteArgs, ctx: Context) -> TextResult:
    film = _film_mod()
    film.load(a.film)
    path = film.film_path(a.film)
    ctx.tx.touch(path)
    os.remove(path)
    return TextResult(body=f"Deleted the film {a.film}.\n")


# ─────────────────────────── film templates ───────────────────────────

class TemplateInfo(BaseModel):
    name: str
    length: float
    looks: int
    lanes: list[str]

    def text(self) -> str:
        lanes = ", ".join(self.lanes) or "none"
        return f"{self.name}  keyed over {self.length:.1f} s, {self.looks} look(s), lanes: {lanes}"


def _template_info(t: dict) -> TemplateInfo:
    return TemplateInfo(name=t["name"], length=float(t.get("length") or 0), looks=len(t.get("looks") or []),
                        lanes=list(t.get("lanes") or {}))


class TemplateListResult(Result):
    templates: list[TemplateInfo]

    def text(self) -> str:
        return ("\n".join(t.text() for t in self.templates) or "No film templates kept yet.") + "\n"


class AlchemyFilmTemplateListArgs(Args):
    pass


@op("alchemy template list", "read", AlchemyFilmTemplateListArgs, TemplateListResult,
    doc="""
    The film templates kept so far.

    A template holds a film's settings, looks and lanes, without its pictures,
    ready to apply to another film.
    """,
    examples=("alchemy template list",))
def _alchemy_film_template_list(a: AlchemyFilmTemplateListArgs, ctx: Context) -> TemplateListResult:
    film = _film_mod()
    out = []
    for name in film.template_listing():
        try:
            out.append(_template_info(film.load_template(name)))
        except (OSError, ValueError, KeyError):
            continue
    return TemplateListResult(templates=out)


class AlchemyFilmTemplateSaveArgs(FilmSpecArgs):
    name: str = Field(description="The template's name. A template of the same name is replaced.")


class TemplateResult(Result):
    template: TemplateInfo

    def text(self) -> str:
        return "Kept the template " + self.template.text() + "\n"


@op("alchemy template save", "write", AlchemyFilmTemplateSaveArgs, TemplateResult,
    doc="""
    Keep a film's settings, looks and lanes as a template, without its
    pictures. Applied to another film, the keys stretch to that film's
    length. Undo takes it back.
    """,
    examples=("alchemy template save --name \"slow fade\" --spec " + EXAMPLE_SPEC,))
def _alchemy_film_template_save(a: AlchemyFilmTemplateSaveArgs, ctx: Context) -> TemplateResult:
    film = _film_mod()
    t = film.as_template(_film_of(a.film, a.spec, ctx), a.name)
    ctx.tx.touch(film.template_path(t["name"]))
    return TemplateResult(template=_template_info(film.save_template(t)))


class AlchemyFilmTemplateApplyArgs(FilmSpecArgs):
    template: str = Field(description="The kept template to apply, by name.")


class TemplateAppliedResult(Result):
    template: str
    spec: dict

    def text(self) -> str:
        lanes = ", ".join(self.spec.get("lanes") or {}) or "none"
        return (f"The film with {self.template}: {len(self.spec.get('looks') or [])} look(s), lanes: {lanes}.\n"
                "alchemy film save --spec takes the spec.\n")


@op("alchemy template apply", "read", AlchemyFilmTemplateApplyArgs, TemplateAppliedResult,
    doc="""
    A film with a template's settings, looks and lanes in place of its own.
    Its pictures stay. The keys stretch from the template's length to the
    film's. It changes nothing kept, and alchemy film save keeps the result.
    """,
    examples=("alchemy template apply --template \"slow fade\" --spec " + EXAMPLE_SPEC,))
def _alchemy_film_template_apply(a: AlchemyFilmTemplateApplyArgs, ctx: Context) -> TemplateAppliedResult:
    from . import archive

    film = _film_mod()
    spec = a.spec if a.spec else film.load(a.film) if a.film else {}
    for p in spec.get("pictures") or []:
        archive.resolve_item(str(p.get("ref", "")), ctx.actor)
    t = film.load_template(a.template)
    return TemplateAppliedResult(template=t["name"], spec=film.templated(spec, t))


class AlchemyFilmTemplateDeleteArgs(Args):
    template: str = positional(0, description="The kept template to delete, by name.",
                               json_schema_extra={"metavar": "TEMPLATE"})


@op("alchemy template delete", "write", AlchemyFilmTemplateDeleteArgs, TextResult,
    doc="""
    Delete a kept film template. Films made with it keep their keys, and undo
    brings the template back.
    """,
    examples=('alchemy template delete "slow fade"',), actors=("person", "cli"))
def _alchemy_film_template_delete(a: AlchemyFilmTemplateDeleteArgs, ctx: Context) -> TextResult:
    film = _film_mod()
    t = film.load_template(a.template)
    path = film.template_path(a.template)
    ctx.tx.touch(path)
    os.remove(path)
    return TextResult(body=f"Deleted the film template {t['name']}.\n")


class FrameResult(Result):
    film: str
    at: float
    length: float
    size: list[int]
    ms: int
    withheld: bool = False

    def text(self) -> str:
        t = f"{self.film} at {self.at:.2f} s of {self.length:.2f}, {self.size[0]} x {self.size[1]}, {self.ms} ms"
        if self.withheld:
            t += "\nThe picture stays on this machine: the person has not allowed pictures to reach agents."
        return t + "\n"


class AlchemyFilmPreviewArgs(FilmSpecArgs):
    at: float = Field(0.0, ge=0, description="The moment to draw, in seconds.")
    size: int = Field(512, ge=256, le=1024, description="Longest side of the frame, in pixels.")


@op("alchemy film preview", "read", AlchemyFilmPreviewArgs, FrameResult,
    doc="""
    One frame of a film at any moment, kept nowhere. It is the morph at that
    time, the look's step and its finish. The pictures are encoded once and
    kept while only the looks and lanes change, so scrubbing is quick.
    """,
    examples=("alchemy film preview --spec " + EXAMPLE_SPEC + " --at 2",))
def _alchemy_film_preview(a: AlchemyFilmPreviewArgs, ctx: Context) -> FrameResult:
    import time as _time

    film = _film_mod()
    f = _film_of(a.film, a.spec, ctx)
    t0 = _time.perf_counter()
    try:
        ready = film.prepared(f, a.size)
        image = film.to_image(ready.frame(min(a.at, f["length"])))
    except MemoryError as e:
        raise ValueError(str(e)) from None
    out = FrameResult(film=f["name"], at=a.at, length=f["length"], size=[image.width, image.height],
                      ms=round((_time.perf_counter() - t0) * 1000), withheld=not ctx.pixels)
    if ctx.pixels:
        out.attachments.append(Attachment(caption=f"{f['name']} at {a.at:.2f} s", data=_jpeg(image)))
    return out


class FilmMadeResult(Result):
    id: str
    path: str
    frames: int
    fps: int
    seconds: float
    size: list[int]

    def text(self) -> str:
        return (f"{self.id}  {self.path}\n  {self.frames} frames, {self.seconds:.1f} s at {self.fps} fps, "
                f"{self.size[0]} x {self.size[1]}\n")


class AlchemyFilmMakeArgs(FilmSpecArgs):
    size: int | None = Field(None, ge=256, le=2048, description="Long side the frames are drawn at. "
                             "Defaults to the film's.")
    target: int | None = Field(None, ge=256, le=8192, description="Long side of the video. Past the drawn "
                               "size, Real-ESRGAN enlarges every frame.")
    fps: int | None = Field(None, ge=6, le=60, description="Frames a second. Defaults to the film's.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("alchemy film make", "effect", AlchemyFilmMakeArgs, FilmMadeResult,
    doc="""
    Render a film to an H.264 video in output/edited/video/<date>/. The film
    is written beside it as JSON, and the ledger records the video as derived
    from its pictures. With --background it runs as a job, and `job show`
    reports its progress frame by frame.
    """,
    examples=("alchemy film make rites --background",
              "alchemy film make rites --size 768 --target 1920 --fps 30"))
def _alchemy_film_make(a: AlchemyFilmMakeArgs, ctx: Context) -> Result:
    import datetime

    from .effects import run_effect

    try:
        from .. import ledger as _ledger
        from .. import paths as _paths
    except ImportError:                               # pragma: no cover
        import ledger as _ledger
        import paths as _paths
    film = _film_mod()
    f = _film_of(a.film, a.spec, ctx)
    folder = _paths.edited_dir("video")
    path = os.path.join(folder, f"{film.slug(f['name'])}__{datetime.datetime.now().strftime('%H%M%S')}.mp4")

    def compute(progress):
        made = film.render(f, path, progress, size=a.size, fps=a.fps, target=a.target)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump({"film": f, "made": made}, fh, indent=1)
        return made

    def record(tx, made):
        entry = _ledger.record(tx, made["path"], op="alchemy film make",
                               args={"film": f["name"], "size": a.size, "target": a.target, "fps": made["fps"]},
                               sources=made["pictures"], derived_from=made["pictures"][0])
        return FilmMadeResult(id=entry["handle"], path=made["path"], frames=made["frames"], fps=made["fps"],
                              seconds=made["seconds"], size=made["size"])

    return run_effect(ctx, f"alchemy film make {f['name']}", compute, record, a.background, kind="film")
