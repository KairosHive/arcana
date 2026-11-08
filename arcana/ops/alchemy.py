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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .registry import Args, Attachment, Context, Result, op, positional

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


def _with_pictures(a: StepInputs, spec: dict) -> dict:
    return {**spec, "picture": a.picture, "materials": list(a.material)}


def _jpeg(image, side=None, quality=90) -> bytes:
    im = image.convert("RGB")
    if side and max(im.size) > side:
        im = im.copy()
        im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _run(spec, long_side, view):
    _, _, _, _, pipeline, _ = _mods()
    try:
        return pipeline.run(spec, long_side=long_side, view=view)
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


# ─────────────────────────── alchemy preview, alchemy make ───────────────────────────

class StepResult(Result):
    method: str
    family: str
    size: list[int]
    ms: int
    mask_share: float | None = None
    recipe: str | None = None
    view: str = "result"
    id: str | None = None
    path: str | None = None
    withheld: bool = False

    def text(self) -> str:
        bits = [f"{self.method}", f"{self.size[0]} x {self.size[1]}", f"{self.ms} ms"]
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
    spec = _with_pictures(a, settings_of(a))
    image, info = _run(spec, a.size, "result" if a.view == "compare" else a.view)
    out = StepResult(**info, view=a.view, withheld=not ctx.pixels)
    if ctx.pixels:
        out.attachments.append(Attachment(caption=f"{info['method']} {a.view}", data=_jpeg(image)))
        if a.view == "compare":
            before, _ = _run(spec, a.size, "before")
            out.attachments.append(Attachment(caption="before", data=_jpeg(before)))
    return out


class AlchemyMakeArgs(StepInputs):
    size: int = Field(1024, ge=256, le=2048, description="Longest side of the picture made, in pixels, "
                      "before any enlarging.")
    enlarge: bool = Field(False, description="Enlarge the result four times with Real-ESRGAN.")


@op("alchemy make", "effect", AlchemyMakeArgs, StepResult,
    doc="""
    One Alchemy step, drawn at full size and kept. The picture is written to
    output/alchemy with a sidecar naming this op, its arguments and its
    sources. It answers with an out: handle that any step, or a board, can
    take as an input.
    """,
    examples=("alchemy make japan:12 --material japan:7 --recipe \"water light\"",
              "alchemy make japan:12 --material japan:7 --material japan:3 "
              "--weight 0.7 --weight 0.3 --method tokens.weave --size 1536 --enlarge"))
def _alchemy_make(a: AlchemyMakeArgs, ctx: Context) -> StepResult:
    _, inputs, _, _, _, _ = _mods()
    settings = settings_of(a)
    spec = {**_with_pictures(a, settings), "enlarge": a.enlarge}
    image, info = _run(spec, a.size, "result")
    sources = [r for r in [a.picture, *a.material] if r]
    oid, path = inputs.save(image, "alchemy make", {**settings, "size": a.size, "enlarge": a.enlarge}, sources)
    out = StepResult(**info, id=oid, path=path, withheld=ctx.actor == "agent" and not ctx.pixels)
    if ctx.actor == "agent" and ctx.pixels:
        out.attachments.append(Attachment(caption=oid, data=_jpeg(image, side=1024)))
    return out


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
    group: str | None = Field(None, description="Only this group: tokens, hidden, pixels, masks, "
                                                "finish, or mine.")


@op("alchemy recipe list", "read", AlchemyRecipeListArgs, RecipeListResult,
    doc="""
    The recipes, named settings for Alchemy's flagship setups and the person's
    own. A recipe sets only what it holds, so a mask recipe and
    a finish recipe stack, and the pictures chosen stay.
    """,
    examples=("alchemy recipe list", "alchemy recipe list --group tokens"))
def _alchemy_recipe_list(a: AlchemyRecipeListArgs, ctx: Context) -> RecipeListResult:
    *_, recipes = _mods()
    return RecipeListResult(recipes=[RecipeInfo(**r.public()) for r in recipes.every()
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


@op("alchemy recipe save", "effect", AlchemyRecipeSaveArgs, RecipeShowResult,
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
    return RecipeShowResult(recipe=RecipeInfo(**recipes.save(a.name, spec, a.doc, a.group).public()))
