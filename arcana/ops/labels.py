# labels.py -- IP labels as ops: named positions in IP-Adapter space (ip_labels.py)
#
#   label make ice --set ice                    the centre of a set's pictures
#   label make "frost" --words "frost on glass" --dataset textures --n 50
#   label make "this rock" --picture textures:12
#   label make ice-grain --set ice --crops 4    each picture as 4 crops, which keeps the grain
#   label list / label show ice / label delete ice
#
# Alchemy's diffusion methods take a label in their `label` knob; Search takes
# one in place of words (`search label`).

from __future__ import annotations

from pydantic import BaseModel, Field

from . import archive
from .registry import Args, Context, Result, op, positional

try:
    from .. import boards as _boards
    from .. import ip_labels as _labels
except ImportError:                                   # pragma: no cover
    import boards as _boards
    import ip_labels as _labels


def label_arg(index: int = 0):
    return positional(index, description="The label's id, as `label list` shows it.",
                      json_schema_extra={"metavar": "LABEL"})


class LabelRow(BaseModel):
    slug: str
    name: str
    origin: dict = Field(default_factory=dict)
    members: int = 0
    pictures: int = 0
    created: float = 0
    nearest: list[str] = Field(default_factory=list)
    tightness: float | None = None
    error: str = ""


def _origin_text(o: dict) -> str:
    kind = o.get("kind")
    if kind == "set":
        return f"set {o.get('set')}" + (f", {o.get('crops')} crops each" if o.get("crops") else "")
    if kind == "words":
        return f"\"{o.get('text')}\" in {o.get('dataset')}, top {o.get('n')}"
    if kind == "picture":
        return f"picture {o.get('picture')}"
    return kind or "?"


class LabelListResult(Result):
    labels: list[LabelRow]

    def text(self) -> str:
        if not self.labels:
            return "No IP labels yet. `label make ice --set ice` measures one on a set.\n"
        width = max(len(r.slug) for r in self.labels) + 2
        return "".join(f"{r.slug:<{width}}{r.pictures} pictures  ·  {_origin_text(r.origin)}"
                       f"{f'  ·  alike {r.tightness:.2f}' if r.tightness is not None else ''}\n"
                       for r in self.labels)


class LabelListArgs(Args):
    pass


@op("label list", "read", LabelListArgs, LabelListResult,
    doc="""
    Every IP label, newest first. Each line gives what it was measured on, how
    many pictures, and how alike they are (mean cosine of a member to the
    centre). Near 1 is one look, lower a mixed bag that draws as an average.
    """,
    examples=("label list",))
def _label_list(a: LabelListArgs, ctx: Context) -> LabelListResult:
    return LabelListResult(labels=[LabelRow(**r) for r in _labels.listing()])


class LabelShowArgs(Args):
    label: str = label_arg()
    sheet: bool = Field(False, description="Add a contact sheet of the members nearest the centre.")


class LabelShowResult(Result):
    label: LabelRow

    def text(self) -> str:
        r = self.label
        lines = [f"{r.name} ({r.slug}): {r.members} vectors from {r.pictures} pictures, {_origin_text(r.origin)}"]
        if r.tightness is not None:
            lines.append(f"  alike {r.tightness:.2f} (mean cosine of a member to the centre)")
        lines += [f"  nearest the centre: {p}" for p in r.nearest[:3]]
        return "\n".join(lines) + "\n"


@op("label show", "read", LabelShowArgs, LabelShowResult,
    doc="""
    One IP label in full. It lists its origin, its size, and the pictures
    nearest its centre, which show what it draws.
    """,
    examples=("label show ice", "label show ice --sheet"))
def _label_show(a: LabelShowArgs, ctx: Context) -> LabelShowResult:
    lab = _labels.load(a.label)
    out = LabelShowResult(label=LabelRow(**lab.public()))
    if a.sheet and ctx.pixels:
        near = lab.nearest(12)
        if near:
            out.attachments.append(archive.sheet(near, list(range(1, len(near) + 1)),
                                                 caption=f"label {a.label}", title=lab.name))
    return out


class LabelMakeArgs(Args):
    name: str = positional(0, description="The label's name, e.g. \"ice\".")
    set: str | None = Field(None, description="Measure it on a set's pictures.")
    words: str | None = Field(None, description="Measure it on the pictures that best match these words.")
    dataset: str | None = Field(None, description="With --words, the dataset searched.")
    n: int = Field(50, ge=1, le=1000, description="With --words, how many of the best matches.")
    picture: str | None = Field(None, description="One picture's own position.")
    crops: int = Field(0, ge=0, le=16, description="Each picture as this many crops at a third of its "
                       "short side, encoded: keeps the grain a whole picture loses. 0 reads each picture "
                       "whole, from a ViT-H index when one holds it.")
    id: str | None = Field(None, description="The label's id. Defaults to the name as a slug.")
    background: bool = Field(False, description="Return a job id at once instead of waiting.")


@op("label make", "effect", LabelMakeArgs, LabelShowResult,
    doc="""
    Measure an IP label, a named place in IP-Adapter space. Alchemy's
    diffusion methods draw from it. Give exactly one source: --set, the
    centre of a set's pictures; --words with --dataset, the centre of the best
    matches; --picture, one picture's position. Pictures in a dataset indexed
    with CLIP ViT-H/14 are read from its index, so a label of a few hundred
    takes a second; others are encoded by the adapter's encoder. Making a label
    under an existing id replaces it.
    """,
    examples=("label make ice --set ice", 'label make frost --words "frost on glass" --dataset textures',
              "label make ice-grain --set ice --crops 4 --background"))
def _label_make(a: LabelMakeArgs, ctx: Context) -> Result:
    from .effects import run_effect

    given = [x for x in (a.set, a.words, a.picture) if x]
    if len(given) != 1:
        raise ValueError("Give exactly one of --set, --words or --picture.")
    if a.words and not a.dataset:
        raise ValueError("--words searches a dataset: add --dataset.")
    if a.set:
        rec = _boards.load(a.set)
        if rec is None:
            raise ValueError(f"No set called `{a.set}`.")
        paths, _ = _boards.present(rec)
        origin = {"kind": "set", "set": a.set, "crops": a.crops}
    elif a.words:
        paths = _labels.search_paths(a.words, a.dataset, a.n)
        origin = {"kind": "words", "text": a.words, "dataset": a.dataset, "n": a.n, "crops": a.crops}
    else:
        item = archive.resolve_item(a.picture, ctx.actor)
        paths = [item.path]
        origin = {"kind": "picture", "picture": item.handle or a.picture, "crops": a.crops}
    if not paths:
        raise ValueError("There are no pictures to measure.")

    def compute(progress):
        cancelled = getattr(progress, "cancelled", None)
        members, files, world, stats = _labels.vectors_for(paths, a.crops, progress, cancelled)
        lab = _labels.save(a.name, members, files, {**origin, **stats}, world, slug=a.id)
        return lab.public()

    def record(tx, payload):
        return LabelShowResult(label=LabelRow(**payload))

    return run_effect(ctx, f"label make {a.name}", compute, record, a.background, kind="label")


class LabelDeleteArgs(Args):
    label: str = label_arg()


class LabelDeleteResult(Result):
    deleted: bool

    def text(self) -> str:
        return ("Deleted.\n" if self.deleted else "There was no such label.\n")


@op("label delete", "write", LabelDeleteArgs, LabelDeleteResult,
    doc="Delete an IP label's file. Recipes that name it fall back to no label.",
    examples=("label delete ice",))
def _label_delete(a: LabelDeleteArgs, ctx: Context) -> LabelDeleteResult:
    return LabelDeleteResult(deleted=_labels.delete(a.label))


# ─────────────────────────── concept axes ───────────────────────────

def _end_members(end: str, dataset: str | None, n: int):
    """An axis end's member vectors: set:<id>, label:<id> or words:<words> (matched in --dataset)."""
    kind, _, ident = end.partition(":")
    kind, ident = kind.strip(), ident.strip()
    if kind in ("set", "label"):
        try:
            from .. import material_draw as _md
        except ImportError:                           # pragma: no cover
            import material_draw as _md
        return _md.source_label({"kind": kind, "id": ident}).members
    if kind == "words":
        if not dataset:
            raise ValueError("An end given in words is matched in a dataset: add --dataset.")
        members, _, _, _ = _labels.vectors_for(_labels.search_paths(ident, dataset, n))
        return members
    raise ValueError(f"An axis end is set:<id>, label:<id> or words:<words>. Got {end!r}.")


class AxisRow(BaseModel):
    slug: str
    name: str
    pos: str | None = None
    neg: str | None = None
    apart: float | None = None
    created: float = 0


class AxisListResult(Result):
    axes: list[AxisRow]

    def text(self) -> str:
        if not self.axes:
            return "No axes yet. `axis make wet --pos set:wet --neg set:dry` fits one.\n"
        return "".join(f"{a.slug}: {a.neg} → {a.pos}  ·  {a.apart} spreads apart\n" for a in self.axes)


class AxisListArgs(Args):
    pass


@op("axis list", "read", AxisListArgs, AxisListResult,
    doc="""
    Every concept axis, with its two ends. Each also says how many of the
    archive's spreads separate them; near 0, the encoder cannot tell them apart.
    """,
    examples=("axis list",))
def _axis_list(a: AxisListArgs, ctx: Context) -> AxisListResult:
    return AxisListResult(axes=[AxisRow(**r) for r in _labels.axis_listing()])


class AxisMakeArgs(Args):
    name: str = positional(0, description="The axis's name, the quality at its positive end, e.g. \"wet\".")
    pos: str = Field(description="The positive end: set:<id>, label:<id> or words:<words>.")
    neg: str = Field(description="The negative end, the same way.")
    dataset: str | None = Field(None, description="Where ends given in words are matched.")
    n: int = Field(40, ge=5, le=500, description="With words, how many of the best matches stand for an end.")
    id: str | None = Field(None, description="The axis's id. Defaults to the name as a slug.")


class AxisMakeResult(Result):
    axis: AxisRow

    def text(self) -> str:
        return f"Fitted {self.axis.slug}: {self.axis.neg} → {self.axis.pos}, {self.axis.apart} spreads apart.\n"


@op("axis make", "effect", AxisMakeArgs, AxisMakeResult,
    doc="""
    Fit a concept axis, the direction in IP space from one place to another.
    It sorts the archive in Search and moves a material in Alchemy and Generate.
    """,
    examples=('axis make wet --pos "words:wet and glistening" --neg "words:bone dry, dusty" --dataset japan',),
    needs=("diffusers",))
def _axis_make(a: AxisMakeArgs, ctx: Context) -> AxisMakeResult:
    pos, neg = _end_members(a.pos, a.dataset, a.n), _end_members(a.neg, a.dataset, a.n)
    ax = _labels.make_axis(a.name, pos, neg, {"pos": a.pos, "neg": a.neg}, slug=a.id)
    return AxisMakeResult(axis=AxisRow(**ax.public()))


class AxisDeleteArgs(Args):
    axis: str = positional(0, description="The axis's id.", json_schema_extra={"metavar": "AXIS"})


@op("axis delete", "write", AxisDeleteArgs, LabelDeleteResult,
    doc="Delete a concept axis.", examples=("axis delete wet",))
def _axis_delete(a: AxisDeleteArgs, ctx: Context) -> LabelDeleteResult:
    return LabelDeleteResult(deleted=_labels.delete_axis(a.axis))
