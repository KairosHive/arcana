# shared.py -- the shared space: embed a dataset into it, search across kinds
#
# `dataset embed` gives a dataset vectors in the space text, pictures, sound
# and video share. `search across` then ranks any embedded dataset against a
# query made of words and of items from anywhere: a sound against the photos,
# a photo against the clips, several of them at once with some pushed away.
# See spaces.py and VIDEO_ROADMAP.md, 2.5.

from __future__ import annotations

from pydantic import Field

from . import archive
from .archive import Hit, dataset_arg
from .effects import run_effect
from .registry import Args, Context, Result, op

try:
    from .. import modality as _modality
    from .. import omni as _omni
    from .. import spaces as _spaces
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import omni as _omni
    import spaces as _spaces

EVERY = ("image", "audio", "video")


def _need_encoder() -> None:
    ok, why = _omni.available()
    if not ok:
        raise ValueError(why)


# ─────────────────────────── dataset embed ───────────────────────────

class EmbedResult(Result):
    dataset: str
    added: int = 0
    kept: int = 0
    skipped: int = 0
    total: int = 0

    def text(self) -> str:
        t = (f"{self.dataset}: {self.added:,} embedded into the shared space"
             + (f", {self.kept:,} were there already" if self.kept else "")
             + (f", {self.skipped:,} could not be read" if self.skipped else "") + ".")
        left = self.total - self.added - self.kept - self.skipped
        if left > 0:
            t += f" {left:,} are left; run it again to go on."
        return t + "\n"


class EmbedArgs(Args):
    dataset: str = dataset_arg()
    limit: int = Field(0, ge=0, description="Embed at most this many items now. 0 does them all.")
    sound: bool = Field(False, description="For a video dataset: embed each section's sound track "
                                           "instead of its picture, so searches can listen.")
    background: bool = Field(True, description="Run as a job. --no-background waits.")


@op("dataset embed", "effect", EmbedArgs, Result,
    modalities=EVERY,
    doc="""
    Embed a dataset into the space text, pictures, sound and video share.
    There they are all comparable. The dataset's own index is untouched, and this adds a
    second set of vectors beside it, which `search across` reads. It picks up
    where it stopped, so running it again embeds only what is new. The first
    run downloads the encoder (1.5 GB) into an environment of its own.
    """,
    examples=("dataset embed japan", "dataset embed field@audio --limit 200"))
def _embed(a: EmbedArgs, ctx: Context) -> Result:
    _need_encoder()
    ds = archive.open_dataset(a.dataset)
    name, modality = archive.split_dataset_id(a.dataset)

    def compute(progress):
        return _spaces.build(name, modality, ds.idx2path, progress=progress, limit=a.limit,
                             should_cancel=getattr(progress, "cancelled", None),
                             channel="sound" if a.sound else "picture")

    def record(tx, got):
        return EmbedResult(dataset=a.dataset, added=got["added"], kept=got["kept"],
                           skipped=got["skipped"], total=got["total"])

    return run_effect(ctx, f"dataset embed {a.dataset}", compute, record, a.background, kind="index")


# ─────────────────────────── search across ───────────────────────────

class AcrossResult(Result):
    dataset: str
    query: str
    hits: list[Hit]

    def text(self) -> str:
        lines = [f"In {self.dataset}, across kinds: {self.query}."]
        lines += [h.line() for h in self.hits] or ["Nothing found."]
        return "\n".join(lines) + "\n"


class AcrossArgs(Args):
    dataset: str = dataset_arg()
    text: list[str] = Field(default_factory=list, max_length=6,
                            description="Words for what to find. Repeat for several.")
    like: list[str] = Field(default_factory=list, max_length=8,
                            description="An item to search by, from any dataset: a picture, a sound "
                                        "or a clip, as dataset:key.",
                            json_schema_extra={"metavar": "HANDLE"})
    not_text: list[str] = Field(default_factory=list, max_length=4,
                                description="Words for something to move away from.")
    not_like: list[str] = Field(default_factory=list, max_length=4,
                                description="An item to move away from, as dataset:key.",
                                json_schema_extra={"metavar": "HANDLE"})
    away: float = Field(0.4, ge=0.0, le=1.0, description="How hard the things to avoid push.")
    listen: bool = Field(False, description="For a video dataset: match the sections' sound tracks "
                                            "instead of their pictures. Needs `dataset embed --sound`.")
    k: int = Field(12, ge=1, le=96, description="Items to return.")


def _item_vector(handle: str, actor: str):
    """The shared-space vector of an item: read from its dataset when it was embedded, made now otherwise."""
    it = archive.resolve_item(handle, actor)
    if it.dataset and it.handle:
        name, modality = archive.split_dataset_id(it.dataset)
        got = _spaces.load(name, modality)
        if got is not None:
            key = archive.parse_handle(it.handle)[1]
            at = (got[0] == key).nonzero()[0]
            if len(at):
                return got[1][int(at[0])]
    return _omni.encode([_omni.item_for(it.path)])[0]


@op("search across", "read", AcrossArgs, AcrossResult,
    modalities=EVERY,
    doc="""
    Search a dataset with words and with items of any kind, in the shared space.
    A sound finds pictures, a picture finds clips, and several terms mix in one
    query. Each term is weighed on its own scale before they are added,
    so a sound and a phrase count equally. The dataset must have been embedded
    with `dataset embed` first. An item used as a query is embedded on the spot
    when its own dataset has not been.
    """,
    examples=('search across japan --text "rain on a tin roof"',
              "search across japan --like field@audio:12",
              'search across japan --like field@audio:12 --text "fog" --not-text "people" --k 8'))
def _across(a: AcrossArgs, ctx: Context) -> AcrossResult:
    if not (a.text or a.like):
        raise ValueError("Give at least one --text or one --like.")
    name, modality = archive.split_dataset_id(a.dataset)
    ds = archive.open_dataset(a.dataset)
    got = _spaces.load(name, modality, _spaces.SOUND if a.listen else _spaces.DEFAULT)
    if got is None:
        how = f"dataset embed {a.dataset}" + (" --sound" if a.listen else "")
        raise ValueError(f"{a.dataset} is not in the shared space yet"
                         + (" by its sound" if a.listen else "") + f". Run `{how}` first.")
    _need_encoder()
    keys, vectors = got
    terms = []
    words = list(a.text) + list(a.not_text)
    if words:
        vecs = _omni.encode([_omni.text(w) for w in words])
        terms += [(v, 1.0) for v in vecs[:len(a.text)]]
        terms += [(v, -a.away) for v in vecs[len(a.text):]]
    terms += [(_item_vector(h, ctx.actor), 1.0) for h in a.like]
    terms += [(_item_vector(h, ctx.actor), -a.away) for h in a.not_like]
    # An item of this same dataset used as the query should not come back as its own best match.
    own = {archive.parse_handle(h)[1] for h in a.like if archive.split_handle_ds(h) == a.dataset}
    allowed = (set(int(k) for k in keys) - own) if own else None
    ranked = _spaces.rank(keys, vectors, terms, a.k, allowed)
    hits = [archive.make_hit(n + 1, a.dataset, ds, key, ds.idx2path[key], f"z {score:.1f}", score)
            for n, (key, score) in enumerate(ranked) if key in ds.idx2path]
    said = [f'"{t}"' for t in a.text] + list(a.like)
    less = [f'"{t}"' for t in a.not_text] + list(a.not_like)
    query = " + ".join(said) + (" without " + ", ".join(less) if less else "")
    return AcrossResult(dataset=a.dataset, query=query, hits=hits)
