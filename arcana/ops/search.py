# search.py -- finding pictures by words, by example and by colour
#
# The ranking itself lives in core.py and db.py, shared with the Dash app.
# These ops add the agent's needs on top: several phrasings and negatives in
# one query, a page that is not twenty versions of one photograph, a verdict
# on whether the idea is in the archive at all, and the person's own marks as
# filters.

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from . import archive
from .archive import Hit, dataset_arg
from .registry import Args, Context, Result, op, positional

try:
    from .. import core
except ImportError:                                   # pragma: no cover
    import core


# ─────────────────────────── shared pieces ───────────────────────────

class Facet(BaseModel):
    """One visual idea, said a few ways."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="A short name for this angle, e.g. 'water as weight'.")
    phrases: list[str] = Field(
        min_length=1, max_length=6,
        description="1-6 concrete visual phrasings of the same idea: things a camera "
                    "could see ('rain on a dark street at night', 'wet asphalt reflecting "
                    "lights'), not abstractions ('melancholy').")
    negatives: list[str] = Field(
        default_factory=list, max_length=4,
        description="Things that keep intruding and should be pushed away, e.g. 'umbrella', "
                    "'people'.")
    negative_weight: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="How hard negatives push. 0.3-0.5 works; 1.0 tends to return noise.")


class FilterArgs(Args):
    """The person's own decisions about their pictures, from ON1 and Lightroom sidecars."""
    min_stars: int = Field(0, ge=-1, le=5,
                           description="Only pictures rated at least this. -1 keeps only unrated ones.")
    tag: list[str] = Field(default_factory=list,
                           description="Colour tags to keep, e.g. Red. (none) means untagged.")
    date_from: str | None = Field(None, description="Captured on or after, YYYY-MM-DD.",
                                  json_schema_extra={"metavar": "DATE"})
    date_to: str | None = Field(None, description="Captured on or before, YYYY-MM-DD.",
                                json_schema_extra={"metavar": "DATE"})

    def allowed(self, ds) -> set | None:
        return core.allowed_keys(ds, self.min_stars, self.tag or None, self.date_from, self.date_to)


Per = dict(ge=1, le=48)


class FacetResult(BaseModel):
    name: str
    verdict: str                       # strong, fair, weak, none, unrated
    best_p: float | None = None        # mean p of the best five
    hits: list[Hit] = Field(default_factory=list)
    lone: list[int] = Field(default_factory=list)   # numbers that score well inside a weak facet
    skipped_duplicates: int = 0
    considered: int | None = None      # set when filters narrowed the dataset
    total: int = 0

    def text(self) -> str:
        head = f"## {self.name} — {self.verdict}"
        if self.best_p is not None and self.hits:
            head += f" (best five average p {self.best_p:.2f})"
        lines = [head]
        if self.verdict in ("weak", "none"):
            lines.append("The archive has little for this. Say so rather than presenting these as a match.")
            if self.lone:
                nums = ", ".join(f"#{n}" for n in self.lone)
                one = len(self.lone) == 1
                lines.append(f"Though {nums} score{'s' if one else ''} well on "
                             f"{'its' if one else 'their'} own: look before using.")
        lines += [h.line() for h in self.hits]
        if self.skipped_duplicates:
            lines.append(f"({self.skipped_duplicates} near-duplicates skipped)")
        if self.considered is not None:
            lines.append(f"(filters left {self.considered:,} of {self.total:,})")
        return "\n".join(lines)


class SearchResult(Result):
    dataset: str
    facets: list[FacetResult]

    def text(self) -> str:
        blocks = [f.text() for f in self.facets]
        summary = "; ".join(f"{f.name}: {f.verdict}" for f in self.facets)
        blocks.append("Summary: " + summary + ". p is how sure CLIP is that a picture shows "
                      "the facet's phrases rather than something generic.")
        return "\n\n".join(blocks) + "\n"


def run_facets(ds_id: str, facets: list[Facet], k: int, diversity: float,
               filters: FilterArgs, ctx: Context) -> SearchResult:
    ds = archive.open_dataset(ds_id)
    allowed = filters.allowed(ds)
    shown: set = set()
    num = 0
    out = SearchResult(dataset=ds_id, facets=[])
    for facet in facets:
        q, pos = core.query_vectors(facet.phrases, ds.index, ds.modality,
                                    negatives=facet.negatives,
                                    negative_weight=facet.negative_weight)
        r = core.rank(ds, q, k, allowed=allowed, exclude=shown, diversity=diversity)
        p = core.confidence(ds, pos, [h.key for h in r.hits])
        nums = list(range(num + 1, num + 1 + len(r.hits)))
        num += len(r.hits)
        fr = FacetResult(name=facet.name, verdict="none", total=len(ds.idx2path),
                         skipped_duplicates=r.skipped_duplicates,
                         considered=r.considered if allowed is not None else None)
        if p is None:
            fr.verdict = "unrated" if r.hits else "none"
        else:
            # Judge the five best matches, not the first five shown: with
            # diversity on, the page interleaves weaker picks for variety.
            best = p[np.argsort([-h.score for h in r.hits])]
            fr.verdict = core.strength(best)
            if r.hits:
                fr.best_p = float(np.mean(best[:5]))
            if fr.verdict in ("weak", "none"):
                # A weak idea can still have one real instance -- the one cat in
                # an archive of temples -- and on a moodboard one can be enough.
                # A lone high score can also be a false positive, hence "look".
                fr.lone = [n for n, x in zip(nums, p) if x >= core.STRONG_P]
        pal = {} if ctx.sheets else archive.palettes(ds, [h.key for h in r.hits])
        for i, h in enumerate(r.hits):
            shown.add(h.key)
            if p is not None:
                metric, score = f"p {p[i]:.2f}", float(p[i])
            else:
                metric, score = f"z {h.z:.1f}", float(h.z)
            fr.hits.append(archive.make_hit(nums[i], ds_id, ds, h.key, h.path, metric, score, pal))
        out.facets.append(fr)
        if ctx.sheets and fr.hits:
            out.attachments.append(archive.sheet([h.path for h in fr.hits], nums,
                                                 title=facet.name, caption=facet.name))
    return out


# ─────────────────────────── search text ───────────────────────────

class SearchTextArgs(FilterArgs):
    dataset: str = dataset_arg()
    phrase: list[str] = Field(min_length=1, max_length=6,
                              description="A concrete visual phrasing of the idea. Repeat for "
                                          "several phrasings of the same idea.")
    negative: list[str] = Field(default_factory=list, max_length=4,
                                description="Something that keeps intruding, to push away.")
    negative_weight: float = Field(0.4, ge=0.0, le=1.0,
                                   description="How hard negatives push. 0.3 to 0.5 works.")
    k: int = Field(12, **Per, description="Pictures to return.")
    diversity: float = Field(0.0, ge=0.0, le=1.0,
                             description="0 ranks by match alone, as the app's search does. "
                                         "Higher spreads the page across different takes.")
    name: str | None = Field(None, description="A name for the idea in the result. "
                                               "Defaults to the first phrase.")


@op("search text", "read", SearchTextArgs, SearchResult,
    modalities=("image", "audio", "video"),
    doc="""
    Pictures matching one visual idea, said one or several ways. With one
    phrase and the defaults it returns the hits the app's Prompt Search shows,
    near-twins folded into one. Each picture has p, how sure CLIP is that it
    shows the phrases rather than something generic, and the idea is marked
    strong, fair or weak. Weak means the archive has little for it.

    CLIP sees concrete things (objects, light, weather, texture, technique) and
    misses abstractions, so write what a camera would see. For several ideas at
    once, with no picture repeated across them, use `search facets`.
    """,
    examples=('search text japan --phrase "rain on a dark street"',
              'search text japan --phrase "wet asphalt" --phrase "neon reflections" '
              '--negative umbrella --k 8 --diversity 0.3',
              'search text japan --phrase "temple roof" --min-stars 3'))
def _search_text(a: SearchTextArgs, ctx: Context) -> SearchResult:
    facet = Facet(name=a.name or a.phrase[0], phrases=a.phrase, negatives=a.negative,
                  negative_weight=a.negative_weight)
    return run_facets(a.dataset, [facet], a.k, a.diversity, a, ctx)


# ─────────────────────────── search facets ───────────────────────────

class SearchFacetsArgs(FilterArgs):
    dataset: str = dataset_arg()
    facet: list[Facet] = Field(min_length=1, max_length=8,
                               description='One visual idea as JSON, '
                                           '{"name": ..., "phrases": [...], "negatives": [...]}. '
                                           'Repeat for each idea.')
    k: int = Field(12, ge=1, le=24, description="Pictures per facet.")
    diversity: float = Field(0.3, ge=0.0, le=1.0,
                             description="0 is closest matches only. Higher spreads results "
                                         "across different takes on the idea.")


@op("search facets", "read", SearchFacetsArgs, SearchResult,
    modalities=("image", "audio", "video"),
    doc="""
    Several visual ideas at once, one facet per idea. Each facet returns up to
    k pictures, near-duplicates removed, and never repeats a picture an earlier
    facet already showed, so the facets of a moodboard draw on different
    pictures. Each facet is marked strong, fair or weak.
    """,
    examples=('search facets japan '
              '--facet \'{"name": "water as weight", "phrases": ["heavy rain", "flooded street"]}\' '
              '--facet \'{"name": "shelter", "phrases": ["empty bus stop at night"], '
              '"negatives": ["people"]}\' --k 6',))
def _search_facets(a: SearchFacetsArgs, ctx: Context) -> SearchResult:
    return run_facets(a.dataset, a.facet, a.k, a.diversity, a, ctx)


# ─────────────────────────── search like ───────────────────────────

class LikeResult(Result):
    handles: list[str]
    mode: str
    note: str
    hits: list[Hit]

    def text(self) -> str:
        lines = [f"More like {', '.join(self.handles)}. {self.note}"]
        lines += [h.line() for h in self.hits]
        return "\n".join(lines) + "\n"


class SearchLikeArgs(Args):
    handles: list[str] = positional(0, min_length=1, max_length=8,
                                    description="Pictures to search by, as dataset:key.",
                                    json_schema_extra={"metavar": "HANDLE"})
    mode: Literal["semantic", "palette", "style"] = Field(
        "semantic", description="semantic (what the pictures show), palette (their colours) "
                                "or style (texture and grain).")
    k: int = Field(12, **Per, description="Pictures to return.")
    diversity: float = Field(0.3, ge=0.0, le=1.0,
                             description="Semantic mode only. Higher spreads the results.")
    exclude: list[str] = Field(default_factory=list,
                               description="A handle not to return again.",
                               json_schema_extra={"metavar": "HANDLE"})


@op("search like", "read", SearchLikeArgs, LikeResult,
    modalities=("image", "audio", "video"),
    doc="""
    Pictures like these, searched by example. Semantic mode blends several
    examples and needs no encoder, since it reads the vectors already in the
    index. Palette and style modes use the first picture and need the
    dataset's palette or style features.
    """,
    examples=("search like japan:12 japan:7", "search like japan:12 --mode palette --k 6"))
def _search_like(a: SearchLikeArgs, ctx: Context) -> LikeResult:
    ds_id, ds, seeds = archive.resolve_handles(a.handles)
    skip = set(seeds) | {archive.parse_handle(h)[1] for h in a.exclude}
    if a.mode == "semantic":
        r = core.rank(ds, core.example_vector(ds.index, seeds), a.k, exclude=skip,
                      diversity=a.diversity)
        found = [(h.key, h.path, f"sim {h.score:.2f}", h.score) for h in r.hits]
        note = ("Ranked by what the pictures show (CLIP); sim is picture-to-picture "
                "similarity, 1 being identical.")
    else:
        try:
            from .. import db as _db
        except ImportError:                           # pragma: no cover
            import db as _db
        query = ds.idx2path[seeds[0]]
        fn = _db.search_by_palette if a.mode == "palette" else _db.search_by_style
        kwargs = {"method": "emd"} if a.mode == "palette" else {"method": "gram"}
        try:
            ranked = fn(query, ds.name, ds.idx2path, top_k=a.k + len(skip) + 8, **kwargs)
        except FileNotFoundError:
            raise ValueError(f"{ds_id} has no {a.mode} features. The person can add them "
                             f"with Rework on the Datasets tab.")
        key_of = {p: kk for kk, p in ds.idx2path.items()}
        found = []
        for p, sim in ranked:
            kk = key_of.get(p)
            if kk is None or kk in skip:
                continue
            found.append((kk, p, None, float(sim)))
            if len(found) >= a.k:
                break
        note = (f"Ranked by {'colour palette' if a.mode == 'palette' else 'texture and grain'} "
                f"of the first picture only.")
    pal = {} if ctx.sheets else archive.palettes(ds, [kk for kk, *_ in found])
    hits = [archive.make_hit(i, ds_id, ds, kk, p, metric, score, pal)
            for i, (kk, p, metric, score) in enumerate(found, 1)]
    out = LikeResult(handles=list(a.handles), mode=a.mode, note=note, hits=hits)
    if ctx.sheets and hits:
        out.attachments.append(archive.sheet([h.path for h in hits],
                                             caption=f"more like {', '.join(a.handles)}"))
    return out


# ─────────────────────────── search palette ───────────────────────────

def parse_colour(entry: str) -> tuple[str, float]:
    """'#1f2a36' or '#1f2a36:0.4' to ('#1f2a36', weight)."""
    hexpart, _, w = entry.strip().partition(":")
    h = hexpart.strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"{entry!r} is not a colour. Write #1f2a36, or #1f2a36:0.4 with a weight.")
    try:
        weight = float(w) if w else 1.0
    except ValueError:
        raise ValueError(f"{entry!r} has a weight that is not a number.") from None
    if weight <= 0:
        raise ValueError(f"{entry!r} needs a positive weight.")
    return "#" + h.lower(), weight


def swatch(hexes: list[str], weights: list[float] | None) -> np.ndarray:
    """A palette as a picture (BGR), each colour's band as wide as its weight."""
    cols = []
    for h in hexes:
        h = h.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"'{h}' is not a hex colour like #1f2a36.")
        cols.append((int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)))     # BGR
    w = np.asarray(weights if weights else [1.0] * len(cols), dtype=np.float64)
    if len(w) != len(cols) or (w <= 0).any():
        raise ValueError("weights, when given, are one positive number per colour.")
    widths = np.maximum(1, np.round(256 * w / w.sum())).astype(int)
    return np.concatenate([np.tile(np.array(c, np.uint8), (128, n, 1))
                           for c, n in zip(cols, widths)], axis=1)


class PaletteSearchResult(Result):
    dataset: str
    colours: list[str]
    hits: list[Hit]

    def text(self) -> str:
        lines = [f"Pictures closest to {' '.join(self.colours)} in {self.dataset}, best first."]
        lines += [h.line() for h in self.hits]
        return "\n".join(lines) + "\n"


class SearchPaletteArgs(Args):
    dataset: str = dataset_arg()
    colour: list[str] = Field(min_length=1, max_length=8,
                              description="A colour as #rrggbb, optionally with its share, "
                                          "#1f2a36:0.4. Repeat for each colour.",
                              json_schema_extra={"metavar": "HEX"})
    k: int = Field(12, **Per, description="Pictures to return.")

    @field_validator("colour")
    @classmethod
    def _colours_parse(cls, v):
        for entry in v:
            parse_colour(entry)
        return v


@op("search palette", "read", SearchPaletteArgs, PaletteSearchResult,
    modalities=("image",),
    doc="""
    Pictures whose colours match a palette you choose. They are ranked by earth
    mover's distance between dominant colours, which needs the dataset's
    palette features.
    """,
    examples=("search palette japan --colour '#1f2a36:0.6' --colour '#b5651d:0.4' --k 8",))
def _search_palette(a: SearchPaletteArgs, ctx: Context) -> PaletteSearchResult:
    ds = archive.open_dataset(a.dataset)
    parsed = [parse_colour(c) for c in a.colour]
    hexes, weights = [h for h, _ in parsed], [w for _, w in parsed]
    try:
        from .. import db as _db
    except ImportError:                               # pragma: no cover
        import db as _db
    try:
        ranked = _db.search_by_palette(swatch(hexes, weights), ds.name, ds.idx2path,
                                       method="emd", n_colors=max(len(hexes), 4), top_k=a.k)
    except FileNotFoundError:
        raise ValueError(f"{a.dataset} has no palette features. The person can add them with "
                         f"Rework on the Datasets tab.")
    key_of = {p: kk for kk, p in ds.idx2path.items()}
    found = [(key_of[p], p, float(sim)) for p, sim in ranked if p in key_of]
    pal = {} if ctx.sheets else archive.palettes(ds, [kk for kk, _, _ in found])
    hits = [archive.make_hit(i, a.dataset, ds, kk, p, None, sim, pal)
            for i, (kk, p, sim) in enumerate(found, 1)]
    out = PaletteSearchResult(dataset=a.dataset, colours=hexes, hits=hits)
    if ctx.sheets and hits:
        out.attachments.append(archive.sheet([h.path for h in hits],
                                             caption=f"closest to {' '.join(hexes)}"))
    return out
