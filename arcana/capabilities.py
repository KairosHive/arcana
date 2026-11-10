# capabilities.py -- what each feature of the page works with, written once
#
# The page draws a tag (photo, audio, video) on every tab, panel and card
# action, and disables what the current dataset cannot use. The routes that
# serve those features refuse the same cases in the same sentence. Both read
# this table, so the page and the server cannot disagree.
#
# A feature that rests on ops takes its modalities from them (the intersection:
# a feature works with a modality when every op it needs does). A feature the
# page reaches without an op (search, the map, marks: see OPS_PLAN.md, "what
# stays out of ops") states its modalities here.
#
# Scope says what the feature reads:
#   dataset  the dataset chosen in the context bar
#   item     the pictures it is handed (the collection can mix datasets)
#   none     nothing of either (the Archive tree, the Agent terminal)
#
# See VIDEO_ROADMAP.md, part 1.

from __future__ import annotations

from dataclasses import dataclass

try:
    from . import modality as _modality
except ImportError:                                   # pragma: no cover
    import modality as _modality

IMAGE, AUDIO, VIDEO = _modality.IMAGE, _modality.AUDIO, _modality.VIDEO
PHOTO = (IMAGE,)
SOUND = (AUDIO,)
CLIPS = (VIDEO,)
BOTH = (IMAGE, AUDIO)
ALL = (IMAGE, AUDIO, VIDEO)

TABS = ("datasets", "prompt", "story", "moodboard", "alchemy", "generate", "archive", "agent")


@dataclass(frozen=True)
class Feature:
    id: str                          # "search.text": tab-ish prefix, then the thing
    tabs: tuple[str, ...]            # the tabs that show it
    label: str                       # what a person reads
    scope: str                       # dataset, item or none
    modalities: tuple[str, ...] | None = None    # stated here, or taken from `ops`
    ops: tuple[str, ...] = ()        # the ops it rests on


FEATURES: tuple[Feature, ...] = (
    # ── Datasets
    Feature("dataset.build", ("datasets",), "Build a dataset", "dataset", ops=("dataset create",)),
    Feature("dataset.extras", ("datasets",), "Palette, style and thumbnails", "dataset", PHOTO),
    Feature("dataset.rework", ("datasets",), "Rework", "dataset", ops=("dataset finish",)),
    Feature("dataset.extend", ("datasets",), "Add new", "dataset", ops=("dataset extend",)),
    Feature("dataset.marks", ("datasets",), "Refresh marks", "dataset", ops=("dataset marks",)),
    Feature("dataset.relocate", ("datasets",), "Relocate", "dataset", ops=("dataset relocate",)),
    Feature("dataset.delete", ("datasets",), "Delete", "dataset", ops=("dataset delete",)),
    # ── Prompt Search and Generate Story
    Feature("search.text", ("prompt",), "Text search", "dataset", ALL),
    Feature("search.map", ("prompt", "story"), "Latent map", "dataset", ALL),
    Feature("search.twins", ("prompt",), "Group twins", "dataset", BOTH),
    Feature("search.levels", ("prompt", "story"), "Section level and length", "dataset", CLIPS),
    Feature("search.motion", ("prompt",), "Search by movement", "dataset", CLIPS),
    Feature("search.cut", ("prompt", "story"), "Cut clips together", "dataset", CLIPS),
    Feature("search.across", ("prompt",), "Search across kinds, in the shared space", "dataset",
            ops=("search across",)),
    Feature("search.marks", ("prompt", "story"), "Stars, colour tags and dates", "dataset", PHOTO),
    Feature("search.spectrogram", ("prompt", "story"), "Spectrogram", "dataset", SOUND),
    Feature("search.sets", ("prompt",), "Save to a set", "dataset", ops=("set save",)),
    Feature("search.labels", ("prompt",), "Search by an IP label or sort along an axis", "dataset", PHOTO),
    Feature("story.search", ("story",), "Story, one search per line", "dataset", ALL),
    Feature("story.poetry", ("story",), "Inject Poetry", "dataset", ops=("generate",)),
    Feature("story.save", ("story",), "Save story", "dataset", PHOTO),
    # ── Moodboard
    Feature("moodboard.collection", ("moodboard", "prompt", "story"), "Collection and boards", "item",
            ops=("board add", "board remove")),
    Feature("moodboard.similar", ("moodboard",), "Similarity search by palette and style", "dataset", PHOTO),
    Feature("moodboard.palette", ("moodboard", "prompt"), "Palette strip", "item", PHOTO),
    Feature("moodboard.colour", ("moodboard",), "Colour transfer", "item", ops=("colour transfer",)),
    Feature("moodboard.render", ("moodboard",), "Board page and PDF", "item", ops=("board render", "board pdf")),
    # ── Alchemy
    Feature("alchemy.step", ("alchemy",), "Alchemy step", "item", ops=("alchemy preview", "alchemy make")),
    Feature("alchemy.mask", ("alchemy",), "Masks", "item", ops=("alchemy mask preview",)),
    Feature("alchemy.film", ("alchemy",), "Film", "item", ops=("alchemy film preview", "alchemy film make")),
    Feature("alchemy.pattern", ("alchemy",), "Pattern", "item", ops=("pattern make",)),
    Feature("alchemy.learn", ("alchemy",), "IP labels and axes", "item", PHOTO),
    # ── Generate
    Feature("generate.material", ("generate",), "New material from places in the archive", "none", PHOTO),
    # ── Archive and Agent
    Feature("archive.tree", ("archive",), "What Arcana has made", "none", (IMAGE, AUDIO, VIDEO)),
    Feature("agent.terminal", ("agent",), "Agent", "none", (IMAGE, AUDIO, VIDEO)),
)

BY_ID = {f.id: f for f in FEATURES}


def modalities(feature_id: str) -> tuple[str, ...]:
    """What a feature works with, in display order."""
    f = BY_ID[feature_id]
    if f.modalities is not None:
        return tuple(_modality.tags(f.modalities))
    try:
        from . import ops as _ops
    except ImportError:                               # pragma: no cover
        import ops as _ops
    sets = []
    for name in f.ops:
        found = _ops.get(name)
        if found is None:
            raise KeyError(f"capability {feature_id!r} names an op that does not exist: {name!r}")
        if found.modalities is not None:
            sets.append(set(found.modalities))
    if not sets:
        return _modality.ALL
    return tuple(_modality.tags(set.intersection(*sets)))


def supports(feature_id: str, modality: str) -> bool:
    return modality in modalities(feature_id)


def reason(feature_id: str, modality: str, what: str | None = None) -> str:
    """The one sentence every refusal uses: what the feature takes, and what it was given."""
    f = BY_ID[feature_id]
    takes = _modality.words(modalities(feature_id))
    given = what or (f"This dataset holds {_modality.PLURAL.get(modality, modality)}"
                     if f.scope == "dataset" else f"This is a {_modality.NOUN.get(modality, modality)}")
    return f"{f.label} works with {takes}. {given}."


class Unsupported(ValueError):
    """A feature was asked to work on media it does not take."""

    def __init__(self, feature_id: str, modality: str, what: str | None = None):
        super().__init__(reason(feature_id, modality, what))
        self.feature = feature_id
        self.modality = modality


def require(feature_id: str, modality: str, what: str | None = None) -> None:
    if not supports(feature_id, modality):
        raise Unsupported(feature_id, modality, what)


def require_paths(feature_id: str, paths) -> None:
    """The item-scoped check: every file handed to the feature is of a kind it takes."""
    allowed = modalities(feature_id)
    for p in paths:
        got = _modality.of_path(p)
        if got is not None and got not in allowed:
            import os
            raise Unsupported(feature_id, got,
                              f"{os.path.basename(p)} is a {_modality.NOUN[got]}")


def table() -> dict:
    """Everything the page needs, in one object, served at /api/bootstrap."""
    feats = [{"id": f.id, "tabs": list(f.tabs), "label": f.label, "scope": f.scope,
              "modalities": list(modalities(f.id))} for f in FEATURES]
    tabs = {}
    for tab in TABS:
        seen = set()
        for f in feats:
            if tab in f["tabs"]:
                seen.update(f["modalities"])
        tabs[tab] = _modality.tags(seen)
    # A tab is about a dataset when any of its features is; the page dims such
    # a tab when the chosen dataset has none of them.
    dataset_tabs = sorted({t for f in FEATURES if f.scope == "dataset" for t in f.tabs
                           if t not in ("datasets",)})
    return {"features": feats, "tabs": tabs, "datasetTabs": dataset_tabs,
            "labels": dict(_modality.LABEL), "plural": dict(_modality.PLURAL),
            "order": list(_modality.ALL)}
