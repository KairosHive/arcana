"""
The read ops' behaviour, as an agent meets it.

These were the MCP server's tool tests before the tools became ops: search
verdicts, negatives, the person's filters, survey, close looks, moved files,
handles. They run each op line the way the MCP tool does (server.run_lines,
actor agent) against the synthetic dataset, so no model is downloaded.

Run with:  python -m pytest tests/test_read_ops.py -q
"""

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import core, ops, server  # noqa: E402
from arcana.ops import archive  # noqa: E402

import synth  # noqa: E402
from synth import N_FOREST, N_MIXED, N_WATER  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def agent(*lines):
    """(text, number of pictures sent) for a batch run as the agent."""
    res = server.run_lines(list(lines), "agent")
    images = sum(len(s.get("attachments") or []) for s in res["steps"])
    return res["text"], images


def keys(text, ds_id="synth"):
    return [int(m) for m in re.findall(rf"\b{ds_id}:(\d+)", text)]


# ─────────────────────────── privacy ───────────────────────────

def test_blind_is_the_default(tmp, ds):
    text, images = agent("search text synth --phrase water")
    assert images == 0 and "## water" in text


def test_pixels_is_the_persons_switch(tmp, ds):
    synth.pixels(tmp)
    text, images = agent("search text synth --phrase water")
    assert images == 1


def test_no_read_op_writes_anything(tmp, ds):
    synth.pixels(tmp)
    before = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    agent("dataset list", "dataset survey synth", "search text synth --phrase water",
          "search like synth:0", "item show synth:0 --no-describe", "item sheet synth:1 synth:2")
    after = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    assert after == before


# ─────────────────────────── search ───────────────────────────

def test_a_present_idea_is_strong_and_an_absent_one_weak(tmp, ds):
    text, _ = agent('search facets synth --facet \'{"name": "water", "phrases": ["water"]}\' '
                    '--facet \'{"name": "space", "phrases": ["spaceship"]}\' --k 5')
    assert "## water — strong" in text
    assert "## space — weak" in text
    assert "say so" in text.lower()
    assert set(keys(text)[:5]) == set(range(N_WATER))


def test_facets_never_repeat_a_picture(tmp, ds):
    text, _ = agent('search facets synth --facet \'{"name": "a", "phrases": ["water"]}\' '
                    '--facet \'{"name": "b", "phrases": ["water"]}\' --k 6')
    found = keys(text)
    assert len(found) == len(set(found))


def test_negatives_push_the_intruder_down(tmp, ds):
    """Water-and-forest pictures follow pure water; asked to avoid forest, they sink."""
    mixed = set(range(N_WATER + N_FOREST, N_WATER + N_FOREST + N_MIXED))
    plain, _ = agent("search text synth --phrase water --k 12")
    pushed, _ = agent("search text synth --phrase water --negative forest "
                      "--negative-weight 0.6 --k 12")
    assert set(keys(plain)[N_WATER:N_WATER + N_MIXED]) == mixed

    def position(text, key):
        found = keys(text)
        return found.index(key) if key in found else len(found)
    assert sum(position(pushed, m) for m in mixed) > sum(position(plain, m) for m in mixed)


def test_filters_are_the_persons_marks(tmp, ds):
    text, _ = agent("search text synth --phrase water --k 24 --min-stars 4")
    frame = core.open_dataset("synth").frame
    for key in keys(text):
        assert frame.iloc[key]["rating"] >= 4
    assert "filters left" in text


def test_search_like_leaves_out_the_examples(tmp, ds):
    text, _ = agent("search like synth:0 synth:1 --k 3")
    found = keys(text)[2:]                    # the first two are the request line
    assert 0 not in found and 1 not in found
    assert set(found) <= set(range(N_WATER)) | set(range(N_WATER + N_FOREST,
                                                         N_WATER + N_FOREST + N_MIXED))


def test_palette_search_ranks_by_colour(tmp, monkeypatch):
    synth.use(tmp, monkeypatch, palette_features=True)
    text, _ = agent("search palette synth --colour '#2850a0' --k 3")
    assert len(keys(text)) == 3


# ─────────────────────────── survey and close looks ───────────────────────────

def test_survey_lists_clusters_largest_first_with_exemplars(tmp, ds):
    synth.pixels(tmp)
    text, images = agent("dataset survey synth --per-cluster 2")
    assert text.index("Noise") < text.index("Water")
    assert images == 1


def test_item_show_describes_in_blind_mode_and_shows_in_pixels(tmp, ds, monkeypatch):
    from arcana import style_transfer
    monkeypatch.setattr(style_transfer, "describe", lambda *a, **k: "lake, mist")
    text, images = agent("item show synth:0")
    assert "looks like: lake, mist" in text and images == 0
    assert "palette: #" in text

    synth.pixels(tmp)
    _, images = agent("item show synth:0")
    assert images == 1


def test_dataset_list_describes_each_one(tmp, ds):
    text, _ = agent("dataset list")
    assert "synth" in text and "43 pictures" in text and "4 clusters" in text
    assert "marks: stars, colour tags" in text
    assert "FILES MOVED" not in text


def test_moved_files_are_named_not_left_as_grey_tiles(tmp, ds):
    """Search survives a move; pictures do not. The agent should know which is which."""
    for k in range(5):
        os.remove(ds[k])
    assert "FILES MOVED: about 12%" in agent("dataset list")[0]
    assert "(file missing)" in agent("search text synth --phrase water --k 6")[0]


def test_nullable_ratings_read_as_missing_not_an_error(tmp, ds):
    """A frame with pandas' nullable integers holds pd.NA, which == cannot compare."""
    import pandas as pd
    lat = os.path.join(tmp, "latents", "latent_space_synth_image_2d.pkl")
    frame = pd.read_pickle(lat)
    frame["rating"] = frame["rating"].astype("Int64")
    frame.to_pickle(lat)
    text, _ = agent("search text synth --phrase water --k 5")
    assert "★" in text


# ─────────────────────────── small parts ───────────────────────────

def test_handles():
    assert archive.parse_handle("japan:1423") == ("japan", 1423)
    assert archive.parse_handle(archive.handle("field@audio", 7)) == ("field@audio", 7)
    assert archive.split_dataset_id("field@audio") == ("field", "audio")
    for bad in ("japan", "japan:x", ":3", ""):
        with pytest.raises(ValueError, match="handle"):
            archive.parse_handle(bad)


def test_an_unknown_dataset_names_the_known_ones(tmp, ds):
    with pytest.raises(ValueError, match="Known datasets: synth"):
        archive.open_dataset("nope")


def test_an_agent_names_pictures_by_handle_not_path(tmp, ds):
    with pytest.raises(ValueError, match="by handle"):
        archive.resolve_item(ds[0], "agent")
    assert archive.resolve_item(ds[0], "cli").handle == "synth:0"


def test_swatch_bands_follow_weights():
    from arcana.ops.search import swatch
    img = swatch(["#ff0000", "#0000ff"], [3, 1])
    assert img.shape[0] == 128 and abs(img.shape[1] - 256) <= 2
    assert tuple(img[0, 0]) == (0, 0, 255)            # BGR red
    assert tuple(img[0, -1]) == (255, 0, 0)           # BGR blue
    assert 180 <= int((img[0, :, 2] == 255).sum()) <= 200
    with pytest.raises(ValueError):
        swatch(["red"], None)
