"""
Tests for the MCP server.

The tools are called as plain functions against a small synthetic dataset in a
scratch data dir, with the text encoder replaced by fixed vectors: no model is
downloaded. One test at the end starts the real server as a subprocess and
talks to it through an MCP client, because the failure that matters most there
-- a stray print() corrupting the protocol -- only shows up over a real pipe.

Run with:  python -m pytest tests/test_mcp_server.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mcp")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arcana import core, mcp_server as srv  # noqa: E402

DIM = 64
N_WATER, N_FOREST, N_MIXED, N_NOISE = 5, 5, 3, 30


def _axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


# The fixed "text encoder": each word is a direction, and every other phrase --
# which includes the generic descriptions core.confidence() weighs a query
# against -- is one shared "ordinary picture" direction.
WORDS = {"water": _axis(0), "forest": _axis(1), "spaceship": _axis(63)}
GENERIC = _axis(62)


def _write_dataset(tmp):
    """
    A dataset of water, forest, water-and-forest, and ordinary pictures, laid
    out as db.py does.

    The ordinary pictures lean towards GENERIC, the way real photographs of
    nothing in particular sit closest to "a photo" or "a landscape" in CLIP:
    that is what lets a query for something absent come back weak. The mixed
    pictures carry enough noise not to be near-copies of each other, so the
    negatives test has three of them to watch.
    """
    from PIL import Image
    from usearch.index import Index

    rng = np.random.default_rng(7)
    photos = os.path.join(tmp, "photos")
    os.makedirs(photos)
    db = os.path.join(tmp, "databases")
    lat = os.path.join(tmp, "latents")
    os.makedirs(db)
    os.makedirs(lat)

    vecs, labels, colours = [], [], []
    for _ in range(N_WATER):
        vecs.append(_axis(0) + 0.05 * rng.standard_normal(DIM)); labels.append("Water")
    for _ in range(N_FOREST):
        vecs.append(_axis(1) + 0.05 * rng.standard_normal(DIM)); labels.append("Forest")
    for _ in range(N_MIXED):
        vecs.append(_axis(0) + _axis(1) + 0.15 * rng.standard_normal(DIM)); labels.append("Mixed")
    for _ in range(N_NOISE):
        vecs.append(GENERIC + 0.3 * rng.standard_normal(DIM)); labels.append("Noise")

    n = len(vecs)
    index = Index(ndim=DIM, metric="cos")
    idx2path = {}
    for k, v in enumerate(vecs):
        index.add(k, np.asarray(v, np.float32))
        p = os.path.join(photos, f"img_{k:03d}.jpg")
        shade = (40 + 5 * k) % 255
        Image.new("RGB", (64, 48), (shade, 90, 200 - shade % 150)).save(p, "JPEG")
        idx2path[k] = p
    with open(os.path.join(db, "index_synth_image.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)

    ratings = [5, 4, None, 3, 2] * (n // 5) + [None] * (n % 5)
    pd.DataFrame({
        "x": np.arange(n, dtype=np.float32), "y": np.zeros(n, np.float32),
        "path": list(idx2path.values()), "label": labels,
        "rating": ratings[:n], "colour": ["Red" if k % 2 else "" for k in range(n)],
        "captured": pd.to_datetime(["2025-06-05"] * n),
    }).to_pickle(os.path.join(lat, "latent_space_synth_image_2d.pkl"))
    return idx2path


@pytest.fixture
def ds(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    monkeypatch.setattr(core, "encode_query",
                        lambda q, index, modality="image": WORDS.get(q, GENERIC))
    monkeypatch.setattr(core, "_DISTRACTOR_VECS", {})
    return _write_dataset(tmp)


def _pixels(tmp, on=True):
    with open(os.path.join(tmp, "agent.json"), "w", encoding="utf-8") as fh:
        json.dump({"pixels": on}, fh)


def _texts(out):
    return "\n".join(c.text for c in out if c.type == "text")


def _images(out):
    return [c for c in out if c.type == "image"]


def _keys(out, ds_id="synth"):
    """Handles in a result, in order, as keys."""
    import re
    return [int(m) for m in re.findall(rf"\b{ds_id}:(\d+)", _texts(out))]


# ─────────────────────────── privacy ───────────────────────────

def test_blind_is_the_default_and_says_so(tmp, ds):
    out = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])])
    assert not _images(out)
    assert "Mode: blind" in _texts(out)


def test_pixels_is_the_persons_switch(tmp, ds):
    _pixels(tmp)
    out = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])])
    assert len(_images(out)) == 1
    assert "Mode: pixels" in _texts(out)


def test_no_tool_can_change_the_setting_or_write_anything(tmp, ds):
    """Every tool is read-only, and declares itself so to the client."""
    import anyio
    tools = anyio.run(srv.mcp.list_tools)
    assert {t.name for t in tools} == {"list_datasets", "survey", "search",
                                       "more_like", "find_by_palette", "inspect"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)

    _pixels(tmp)
    before = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    srv.do_list_datasets()
    srv.do_survey("synth")
    srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])])
    srv.do_more_like(["synth:0"])
    srv.do_inspect(["synth:0"])
    after = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    assert after == before


# ─────────────────────────── search ───────────────────────────

def test_a_present_idea_is_strong_and_an_absent_one_weak(tmp, ds):
    out = srv.do_search("synth", [srv.Facet(name="water", phrases=["water"]),
                                  srv.Facet(name="space", phrases=["spaceship"])], k=5)
    text = _texts(out)
    assert "## water — strong" in text
    assert "## space — weak" in text
    assert "say so" in text.lower()
    assert set(_keys(out)[:5]) == set(range(N_WATER))


def test_facets_never_repeat_a_picture(tmp, ds):
    out = srv.do_search("synth", [srv.Facet(name="a", phrases=["water"]),
                                  srv.Facet(name="b", phrases=["water"])], k=6)
    keys = _keys(out)
    assert len(keys) == len(set(keys))


def test_negatives_push_the_intruder_down(tmp, ds):
    """Water-and-forest pictures follow pure water; asked to avoid forest, they sink."""
    mixed = set(range(N_WATER + N_FOREST, N_WATER + N_FOREST + N_MIXED))
    plain = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])], k=12, diversity=0)
    pushed = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"],
                                               negatives=["forest"], negative_weight=0.6)],
                           k=12, diversity=0)
    assert set(_keys(plain)[N_WATER:N_WATER + N_MIXED]) == mixed

    def position(out, key):
        keys = _keys(out)
        return keys.index(key) if key in keys else len(keys)
    assert sum(position(pushed, m) for m in mixed) > sum(position(plain, m) for m in mixed)


def test_filters_are_the_persons_marks(tmp, ds):
    out = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])], k=24,
                        filters=srv.Filters(min_stars=4))
    frame = core.open_dataset("synth").frame
    for key in _keys(out):
        assert frame.iloc[key]["rating"] >= 4
    assert "filters left" in _texts(out)


def test_more_like_leaves_out_the_examples(tmp, ds):
    out = srv.do_more_like(["synth:0", "synth:1"], k=3)
    keys = _keys(out)[2:]                       # the first two are the request line
    assert 0 not in keys and 1 not in keys
    assert set(keys) <= set(range(N_WATER)) | set(range(N_WATER + N_FOREST, N_WATER + N_FOREST + N_MIXED))


# ─────────────────────────── survey and inspect ───────────────────────────

def test_survey_lists_clusters_largest_first_with_exemplars(tmp, ds):
    _pixels(tmp)
    out = srv.do_survey("synth", per_cluster=2)
    text = _texts(out)
    assert text.index("Noise") < text.index("Water")
    assert len(_images(out)) == 1


def test_inspect_describes_in_blind_mode_and_shows_in_pixels(tmp, ds, monkeypatch):
    from arcana import style_transfer
    monkeypatch.setattr(style_transfer, "describe", lambda *a, **k: "lake, mist")
    blind = srv.do_inspect(["synth:0"])
    assert "looks like: lake, mist" in _texts(blind) and not _images(blind)
    assert "palette: #" in _texts(blind)

    _pixels(tmp)
    seen = srv.do_inspect(["synth:0"])
    assert len(_images(seen)) == 1


def test_list_datasets_describes_each_one(tmp, ds):
    text = _texts(srv.do_list_datasets())
    assert "synth" in text and "43 pictures" in text and "4 clusters" in text
    assert "marks: stars, colour tags" in text
    assert "FILES MOVED" not in text


def test_moved_files_are_named_not_left_as_grey_tiles(tmp, ds):
    """Search survives a move; pictures do not. The agent should know which is which."""
    for k in range(5):
        os.remove(ds[k])
    assert "FILES MOVED: about 12%" in _texts(srv.do_list_datasets())
    out = srv.do_search("synth", [srv.Facet(name="w", phrases=["water"])], k=6)
    assert "(file missing)" in _texts(out)


# ─────────────────────────── small parts ───────────────────────────

def test_handles(tmp):
    assert srv.parse_handle("japan:1423") == ("japan", 1423)
    assert srv.parse_handle(srv.handle("field@audio", 7)) == ("field@audio", 7)
    assert srv.split_dataset_id("field@audio") == ("field", "audio")
    for bad in ("japan", "japan:x", ":3", ""):
        with pytest.raises(ValueError, match="handle"):
            srv.parse_handle(bad)


def test_an_unknown_dataset_names_the_known_ones(tmp, ds):
    with pytest.raises(ValueError, match="Known datasets: synth"):
        srv.open_dataset("nope")


def test_swatch_bands_follow_weights(tmp):
    img = srv._swatch(["#ff0000", "#0000ff"], [3, 1])
    assert img.shape[0] == 128 and abs(img.shape[1] - 256) <= 2
    assert tuple(img[0, 0]) == (0, 0, 255)            # BGR red
    assert tuple(img[0, -1]) == (255, 0, 0)           # BGR blue
    assert 180 <= int((img[0, :, 2] == 255).sum()) <= 200
    with pytest.raises(ValueError):
        srv._swatch(["red"], None)


# ─────────────────────────── over a real pipe ───────────────────────────

# Starts the real server, except that list_datasets first prints to stdout the
# way db.py's progress messages do. On a pipe that line would be read as a
# malformed protocol message unless the transport keeps fd 1 off the wire.
_NOISY_SERVER = (
    "import arcana.mcp_server as s\n"
    "real = s.do_list_datasets\n"
    "def noisy():\n"
    "    print('Encoding 3/10 ...', flush=True)\n"
    "    return real()\n"
    "s.do_list_datasets = noisy\n"
    "s.main()\n"
)


def test_the_server_answers_over_stdio_even_when_a_tool_prints(tmp, ds):
    """Start the server as a client would and call a tool through the protocol."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, ARCANA_DATA_DIR=tmp, PYTHONPATH=ROOT)
    params = StdioServerParameters(command=sys.executable, args=["-c", _NOISY_SERVER],
                                   env=env, cwd=ROOT)

    async def talk():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool("list_datasets", {})
                wrong = await session.call_tool(
                    "search", {"dataset": "nope", "facets": [{"name": "x", "phrases": ["x"]}]})
                return {t.name for t in tools.tools}, result, wrong

    names, result, wrong = anyio.run(talk)
    assert "search" in names
    assert not result.is_error
    assert "synth" in result.content[0].text

    # A mistake the agent can fix has to reach it in words, not as "Error
    # executing tool search".
    assert wrong.is_error
    assert "Known datasets: synth" in wrong.content[0].text
