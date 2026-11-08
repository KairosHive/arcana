"""
Tests for the MCP server.

The tools are called as plain functions against a small synthetic dataset in a
scratch data dir, with the text encoder replaced by fixed vectors: no model is
downloaded. One test at the end starts the real server as a subprocess and
talks to it through an MCP client, because the failure that matters most there
-- a stray print() corrupting the protocol -- only shows up over a real pipe.

Run with:  python -m pytest tests/test_mcp_server.py
"""

import os
import sys

import pytest

pytest.importorskip("mcp")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcana import core, mcp_server as srv  # noqa: E402

from synth import N_FOREST, N_MIXED, N_WATER  # noqa: E402
import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def _pixels(tmp, on=True):
    synth.pixels(tmp, on)


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
