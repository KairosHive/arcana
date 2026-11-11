"""
The open film as shared state: every edit an op, undone like any write, and seen by every caller.
See VIDEO_ROADMAP.md, 3.7.

Run with:  python -m pytest tests/test_film_ops.py -q
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import ops, state  # noqa: E402
from arcana.ops import film as FO, registry  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    return out


def refs():
    return [p["ref"] for p in sorted(FO.current()["pictures"], key=lambda p: p["t"])]


def test_an_agents_edits_land_in_the_open_film_and_undo_takes_them_back(ds):
    before = state.STORE.signature()
    ops.call("film open", {"name": "walk"}, actor="agent")
    ops.call("film add", {"item": "synth:12", "at": 0}, actor="agent")
    ops.call("film add", {"item": "synth:7", "at": 4, "ease": "smooth"}, actor="agent")
    shown = ops.call("film current", {}, actor="person")
    assert shown.spec["name"] == "walk" and refs() == ["synth:12", "synth:7"] and shown.length
    assert state.STORE.signature() != before                     # the page's stream fires on this
    assert "key 2" in shown.text() and "synth:7" in shown.text()
    ops.call("film lane", {"path": "finish.saturation", "at": 4, "value": -100}, actor="agent")
    ops.call("film look", {"at": 0, "method": "pixels.grain", "material": ["synth:3"]}, actor="agent")
    spec = FO.current()
    assert spec["lanes"]["finish.saturation"] == [{"t": 4.0, "v": -100.0, "ease": "linear"}]
    assert spec["looks"][0]["settings"]["method"] == "pixels.grain" and spec["looks"][0]["settings"]["materials"] == ["synth:3"]
    ops.call("undo", {}, actor="agent")                           # the look
    ops.call("undo", {}, actor="agent")                           # the lane key
    ops.call("undo", {}, actor="agent")                           # the second picture
    assert refs() == ["synth:12"] and not FO.current().get("looks")
    ops.call("film remove", {"key": 1}, actor="agent")
    assert refs() == []
    with pytest.raises(ops.Refusal, match="1 key|0 keys"):
        ops.call("film remove", {"key": 1}, actor="agent")


def test_the_page_sets_the_whole_film_and_the_agent_names_pictures_by_handle(ds):
    path = ds[0]
    ops.call("film set", {"spec": {"name": "page", "pictures": [{"t": 0, "ref": path}]}}, actor="person")
    assert refs() == [path]
    with pytest.raises(ops.Refusal, match="by handle"):
        ops.call("film set", {"spec": {"name": "x", "pictures": [{"t": 0, "ref": path}]}}, actor="agent")
    with pytest.raises(ops.Refusal, match="by handle"):
        ops.call("film add", {"item": path}, actor="agent")


def test_a_kept_film_opens_and_the_open_film_is_kept_and_rendered_by_default(ds, monkeypatch):
    ops.call("film open", {"name": "rites"}, actor="cli")
    ops.call("film add", {"item": "synth:12", "at": 0}, actor="cli")
    ops.call("film add", {"item": "synth:7", "at": 3}, actor="cli")
    ops.call("film keep", {}, actor="cli")
    assert [f.name for f in ops.call("alchemy film list", {}, actor="cli").films] == ["rites"]
    ops.call("film open", {"name": "other"}, actor="cli")
    assert refs() == []
    ops.call("film open", {"film": "rites"}, actor="cli")
    assert refs() == ["synth:12", "synth:7"]
    shown = ops.call("alchemy film show", {}, actor="cli")      # neither a name nor a spec: the open film
    assert shown.spec["name"] == "rites" and shown.film.passes == [0.0, 3.0]


def test_an_agents_step_opens_on_the_stage_with_its_files(ds, monkeypatch):
    from flask import Flask

    from arcana import api

    app = Flask("t")
    api.register(app)
    client = app.test_client()
    headers = {"X-Arcana-UI": api.UI_TOKEN}
    r = client.post("/api/alchemy/from-op", headers=headers,
                    json={"args": {"picture": "synth:12", "material": ["synth:7"], "method": "pixels.grain",
                                   "knobs": {"amount": 0.4}}})
    got = r.get_json()
    assert r.status_code == 200, got
    assert got["picture"] == ds[12] and got["materials"] == [ds[7]] and got["method"] == "pixels.grain"
    bad = client.post("/api/alchemy/from-op", headers=headers, json={"args": {"picture": "synth:99999"}})
    assert bad.status_code == 400


def test_a_sound_is_laid_under_the_open_film(ds):
    ops.call("film open", {"name": "sound"}, actor="cli")
    with pytest.raises(ops.Refusal):
        ops.call("film audio add", {"item": "synth:12"}, actor="cli")          # a photo has no sound
    with pytest.raises(ops.Refusal, match="-60|less than or equal|greater"):
        ops.call("film audio add", {"item": "synth:12", "gain": 90}, actor="cli")
