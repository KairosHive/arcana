"""
The active collection, shared by every tab, and where outputs are written.

A collection is a board: the person chooses the active one with `settings set
--collection`, and the app adds and removes pictures with the board ops, so a
terminal, an agent and the app on any port all see the same pictures.

Run with:  python -m pytest tests/test_collections.py -q
"""

import datetime
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import ops, paths  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    return out


# ─────────────────────────── the setting ───────────────────────────

def test_the_active_collection_is_a_setting_only_the_person_changes(tmp, ds):
    with pytest.raises(ops.Refusal, match="No board called"):
        ops.call("settings set", {"collection": "nope"}, actor="cli")
    ops.call("board create", {"name": "Monsoon"}, actor="cli")
    shown = ops.call("settings set", {"collection": "monsoon"}, actor="cli")
    assert shown.collection == "monsoon" and "Collection: monsoon" in shown.text()
    assert ops.call("settings show", {}, actor="agent").collection == "monsoon"
    with pytest.raises(ops.Refusal):
        ops.call("settings set", {"collection": "monsoon"}, actor="agent")
    assert ops.call("settings set", {"collection": ""}, actor="cli").collection is None
    with pytest.raises(ops.Refusal, match="Give --pixels, --collection"):
        ops.call("settings set", {}, actor="cli")


# ─────────────────────────── the app's collection ───────────────────────────

def test_adding_with_no_collection_makes_one_and_every_change_is_a_board_op(tmp, ds):
    from arcana import ui_collection as col
    assert col.active() is None
    slug = col.add([ds[12], ds[7], "synth:3"])
    assert slug == col.active() == "collection"
    items, details, name = col.contents(slug)
    assert items == [ds[12], ds[7], ds[3]] and name == "Collection"
    assert set(details) == set(items)              # every picture has its badge entry
    # The same board, as a terminal sees it.
    assert [i.path for i in ops.call("board show", {"board": slug}, actor="cli").items] == items
    col.remove([ds[7]])
    assert col.contents(slug)[0] == [ds[12], ds[3]]
    col.clear()
    assert col.contents(slug)[0] == []
    # A step back undoes the person's last change, as it does for any write.
    col.add([ds[1]])
    ops.call("undo", {}, actor="person")
    assert col.contents(slug)[0] == []


def test_a_deleted_active_collection_is_none_and_the_next_add_makes_one(tmp, ds):
    from arcana import ui_collection as col
    ops.call("board create", {"name": "Dunes"}, actor="cli")
    col.use("dunes")
    ops.call("board delete", {"board": "dunes"}, actor="cli")
    assert col.active() is None
    assert col.add([ds[2]]) == "collection"


def test_a_kept_alchemy_picture_joins_the_collection_by_handle(tmp, ds):
    from arcana import ui_collection as col
    m = ops.call("alchemy make", {"picture": "synth:12", "material": ["synth:7"], "method": "colour.lab"},
                 actor="person")
    col.add([m.id])
    rec = ops.call("board show", {"board": col.active()}, actor="person")
    assert [i.path for i in rec.items] == [m.path] and rec.items[0].kind == "derived"


# ─────────────────────────── where outputs go ───────────────────────────

def test_outputs_are_grouped_by_where_they_came_from(tmp, ds):
    today = datetime.date.today().isoformat()
    assert paths.edited_dir("colour") == os.path.join(tmp, "output", "edited", "colour", today)
    assert paths.dated_name("rain on: glass?") == f"{today} rain on glass"
    picks = ops.call("item export", {"items": ["synth:12"], "folder": "picks"}, actor="cli")
    assert picks.folder == os.path.join(tmp, "output", "search", "picks")
    ops.call("board create", {"name": "Monsoon"}, actor="cli")
    ops.call("board add", {"board": "monsoon", "items": ["synth:12"]}, actor="cli")
    copy = ops.call("board export", {"board": "monsoon"}, actor="cli")
    assert copy.folder == os.path.join(tmp, "output", "collections", "monsoon")
    t = ops.call("colour transfer", {"target": "synth:12", "reference": "synth:7", "method": "lab"},
                 actor="cli")
    assert os.path.dirname(t.path) == os.path.join(tmp, "output", "edited", "colour", today)
