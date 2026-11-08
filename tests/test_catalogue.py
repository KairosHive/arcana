"""The archive (arcana/catalogue.py, ops/catalogue.py): one tree of what is kept, organised by name
and folder in a catalogue while files stay where they are."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcana import boards, ops  # noqa: E402
from arcana import catalogue as cat  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    return out


def _board(slug, name, items):
    with open(os.path.join(boards.boards_dir(), slug + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": 2, "name": name, "slug": slug, "author": "person", "items": items,
                   "details": {}, "sections": []}, fh)


def _picture(rel):
    p = os.path.join(cat.output_root(), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
    return p


def run(line, actor="person"):
    batch = ops.exec_lines([line], ctx=ops.context(actor))
    step = batch.steps[0]
    assert step.refusal is None, (line, step.refusal and step.refusal["reason"])
    return step.result


def refused(line, actor="person"):
    step = ops.exec_lines([line], ctx=ops.context(actor)).steps[0]
    assert step.refusal is not None, line
    return step.refusal["reason"]


def _ids(node):
    return [i["id"] for i in node["items"]]


def test_the_tree_lists_each_kind_and_lays_names_over_files(ds):
    _board("monsoon", "Monsoon", [])
    pic = _picture("edited/colour/2026-11-08/a.png")
    roots = cat.tree()
    assert set(roots) == set(cat.KINDS)
    assert "collection:monsoon" in _ids(roots["collections"])
    fid = "file:" + os.path.relpath(pic, cat.output_root()).replace(os.sep, "/")
    assert fid in _ids(roots["pictures"])
    run(f'archive rename {fid} "First light"')
    shown = {i["id"]: i["shown"] for i in cat.tree()["pictures"]["items"]}
    assert shown[fid] == "First light"
    assert os.path.exists(pic)                       # the file keeps its name on disk


def test_a_collection_rename_writes_its_own_record(ds):
    _board("monsoon", "Monsoon", [])
    run('archive rename collection:monsoon "Monsoon rains"')
    assert boards.load("monsoon")["name"] == "Monsoon rains"


def test_folders_hold_things_and_undo_takes_a_move_back(ds):
    _board("monsoon", "Monsoon", [])
    fid = run("archive folder add Rituals --kind collections").id
    run(f"archive move collection:monsoon --into {fid}")
    roots = cat.tree()
    folder = roots["collections"]["folders"][0]
    assert folder["id"] == fid and _ids(folder) == ["collection:monsoon"]
    assert "collection:monsoon" not in _ids(roots["collections"])
    run("undo")
    assert "collection:monsoon" in _ids(cat.tree()["collections"])


def test_a_collection_sits_inside_another_and_keeps_its_pictures(ds):
    _board("monsoon", "Monsoon", ["a.png"])
    _board("rains", "Rains", ["b.png"])
    run("archive move collection:monsoon --into collection:rains")
    rains = next(i for i in cat.tree()["collections"]["items"] if i["id"] == "collection:rains")
    assert _ids(rains) == ["collection:monsoon"]
    assert boards.load("rains")["items"] == ["b.png"]
    assert "inside itself" in refused("archive move collection:rains --into collection:monsoon")


def test_a_thing_goes_only_into_a_folder_of_its_kind(ds):
    _board("monsoon", "Monsoon", [])
    fid = run("archive folder add Stills --kind pictures").id
    assert "folder of" in refused(f"archive move collection:monsoon --into {fid}")


def test_merging_collections_moves_the_pictures_and_drops_the_first(ds):
    _board("monsoon", "Monsoon", [])
    _board("rains", "Rains", [])
    _board("drizzle", "Drizzle", [])
    run("archive move collection:drizzle --into collection:monsoon")
    run("archive merge collection:monsoon --into collection:rains")
    assert boards.load("monsoon") is None
    rains = next(i for i in cat.tree()["collections"]["items"] if i["id"] == "collection:rains")
    assert _ids(rains) == ["collection:drizzle"]     # what sat in the first now sits in the second


def test_merging_folders_keeps_everything_they_held(ds):
    _board("monsoon", "Monsoon", [])
    f1 = run("archive folder add One --kind collections").id
    f2 = run("archive folder add Two --kind collections").id
    run(f"archive move collection:monsoon --into {f1}")
    run(f"archive merge {f1} --into {f2}")
    folders = cat.tree()["collections"]["folders"]
    assert [f["id"] for f in folders] == [f2] and _ids(folders[0]) == ["collection:monsoon"]
    assert "only an empty folder" in refused(f"archive delete {f2}").lower()
    run(f"archive move collection:monsoon --into kind:collections")
    assert "went" in run(f"archive delete {f2}").text()
    assert cat.tree()["collections"]["folders"] == []


def test_delete_restore_and_empty(ds):
    pic = _picture("edited/colour/2026-11-08/a.png")
    fid = "file:" + os.path.relpath(pic, cat.output_root()).replace(os.sep, "/")
    run(f"archive delete {fid}")
    assert not os.path.exists(pic)
    assert fid not in _ids(cat.tree()["pictures"])
    run(f"archive restore {fid}")
    assert os.path.exists(pic)
    run(f"archive delete {fid}")
    assert "--yes" in refused("archive empty")
    assert "Deleted 1 thing for good" in run("archive empty --yes").text()
    assert cat.read()["trash"] == {}
    assert not os.listdir(cat.trash_dir()) or all(not os.listdir(os.path.join(cat.trash_dir(), d))
                                                  for d in os.listdir(cat.trash_dir()))


def test_an_agent_organises_but_leaves_deleting_to_the_person(ds):
    _board("monsoon", "Monsoon", [])
    assert "agents change only boards an agent made" in refused('archive rename collection:monsoon "Wet"', actor="agent")
    run("archive move collection:monsoon --into kind:collections", actor="agent")
    for name in ("archive delete", "archive restore", "archive empty", "archive merge"):
        assert "agent" not in ops.get(name).actors, name
