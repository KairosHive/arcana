"""Named moodboards: saved to the data directory, loaded back to work on."""

import json
import os

import pytest


@pytest.fixture
def bd(tmp, monkeypatch):
    """A boards module pointed at a scratch data directory."""
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    from arcana import boards, paths
    monkeypatch.setattr(paths, "_CACHED_DATA_DIR", None, raising=False)
    monkeypatch.setattr(boards, "boards_dir", lambda: paths.ensure_dir(
        os.path.join(tmp, "boards")))
    return boards


def test_save_and_load_round_trip(bd, tmp):
    """A board comes back with its items, its reference and its target."""
    a = os.path.join(tmp, "a.jpg")
    b = os.path.join(tmp, "b.jpg")
    for p in (a, b):
        open(p, "wb").close()

    rec = bd.save("Japan cool tones", [a, b], reference=a, transfer=b,
                  dataset="japan")
    assert rec["slug"] == "japan-cool-tones"

    back = bd.load(rec["slug"])
    assert back["name"] == "Japan cool tones"
    assert back["items"] == [a, b]
    assert back["reference"] == a
    assert back["transfer"] == b
    assert back["dataset"] == "japan"


def test_missing_files_are_reported_not_dropped(bd, tmp):
    """
    A board records paths, so originals can move out from under it.

    Loading the pictures that remain and naming how many are gone beats
    silently returning a shorter board than the one that was saved.
    """
    here = os.path.join(tmp, "here.jpg")
    open(here, "wb").close()
    gone = os.path.join(tmp, "gone.jpg")

    rec = bd.save("Half there", [here, gone])
    present, absent = bd.present(bd.load(rec["slug"]))
    assert present == [here]
    assert absent == [gone]
    assert bd.listing()[0]["missing"] == 1


def test_resaving_keeps_the_original_created_time(bd):
    """A board you keep working on should not look new every time."""
    import time
    first = bd.save("Ongoing", ["x.jpg"])
    time.sleep(0.02)
    second = bd.save("Ongoing", ["x.jpg", "y.jpg"])
    assert second["created"] == first["created"]
    assert second["updated"] > first["updated"]
    assert len(bd.load("ongoing")["items"]) == 2


def test_listing_is_newest_first(bd):
    import time
    bd.save("One", ["a"])
    time.sleep(0.02)
    bd.save("Two", ["b"])
    names = [b["name"] for b in bd.listing()]
    assert names[0] == "Two"


def test_delete_removes_only_boards(bd):
    """
    The slug arrives from the browser, so delete must not follow it out of the
    boards folder.
    """
    bd.save("Doomed", ["a"])
    assert bd.delete("doomed") is True
    assert bd.listing() == []

    for hostile in ("../../../etc/passwd", "..\\..\\secrets", "/etc/passwd"):
        assert bd.delete(hostile) is False


def test_unreadable_board_returns_none_rather_than_raising(bd, tmp):
    """A truncated or hand-edited file must not take the app down."""
    d = bd.boards_dir()
    with open(os.path.join(d, "broken.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json at all")
    assert bd.load("broken") is None
    assert bd.listing() == []          # skipped, not fatal


def test_save_is_atomic(bd, tmp):
    """
    Written beside and renamed, so an interrupted save cannot leave a
    truncated board where a good one was.
    """
    rec = bd.save("Atomic", ["a.jpg", "b.jpg"])
    path = os.path.join(bd.boards_dir(), rec["slug"] + ".json")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["items"] == ["a.jpg", "b.jpg"]


def test_slugify_handles_punctuation_and_emptiness(bd):
    assert bd.slugify("Japan — cool/tones!!") == "japan-cooltones"
    assert bd.slugify("   ") == "board"
    assert bd.slugify("Same Name") == bd.slugify("same name")


# ─────────────────────────── schema 2 ───────────────────────────

def _agent_board(bd, tmp):
    a, b = os.path.join(tmp, "a.jpg"), os.path.join(tmp, "b.jpg")
    for p in (a, b):
        open(p, "wb").close()
    rec = bd.save(
        "Monsoon", [
            {"path": a, "role": "hero", "section": "s1", "note": "the weight of it"},
            {"path": b, "role": "texture", "section": "s1"},
        ],
        author="agent", brief="monsoon melancholy", concept="water as weight",
        sections=[{"id": "s1", "title": "Water as weight"}],
        palette=[{"hex": "#1f2a36", "weight": 0.4}],
    )
    return rec, a, b


def test_a_schema_1_board_loads_as_it_always_did(bd):
    """Every board saved before schema 2 must load unchanged, as a person's."""
    path = os.path.join(bd.boards_dir(), "old.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": 1, "name": "Old", "items": ["a.jpg", "b.jpg"],
                   "reference": "a.jpg", "created": 1, "updated": 1}, fh)
    rec = bd.load("old")
    assert rec["items"] == ["a.jpg", "b.jpg"]
    assert rec["details"] == {}
    assert bd.listing()[0]["author"] == "person"


def test_details_and_board_fields_round_trip(bd, tmp):
    rec, a, b = _agent_board(bd, tmp)
    back = bd.load(rec["slug"])
    assert back["items"] == [a, b]
    assert bd.details_for(back, a) == {"role": "hero", "section": "s1",
                                       "note": "the weight of it"}
    assert back["author"] == "agent"
    assert back["concept"] == "water as weight"
    assert back["sections"][0]["title"] == "Water as weight"
    assert bd.listing()[0]["author"] == "agent"


def test_an_older_reader_still_finds_plain_paths_on_disk(bd, tmp):
    """
    The installed app and a development checkout can share one data dir. The
    older one reads `items` as paths; objects there would crash it on every
    board an agent wrote.
    """
    rec, a, b = _agent_board(bd, tmp)
    with open(os.path.join(bd.boards_dir(), rec["slug"] + ".json"), encoding="utf-8") as fh:
        assert json.load(fh)["items"] == [a, b]


def test_resaving_from_the_moodboard_tab_keeps_the_agents_work(bd, tmp):
    """
    The tab only knows paths. Tidying an agent's board there -- dropping one
    picture, adding another -- must not wipe the notes, sections or concept.
    """
    rec, a, b = _agent_board(bd, tmp)
    c = os.path.join(tmp, "c.jpg")
    open(c, "wb").close()

    bd.save("Monsoon", [a, c], reference=a, transfer="", dataset="japan")
    back = bd.load(rec["slug"])

    assert back["items"] == [a, c]
    assert bd.details_for(back, a)["note"] == "the weight of it"
    assert b not in back["details"], "a removed picture's details go with it"
    assert bd.details_for(back, c) == {}
    assert back["concept"] == "water as weight"
    assert back["author"] == "agent"


def test_new_details_merge_over_old_ones(bd, tmp):
    rec, a, _ = _agent_board(bd, tmp)
    bd.save("Monsoon", [a], details={a: {"note": "rewritten"}})
    d = bd.details_for(bd.load(rec["slug"]), a)
    assert d == {"role": "hero", "section": "s1", "note": "rewritten"}


def test_unknown_roles_and_authors_are_refused(bd):
    import pytest
    with pytest.raises(ValueError):
        bd.save("Bad role", [{"path": "a.jpg", "role": "protagonist"}])
    with pytest.raises(ValueError):
        bd.save("Bad author", ["a.jpg"], author="robot")


def test_fingerprints_are_recorded_only_when_asked(bd, tmp):
    """Hashing reads each file whole, so the tab's quick save does not do it."""
    from arcana import bundle
    a = os.path.join(tmp, "a.jpg")
    with open(a, "wb") as fh:
        fh.write(b"JPEG" + b"\x01" * 4096)
    gone = os.path.join(tmp, "gone.jpg")

    rec = bd.save("Quick", [a])
    assert "fp" not in bd.details_for(bd.load(rec["slug"]), a)

    rec = bd.save("Hashed", [a, gone], fingerprints=True)
    back = bd.load(rec["slug"])
    assert bd.details_for(back, a)["fp"] == bundle.fingerprint(a)
    assert "fp" not in bd.details_for(back, gone)


def test_hand_edited_object_items_are_reduced_to_paths(bd, tmp):
    """os.path.exists on a dict raises, so load() must not hand objects onward."""
    a = os.path.join(tmp, "a.jpg")
    open(a, "wb").close()
    with open(os.path.join(bd.boards_dir(), "edited.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "Edited", "items": [{"path": a, "role": "hero"}, "gone.jpg"]}, fh)
    rec = bd.load("edited")
    assert rec["items"] == [a, "gone.jpg"]
    assert bd.details_for(rec, a) == {"role": "hero"}
    assert bd.present(rec) == ([a], ["gone.jpg"])
