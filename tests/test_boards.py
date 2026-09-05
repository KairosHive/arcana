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
