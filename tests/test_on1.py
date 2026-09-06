"""
Tests for reading ON1 sidecars and filtering on what they say.

Run with:  python -m pytest tests/test_on1.py -q
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import on1  # noqa: E402


def _sidecar(folder, stem, entries, version=2026):
    """
    Write a .on1 beside `stem`.

    entries: list of (name, metadata dict). More than one is the RAW+JPEG case.
    """
    photos = {}
    for i, (name, meta) in enumerate(entries):
        photos[f"guid-{stem}-{i}"] = {"name": name, "metadata": meta, "type": 2}
    path = os.path.join(folder, stem + ".on1")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"photos": photos, "type": 1, "version": version}, fh)
    return path


def _jpeg(folder, name, size=(8, 8)):
    path = os.path.join(folder, name)
    Image.new("RGB", size, (40, 90, 140)).save(path, "JPEG")
    return path


# ───────────────────────── date parsing ─────────────────────────
def test_reads_asctime_capture_date(tmp):
    _jpeg(tmp, "a.JPG")
    _sidecar(tmp, "a", [("a.JPG", {"CaptureDate": "Sat Oct 28 14:26:25 2023"})])
    assert on1.read_marks(os.path.join(tmp, "a.JPG")).captured.startswith("2023-10-28T14:26:25")


def test_on1_writes_thur_and_it_must_still_parse(tmp):
    """
    ON1's Thursday is "Thur", which no locale defines and %a rejects.

    7,739 of the 55,783 sidecars this was written against say Thur. Losing them
    is not a rounding error in a date filter, it is a seventh of the archive
    with no date at all.
    """
    _jpeg(tmp, "t.JPG")
    _sidecar(tmp, "t", [("t.JPG", {"CaptureDate": "Thur Apr 02 15:31:18 2026"})])
    assert on1.read_marks(os.path.join(tmp, "t.JPG")).captured.startswith("2026-04-02T15:31:18")


def test_space_padded_single_digit_day(tmp):
    _jpeg(tmp, "s.JPG")
    _sidecar(tmp, "s", [("s.JPG", {"CaptureDate": "Wed Jun  4 19:15:30 2025"})])
    assert on1.read_marks(os.path.join(tmp, "s.JPG")).captured.startswith("2025-06-04T19:15:30")


def test_capture_offset_is_kept_not_converted(tmp):
    """
    A photograph shot at 19:15 in Montreal was shot in the evening.

    Normalising to UTC would file it under the following day, which is what the
    date filter would then report.
    """
    _jpeg(tmp, "o.JPG")
    _sidecar(tmp, "o", [("o.JPG", {"CaptureDate": "Wed Jun 04 19:15:30 2025",
                                   "CaptureDateOffset": -18000})])
    captured = on1.read_marks(os.path.join(tmp, "o.JPG")).captured
    assert captured.startswith("2025-06-04T19:15:30")
    assert captured.endswith("-05:00")


# ───────────────────────── stars and tags ─────────────────────────
def test_absent_rating_is_none_not_zero(tmp):
    """
    "Never opened in ON1" and "opened and given no stars" are different facts.

    Flattening them to 0 makes a "Never rated" filter impossible and quietly
    tells you a picture was considered and rejected when it was never seen.
    """
    _jpeg(tmp, "n.JPG")
    _sidecar(tmp, "n", [("n.JPG", {"CameraMake": "SONY"})])
    marks = on1.read_marks(os.path.join(tmp, "n.JPG"))
    assert marks.rating is None

    _jpeg(tmp, "z.JPG")
    _sidecar(tmp, "z", [("z.JPG", {"Rating": 0})])
    assert on1.read_marks(os.path.join(tmp, "z.JPG")).rating == 0


def test_reads_rating_label_and_camera(tmp):
    _jpeg(tmp, "m.JPG")
    _sidecar(tmp, "m", [("m.JPG", {"Rating": 4, "Label": "Yellow",
                                   "CameraMake": "SONY", "CameraModel": "ILCE-7M3"})])
    marks = on1.read_marks(os.path.join(tmp, "m.JPG"))
    assert (marks.rating, marks.label, marks.camera) == (4, "Yellow", "SONY ILCE-7M3")
    assert marks.source == "on1"


def test_unidentified_camera_dashes_are_dropped(tmp):
    """ON1 writes "----" for a body or lens it cannot name; that is not a camera."""
    _jpeg(tmp, "d.JPG")
    _sidecar(tmp, "d", [("d.JPG", {"CameraMake": "SONY", "CameraModel": "----"})])
    assert on1.read_marks(os.path.join(tmp, "d.JPG")).camera == "SONY"


# ───────────────────────── one sidecar, two photographs ─────────────────────────
def test_raw_and_jpeg_share_a_sidecar_and_get_their_own_marks(tmp):
    """
    Shooting RAW+JPEG gives DSC1.ARW and DSC1.JPG one sidecar with two entries.

    4,114 of 55,783 sidecars in the archive this was built against are such
    pairs. Taking the first entry is right half the time by luck.
    """
    _jpeg(tmp, "DSC1.JPG")
    open(os.path.join(tmp, "DSC1.ARW"), "wb").write(b"not really a raw file")
    _sidecar(tmp, "DSC1", [("DSC1.ARW", {"Rating": 1, "Label": "Red"}),
                           ("DSC1.JPG", {"Rating": 5, "Label": "Yellow"})])

    jpg = on1.read_marks(os.path.join(tmp, "DSC1.JPG"))
    raw = on1.read_marks(os.path.join(tmp, "DSC1.ARW"))
    assert (jpg.rating, jpg.label) == (5, "Yellow")
    assert (raw.rating, raw.label) == (1, "Red")


def test_two_entries_neither_matching_declines_to_guess(tmp):
    _jpeg(tmp, "renamed.JPG")
    _sidecar(tmp, "renamed", [("old-a.ARW", {"Rating": 1}), ("old-b.JPG", {"Rating": 5})])
    marks = on1.read_marks(os.path.join(tmp, "renamed.JPG"), exif_fallback=False)
    assert marks is None


def test_single_entry_with_a_stale_name_is_still_used(tmp):
    """ON1 records the name it last saw; a renamed file keeps its stars."""
    _jpeg(tmp, "new-name.JPG")
    _sidecar(tmp, "new-name", [("old-name.JPG", {"Rating": 3})])
    assert on1.read_marks(os.path.join(tmp, "new-name.JPG")).rating == 3


# ───────────────────────── malformed input ─────────────────────────
def test_a_corrupt_sidecar_does_not_take_the_scan_down(tmp):
    good = _jpeg(tmp, "good.JPG")
    _sidecar(tmp, "good", [("good.JPG", {"Rating": 2})])
    bad = _jpeg(tmp, "bad.JPG")
    with open(os.path.join(tmp, "bad.on1"), "w", encoding="utf-8") as fh:
        fh.write("{not json at all")

    marks = on1.scan([good, bad])
    assert marks[good].rating == 2
    # bad.JPG still yields its EXIF-derived record or nothing, but never raises.
    assert marks.get(bad) is None or marks[bad].rating is None


def test_byte_order_mark_is_tolerated(tmp):
    path = _jpeg(tmp, "bom.JPG")
    doc = {"photos": {"g": {"name": "bom.JPG", "metadata": {"Rating": 3}}},
           "type": 1, "version": 2026}
    with open(os.path.join(tmp, "bom.on1"), "w", encoding="utf-8-sig") as fh:
        json.dump(doc, fh)
    assert on1.read_marks(path).rating == 3


def test_no_sidecar_and_no_exif_is_none(tmp):
    path = os.path.join(tmp, "plain.txt")
    with open(path, "w") as fh:
        fh.write("hello")
    assert on1.read_marks(path) is None


# ───────────────────────── the xmp sidecar ─────────────────────────
_XMP = """<?xml version="1.0"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmp:Rating="{rating}" xmp:Label="{label}"
    exif:DateTimeOriginal="2024-03-09T08:30:00"
    tiff:Make="SONY" tiff:Model="ILCE-7M3"/>
 </rdf:RDF>
</x:xmpmeta>"""


def test_xmp_sidecar_is_read_when_there_is_no_on1(tmp):
    path = _jpeg(tmp, "x.JPG")
    with open(os.path.join(tmp, "x.xmp"), "w", encoding="utf-8") as fh:
        fh.write(_XMP.format(rating=4, label="red"))
    marks = on1.read_marks(path)
    assert (marks.rating, marks.label, marks.source) == (4, "Red", "xmp")
    assert marks.captured.startswith("2024-03-09T08:30:00")


def test_lightroom_reject_is_not_a_star_count(tmp):
    """xmp:Rating of -1 means rejected. Read as a rating it outranks unrated."""
    path = _jpeg(tmp, "r.JPG")
    with open(os.path.join(tmp, "r.xmp"), "w", encoding="utf-8") as fh:
        fh.write(_XMP.format(rating=-1, label=""))
    assert on1.read_marks(path).rating is None


def test_on1_wins_over_xmp(tmp):
    path = _jpeg(tmp, "both.JPG")
    _sidecar(tmp, "both", [("both.JPG", {"Rating": 2, "Label": "Red"})])
    with open(os.path.join(tmp, "both.xmp"), "w", encoding="utf-8") as fh:
        fh.write(_XMP.format(rating=5, label="blue"))
    assert on1.read_marks(path).rating == 2


# ───────────────────────── scanning ─────────────────────────
def test_scan_summarises_a_folder(tmp):
    paths = []
    for i, (rating, label) in enumerate([(4, "Yellow"), (4, "Red"), (0, ""), (3, "Yellow")]):
        p = _jpeg(tmp, f"f{i}.JPG")
        meta = {"Rating": rating, "CaptureDate": "Sat Oct 28 14:26:25 2023"}
        if label:
            meta["Label"] = label
        _sidecar(tmp, f"f{i}", [(f"f{i}.JPG", meta)])
        paths.append(p)

    marks = on1.scan(paths)
    summary = on1.summarise(marks)
    assert summary["n"] == 4
    assert summary["sidecars"] == 4
    assert summary["dated"] == 4
    assert summary["stars"] == {0: 1, 3: 1, 4: 2}
    assert summary["labels"] == {"Yellow": 2, "Red": 1}


def test_scan_of_nothing_is_empty_not_an_error():
    assert on1.scan([]) == {}


# ───────────────────────── the latent-frame columns ─────────────────────────
def test_attach_marks_keeps_unrated_distinct_from_zero(tmp):
    from arcana import db

    rated = _jpeg(tmp, "rated.JPG")
    _sidecar(tmp, "rated", [("rated.JPG", {"Rating": 0})])
    unseen = _jpeg(tmp, "unseen.JPG")

    df = pd.DataFrame({"path": [rated, unseen], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    db.attach_marks(df, on1.scan([rated, unseen]))

    assert df.loc[0, "rating"] == 0
    assert pd.isna(df.loc[1, "rating"])


def test_attached_dates_are_the_cameras_own_clock(tmp):
    from arcana import db

    path = _jpeg(tmp, "evening.JPG")
    _sidecar(tmp, "evening", [("evening.JPG", {"CaptureDate": "Wed Jun 04 19:15:30 2025",
                                               "CaptureDateOffset": -18000})])
    df = pd.DataFrame({"path": [path]})
    db.attach_marks(df, on1.scan([path]))
    # Not the 5th, which is what a UTC conversion would file it under.
    assert df.loc[0, "captured"].date().isoformat() == "2025-06-04"


# ───────────────────────── filtering ─────────────────────────
@pytest.fixture
def frame():
    """Four frames: 4-star yellow, 3-star red, 0-star untagged, and unrated."""
    return pd.DataFrame({
        "path": ["a", "b", "c", "d"],
        "rating": pd.array([4, 3, 0, None], dtype="Int64"),
        "colour": ["Yellow", "Red", "", ""],
        "captured": pd.to_datetime(["2025-06-04 19:15", "2025-06-05 08:00",
                                    "2025-06-06 12:00", None]),
    })


def test_no_filter_keeps_everything(frame):
    from arcana import arcana as app
    assert app.marks_mask(frame).all()


def test_star_floor_excludes_the_never_rated(frame):
    """"Three and up" is a request for pictures somebody chose."""
    from arcana import arcana as app
    assert list(app.marks_mask(frame, stars=3)) == [True, True, False, False]


def test_unrated_is_its_own_selection(frame):
    from arcana import arcana as app
    assert list(app.marks_mask(frame, stars=app.UNRATED)) == [False, False, False, True]


def test_colour_filter_and_the_untagged_remainder(frame):
    from arcana import arcana as app
    assert list(app.marks_mask(frame, colours=["Yellow"])) == [True, False, False, False]
    assert list(app.marks_mask(frame, colours=[app.NO_COLOUR])) == [False, False, True, True]
    assert list(app.marks_mask(frame, colours=["Yellow", app.NO_COLOUR])) == [True, False, True, True]


def test_date_range_includes_the_whole_closing_day(frame):
    """The picker gives a day, and a day means all of it, not midnight."""
    from arcana import arcana as app
    kept = app.marks_mask(frame, date_from="2025-06-05", date_to="2025-06-05")
    assert list(kept) == [False, True, False, False]


def test_undated_frames_cannot_satisfy_a_date_range(frame):
    from arcana import arcana as app
    kept = app.marks_mask(frame, date_from="2020-01-01", date_to="2030-01-01")
    assert list(kept) == [True, True, True, False]


def test_filters_combine(frame):
    from arcana import arcana as app
    kept = app.marks_mask(frame, stars=3, colours=["Yellow"])
    assert list(kept) == [True, False, False, False]


def test_a_dataset_without_mark_columns_is_unaffected():
    """Datasets indexed before any of this existed must behave exactly as before."""
    from arcana import arcana as app
    old = pd.DataFrame({"path": ["a", "b"], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    assert not app.has_marks(old)
    assert app.marks_mask(old, stars=4, colours=["Red"], date_from="2020-01-01").all()


def test_options_carry_counts(frame):
    from arcana import arcana as app
    stars = {o["value"]: o["label"] for o in app.star_options(frame)}
    assert "· 2" in stars[3]                       # 4-star and 3-star
    assert app.UNRATED in stars and "· 1" in stars[app.UNRATED]
    colours = {o["value"]: o["label"] for o in app.colour_options(frame)}
    assert colours["Yellow"].endswith("· 1")
    assert app.NO_COLOUR in colours


def test_matrix_totals_match_the_frame(frame):
    from arcana import arcana as app
    grid, rows, cols = app._matrix_counts(frame)
    assert sum(grid.values()) == len(frame)
    assert app.UNRATED in rows
    assert app.NO_COLOUR in cols


def test_allowed_keys_follow_index_order_not_row_labels():
    """
    The latent frame is walked in idx2path order, so a filtered frame's own
    labels are not usable as keys -- which is the bug this guards.
    """
    from arcana import arcana as app
    idx2path = {7: "a", 9: "b", 11: "c"}
    keep = pd.Series([False, True, True], index=[100, 101, 102])
    assert app._allowed_keys(idx2path, keep) == {9, 11}


def main():
    """Standalone runner, so this file works without pytest like the rest do."""
    import tempfile
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        args = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        kwargs = {}
        if "tmp" in args:
            kwargs["tmp"] = tempfile.mkdtemp()
        if "frame" in args:
            kwargs["frame"] = pd.DataFrame({
                "path": ["a", "b", "c", "d"],
                "rating": pd.array([4, 3, 0, None], dtype="Int64"),
                "colour": ["Yellow", "Red", "", ""],
                "captured": pd.to_datetime(["2025-06-04 19:15", "2025-06-05 08:00",
                                            "2025-06-06 12:00", None]),
            })
        try:
            fn(**kwargs)
            print(f"  ok   {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
