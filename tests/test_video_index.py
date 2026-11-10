"""
A folder of videos as an index of sections: built, read back, searched by level.
See VIDEO_ROADMAP.md, 2.3 to 2.8.

The encoder is replaced by the colour signature, so no model loads: a red part
is near "red" because both are the red histogram bin.

Run with:  python -m pytest tests/test_video_index.py -q
"""

import os
import pickle
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import modality, video as V, video_index as VI  # noqa: E402
from test_video import make_clip  # noqa: E402

BLUE, GREY, GREEN, RED = "0x1b4f8f", "0x9a9a9a", "0x2f7d3a", "0xb02a2a"


def colour_encode(frames):
    return np.stack([V.colour_signature(f) for f in frames])


def colour_query(hex_colour):
    rgb = np.array([[[int(hex_colour[i:i + 2], 16) for i in (2, 4, 6)]]], np.uint8)
    return V.colour_signature(np.repeat(np.repeat(rgb, 8, 0), 8, 1))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("vidset"))
    media = os.path.join(root, "media")
    db = os.path.join(root, "databases")
    os.makedirs(media)
    os.makedirs(db)
    make_clip(os.path.join(media, "a_three_parts.mp4"), [(BLUE, 10), (GREY, 14), (GREEN, 8)])
    make_clip(os.path.join(media, "b_one_take.mp4"), [(RED, 6)], sound=False)
    with open(os.path.join(media, "c_broken.mp4"), "wb") as f:
        f.write(b"not a video at all")
    paths = [os.path.join(media, f) for f in os.listdir(media)]
    index_path = os.path.join(db, "index_clips_video.pkl")
    index, idx2path = VI.build(paths, index_path, "clips", encode=colour_encode, db_dir=db)
    return {"db": db, "media": media, "index": index, "idx2path": idx2path, "index_path": index_path}


def test_section_refs_round_trip():
    ref = modality.section_ref("C:\\footage\\walk.mp4", 12.0, 22.5)
    assert ref == "C:\\footage\\walk.mp4#t=12.000,22.500"
    assert modality.split_ref(ref) == ("C:\\footage\\walk.mp4", 12.0, 22.5)
    assert modality.split_ref("C:\\a\\b.jpg") == ("C:\\a\\b.jpg", None, None)
    assert modality.of_path(ref) == "video" and modality.file_of(ref).endswith("walk.mp4")


def test_the_index_holds_one_key_per_distinct_section(built):
    sec = VI.load_sections("clips", built["db"])
    assert set(sec["key"]) == set(built["idx2path"])
    assert len(sec) == len(built["idx2path"]) == len(built["index"])
    three = sec[sec["file"].str.endswith("a_three_parts.mp4")]
    spans = sorted((round(a), round(b)) for a, b in zip(three["t0"], three["t1"]))
    assert spans == [(0, 10), (0, 32), (10, 24), (24, 32)]
    # The three parts hold no finer change, so each is coarse, medium and fine at once.
    part = three[three["t0"].round() == 10].iloc[0]
    assert part["levels"] == "coarse,medium,fine"
    assert three[three["seconds"] > 30].iloc[0]["levels"] == "file"
    # A single take is one section at every level, stored once.
    one = sec[sec["file"].str.endswith("b_one_take.mp4")]
    assert len(one) == 1 and one.iloc[0]["levels"] == "file,coarse,medium,fine"
    # Every section says how it moves. Flat colour does not move.
    assert set(sec["motion"]) == {"still camera, nothing moves"}
    moved = np.load(VI.motion_path("clips", built["db"]))
    assert moved["motion"].shape == (len(sec), len(moved["names"])) and list(moved["ids"]) == list(sec["key"])
    # The broken file was skipped and the rest indexed.
    assert not sec["file"].str.endswith("c_broken.mp4").any()


def test_paths_carry_the_time_range(built):
    for ref in built["idx2path"].values():
        file, t0, t1 = modality.split_ref(ref)
        assert os.path.isfile(file) and t0 is not None and t1 > t0


def test_the_tree_is_kept_for_resectioning(built):
    with open(VI.tree_path("clips", built["db"]), "rb") as f:
        tree = pickle.load(f)
    a = next(v for k, v in tree.items() if k.endswith("a_three_parts.mp4"))
    assert a["cuts"] == [19, 47] and len(a["strengths"]) == a["ticks"] - 1
    assert a["has_audio"] is True and abs(a["duration"] - 32) < 0.3


def _search(built, colour, allowed, n=10):
    hits = built["index"].search(colour_query(colour), len(built["idx2path"]), exact=True)
    return [(int(h.key), built["idx2path"][int(h.key)], float(h.distance)) for h in hits
            if int(h.key) in allowed][:n]


def test_levels_and_lengths_choose_what_a_search_returns(built):
    sec = VI.load_sections("clips", built["db"])
    files = VI.keys_at(sec, "file")
    assert len(files) == 2
    coarse = VI.keys_at(sec, "coarse")
    assert len(coarse) == 4                      # three parts, and the single take
    assert VI.keys_at(sec, "auto") == set(sec["key"])
    # "At least twelve seconds" leaves the grey part and the whole three-part file.
    long = VI.keys_at(sec, "auto", min_seconds=12)
    assert sorted(round(float(s)) for s in sec[sec["key"].isin(long)]["seconds"]) == [14, 32]
    assert len(VI.keys_at(sec, "coarse", max_seconds=9)) == 2

    best = _search(built, GREEN, coarse)[0]
    file, t0, t1 = modality.split_ref(best[1])
    assert file.endswith("a_three_parts.mp4") and (round(t0), round(t1)) == (24, 32)


def test_auto_returns_each_match_once_at_its_own_length(built):
    sec = VI.load_sections("clips", built["db"])
    ranked = _search(built, GREEN, VI.keys_at(sec, "auto"), n=100)
    kept = VI.best_per_video(sec, ranked, 10)
    spans = [(os.path.basename(modality.file_of(p)), round(modality.split_ref(p)[1]), round(modality.split_ref(p)[2]))
             for _, p, _ in kept]
    assert spans[0] == ("a_three_parts.mp4", 24, 32)
    # The whole file overlaps the green part and ranks below it, so it is left out.
    assert ("a_three_parts.mp4", 0, 32) not in spans
    # The other parts of that video do not overlap it and may follow.
    assert len({s for s in spans if s[0] == "a_three_parts.mp4"}) == 3


def test_media_routes_serve_a_section(built, monkeypatch):
    from flask import Flask
    from arcana import media
    media.register_dataset_files(built["idx2path"])
    app = Flask("t")
    media.register(app)
    c = app.test_client()
    ref = next(p for p in built["idx2path"].values() if "#t=24.000" in p)
    thumb = c.get("/thumb", query_string={"p": ref})
    assert thumb.status_code == 200 and thumb.mimetype == "image/jpeg"
    import cv2
    img = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
    b, g, r = img[96, 96].astype(int)
    assert g > r + 30 and g > b + 30                 # the middle of the green part
    assert c.get("/preview", query_string={"p": ref, "w": 400}).status_code == 200
    whole = c.get("/video", query_string={"p": ref})
    assert whole.status_code == 200 and whole.mimetype == "video/mp4"
    part = c.get("/video", query_string={"p": ref}, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.data) == 100
    # A file no dataset names is still refused, with or without a time range.
    other = os.path.join(built["media"], "..", "elsewhere.mp4")
    assert c.get("/video", query_string={"p": other + "#t=0.000,1.000"}).status_code == 404


def test_movement_reranks_what_content_found():
    from arcana import motion as M
    names = list(M.NAMES)

    def moving(**kw):
        v = np.zeros(len(names), np.float32)
        for k, x in kw.items():
            v[names.index(k)] = x
        return v

    # Three hits, best content first. The third is the only one panning right.
    hits = [(1, "a", 0.10), (2, "b", 0.12), (3, "c", 0.30)]
    motion = {1: moving(), 2: moving(pan_x=-0.2, pan=0.2), 3: moving(pan_x=0.3, pan=0.3)}
    same, read = VI.rank_with_motion(hits, motion, "a red door", 0.5)
    assert same == hits and read == []
    by_content, read = VI.rank_with_motion(hits, motion, "pan right", 0.0)
    assert [h[0] for h in by_content] == [1, 2, 3] and read == ["pan right"]
    by_motion, _ = VI.rank_with_motion(hits, motion, "pan right", 1.0)
    assert by_motion[0][0] == 3
    both, _ = VI.rank_with_motion(hits, motion, "pan left", 0.5)
    assert both[0][0] == 2


def test_tags_filter_the_keys(built):
    sec = VI.load_sections("clips", built["db"])
    assert VI.keys_at(sec, "auto", tags=["still"]) == set(sec["key"])
    assert VI.keys_at(sec, "auto", tags=["handheld"]) == set()
    assert set(VI.load_motion("clips", built["db"])) == set(sec["key"])


def test_whole_videos_come_back_once_with_a_strip_of_where_they_match(built):
    sec = VI.load_sections("clips", built["db"])
    ranked = _search(built, GREEN, set(sec["key"]), n=100)
    hits, strips = VI.whole_videos(sec, ranked, 5)
    files = [os.path.basename(modality.file_of(p)) for _, p, _ in hits]
    assert files == ["a_three_parts.mp4", "b_one_take.mp4"]
    # Each hit is the whole file, and its strip covers the file with its parts in order.
    path = hits[0][1]
    assert round(modality.split_ref(path)[1]) == 0 and round(modality.split_ref(path)[2]) == 32
    strip = strips[path]
    assert [(round(a), round(b)) for a, b, _ in strip["parts"]] == [(0, 10), (10, 24), (24, 32)]
    assert [round(x) for x in strip["best"]] == [24, 32]
    scores = [s for _, _, s in strip["parts"]]
    assert scores[2] == max(scores) == 1.0 and min(scores) >= 0.0
    assert VI.whole_videos(sec, [], 5) == ([], {})


def test_new_videos_are_added_without_moving_what_is_there(built, tmp):
    import shutil
    db = os.path.join(tmp, "databases")
    shutil.copytree(built["db"], db)
    media = os.path.join(tmp, "media")
    shutil.copytree(built["media"], media)
    index_path = os.path.join(db, "index_clips_video.pkl")
    before = VI.load_sections("clips", db).copy()
    # Nothing new: the files of the first build live elsewhere, so point at a folder holding only a new one.
    fresh = os.path.join(tmp, "fresh")
    os.makedirs(fresh)
    make_clip(os.path.join(fresh, "d_new.mp4"), [(GREY, 4), (RED, 4)], sound=False)
    got = VI.extend("clips", fresh, index_path, encode=colour_encode, db_dir=db)
    assert got["added"] == 1 and got["sections"] >= 3 and got["skipped"] == 0
    after = VI.load_sections("clips", db)
    assert list(after["key"][:len(before)]) == list(before["key"])              # old keys untouched
    assert list(after["path"][:len(before)]) == list(before["path"])
    new = after[after["file"].str.endswith("d_new.mp4")]
    assert len(new) == got["sections"] and new["key"].min() == before["key"].max() + 1
    assert new["video"].nunique() == 1 and int(new["video"].iloc[0]) == int(before["video"].max()) + 1
    assert set(VI.load_motion("clips", db)) == set(after["key"])
    with open(index_path, "rb") as fh:
        _, idx2path = pickle.load(fh)
    assert set(idx2path) == set(after["key"])
    # A second run finds nothing new.
    assert VI.extend("clips", fresh, index_path, encode=colour_encode, db_dir=db)["added"] == 0

