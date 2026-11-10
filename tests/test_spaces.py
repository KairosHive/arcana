"""
The shared space: a second set of vectors for a dataset, and searches that mix
words with pictures and sounds. See VIDEO_ROADMAP.md, 2.5.

The encoder is a stub, so nothing is downloaded: a file's vector is read from
a table by its name, and a phrase's from its words.

Run with:  python -m pytest tests/test_spaces.py -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import omni, ops, spaces  # noqa: E402

import synth  # noqa: E402
from test_modality import write_audio_dataset  # noqa: E402

DIM = omni.DIM
RAIN, FOREST, CROWD = 0, 1, 2


def axis(i, more=None, w=0.0):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    if more is not None:
        v[more] = w
    return v / np.linalg.norm(v)


WORDS = {"rain": axis(RAIN), "forest": axis(FOREST), "people": axis(CROWD)}


@pytest.fixture
def world(tmp, monkeypatch):
    """Photos and sounds named `synth`, and an encoder that knows a few of them."""
    photos = synth.use(tmp, monkeypatch)
    sounds = write_audio_dataset(tmp)
    table = {}
    for k, p in photos.items():
        # Photos 0-4 show rain, 5-9 forest, 10-12 rain with a crowd, the rest nothing in particular.
        table[p] = axis(RAIN) if k < 5 else axis(FOREST) if k < 10 else axis(RAIN, CROWD, 1.0) if k < 13 else axis(40 + k)
    for k, p in sounds.items():
        table[p] = axis(RAIN) if k < 3 else axis(FOREST)       # three sounds of rain, three of forest
    calls = {"n": 0}

    def encode(items, batch=8):
        calls["n"] += len(items)
        out = []
        for it in items:
            if it["kind"] == "text":
                v = sum((WORDS[w] for w in it["text"].split() if w in WORDS), np.zeros(DIM, np.float32))
                out.append(v if np.linalg.norm(v) else axis(700))
            else:
                if it["path"].endswith("img_020.jpg"):
                    raise omni.OmniError("unreadable")
                out.append(table[it["path"]])
        return np.stack(out).astype(np.float32)

    monkeypatch.setattr(omni, "encode", encode)
    monkeypatch.setattr(omni, "available", lambda: (True, ""))
    return {"photos": photos, "sounds": sounds, "calls": calls, "tmp": tmp}


def nums(result):
    return [int(h.handle.rpartition(":")[2]) for h in result.hits]


def test_embedding_adds_vectors_beside_the_index_and_resumes(world):
    index_file = os.path.join(world["tmp"], "databases", "index_synth_image.pkl")
    before = os.path.getmtime(index_file)
    first = ops.call("dataset embed", {"dataset": "synth", "limit": 10, "background": False})
    assert (first.added, first.kept, first.total) == (10, 0, synth.N)
    assert "are left" in first.text()
    rest = ops.call("dataset embed", {"dataset": "synth", "background": False})
    # One photo cannot be read: it is reported and left out, and the rest are kept.
    assert (rest.added, rest.kept, rest.skipped) == (synth.N - 11, 10, 1)
    again = ops.call("dataset embed", {"dataset": "synth", "background": False})
    assert again.kept == synth.N - 1 and again.added == 0 and again.skipped == 1
    keys, vectors = spaces.load("synth", "image")
    assert len(keys) == synth.N - 1 and vectors.shape == (synth.N - 1, DIM) and 20 not in set(keys)
    assert os.path.getmtime(index_file) == before          # the dataset's own index is untouched
    assert spaces.has("synth", "image") and not spaces.has("synth", "audio")


def test_a_dataset_must_be_embedded_before_it_is_searched_across(world):
    with pytest.raises(ops.Refusal, match="dataset embed synth"):
        ops.call("search across", {"dataset": "synth", "text": ["rain"]})
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    with pytest.raises(ops.Refusal, match="at least one"):
        ops.call("search across", {"dataset": "synth"})


def test_words_find_pictures_in_the_shared_space(world):
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    got = ops.call("search across", {"dataset": "synth", "text": ["forest"], "k": 5})
    assert sorted(nums(got)) == [5, 6, 7, 8, 9]
    assert got.hits[0].metric.startswith("z ")


def test_a_sound_finds_pictures_and_a_picture_finds_sounds(world):
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    # The sound's own dataset is not embedded: its vector is made on the spot.
    by_sound = ops.call("search across", {"dataset": "synth", "like": ["synth@audio:4"], "k": 5})
    assert sorted(nums(by_sound)) == [5, 6, 7, 8, 9]
    ops.call("dataset embed", {"dataset": "synth@audio", "background": False})
    n = world["calls"]["n"]
    by_picture = ops.call("search across", {"dataset": "synth@audio", "like": ["synth:2"], "k": 3})
    assert sorted(nums(by_picture)) == [0, 1, 2] and by_picture.hits[0].handle.startswith("synth@audio:")
    # Both sides were embedded, so the query's vector was read, and nothing was encoded.
    assert world["calls"]["n"] == n


def test_terms_mix_and_things_can_be_pushed_away(world):
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    rain = ops.call("search across", {"dataset": "synth", "like": ["synth@audio:0"], "k": 8})
    assert set(nums(rain)) == set(range(5)) | {10, 11, 12}
    dry = ops.call("search across", {"dataset": "synth", "like": ["synth@audio:0"],
                                     "not_text": ["people"], "away": 1.0, "k": 5})
    assert sorted(nums(dry)) == [0, 1, 2, 3, 4]
    assert "without" in dry.text()


def test_an_item_does_not_come_back_as_its_own_match(world):
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    got = ops.call("search across", {"dataset": "synth", "like": ["synth:6"], "k": 4})
    assert 6 not in nums(got) and sorted(nums(got)) == [5, 7, 8, 9]


def test_each_term_counts_for_its_weight_whatever_its_scale():
    keys = np.arange(4)
    V = np.stack([axis(0), axis(1), axis(0, 1, 1.0), axis(5)])
    # The first term's similarities are ten times smaller than the second's; it still decides half.
    small, big = axis(0) * 0.1, axis(1)
    ranked = spaces.rank(keys, V, [(small, 1.0), (big, 1.0)], k=1)
    assert ranked[0][0] == 2
    assert spaces.rank(keys, V, [], k=3) == []
    # Only allowed keys come back, however well the others match.
    assert {k for k, _ in spaces.rank(keys, V, [(axis(0), 1.0)], k=4, allowed={1, 3})} == {1, 3}


def test_the_tags_and_the_encoder_being_absent(world, monkeypatch):
    assert ops.get("search across").tags == ["image", "audio", "video"]
    monkeypatch.setattr(omni, "available", lambda: (False, "needs `uv` on the PATH."))
    with pytest.raises(ops.Refusal, match="uv"):
        ops.call("dataset embed", {"dataset": "synth", "background": False})


def test_deleting_a_dataset_takes_its_shared_vectors(world):
    from arcana import legacy
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    db = os.path.join(world["tmp"], "databases")
    files = legacy.dataset_files("synth", "image", db_dir=db, latents_dir=os.path.join(world["tmp"], "latents"),
                                 bundles_dir=os.path.join(world["tmp"], "bundles"))
    assert any(os.path.basename(f).startswith("space_synth_image.") for f in files)


def test_the_sound_channel_is_for_video_and_kept_apart(world):
    with pytest.raises(ops.Refusal, match="sound channel is for video"):
        ops.call("dataset embed", {"dataset": "synth", "sound": True, "background": False})
    ops.call("dataset embed", {"dataset": "synth", "background": False})
    with pytest.raises(ops.Refusal, match="by its sound"):
        ops.call("search across", {"dataset": "synth", "text": ["rain"], "listen": True})
    assert spaces.space_path("x", "video", spaces.SOUND).endswith("space_x_video.embeddinggemma-2-sound.npz")
    item = spaces._sound_item("C:/v/a.mp4#t=10.000,100.000")
    assert item["kind"] == "audio" and item["t0"] == 10.0 and item["t1"] == 10.0 + spaces.SOUND_SECONDS

