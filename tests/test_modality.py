"""
Modality as a declared property: one vocabulary, a tag on every op that takes
media, one refusal, and a capability table the page and the routes both read.
See VIDEO_ROADMAP.md, part 1.

Run with:  python -m pytest tests/test_modality.py -q
"""

import os
import pickle
import sys
import wave

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import capabilities, modality, ops  # noqa: E402
from arcana.ops import archive, registry  # noqa: E402

import synth  # noqa: E402


def write_audio_dataset(tmp, name="synth", n=6):
    """A few silent wavs indexed as `<name>@audio`, beside whatever else is in tmp."""
    from usearch.index import Index
    sounds = os.path.join(tmp, "sounds")
    os.makedirs(sounds, exist_ok=True)
    index = Index(ndim=synth.DIM, metric="cos")
    idx2path = {}
    rng = np.random.default_rng(3)
    for k in range(n):
        p = os.path.join(sounds, f"snd_{k:03d}.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        index.add(k, (synth.axis(0) + 0.05 * rng.standard_normal(synth.DIM)).astype(np.float32))
        idx2path[k] = p
    with open(os.path.join(tmp, "databases", f"index_{name}_audio.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)
    pd.DataFrame({"x": np.arange(n, dtype=np.float32), "y": np.zeros(n, np.float32),
                  "path": list(idx2path.values()), "label": ["Water"] * n,
                  }).to_pickle(os.path.join(tmp, "latents", f"latent_space_{name}_audio_2d.pkl"))
    return idx2path


@pytest.fixture
def both(tmp, monkeypatch):
    """A photo dataset and an audio dataset that share the name `synth`."""
    photos = synth.use(tmp, monkeypatch, palette_features=True)
    sounds = write_audio_dataset(tmp)
    return photos, sounds


# ─────────────────────────── the vocabulary ───────────────────────────

def test_dataset_ids_round_trip_for_every_modality():
    assert modality.dataset_id("japan", "image") == "japan"
    assert modality.dataset_id("field", "audio") == "field@audio"
    assert modality.dataset_id("rushes", "video") == "rushes@video"
    for m in modality.ALL:
        assert modality.split_dataset_id(modality.dataset_id("a_b", m)) == ("a_b", m)
    # A name that merely holds an @ is still a photo dataset.
    assert modality.split_dataset_id("me@home") == ("me@home", "image")


def test_handles_and_paths_say_their_modality():
    assert modality.of_handle("japan:12") == "image"
    assert modality.of_handle("field@audio:3") == "audio"
    assert modality.of_handle("#3") is None
    assert modality.of_path("a/b/c.WAV") == "audio"
    assert modality.of_path("clip.mov") == "video"
    assert modality.of_path("notes.txt") is None


def test_dataset_files_of_every_modality_are_found():
    for m in modality.ALL:
        assert modality.INDEX_RE.match(f"index_my_set_{m}.pkl").groups() == ("my_set", m)
        assert modality.LATENT_RE.match(f"latent_space_my_set_{m}_3d.pkl").groups() == ("my_set", m, "3")


def test_words():
    assert modality.words({"image"}) == "photos"
    assert modality.words({"audio", "image"}) == "photos and sounds"
    assert modality.words(modality.ALL) == "photos, sounds and videos"


# ─────────────────────────── ops carry a tag ───────────────────────────

def test_every_op_that_takes_media_declares_its_modalities():
    for o in ops.all_ops():
        if registry.media_fields(o.args):
            assert o.modalities, f"{o.name} takes media and declares no modalities"
            assert o.modalities <= set(modality.ALL)


def test_an_op_that_takes_media_without_a_tag_is_refused_at_registration():
    class A(ops.Args):
        dataset: str = archive.dataset_arg()

    with pytest.raises(TypeError, match="declares its modalities"):
        registry.register(ops.Op("scratch untagged", "read", A, ops.TextResult,
                                 lambda a, c: ops.TextResult(body=""), "A scratch op."))
    assert ops.get("scratch untagged") is None


def test_the_tag_reaches_help_the_op_list_and_the_schema(both):
    assert "works with photos, sounds and videos" in ops.help_text(ops.get("search text"))
    assert "works with photos and sounds" in ops.help_text(ops.get("board note"))
    assert "works with photos" in ops.help_text(ops.get("colour transfer"))
    assert "works with" not in ops.help_text(ops.get("undo"))
    rows = {r.name: r for r in ops.call("op list").ops}
    assert rows["search palette"].modalities == ["image"]
    assert rows["search text"].modalities == ["image", "audio", "video"]
    assert rows["dataset marks"].modalities == ["image"]
    assert rows["undo"].modalities == []
    assert ops.grammar.schema()["item show"]["modalities"] == ["image"]


def _args_with(op, ref_for):
    """An args object with only the media fields filled, which is all the check reads."""
    values = {}
    for name, role in registry.media_fields(op.args):
        info = op.args.model_fields[name]
        ref = ref_for(role)
        values[name] = [ref] if "list" in str(info.annotation) else ref
    return op.args.model_construct(**values)


@pytest.mark.parametrize("other", ["audio", "video"])
def test_every_photo_only_op_refuses_other_media_in_one_sentence(other):
    photo_only = [o for o in ops.all_ops() if o.modalities == frozenset({"image"})
                  and registry.media_fields(o.args)]
    assert len(photo_only) >= 10
    for o in photo_only:
        parsed = _args_with(o, lambda role: f"set@{other}" if role == "dataset" else f"set@{other}:3")
        with pytest.raises(ops.Refusal) as e:
            registry.check_modalities(o, parsed)
        assert e.value.reason.startswith(f"`{o.name}` works with photos. "), e.value.reason
        # And the same op takes a photo without complaint.
        registry.check_modalities(o, _args_with(o, lambda role: "set" if role == "dataset" else "set:3"))


def test_a_file_is_judged_by_its_extension():
    o = ops.get("colour transfer")
    parsed = o.args.model_construct(target="C:\\sounds\\rain.wav", reference="japan:7")
    with pytest.raises(ops.Refusal, match="rain.wav is a sound"):
        registry.check_modalities(o, parsed)


def test_ops_on_the_two_datasets(both):
    listed = {d.id for d in ops.call("dataset list").datasets}
    assert {"synth", "synth@audio"} <= listed

    # Text search runs on both.
    assert ops.call("search text", {"dataset": "synth", "phrase": ["water"]}).facets[0].hits
    assert ops.call("search text", {"dataset": "synth@audio", "phrase": ["water"]}).facets[0].hits

    # Palette search, a contact sheet and a colour transfer are for photos.
    for name, args in (("search palette", {"dataset": "synth@audio", "colour": ["#336699"]}),
                       ("item sheet", {"handles": ["synth@audio:1"]}),
                       ("colour transfer", {"target": "synth@audio:1", "reference": "synth:2"}),
                       ("alchemy preview", {"picture": "synth@audio:1"})):
        with pytest.raises(ops.Refusal) as e:
            ops.call(name, args)
        assert "works with photos" in e.value.reason, (name, e.value.reason)
        assert e.value.help, "a refusal carries the op's help"


# ─────────────────────────── features stay with their dataset ───────────────────────────

def test_a_photo_sets_features_are_not_read_by_an_audio_set_of_the_same_name(both, tmp):
    from arcana import db as _db
    assert archive.feature_blocks("synth", "image") == ["palette"]
    assert archive.feature_blocks("synth", "audio") == []
    assert set(_db.existing_feature_paths("synth", "image")) == {"palette"}
    assert _db.existing_feature_paths("synth", "audio") == {}
    assert archive.describe_dataset("synth@audio").features == []
    assert modality.feature_file("synth", "palette") == "features_synth_palette.npz"
    assert modality.feature_file("synth", "motion", "video") == "features_synth_video_motion.npz"


# ─────────────────────────── the capability table ───────────────────────────

def test_every_capability_is_well_formed():
    for f in capabilities.FEATURES:
        assert f.scope in ("dataset", "item", "none")
        assert set(f.tabs) <= set(capabilities.TABS)
        assert f.modalities is not None or f.ops, f"{f.id} states no modalities and no ops"
        for name in f.ops:
            assert ops.get(name) is not None, f"{f.id} names a missing op: {name}"
        assert capabilities.modalities(f.id), f"{f.id} works with nothing"


def test_a_capability_backed_by_ops_takes_their_modalities():
    assert capabilities.modalities("alchemy.step") == ("image",)
    assert capabilities.modalities("dataset.build") == ("image", "audio", "video")
    assert capabilities.modalities("dataset.marks") == ("image",)
    assert capabilities.modalities("moodboard.collection") == ("image", "audio", "video")
    assert capabilities.modalities("story.poetry") == ("image",)


def test_the_table_the_page_reads():
    t = capabilities.table()
    assert set(t["tabs"]) == set(capabilities.TABS)
    assert t["tabs"]["prompt"] == ["image", "audio", "video"]
    assert t["tabs"]["moodboard"] == ["image", "audio", "video"]
    assert t["tabs"]["alchemy"] == ["image", "video"]
    assert t["labels"]["image"] == "photo"
    assert "alchemy" not in t["datasetTabs"] and "prompt" in t["datasetTabs"]
    by_id = {f["id"]: f for f in t["features"]}
    assert by_id["moodboard.similar"]["modalities"] == ["image"]
    assert by_id["search.spectrogram"]["modalities"] == ["audio"]


def test_the_refusal_sentence():
    assert (capabilities.reason("moodboard.similar", "audio")
            == "Similarity search by palette and style works with photos. This dataset holds sounds.")
    with pytest.raises(capabilities.Unsupported, match="rain.wav is a sound"):
        capabilities.require_paths("story.save", ["C:/x/rain.wav"])
    capabilities.require_paths("story.save", ["C:/x/rain.jpg"])


# ─────────────────────────── the page's routes ───────────────────────────

@pytest.fixture
def page(both, tmp, monkeypatch):
    from flask import Flask
    from arcana import api, media
    monkeypatch.setattr(media, "DB_DIR", os.path.join(tmp, "databases"))
    monkeypatch.setattr(media, "LATENTS_DIR", os.path.join(tmp, "latents"))
    app = Flask("t")
    api.register(app)
    c = app.test_client()
    c.environ_base["HTTP_X_ARCANA_UI"] = api.UI_TOKEN
    return c, both


def test_bootstrap_serves_the_capabilities(page):
    c, _ = page
    b = c.get("/api/bootstrap").get_json()
    assert {f["id"] for f in b["capabilities"]["features"]} >= {"search.text", "alchemy.step"}
    assert {d["modality"] for d in b["datasets"]} == {"image", "audio"}


def test_routes_refuse_what_the_table_says_they_do_not_take(page):
    c, (photos, sounds) = page
    r = c.post("/api/similar", json={"dataset": "synth::2::audio", "reference": sounds[0]})
    assert r.status_code == 409
    assert r.get_json()["error"].endswith("works with photos. This dataset holds sounds.")
    assert r.get_json()["feature"] == "moodboard.similar"

    scenes = [{"text": "rain", "path": sounds[0]}]
    r = c.post("/api/story/save", json={"scenes": scenes, "text": "rain"})
    assert r.status_code == 409 and "is a sound" in r.get_json()["error"]
    assert not os.path.isdir(os.path.join(os.environ["ARCANA_DATA_DIR"], "output", "stories"))

    r = c.post("/api/story/poetry", json={"scenes": scenes})
    assert r.status_code == 409 and "Inject Poetry works with photos" in r.get_json()["error"]


def test_features_are_reported_per_modality(page):
    c, _ = page
    assert c.get("/api/features?name=synth").get_json() == {"palette": True, "style": False}
    assert c.get("/api/features?name=synth&modality=audio").get_json() == {"palette": False, "style": False}

