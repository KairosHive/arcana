"""
Sets, IP labels and LoRAs: what Alchemy learns from pictures gathered in Search.

Run with:  python -m pytest tests/test_learn.py -q
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import boards, ip_labels, loras, lora_train, ops  # noqa: E402
from arcana.alchemy import methods as mt  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def _vec(seed, scale=20.0):
    v = np.random.default_rng(seed).normal(size=ip_labels.DIM)
    return v / np.linalg.norm(v) * scale


# ─────────────────────────── sets ───────────────────────────

def test_a_set_grows_over_searches_and_skips_twins(ds):
    r = ops.call("set save", {"name": "Ice", "items": ["synth:1", "synth:2"], "query": "ice", "dataset": "synth"})
    assert r.added == 2 and r.set.kind == "set"
    r = ops.call("set save", {"name": "Ice", "items": ["synth:2", "synth:3"], "query": "frost", "dataset": "synth"})
    assert r.added == 1 and r.set.count == 3
    shown = ops.call("set show", {"set": "ice"})
    assert [q.text for q in shown.queries] == ["ice", "frost"] and len(shown.items) == 3
    r = ops.call("set save", {"name": "Ice", "items": ["synth:4"], "replace": True})
    assert r.set.count == 1


def test_sets_and_boards_are_listed_apart(ds):
    ops.call("board create", {"name": "Monsoon"})
    ops.call("set save", {"name": "Ice", "items": ["synth:1"]})
    assert [s.slug for s in ops.call("set list").sets] == ["ice"]
    kinds = {b.slug: b.kind for b in ops.call("board list").boards}
    assert kinds == {"monsoon": "board", "ice": "set"}
    with pytest.raises(ops.Refusal, match="is a board"):
        ops.call("set save", {"name": "Monsoon", "items": ["synth:1"]})
    ops.call("board set", {"board": "monsoon", "kind": "set"})
    assert {s.slug for s in ops.call("set list").sets} == {"ice", "monsoon"}


def test_the_board_tab_keeps_a_sets_kind_and_searches(ds):
    ops.call("set save", {"name": "Ice", "items": ["synth:1"], "query": "ice"})
    rec = boards.load("ice")
    boards.save("Ice", rec["items"], slug="ice")          # the Moodboard tab's save knows nothing of sets
    rec = boards.load("ice")
    assert rec["kind"] == "set" and rec["queries"][0]["text"] == "ice"


def test_export_copies_a_sets_files(ds, tmp):
    ops.call("set save", {"name": "Ice", "items": ["synth:1", "synth:2"]})
    r = ops.call("set export", {"set": "ice", "folder": "ice-out"})
    assert r.copied == 2 and len(os.listdir(r.folder)) == 3          # the pictures and search.json
    with open(os.path.join(r.folder, "search.json"), encoding="utf-8") as fh:
        assert json.load(fh)["set"] == "ice"
    with pytest.raises(ops.Refusal, match="not allowed"):
        ops.call("set export", {"set": "ice", "folder": "../x"})


# ─────────────────────────── labels ───────────────────────────

def test_a_label_round_trips_and_points_at_a_real_pictures_length(ds):
    members = np.stack([_vec(i) for i in range(6)])
    world = np.zeros(ip_labels.DIM)
    world[0] = 5
    lab = ip_labels.save("Ice", members, [f"{i}.jpg" for i in range(6)], {"kind": "set", "set": "ice"}, world)
    assert lab.slug == "ice" and lab.name == "Ice" and len(lab.axes) == 6
    assert np.isclose(np.linalg.norm(ip_labels.point(lab)), lab.norm, rtol=1e-4)
    # the mean of six random vectors is far shorter than any of them, and is drawn at their length
    assert np.linalg.norm(lab.centre) < 0.6 * lab.norm
    near = ip_labels.point(lab, push=1.0)
    far = ip_labels.point(lab, push=3.0)
    w = world / np.linalg.norm(world)
    assert (far / np.linalg.norm(far)) @ w < (near / np.linalg.norm(near)) @ w      # pushed away from the world
    assert ip_labels.options() == ("none", "ice")
    assert [r["slug"] for r in ip_labels.listing()] == ["ice"]


def test_slerp_and_mix_land_between_and_beyond(ds):
    a, b = _vec(1), _vec(2)
    assert np.allclose(ip_labels.slerp(a, b, 0), a, atol=1e-4)
    assert np.allclose(ip_labels.slerp(a, b, 1), b, atol=1e-4)
    mid = ip_labels.slerp(a, b, 0.5)
    ua, ub, um = (x / np.linalg.norm(x) for x in (a, b, mid))
    assert np.isclose(um @ ua, um @ ub, atol=1e-4)
    ip_labels.save("A", a[None], ["a.jpg"], {"kind": "picture"})
    ip_labels.save("B", b[None], ["b.jpg"], {"kind": "picture"})
    assert np.allclose(ip_labels.steer("a", "b", 0.0, push=1.0), ip_labels.point(ip_labels.load("a"), 1.0), atol=1e-3)
    assert ip_labels.steer("none") is None


def test_label_make_reads_a_sets_pictures(ds, monkeypatch):
    ops.call("set save", {"name": "Ice", "items": ["synth:1", "synth:2", "synth:3"]})
    seen = {}

    def fake(paths, per_picture=0, progress=None, cancelled=None):
        seen["paths"], seen["crops"] = list(paths), per_picture
        return np.stack([_vec(i) for i in range(len(paths))]), list(paths), None, {"from_index": len(paths)}

    monkeypatch.setattr(ip_labels, "vectors_for", fake)
    r = ops.call("label make", {"name": "ice", "set": "ice", "crops": 2})
    assert len(seen["paths"]) == 3 and seen["crops"] == 2
    assert r.label.pictures == 3 and r.label.origin["set"] == "ice"
    with pytest.raises(ops.Refusal, match="exactly one"):
        ops.call("label make", {"name": "x", "set": "ice", "picture": "synth:1"})
    assert ops.call("label delete", {"label": "ice"}).deleted


def test_labels_appear_in_the_diffusion_methods_as_they_are_made(ds):
    knob = next(k for k in mt.get("generative.controlnet").knobs if k.key == "label")
    assert mt.options_of(knob) == ("none",)
    ip_labels.save("Moss", _vec(3)[None], ["m.jpg"], {"kind": "picture"})
    assert "moss" in mt.get("generative.controlnet").public()["knobs"][[k.key for k in mt.get("generative.controlnet").knobs].index("label")]["options"]
    assert mt.settle_knobs(mt.get("generative.ip_adapter"), {"label": "moss"})["label"] == "moss"
    for mid in ("generative.controlnet", "generative.ip_adapter"):
        assert mt.get(mid).materials == (0, 1)


def test_a_step_with_neither_material_nor_label_says_so(ds):
    from arcana.alchemy.pipeline import Step

    step = Step(32, 32, np.zeros((32, 32, 3), np.float32), "p", [], [], [], None)
    with pytest.raises(ValueError, match="material \\[1\\] or an IP label"):
        mt.get("generative.ip_adapter").run(step, mt.settle_knobs(mt.get("generative.ip_adapter"), {}))


def test_a_label_reaches_the_adapter_as_its_embedding():
    import torch
    from arcana import style_transfer as st

    class Unet:
        dtype, device = torch.float32, torch.device("cpu")

    class Pipe:
        unet = Unet()

    v = _vec(5)
    got = st._ip_inputs(Pipe(), None, v, guidance=5.0)["ip_adapter_image_embeds"][0]
    assert got.shape == (2, 1, ip_labels.DIM)
    assert torch.all(got[0] == 0) and np.allclose(got[1, 0].numpy(), v, atol=1e-4)
    assert st._ip_inputs(Pipe(), None, v, guidance=1.0)["ip_adapter_image_embeds"][0].shape == (1, 1, ip_labels.DIM)
    assert st._ip_inputs(Pipe(), "picture", None) == {"ip_adapter_image": "picture"}


def test_a_label_searches_a_vith_dataset_only(ds):
    with pytest.raises(ValueError, match="ViT-H/14"):
        ip_labels.nearest_in("synth", _vec(1), 3)


# ─────────────────────────── LoRAs ───────────────────────────

def test_a_lora_trained_here_is_found_with_its_trigger_and_sheets(ds, tmp, monkeypatch):
    root = lora_train.loras_root()
    cp = os.path.join(root, "ice", "step-000500")
    os.makedirs(cp)
    open(os.path.join(cp, loras.WEIGHTS), "wb").close()
    open(os.path.join(cp, "sheet.jpg"), "wb").close()
    with open(os.path.join(cp, lora_train.META), "w", encoding="utf-8") as f:
        json.dump({"trigger": "in ice style", "step": 500}, f)
    monkeypatch.setattr(loras, "roots", lambda: [__import__("pathlib").Path(root)])
    assert "ice/step-000500" in loras.options()
    assert loras.trigger("ice/step-000500") == "in ice style"
    runs = ops.call("lora list").runs
    assert runs[0].name == "ice" and runs[0].checkpoints[0].sheet.endswith("sheet.jpg")
    knob = next(k for k in mt.get("generative.lora").knobs if k.key == "lora")
    assert "ice/step-000500" in mt.options_of(knob)


def test_training_crops_are_square_and_in_range(tmp):
    import random
    from PIL import Image

    p = os.path.join(tmp, "big.jpg")
    Image.fromarray((np.random.default_rng(0).random((600, 900, 3)) * 255).astype(np.uint8)).save(p)
    crop = lora_train._square(p, 128, 3.0, random.Random(0))
    assert crop.shape == (128, 128, 3) and crop.min() >= -1 and crop.max() <= 1


def test_training_refuses_without_a_gpu_or_enough_pictures(ds, monkeypatch):
    from arcana import gpu

    monkeypatch.setattr(gpu, "device", lambda: "cpu")
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        lora_train.train("ice", ["a.jpg"] * 10)
    monkeypatch.setattr(gpu, "device", lambda: "cuda")
    with pytest.raises(ValueError, match="at least 8"):
        lora_train.train("ice", ["missing.jpg"])


# ─────────────────────────── captions ───────────────────────────

def test_colours_are_named_from_the_pixels():
    from PIL import Image
    from arcana import captions

    rust = Image.new("RGB", (64, 64), (150, 76, 38))
    assert captions.colours_of(rust)[0] == "rust"
    half = np.zeros((64, 64, 3), np.uint8)
    half[:, :32] = (232, 228, 218)
    half[:, 32:] = (36, 118, 124)
    named = captions.colours_of(Image.fromarray(half))
    assert set(named) == {"bone white", "teal"}


def test_a_caption_reads_as_a_sentence_with_the_trigger_last():
    from arcana import captions

    c = captions.phrase("rippled water", ["glassy", "rippled"], ["steel blue"], "backlit", "in ice style")
    assert c == "a close-up photograph of rippled water, glassy and rippled, in steel blue, backlit, in ice style"


def test_a_film_lane_on_mix_walks_from_one_label_to_the_other():
    from arcana.alchemy import film

    f = {"looks": [{"t": 0, "ease": "linear", "settings": {"method": "generative.controlnet",
                                                            "knobs": {"label": "ice", "label2": "moss"}}}],
         "lanes": {"knobs.mix": [{"t": 0, "v": 0.0, "ease": "linear"}, {"t": 10, "v": 1.0, "ease": "linear"}]}}
    film.check_path("knobs.mix")
    mixes = [film.look_at(f, t)["knobs"]["mix"] for t in (0, 5, 10)]
    assert mixes[0] == 0.0 and 0.3 < mixes[1] < 0.7 and mixes[2] == 1.0
    assert film.look_at(f, 5)["knobs"]["label2"] == "moss"


def test_a_set_is_summed_up_in_its_typical_caption():
    from arcana import captions

    caps = {"a": captions.phrase("rippled water", ["glassy", "rippled"], ["taupe", "steel blue"], "backlit", "in w style"),
            "b": captions.phrase("rippled water", ["rippled", "smooth"], ["steel blue"], "backlit", "in w style"),
            "c": captions.phrase("still water", ["glassy"], ["navy"], "matte", "in w style")}
    assert captions.typical(caps, "in w style") == \
        "a close-up photograph of rippled water, glassy and rippled, in steel blue and taupe, backlit, in w style"



# ─────────────────────────── the archive ───────────────────────────

def test_the_archive_shows_sets_labels_loras_and_pages_apart(ds, tmp):
    from arcana import catalogue, paths

    ops.call("board create", {"name": "Monsoon"})
    ops.call("set save", {"name": "Ice", "items": ["synth:1", "synth:2"], "query": "ice", "dataset": "synth"})
    ip_labels.save("Ice", _vec(1)[None], ["a.jpg"], {"kind": "set", "set": "ice"})
    pages = os.path.join(paths.subdir("output"), "boards")
    os.makedirs(pages, exist_ok=True)
    for f in ("monsoon.page.html", "monsoon.pdf"):
        open(os.path.join(pages, f), "w").close()
    roots = catalogue.tree()
    assert [i["id"] for i in roots["collections"]["items"]] == ["collection:monsoon"]
    assert [i["id"] for i in roots["sets"]["items"]] == ["collection:ice"]
    assert [i["id"] for i in roots["labels"]["items"]] == ["label:ice"]
    assert [i["id"] for i in roots["pages"]["items"]] == ["page:monsoon"]
    shown = ops.call("archive show", {"id": "collection:ice"})
    assert shown.detail["queries"][0]["text"] == "ice"
    with pytest.raises(ops.Refusal, match="into another set"):
        ops.call("archive move", {"id": "collection:ice", "into": "collection:monsoon"})


def test_a_story_reads_as_sentences_beside_their_pictures(tmp):
    from PIL import Image
    from arcana.ops import catalogue as opcat

    folder = os.path.join(tmp, "story")
    for sub in ("original", "poetry_injected"):
        os.makedirs(os.path.join(folder, sub))
    for i in range(2):
        Image.new("RGB", (8, 8)).save(os.path.join(folder, "original", f"{i:02d}_original.jpg"))
    Image.new("RGB", (8, 8)).save(os.path.join(folder, "poetry_injected", "01_rain_poetry.png"))
    with open(os.path.join(folder, "story.txt"), "w", encoding="utf-8") as fh:
        fh.write("1. The rain begins\n2. The street floods\n")
    scenes = opcat.story_scenes(folder)
    assert [s["text"] for s in scenes] == ["The rain begins", "The street floods"]
    assert scenes[0]["poetry"] is None and scenes[1]["poetry"].endswith("01_rain_poetry.png")


def test_a_search_copied_before_searches_were_recorded_is_read_from_its_name(tmp):
    from arcana.ops import catalogue as opcat

    folder = os.path.join(tmp, "2026-03-01 frost on glass")
    os.makedirs(folder)
    d = opcat.detail_of({"kind": "searches", "path": folder, "id": "dir:x"})
    assert d["queries"][0]["text"] == "frost on glass" and d["guessed"] and d["date"] == "2026-03-01"


# ─────────────────────────── the material editor ───────────────────────────

def test_the_editor_names_what_the_mask_holds():
    from arcana.alchemy import editor as ed

    assert ed.mask_words({"layers": ["thing:dress"]}) == "the dress"
    assert ed.mask_words({"layers": ["thing:the cliff"]}) == "the cliff"
    assert ed.mask_words({"layers": ["faces"], "feather": 2}) == "the face"
    assert ed.mask_words({"layers": ["highlights"]}) is None
    assert ed.mask_words(None) is None
    assert "Replace the surface of the dress" in ed.instruction("the dress")
    assert "the masked area" in ed.instruction(None)


def test_the_editor_works_around_a_small_mask_and_gives_it_pixels():
    from arcana.alchemy import editor as ed

    m = np.zeros((4000, 6000), np.float32)
    m[1000:1300, 2000:2200] = 1
    box = ed.around(m)
    assert box[0] < 2000 and box[2] > 2200 and box[1] < 1000 and box[3] > 1300
    w, h = ed.worksize(box, (6000, 4000))
    assert max(w, h) >= 640 and w % 16 == 0 and h % 16 == 0
    assert ed.around(np.ones((100, 100), np.float32)) is None        # most of the frame: draw it whole


def test_the_editor_is_a_method_with_recipes(ds):
    from arcana.alchemy import pipeline, recipes

    m = mt.get("generative.edit")
    assert m.materials == (0, 1) and m.focused and "label" in [k.key for k in m.knobs]
    step = pipeline.Step(32, 32, np.zeros((32, 32, 3), np.float32), "p", [], [], [], None)
    with pytest.raises(ValueError, match="a picture on \[1\], an IP label, or words"):
        m.run(step, mt.settle_knobs(m, {}))
    spec = recipes.apply(recipes.get("material on the face"), {})
    assert spec["method"] == "generative.edit" and spec["mask"]["layers"] == ["faces"]


# ─────────────────────────── the Generate tab ───────────────────────────

def test_places_blend_by_weight_at_a_real_pictures_length(ds):
    from arcana import material_draw as md

    a, b = _vec(11), _vec(12)
    ip_labels.save("A", np.stack([a, a * 1.01]), ["a1.jpg", "a2.jpg"], {"kind": "picture"})
    ip_labels.save("B", np.stack([b, b * 0.99]), ["b1.jpg", "b2.jpg"], {"kind": "picture"})
    only_a = md.place([{"kind": "label", "id": "a", "weight": 1}], push=1.0)
    ua = a / np.linalg.norm(a)
    assert (only_a / np.linalg.norm(only_a)) @ ua > 0.999
    mostly_a = md.place([{"kind": "label", "id": "a", "weight": 3}, {"kind": "label", "id": "b", "weight": 1}], push=1.0)
    ub = b / np.linalg.norm(b)
    u = mostly_a / np.linalg.norm(mostly_a)
    assert u @ ua > u @ ub
    assert abs(np.linalg.norm(mostly_a) - np.linalg.norm(a)) / np.linalg.norm(a) < 0.05
    with pytest.raises(ValueError, match="at least one source"):
        md.place([{"kind": "label", "id": "a", "weight": 0}])


def test_a_set_becomes_a_place_and_pictures_are_shared_by_weight(ds, monkeypatch):
    from arcana import material_draw as md

    ops.call("set save", {"name": "Ice", "items": ["synth:1", "synth:2", "synth:3"]})
    monkeypatch.setattr(ip_labels, "vectors_for", lambda paths, *a, **k: (
        np.stack([_vec(i) for i in range(len(paths))]), list(paths), None, {}))
    lab = md.source_label({"kind": "set", "id": "ice"})
    assert lab.slug == "ice" and len(lab.files) == 3
    ip_labels.save("B", np.stack([_vec(40), _vec(41)]), ["b1.jpg", "b2.jpg"], {"kind": "picture"})
    refs = md.references([{"kind": "set", "id": "ice", "weight": 3}, {"kind": "label", "id": "b", "weight": 1}], n=4)
    assert len(refs) == 4 and sum(r.startswith("b") for r in refs) == 1


def test_only_drafts_can_be_kept(ds, tmp):
    from arcana import material_draw as md

    stray = os.path.join(tmp, "x.png")
    open(stray, "wb").close()
    with pytest.raises(ValueError, match="not a drawn material"):
        md.keep(stray)


def test_tiling_pads_circularly_and_puts_it_back():
    import torch
    from arcana import material_draw as md

    net = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
    with md.tiling(net):
        assert net[0].padding_mode == "circular"
    assert net[0].padding_mode == "zeros"


# ─────────────────────────── concept axes ───────────────────────────

def test_an_axis_points_from_one_end_to_the_other_and_moves_a_place_by_its_length(ds):
    rng = np.random.default_rng(0)
    base = _vec(50)
    toward = _vec(51)
    pos = np.stack([base + 6 * toward / np.linalg.norm(toward) + rng.normal(size=ip_labels.DIM) * 0.1 for _ in range(8)])
    neg = np.stack([base - 6 * toward / np.linalg.norm(toward) + rng.normal(size=ip_labels.DIM) * 0.1 for _ in range(8)])
    ax = ip_labels.make_axis("wet", pos, neg, {"pos": "words:wet", "neg": "words:dry"})
    t = toward / np.linalg.norm(toward)
    assert ax.direction @ t > 0.9 and ax.meta["apart"] > 5
    v = _vec(52)
    moved = ip_labels.along(v, "wet", 1.0)
    assert np.isclose(np.linalg.norm(moved), np.linalg.norm(v), rtol=1e-4)
    assert (moved / np.linalg.norm(moved)) @ ax.direction > (v / np.linalg.norm(v)) @ ax.direction
    assert np.allclose(ip_labels.along(v, "wet", 0.0), v)
    assert ip_labels.axis_options() == ("none", "wet")
    with pytest.raises(ValueError, match="ViT-H"):
        ip_labels.rank_along("synth", "wet")
    assert ops.call("axis delete", {"axis": "wet"}).deleted
