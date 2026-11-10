"""
Alchemy's films: the timeline, the looks and lanes over time, the morph, the render and the ops.

The token spaces load large models, so the morph and the render run here on a stand-in space whose
tokens are each 16 px patch's mean colour. That keeps what a film does with its space (encode the
pictures once, move along the curve through their grids, decode each frame) and runs in a moment.

Run with:  python -m pytest tests/test_film.py -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import ops  # noqa: E402
from arcana.alchemy import film as F  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402


class PatchSpace:
    """A token space for tests: a token is a 16 x 16 patch's mean colour."""

    side = 16

    def grid(self, width, height, long=None):
        long = long or self.side
        if width >= height:
            return max(1, round(long * height / width)), long
        return long, max(1, round(long * width / height))

    def shell(self, z):
        return z

    def encode(self, images, grid=None):
        import torch
        rows, cols = grid
        x = np.asarray(images, np.float32)
        x = x[None] if x.ndim == 3 else x
        out = []
        for im in x:
            h, w = im.shape[:2]
            ys = np.linspace(0, h, rows + 1).astype(int)
            xs = np.linspace(0, w, cols + 1).astype(int)
            out.append([im[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].reshape(-1, 3).mean(0)
                        for r in range(rows) for c in range(cols)])
        return torch.tensor(np.array(out, np.float32))

    def decode(self, tokens, grid=None):
        rows, cols = grid
        z = tokens.detach().cpu().numpy().reshape(len(tokens), rows, cols, 3)
        return np.clip(np.repeat(np.repeat(z, 16, 1), 16, 2), 0, 1).astype(np.float32)


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    space = PatchSpace()
    monkeypatch.setattr(F.spaces, "space", lambda name: space)
    F._ready.clear()
    return out


def _spec(**more):
    spec = {"name": "test", "fps": 6, "size": 256, "aspect": "4:3", "even": False,
            "pictures": [{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": "synth:7"}, {"t": 4, "ref": "synth:3"}]}
    spec.update(more)
    return spec


# ─────────────────────────── the timeline ───────────────────────────

def test_a_film_fills_its_defaults_and_refuses_what_is_not_a_film():
    f = F.normalised(_spec())
    assert f["loop"] and f["length"] == 6.0           # the run back takes an average passage
    assert F.normalised(_spec(loop=False))["length"] == 4.0
    assert F.normalised(_spec(length=10))["length"] == 10.0
    with pytest.raises(ValueError, match="at least two pictures"):
        F.normalised(_spec(pictures=[{"t": 0, "ref": "x"}]))
    with pytest.raises(ValueError, match="share a time"):
        F.normalised(_spec(pictures=[{"t": 0, "ref": "x"}, {"t": 0, "ref": "y"}]))
    with pytest.raises(ValueError, match="No lane called"):
        F.normalised(_spec(lanes={"knobs.picture.x": [{"t": 0, "v": 1}]}))
    with pytest.raises(ValueError, match="An ease is one of"):
        F.normalised(_spec(pictures=[{"t": 0, "ref": "x", "ease": "bounce"}, {"t": 1, "ref": "y"}]))


def test_the_film_passes_each_picture_at_its_key_and_eases_between():
    f = F.normalised(_spec(pictures=[{"t": 0, "ref": "a"}, {"t": 2, "ref": "b", "ease": "hold"},
                                     {"t": 4, "ref": "c", "ease": "smooth"}]))
    assert F.passage(f, 0) == (0, 0.0) and F.passage(f, 2) == (1, 0.0) and F.passage(f, 4) == (2, 0.0)
    assert F.passage(f, 1)[1] == pytest.approx(0.5)
    assert F.passage(f, 3) == (1, 0.0)                 # hold: stays on b until c
    assert F.passage(f, 5)[1] == pytest.approx(0.5)   # smooth: half way at the middle
    assert F.passage(f, 6) == (0, 0.0)                 # the loop is back at the start
    g = F.normalised(_spec(loop=False))
    assert F.passage(g, 9) == (2, 0.0)                 # an open film ends on its last picture


def test_looks_move_their_numbers_and_switch_the_rest_at_the_next_look():
    looks = [{"t": 0, "ease": "linear", "settings": {"method": "none", "finish": {"saturation": 0}}},
             {"t": 4, "settings": {"method": "pixels.chroma", "finish": {"saturation": -100}}}]
    f = F.normalised(_spec(looks=looks))
    half = F.look_at(f, 2)
    assert half["finish"]["saturation"] == pytest.approx(-50) and half["method"] == "none"
    assert F.look_at(f, 4)["method"] == "pixels.chroma"
    assert F.look_at(F.normalised(_spec()), 1) == {"method": "none"}


def test_lanes_win_over_the_looks_and_move_any_number():
    looks = [{"t": 0, "settings": {"method": "none", "materials": ["synth:1", "synth:2"], "weights": [1, 1],
                                   "mask": {"layers": ["centre spot", "highlights", "gradient:top"]},
                                   "finish": {"saturation": 0}}}]
    lanes = {"finish.saturation": [{"t": 0, "v": 0}, {"t": 4, "v": -100}],
             "mask.1.x": [{"t": 0, "v": 0.2}, {"t": 4, "v": 0.6}],
             "mask.2.coverage": [{"t": 0, "v": 0.1}, {"t": 4, "v": 0.9}],
             "mask.3.angle": [{"t": 0, "v": 0}, {"t": 4, "v": 90}],
             "mask.feather": [{"t": 0, "v": 12}],
             "weights.2": [{"t": 0, "v": 0}, {"t": 4, "v": 0.5}],
             "zoom": [{"t": 0, "v": 1}, {"t": 4, "v": 3}]}
    s = F.look_at(F.normalised(_spec(looks=looks, lanes=lanes)), 2)
    assert s["finish"]["saturation"] == pytest.approx(-50)
    assert s["mask"]["layers"][0] == "spot:0.4,0.5,0.3"
    assert s["mask"]["coverage"][1] == pytest.approx(0.5)
    assert s["mask"]["layers"][2] == "gradient:45"
    assert s["mask"]["feather"] == 12
    assert s["weights"] == [1, pytest.approx(0.25)] and s["zoom"] == pytest.approx(2.0)


def test_a_lane_eases_from_each_key_to_the_next():
    keys = [{"t": 0, "v": 0, "ease": "in"}, {"t": 2, "v": 10, "ease": "hold"}, {"t": 4, "v": 20}]
    assert F.lane_value(keys, 1) == pytest.approx(2.5)
    assert F.lane_value(keys, 3) == 10 and F.lane_value(keys, 4) == 20 and F.lane_value(keys, 9) == 20


def test_spacing_speeds_the_morph_up_or_down_between_the_same_ends():
    spec = _spec(pictures=[{"t": i, "ref": f"p{i}"} for i in range(6)])
    faster = [p["t"] for p in F.spaced(spec, "faster", 0, 10)["pictures"]]
    slower = [p["t"] for p in F.spaced(spec, "slower", 0, 10)["pictures"]]
    gaps = np.diff(faster)
    assert faster[0] == 0 and faster[-1] == 10 and all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))
    assert all(np.diff(np.diff(slower)) > 0)


def test_a_geometric_layer_reads_and_writes_its_numbers():
    assert F.layer_numbers("centre spot") == ("spot", ("x", "y", "r"), [0.5, 0.5, 0.3])
    assert F.layer_numbers("gradient:bottom")[2] == [180.0]
    assert F.layer_numbers("noise:cells,30")[2:] == ([30.0], "cells")
    assert F.layer_numbers("highlights") is None
    assert F.layer_line("band", [0.4, 0.2]) == "band:0.4,0.2"


# ─────────────────────────── the morph and the render ───────────────────────────

def test_the_morph_passes_through_each_picture(ds):
    film = F.Film(_spec())
    first = film.morph(0.0)
    pic = film.grids[0].numpy().reshape(film.rows, film.cols, 3)
    assert np.allclose(first[::16, ::16], pic, atol=1e-5)
    between = film.morph(1.0)
    assert not np.allclose(between, first) and not np.allclose(between, film.morph(2.0))
    assert film.even() and film.morph(2.0).shape == first.shape


def test_a_model_mask_is_read_on_the_key_pictures_and_blended(ds, monkeypatch):
    film = F.Film(_spec())
    monkeypatch.setattr(film, "_map", lambda i, kind, arg: np.full((film.H, film.W), float(i), np.float32))
    assert np.allclose(film.asker(1.0)("subject"), 0.5)
    assert np.allclose(film.asker(5.0)("subject"), 1.0)       # half way from the last back to the first


def test_a_film_renders_every_frame(ds, tmp):
    import imageio_ffmpeg
    out = os.path.join(tmp, "film.mp4")
    made = F.render(_spec(lanes={"finish.saturation": [{"t": 0, "v": 0}, {"t": 6, "v": -100}]}), out)
    assert made["frames"] == 36 and os.path.isfile(out)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(out)
    assert frames == 36


def test_a_render_stops_cleanly_when_cancelled(ds, tmp):
    calls = []

    def progress(f, m):
        calls.append(f)
    progress.cancelled = lambda: len(calls) > 3
    made = F.render(_spec(), os.path.join(tmp, "cut.mp4"), progress)
    assert 0 < made["frames"] < made["of"]


# ─────────────────────────── the ops ───────────────────────────

def test_films_are_kept_shown_and_deleted_with_undo(ds, tmp):
    spec = _spec(name="Rites")
    ops.call("alchemy film save", {"spec": spec}, actor="cli")
    assert [f.name for f in ops.call("alchemy film list", {}, actor="cli").films] == ["Rites"]
    shown = ops.call("alchemy film show", {"film": "rites"}, actor="cli")
    assert shown.film.passes == [0.0, 2.0, 4.0] and shown.spec["length"] == 6.0
    ops.call("alchemy film delete", {"film": "rites"}, actor="cli")
    assert ops.call("alchemy film list", {}, actor="cli").films == []
    ops.call("undo", {}, actor="cli")
    assert [f.name for f in ops.call("alchemy film list", {}, actor="cli").films] == ["Rites"]


def test_an_agent_names_a_films_pictures_by_handle(ds):
    spec = _spec(pictures=[{"t": 0, "ref": ds[12]}, {"t": 2, "ref": "synth:7"}])
    with pytest.raises(ops.Refusal, match="by handle"):
        ops.call("alchemy film show", {"spec": spec}, actor="agent")
    ops.call("alchemy film show", {"spec": _spec()}, actor="agent")


def test_a_preview_draws_a_frame_and_a_make_records_the_video(ds, tmp):
    r = ops.call("alchemy film preview", {"spec": _spec(), "at": 1.0, "size": 256}, actor="cli")
    assert r.attachments and r.size == [256, 192]
    m = ops.call("alchemy film make", {"spec": _spec(name="Rites")}, actor="cli")
    assert m.frames == 36 and os.path.isfile(m.path) and os.path.isfile(os.path.splitext(m.path)[0] + ".json")
    assert os.path.basename(os.path.dirname(os.path.dirname(m.path))) == "video"
    from arcana import ledger
    entry = ledger.get(m.id)
    assert entry["op"] == "alchemy film make" and entry["sources"] == ["synth:12", "synth:7", "synth:3"]


# ─────────────────────────── templates ───────────────────────────

def test_a_template_keeps_the_keys_and_settings_and_stretches_them_to_another_film(ds):
    lanes = {"finish.saturation": [{"t": 0, "v": 0}, {"t": 6, "v": -100, "ease": "in"}]}
    looks = [{"t": 3, "ease": "smooth", "settings": {"method": "none"}}]
    saved = ops.call("alchemy template save", {"name": "Slow fade", "spec": _spec(lanes=lanes, looks=looks)},
                     actor="cli").template
    assert (saved.name, saved.length, saved.lanes) == ("Slow fade", 6.0, ["finish.saturation"])
    assert "pictures" not in F.load_template("slow fade")

    other = {"name": "other", "fps": 30, "size": 512, "even": True,
             "pictures": [{"t": 0, "ref": "synth:1"}, {"t": 6, "ref": "synth:2"}, {"t": 12, "ref": "synth:3"}]}
    got = ops.call("alchemy template apply", {"template": "Slow fade", "spec": other}, actor="agent").spec
    assert [p["ref"] for p in got["pictures"]] == ["synth:1", "synth:2", "synth:3"]
    assert (got["name"], got["fps"], got["size"], got["aspect"], got["even"]) == ("other", 6, 256, "4:3", False)
    assert [(k["t"], k["v"], k["ease"]) for k in got["lanes"]["finish.saturation"]] == [(0, 0, "linear"), (18, -100, "in")]
    assert [k["t"] for k in got["looks"]] == [9]

    empty = ops.call("alchemy template apply", {"template": "Slow fade", "spec": {}}, actor="cli").spec
    assert [k["t"] for k in empty["lanes"]["finish.saturation"]] == [0, 6]


def test_templates_are_listed_deleted_and_undone(ds):
    ops.call("alchemy template save", {"name": "Slow fade", "spec": _spec()}, actor="cli")
    assert [t.name for t in ops.call("alchemy template list", {}, actor="cli").templates] == ["Slow fade"]
    assert "agent" not in ops.get("alchemy template delete").actors
    ops.call("alchemy template delete", {"template": "slow fade"}, actor="cli")
    assert ops.call("alchemy template list", {}, actor="cli").templates == []
    ops.call("undo", {}, actor="cli")
    assert [t.name for t in ops.call("alchemy template list", {}, actor="cli").templates] == ["Slow fade"]
    with pytest.raises(ops.Refusal, match="No film template"):
        ops.call("alchemy template apply", {"template": "rain", "spec": {}}, actor="cli")


# ─────────────────────────── a film inside a film, lane shapes, derivatives ───────────────────────────

def _inner(**more):
    inner = {"name": "inner", "fps": 6, "size": 256, "aspect": "4:3", "even": False, "loop": False,
             "pictures": [{"t": 0, "ref": "synth:5"}, {"t": 2, "ref": "synth:9"}]}
    inner.update(more)
    return F.save(inner)


def test_a_kept_film_plays_on_the_picture_lane_between_two_morphs(ds):
    _inner()
    f = F.normalised(_spec(pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"},
                                     {"t": 8, "ref": "synth:3"}]))
    held = f["pictures"][1]
    assert (held["in"], held["out"], held["speed"]) == (0.0, 2.0, 1.0) and F.span(held) == 2.0
    stops = F.points(f)
    assert [(q["t"], q["plays"]) for q in stops] == [(0, False), (3, True), (5, False), (8, False)]
    assert F.passage(f, 4.0) == (1, 0.5)                      # half way through the held film
    assert F.passage(f, 6.5)[0] == 2                          # morphing from its last frame to the next picture
    film = F.Film(f)
    inner = F.Film(F.load("inner"), 256, 1, aspect="4:3")
    assert np.allclose(film.morph(4.0), film._fit(inner.frame(1.0), film.W, film.H), atol=1e-5)
    assert np.allclose(film.stills[2], film._fit(inner.frame(2.0), film.W, film.H), atol=1e-5)
    assert film.even() and film.table[1][0].tolist() == [0.0, 1.0]   # the held film keeps its own pace
    # The morph lands on the held film's own frame, so nothing jumps where the film starts or ends.
    arrive, leave = film.morph(3.0 - 1e-6), film.morph(5.0 + 1e-6)
    assert np.allclose(arrive, film._fit(inner.frame(0.0), film.W, film.H), atol=2e-3)
    assert np.allclose(leave, film._fit(inner.frame(2.0), film.W, film.H), atol=2e-3)


def test_a_held_film_is_refused_where_it_overlaps_or_holds_itself(ds):
    _inner()
    with pytest.raises(ValueError, match="past the next key"):
        F.normalised(_spec(pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"},
                                     {"t": 4, "ref": "synth:3"}]))
    with pytest.raises(ValueError, match="cannot hold itself"):
        F.normalised(_spec(name="inner", pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"}]))
    with pytest.raises(ValueError, match="in before out"):
        F.normalised(_spec(pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner", "in": 1.5,
                                                                      "out": 1.0}, {"t": 9, "ref": "synth:3"}]))
    F.save({**_spec(name="outer"), "pictures": [{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"}]})
    nested = {**_spec(name="deep"), "pictures": [{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:outer"}]}
    assert F.normalised(nested)["pictures"][1]["out"] > 0     # two deep is allowed
    assert F.stills_of(nested) == ["synth:12", "synth:12", "synth:5", "synth:9"]


def test_the_amount_lane_lays_the_look_over_the_morph_bit_by_bit(ds):
    from arcana.alchemy.pipeline import step_on
    spec = _spec(looks=[{"t": 0, "settings": {"method": "pixels.detail", "materials": ["synth:7"]}}],
                 lanes={"amount": [{"t": 0, "v": 0}, {"t": 4, "v": 1}]})
    f = F.normalised(spec)
    assert F.look_at(f, 2.0)["amount"] == pytest.approx(0.5)
    film = F.Film(f)
    c = film.morph(2.0)
    full, _ = step_on(c, "k", {**F.look_at(f, 2.0), "amount": 1.0})
    half, _ = step_on(c, "k", F.look_at(f, 2.0))
    none, _ = step_on(c, "k", {**F.look_at(f, 2.0), "amount": 0.0})
    assert np.allclose(none, c, atol=1e-5)
    assert np.allclose(half, (c + full) / 2, atol=2e-3)


def test_masks_glide_between_looks_by_their_numbers_or_cross_fade_when_their_layers_differ(ds):
    from arcana.alchemy.pipeline import step_on
    step = {"method": "pixels.detail", "materials": ["synth:7"]}
    glide = [{"t": 0, "ease": "linear", "settings": {**step, "mask": {"layers": ["spot:0.2,0.5,0.2"], "feather": 0}}},
             {"t": 4, "settings": {**step, "mask": {"layers": ["spot:0.8,0.5,0.2"], "feather": 20}}}]
    s = F.look_at(F.normalised(_spec(looks=glide)), 2.0)
    assert s["mask"]["layers"] == ["spot:0.5,0.5,0.2"] and s["mask"]["feather"] == pytest.approx(10)
    assert "mask_to" not in s and s["mask"]["seed"] == 0 and isinstance(s["mask"]["seed"], int)
    fade = [{"t": 0, "ease": "linear", "settings": {**step, "mask": {"layers": ["gradient:left"], "feather": 0}}},
            {"t": 4, "settings": {**step, "mask": {"layers": ["gradient:right"], "invert": True, "feather": 0}}},
            {"t": 6, "settings": {**step, "mask": {"layers": ["highlights"]}}}]
    f = F.normalised(_spec(looks=fade, lanes={"mask.feather": [{"t": 0, "v": 0}]}))
    s = F.look_at(f, 1.0)
    assert s["mask_mix"] == pytest.approx(0.25) and s["mask_to"]["invert"] and s["mask_to"]["feather"] == 0
    film = F.Film(f)
    c = film.morph(1.0)
    _, a = step_on(c, "k", {**s, "mask_mix": 0.0})
    _, b = step_on(c, "k", {**s, "mask_mix": 1.0})
    _, m = step_on(c, "k", s)
    assert not np.allclose(a, b) and np.allclose(m, 0.75 * a + 0.25 * b, atol=1e-5)


def test_a_lane_takes_a_whole_shape_between_two_times():
    from arcana import ui_film as U
    s = U.shape_lane({}, "finish.saturation", "wave", -50, 50, 0, 8, 2)
    keys = s["lanes"]["finish.saturation"]
    assert [(k["t"], k["v"]) for k in keys] == [(0, -50), (2, 50), (4, -50), (6, 50), (8, -50)]
    kept = {"lanes": {"finish.exposure": [{"t": 1, "v": 0.2}, {"t": 9, "v": 0.9}]}}
    s = U.shape_lane(kept, "finish.exposure", "swell", 0, 1, 0, 4)
    assert [(k["t"], k["v"]) for k in s["lanes"]["finish.exposure"]] == [(0, 0), (2, 1), (4, 0), (9, 0.9)]
    s = U.shape_lane({}, "amount", "steps", 0, 1, 0, 6, 3)
    assert [k["v"] for k in s["lanes"]["amount"]] == [0, 0.5, 1, 1] and s["lanes"]["amount"][0]["ease"] == "hold"
    with pytest.raises(ValueError):
        U.shape_lane({}, "nothing.here", "rise", 0, 1, 0, 4)


def test_a_series_is_laid_after_a_picture_and_moves_what_follows():
    from arcana import ui_film as U
    spec = {"pictures": [{"t": 0, "ref": "a"}, {"t": 4, "ref": "b"}]}
    s, end = U.place_series(spec, ["d1", "d2", "a"], 3, every=3)
    assert [(p["t"], p["ref"]) for p in s["pictures"]] == [(0, "a"), (3, "d1"), (6, "d2"), (9, "a"), (13, "b")]
    assert end == 12


def test_the_timeline_lists_a_held_film_with_its_length(ds):
    from arcana import ui_film as U
    _inner()
    m = U.meta(_spec(pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"}]))
    assert m["pictures"][1]["film"] == "inner" and m["pictures"][1]["dur"] == 2.0
    s, t = U.add_picture({"pictures": [{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "synth:7"}]}, "film:inner", 2.0)
    assert t == 7.0                                          # it would run into the next picture, so it goes last
    assert "amount" in U.lane_meta(_spec(looks=[{"t": 0, "settings": {"method": "pixels.detail"}}]))[1]


def test_derivatives_vary_one_thing_and_are_kept(ds):
    made = ops.call("alchemy derive", {"picture": "synth:12", "material": ["synth:7"], "method": "pixels.detail",
                                       "vary": "knob", "knob": "amount", "start": 0.2, "end": 1.0, "count": 3,
                                       "size": 256})
    assert len(made.ids) == 3 and all(os.path.isfile(p) for p in made.paths)
    assert made.labels[0].endswith("0.2") and made.labels[-1].endswith("1")
    made = ops.call("alchemy derive", {"picture": "synth:12", "method": "tokens.mix", "vary": "material",
                                       "each": ["synth:7", "synth:3"], "size": 256})
    assert len(made.ids) == 2
    with pytest.raises(ops.Refusal, match="same picture every time"):
        ops.call("alchemy derive", {"picture": "synth:12", "material": ["synth:7"], "method": "pixels.detail",
                                    "vary": "seed"})


def test_steady_spacing_gives_each_passage_the_time_its_change_takes():
    spec = _spec(loop=False, pictures=[{"t": 0, "ref": "a"}, {"t": 1, "ref": "b"}, {"t": 2, "ref": "c"},
                                       {"t": 9, "ref": "d"}])
    f = F.spaced(spec, "even", by="change", changes=([1.0, 3.0, 2.0], 0.0))
    assert [p["t"] for p in f["pictures"]] == [0, 1.5, 6, 9]          # 1 : 3 : 2 of the nine seconds
    g = F.spaced(spec, "faster", by="change", changes=([1.0, 3.0, 2.0], 0.0))
    assert g["pictures"][1]["t"] > 1.5 and g["pictures"][-1]["t"] == 9    # the rate bent, the ends kept
    h = F.spaced(spec, "even")
    assert [p["t"] for p in h["pictures"]] == [0, 3, 6, 9]


def test_steady_spacing_keeps_a_held_film_whole_and_measures_the_real_change(ds):
    _inner()
    spec = _spec(loop=False, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "film:inner"},
                                       {"t": 9, "ref": "synth:3"}, {"t": 10, "ref": "synth:7"}])
    film = F.Film(spec)
    gaps, back = F.key_changes(film)
    assert len(gaps) == 3 and all(g > 0 for g in gaps) and back == 0
    f = F.spaced(spec, "even", by="change")
    t = [p["t"] for p in f["pictures"]]
    assert t[0] == 0 and t[-1] == 10 and t[2] - t[1] > 2.0          # the film plays its 2 s, then the morph
    held = F.normalised(f)["pictures"][1]
    assert F.span(held) == 2.0
    morphs = [t[1] - t[0], t[2] - t[1] - 2.0, t[3] - t[2]]
    assert sum(morphs) == pytest.approx(8.0, abs=1e-2) and min(morphs) > 0
