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
