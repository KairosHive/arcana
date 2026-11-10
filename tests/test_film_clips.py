"""
Clips on a film's picture lane, and the film's sound. See VIDEO_ROADMAP.md, 3.1.

A clip key holds a video by its file, a section of it or a handle, with in, out and speed, the shape
a kept film has on the lane. The film morphs into the clip's frame at in, plays it, and morphs on.

Run with:  python -m pytest tests/test_film_clips.py -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import modality, video as V  # noqa: E402
from arcana.alchemy import film as F  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402
from test_film import PatchSpace  # noqa: E402
from test_video import make_clip  # noqa: E402


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("filmclips"))
    # Two seconds of red, then two of blue, with a tone all along.
    return make_clip(os.path.join(root, "redblue.mp4"), [("0xc02020", 2), ("0x2040c0", 2)], size="160x120")


@pytest.fixture(scope="module")
def moving(tmp_path_factory):
    """A patterned square crossing a grey frame, 100 px a second, for two seconds."""
    import subprocess

    path = str(tmp_path_factory.mktemp("filmmoving") / "moving.mp4")
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=0x606060:s=320x240:r=24:d=2",
                    "-f", "lavfi", "-i", "testsrc=s=64x64:r=24:d=2",
                    "-filter_complex", "[0:v][1:v]overlay=x='20+100*t':y=88[v]", "-map", "[v]",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], check=True)
    return path


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    space = PatchSpace()
    monkeypatch.setattr(F.spaces, "space", lambda name: space)
    F._ready.clear()
    return out


def _spec(clip, **more):
    spec = {"name": "clips", "fps": 6, "size": 256, "aspect": "4:3", "even": False, "loop": False,
            "pictures": [{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": clip, "in": 0.5, "out": 3.5},
                         {"t": 7, "ref": "synth:3"}]}
    spec.update(more)
    return spec


def middle(frame):
    return tuple(float(c) for c in frame[frame.shape[0] // 2, frame.shape[1] // 2])


def test_a_clip_key_takes_its_range_from_the_section_or_the_file(ds, clip):
    assert F.is_clip(clip) and F.is_clip(modality.section_ref(clip, 1, 2)) and F.is_clip("walks@video:3")
    assert not F.is_clip("synth:12") and not F.is_clip("film:rites") and F.plays("film:rites")
    whole = F.normalised(_spec(clip, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": clip}]))
    key = whole["pictures"][1]
    assert key["in"] == 0.0 and abs(key["out"] - 4.0) < 0.2 and key["speed"] == 1.0
    assert abs(whole["length"] - (2 + key["out"])) < 1e-6          # the film ends when the clip does
    part = F.normalised(_spec(clip, pictures=[{"t": 0, "ref": "synth:12"},
                                              {"t": 2, "ref": modality.section_ref(clip, 1.0, 3.0), "speed": 2}]))
    key = part["pictures"][1]
    assert (key["in"], key["out"]) == (1.0, 3.0) and F.span(key) == 1.0
    stops = F.points(part)
    assert [(q["at"], q["plays"]) for q in stops] == [(None, False), (1.0, True), (3.0, False)]


def test_a_clip_is_refused_in_its_own_words(ds, clip):
    with pytest.raises(ValueError, match="redblue.mp4 runs 0 to"):
        F.normalised(_spec(clip, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": clip, "in": 3, "out": 9}]))
    with pytest.raises(ValueError, match="The clip at 2.00 s plays until 5.00 s"):
        F.normalised(_spec(clip, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": clip, "in": 0.5, "out": 3.5},
                                           {"t": 4, "ref": "synth:3"}]))
    with pytest.raises(ValueError, match="0.1 to 8 times"):
        F.normalised(_spec(clip, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 2, "ref": clip, "speed": 20}]))


def test_the_film_plays_the_clips_own_frames_and_lands_on_them(ds, clip):
    film = F.Film(_spec(clip))
    r, g, b = middle(film.morph(2.5))                  # 1.0 s into the file: red
    assert r > 0.6 and b < 0.3
    r, g, b = middle(film.morph(4.5))                  # 3.0 s into the file: blue
    assert b > 0.6 and r < 0.3
    source = F.Clip(clip, film.W, film.H)
    assert np.allclose(film.morph(3.0), source.frame(1.5), atol=1e-5)
    # The morph arrives on the clip's frame at in and leaves from its frame at out.
    assert np.allclose(film.morph(2.0 - 1e-6), source.frame(0.5), atol=2e-3)
    assert np.allclose(film.morph(5.0 + 1e-6), source.frame(3.5), atol=2e-3)
    # A look applies to the clip's frames as it does to a morph.
    grey = F.Film(_spec(clip, lanes={"finish.saturation": [{"t": 0, "v": -100}]})).frame(2.5)
    r, g, b = middle(grey)
    assert abs(r - b) < 0.03


def test_frames_come_in_order_from_one_decoder_and_a_jump_starts_again(ds, clip):
    source = F.Clip(clip, 160, 120)
    first = source.frame(0.2)
    running = source._gen
    assert middle(source.frame(0.5))[0] > 0.6 and source._gen is running        # read on, same decoder
    assert middle(source.frame(3.0))[2] > 0.6
    assert middle(source.frame(0.2)) == middle(first) and source._gen is not running   # a jump back
    assert source.frame(99.0) is not None                                       # past the end: the last frame


def test_a_rendered_film_carries_the_clips_sound(ds, clip, tmp):
    out = F.render(_spec(clip), os.path.join(tmp, "film.mp4"))
    assert out["sounds"] == 1
    info = V.probe(out["path"])
    assert info["has_audio"] and abs(info["duration"] - 7.0) < 0.3
    quiet = V.audio(out["path"], 16000, t0=0.3, t1=1.7)
    tone = V.audio(out["path"], 16000, t0=2.5, t1=4.5)
    after = V.audio(out["path"], 16000, t0=5.5, t1=6.8)
    assert np.abs(quiet).max() < 0.01 and np.abs(tone).max() > 0.02 and np.abs(after).max() < 0.01
    silent = F.render(_spec(clip, sound=False), os.path.join(tmp, "silent.mp4"))
    assert silent["sounds"] == 0 and V.probe(silent["path"])["has_audio"] is False
    muted = _spec(clip)
    muted["pictures"][1]["mute"] = True
    assert F.sounds(muted) == []


def test_the_audio_lane_places_a_sound_with_gain_and_fades(ds, clip, tmp):
    spec = _spec(clip, audio=[{"t": 0.5, "ref": clip, "in": 0, "out": 1.0, "gain": -6, "fade": [0.2, 0.2]}])
    spec["pictures"][1]["mute"] = True
    plan = F.sounds(spec)
    assert len(plan) == 1 and plan[0]["at"] == 0.5 and plan[0]["length"] == 1.0 and plan[0]["gain"] == -6
    out = F.render(spec, os.path.join(tmp, "lane.mp4"))
    assert out["sounds"] == 1
    before = V.audio(out["path"], 16000, t0=0.0, t1=0.4)
    during = V.audio(out["path"], 16000, t0=0.8, t1=1.2)
    later = V.audio(out["path"], 16000, t0=2.5, t1=4.5)
    assert np.abs(before).max() < 0.01 and np.abs(during).max() > 0.01 and np.abs(later).max() < 0.01
    with pytest.raises(ValueError, match="gain is -60 to 24"):
        F.normalised(_spec(clip, audio=[{"t": 0, "ref": clip, "gain": 90}]))
    with pytest.raises(ValueError, match="holds no sound"):
        F.sounds(_spec(clip, audio=[{"t": 0, "ref": "synth:12"}]))


def test_a_models_mask_follows_what_moves_in_a_clip(ds, moving, monkeypatch):
    from arcana.alchemy import vision as vis

    def found(image):
        """The stand-in model: whatever is not the grey ground."""
        a = np.asarray(image, np.float32) / 255.0
        return (np.abs(a - a[2, 2]).max(-1) > 0.12).astype(np.float32)

    asked = []

    class Stub:
        def ask(self, key, picture, kind, arg=""):
            asked.append(key)
            return found(picture())

    monkeypatch.setattr(vis, "vision", lambda: Stub())
    film = F.Film(_spec(moving, pictures=[{"t": 0, "ref": "synth:12"}, {"t": 1, "ref": moving, "in": 0, "out": 2},
                                          {"t": 5, "ref": "synth:3"}]))

    def overlap(a, b):
        a, b = a > 0.5, b > 0.5
        return (a & b).sum() / max((a | b).sum(), 1)

    for t in (1.25, 1.7, 2.3):                          # between two readings, a quarter second from each or near one
        truth = found(F.to_image(film.morph(t)))
        tracked = film.asker(t)("subject")
        assert overlap(tracked, truth) > 0.7, t
        k, s = F.passage(film.spec, t)
        faded = (1 - s) * film._map(k, "subject", "") + s * film._map(k + 1, "subject", "")
        assert overlap(faded, truth) < 0.5 < overlap(tracked, truth)    # the two ends cross-faded lose the square
    # The model ran on the clip's frames every half second, once each, and on no frame in between.
    readings = [a for a in asked if a.startswith(moving + "@")]
    assert 0 < len(readings) <= 5 and len(set(readings)) == len(readings)


def test_a_look_runs_on_key_frames_of_a_clip_and_is_carried_between_them(ds, moving, monkeypatch):
    from arcana.alchemy import pipeline

    ran = []

    def negative(c, key, spec, asker=None):
        """The stand-in look: the picture inverted. It records the frames it ran on."""
        if spec.get("method") != "none":
            ran.append(key)
        return (1.0 - np.asarray(c, np.float32)) if spec.get("method") != "none" else np.asarray(c, np.float32), None

    monkeypatch.setattr(pipeline, "step_on", negative)
    pictures = [{"t": 0, "ref": "synth:12"}, {"t": 1, "ref": moving, "in": 0, "out": 2}, {"t": 5, "ref": "synth:3"}]
    look = [{"t": 0, "settings": {"method": "pixels.negative"}}]
    film = F.Film(_spec(moving, pictures=pictures, looks=look, temporal="keys"))
    assert film.by_keys(1, {"method": "pixels.negative"}) and not film.by_keys(0, {"method": "pixels.negative"})
    frames = [film.frame(1 + i / 12) for i in range(1, 24)]            # the clip's two seconds, 12 a second
    on_clip = [k for k in ran if ":key:" in k]
    assert len(on_clip) == 5 and len(ran) == 5                          # every half second, and on no frame between
    # Each frame is the look of its own picture: the carried change lands on the square where it is now.
    for i in (3, 8, 15, 20):
        want = 1.0 - film.morph(1 + i / 12)
        assert np.abs(frames[i - 1] - want).mean() < 0.02, i
    # Per frame, the same film runs the look on every frame of the clip.
    ran.clear()
    each = F.Film(_spec(moving, pictures=pictures, looks=look, temporal="frame"))
    assert not each.by_keys(1, {"method": "pixels.negative"})
    assert np.abs(each.frame(1.5) - (1.0 - each.morph(1.5))).mean() < 1e-6 and len(ran) == 1
    # Auto carries the generative methods and runs the others on each frame.
    auto = F.Film(_spec(moving, pictures=pictures))
    assert auto.by_keys(1, {"method": "generative.controlnet"}) and not auto.by_keys(1, {"method": "pixels.blend"})
    assert not auto.by_keys(1, {"method": "none"})
    with pytest.raises(ValueError, match="temporal is one of"):
        F.normalised(_spec(moving, pictures=pictures, temporal="often"))


def test_the_timeline_lists_a_clip_with_its_length_and_lays_one_down(ds, clip):
    from arcana import ui_film as U
    m = U.meta(_spec(clip))
    got = m["pictures"][1]
    assert got["clip"] and got["dur"] == 3.0 and got["name"] == "redblue.mp4"
    assert modality.split_ref(got["path"]) == (clip, 0.5, 3.5)
    s, t = U.add_picture({"pictures": [{"t": 0, "ref": "synth:12"}, {"t": 3, "ref": "synth:7"}]}, clip, 2.0)
    key = [p for p in s["pictures"] if p["ref"] == clip][0]
    assert key["t"] > 3 and key["in"] == 0.0 and abs(key["out"] - 4.0) < 0.2      # after everything: it would overlap
