"""
Patterns: procedural editing, a rule that places the frames.
See VIDEO_ROADMAP.md, 3.5.

Sources here are flat colours, so every frame can be read back by its colour.

Run with:  python -m pytest tests/test_pattern.py -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import pattern as P, video as V  # noqa: E402

RED, GREEN, BLUE, BLACK, WHITE = "#ff0000", "#00ff00", "#0000ff", "#000000", "#ffffff"


def spec(**kw):
    base = {"fps": 30, "seconds": 2, "size": [64, 64], "sources": {"A": RED, "B": GREEN, "I": BLUE}}
    base.update(kw)
    return base


def colours(s):
    """The pattern's frames as letters: r, g, b, k, w, or ? for a mix."""
    names = {(255, 0, 0): "r", (0, 255, 0): "g", (0, 0, 255): "b", (0, 0, 0): "k", (255, 255, 255): "w"}
    out = []
    for f in P.render(s):
        px = {tuple(int(c) for c in f[y, x]) for y in (0, 31, 63) for x in (0, 31, 63)}
        out.append(names.get(next(iter(px)), "?") if len(px) == 1 else "?")
    return "".join(out)


# ─────────────────────────── pulses ───────────────────────────

def test_pulses():
    hit = lambda p, n=12: "".join("x" if b else "." for b in P.pulse_frames(p, n))  # noqa: E731
    assert hit({"every": 3}) == "x..x..x..x.."
    assert hit({"every": 4, "offset": 1}) == ".x...x...x.."
    assert hit({"on": 2, "off": 3}, 10) == "xx...xx..."
    assert hit({"euclid": [3, 8]}, 8) == "x..x..x."
    assert hit({"euclid": [5, 8]}, 8).count("x") == 5
    assert hit({"hits": "x..x.x"}, 12) == "x..x.xx..x.x"
    assert hit({"hits": "x.", "step": 3}, 12) == "x.....x....."
    # The same seed gives the same chance; another seed gives another.
    a = P.pulse_frames({"chance": 0.5}, 200, seed=1)
    assert (a == P.pulse_frames({"chance": 0.5}, 200, seed=1)).all() and (a != P.pulse_frames({"chance": 0.5}, 200, seed=2)).any()
    assert 60 < a.sum() < 140


def test_a_fractional_period_drifts():
    whole = np.flatnonzero(P.pulse_frames({"every": 6}, 600))
    drift = np.flatnonzero(P.pulse_frames({"every": 6.1}, 600))
    assert list(whole[:3]) == [0, 6, 12]
    # They start together and are a whole beat apart after sixty beats.
    assert drift[0] == whole[0] and drift[1] - whole[1] == 1 and drift[59] - whole[59] == 6
    assert set(np.diff(drift)) == {6, 7}


# ─────────────────────────── the frames ───────────────────────────

def test_interlacing_in_time():
    assert colours(spec(seconds=0.2, voices=[{"pulse": {"every": 1}, "cut": ["A", "B"]}])) == "rgrgrg"
    assert colours(spec(seconds=0.4, voices=[{"pulse": {"hits": "x..x", "step": 1}, "cut": ["A", "B"]}])) == "rrrgrrrgrrrg"
    three = colours(spec(seconds=0.3, voices=[{"pulse": {"every": 1}, "cut": ["A", "B", "I"]}]))
    assert three == "rgbrgbrgb"


def test_polyrhythm_of_three_voices():
    s = spec(seconds=2, sources={"A": RED, "B": GREEN, "I": BLUE, "K": BLACK},
             voices=[{"pulse": {"every": 3}, "flash": "B"}, {"pulse": {"every": 4}, "flash": "I"},
                     {"pulse": {"every": 5}, "flash": "K"}])
    got = colours(s)
    assert got[:13] == "krrgbkgrbgkrb"                 # later voices win where they meet
    # The three meet again every 60 frames, and it renders the same twice.
    assert got[0] == got[60 - 60] == "k" and got == colours(s)
    plan = P.plan(s)
    together = np.flatnonzero(plan["fires"][0] & plan["fires"][1] & plan["fires"][2])
    assert list(together) == [0]


def test_a_flash_lasts_its_frames_and_a_euclidean_rhythm_places_it():
    got = colours(spec(seconds=0.4, voices=[{"pulse": {"every": 6}, "flash": "I", "frames": 2}]))
    assert got == "bbrrrrbbrrrr"
    eu = colours(spec(seconds=32 / 30, voices=[{"pulse": {"euclid": [3, 8], "step": 4}, "flash": "I"}]))
    assert [i for i, c in enumerate(eu) if c == "b"] == [0, 12, 24]


def test_interlacing_in_space():
    frames = list(P.render(spec(seconds=0.1, space={"kind": "lines", "size": 2, "shift": 1, "with": "B"})))
    first = ["r" if frames[0][y, 5, 0] == 255 else "g" for y in range(8)]
    assert first == ["r", "r", "g", "g", "r", "r", "g", "g"]
    second = ["r" if frames[1][y, 5, 0] == 255 else "g" for y in range(8)]
    assert second == ["r", "g", "g", "r", "r", "g", "g", "r"]        # the lines moved one row
    cols = list(P.render(spec(seconds=0.1, space={"kind": "columns", "size": 4, "with": "B"})))[0]
    assert cols[10, 0, 0] == 255 and cols[10, 4, 1] == 255
    check = list(P.render(spec(seconds=0.1, space={"kind": "checker", "size": 8, "with": "B"})))[0]
    assert check[0, 0, 0] == 255 and check[0, 8, 1] == 255 and check[8, 8, 0] == 255


def test_a_complementary_pair_cancels_at_speed_and_shows_on_pause():
    # The hidden picture: white on the left, black on the right.
    half = np.zeros((1, 64, 64, 3), np.uint8)
    half[0, :, :32] = 255
    s = spec(seconds=0.2, sources={"A": "#808080", "I": WHITE}, voices=[{"pulse": {"every": 2}, "pair": "I", "amount": 0.2}])
    srcs = {"A": P.Source("#808080", (64, 64), 30, 6), "I": P.Source("#000000", (64, 64), 30, 6)}
    srcs["I"].frames = half
    frames = [f.astype(np.float32) for f in P.render(s, sources=srcs)]
    plain = 128.0
    # There on any single frame: its left side lighter on one, darker on the next.
    assert frames[0][:, :32].mean() > plain + 20 and frames[0][:, 32:].mean() < plain - 20
    assert frames[1][:, :32].mean() < plain - 20 and frames[1][:, 32:].mean() > plain + 20
    # Gone in the average of the two, everywhere.
    assert np.abs((frames[0] + frames[1]) / 2 - plain).max() < 1.0
    # And no frame is brighter or darker overall, so the pair does not flicker.
    assert abs(frames[0].mean() - plain) < 1.0 and abs(frames[1].mean() - plain) < 1.0


def test_source_time(tmp):
    from test_video import make_clip
    clip = make_clip(os.path.join(tmp, "count.mp4"), [("0x000000", 0.2), ("0xffffff", 0.2), ("0xff0000", 0.2)],
                     size="64x64", rate=30, sound=False)
    base = {"fps": 30, "seconds": 0.6, "size": [64, 64], "sources": {"A": GREEN, "C": clip},
            "voices": [{"pulse": {"every": 1}, "cut": ["A", "C"]}]}
    kind = lambda f: "g" if f[32, 32, 1] > 200 and f[32, 32, 0] < 60 else "k" if f[32, 32].max() < 60 else \
        "w" if f[32, 32].min() > 200 else "r"                                   # noqa: E731
    running = "".join(kind(f) for f in P.render(base))
    stepping = "".join(kind(f) for f in P.render({**base, "time": {"C": "stepping"}}))
    frozen = "".join(kind(f) for f in P.render({**base, "time": {"C": "freeze"}}))
    # Shown on every other frame. Running, the clip has moved on underneath; stepping, it waited.
    assert running[1::2] == "kkkwwwrrr"
    assert stepping[1::2] == "kkkkkkwww"
    assert frozen[1::2] == "kkkkkkkkk"


# ─────────────────────────── saying no ───────────────────────────

@pytest.mark.parametrize("bad, words", [
    ({"sources": {}}, "at least one source"),
    ({"voices": [{"pulse": {"every": 2}, "cut": ["A", "Z"]}]}, "names the source 'Z'"),
    ({"voices": [{"pulse": {"every": 0}, "flash": "B"}]}, "above 0"),
    ({"voices": [{"pulse": {"sometimes": 1}, "flash": "B"}]}, "pulse is one of"),
    ({"voices": [{"pulse": {"every": 2}}]}, "does one of"),
    ({"space": {"kind": "spiral", "with": "B"}}, "lines, columns, checker or blocks"),
    ({"seconds": 0}, "seconds"),
])
def test_a_bad_pattern_is_refused_in_its_own_words(bad, words):
    with pytest.raises(P.PatternError, match=words):
        P.plan(spec(**bad))


# ─────────────────────────── the flash meter ───────────────────────────

def test_the_flash_meter_counts_opposing_changes():
    steady = P.flash_report([0.5] * 90, 30)
    assert steady["max_per_second"] == 0 and not steady["over"]
    # Black and white alternating every frame at 30 fps: fifteen flashes a second.
    strobe = P.flash_report([0.0, 1.0] * 45, 30)
    assert strobe["max_per_second"] >= 14 and strobe["over"]
    # One flash a second is under the limit of three.
    slow = P.flash_report(([0.1] * 29 + [0.9]) * 3, 30)
    assert slow["max_per_second"] <= 2 and not slow["over"]
    # A change of less than 10% is not a flash, and neither is one between two bright states.
    assert P.flash_report([0.50, 0.55] * 45, 30)["flashes"] == 0
    assert P.flash_report([0.85, 0.97] * 45, 30)["flashes"] == 0
    assert abs(P.luminance_of(np.full((4, 4, 3), 255, np.uint8)) - 1.0) < 1e-4


def test_writing_a_pattern(tmp):
    s = spec(seconds=1, sources={"A": BLACK, "B": WHITE}, voices=[{"pulse": {"every": 1}, "cut": ["A", "B"]}])
    out = P.write(s, os.path.join(tmp, "out", "strobe.mp4"))
    assert out["frames"] == 30 and out["flash"]["over"]
    info = V.probe(out["path"])
    assert (info["width"], info["height"]) == (64, 64) and abs(info["duration"] - 1.0) < 0.1
    back = [f[32, 32].mean() for _, f in V.frames(out["path"], 30, side=64, hw=False)]
    assert len(back) == 30 and back[0] < 40 and back[1] > 215 and back[2] < 40        # every frame survived the encode
    for name, preset in P.PRESETS.items():
        P.plan(spec(**preset))



# ─────────────────────────── curves ───────────────────────────

def test_waves():
    s = P.wave({"period": 8}, 8)
    assert abs(s[0]) < 1e-6 and abs(s[2] - 1) < 1e-6 and abs(s[6] + 1) < 1e-6
    assert list(P.wave({"period": 4, "shape": "square"}, 8)) == [1, 1, -1, -1, 1, 1, -1, -1]
    tri = P.wave({"period": 8, "shape": "triangle"}, 9)
    assert tri[0] == -1 and tri[4] == 1 and tri[8] == -1
    assert list(P.wave({"period": 4, "shape": "saw"}, 4)) == [-1, -0.5, 0, 0.5]
    # A phase of a quarter starts the sine at its top.
    assert abs(P.wave({"period": 8, "phase": 0.25}, 1)[0] - 1) < 1e-6
    with pytest.raises(P.PatternError, match="shape"):
        P.wave({"period": 8, "shape": "wobble"}, 4)


def test_a_curve_moves_colour_at_its_own_frequency():
    # Red, with the hue swinging: a quarter period in, the wheel has turned half a turn, to cyan.
    frames = list(P.render(spec(seconds=0.3, sources={"A": RED}, curves=[{"wave": {"period": 8}, "hue": 1.0}])))
    assert tuple(frames[0][0, 0]) == (255, 0, 0)
    r, g, b = (int(c) for c in frames[2][0, 0])
    assert r < 40 and g > 200 and b > 200
    assert tuple(frames[8][0, 0]) == (255, 0, 0)            # back where it began, eight frames later


def test_invert_blend_and_shift():
    inv = list(P.render(spec(seconds=0.2, sources={"A": BLACK},
                             curves=[{"wave": {"period": 4, "shape": "square"}, "invert": 1.0}])))
    assert [int(f[0, 0, 0]) for f in inv[:4]] == [255, 255, 0, 0]
    mix = list(P.render(spec(seconds=0.2, sources={"A": BLACK, "B": WHITE},
                             curves=[{"wave": {"period": 8, "shape": "triangle"}, "blend": "B", "amount": 1.0}])))
    levels = [int(f[0, 0, 0]) for f in mix[:5]]
    assert levels[0] == 0 and levels[4] == 255 and levels == sorted(levels)
    # A red bar on black: shifted, the red moves and nothing else does.
    import numpy as np
    bar = np.zeros((1, 64, 64, 3), np.uint8)
    bar[0, :, 30:34] = (255, 0, 0)
    src = P.Source("#000000", (64, 64), 30, 4)
    src.frames = bar
    out = list(P.render(spec(seconds=0.1, sources={"A": "#000000"},
                             curves=[{"wave": {"period": 4, "phase": 0.25}, "shift": 5}]), sources={"A": src}))[0]
    assert out[10, 37, 0] == 255 and out[10, 31, 0] == 0


def test_a_curve_and_a_rhythm_run_together():
    s = spec(seconds=1, voices=[{"pulse": {"every": 2}, "cut": ["A", "B"]}],
             curves=[{"wave": {"period": 30, "shape": "square"}, "invert": 1.0}])
    got = colours(s)
    # The cut every two frames goes on underneath; the curve turns everything negative for half a second.
    assert got[15:19] == "grrg" and got[0] == "?" and "r" not in got[:15] and "g" not in got[:15]
    with pytest.raises(P.PatternError, match="moves one of"):
        P.plan(spec(curves=[{"wave": {"period": 9}}]))
    with pytest.raises(P.PatternError, match="names the source"):
        P.plan(spec(curves=[{"wave": {"period": 9}, "blend": "Z"}]))
