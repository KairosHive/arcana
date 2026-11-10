"""
Motion descriptors: camera and subject movement read from optical flow.
See VIDEO_ROADMAP.md, 2.4.

The frames are synthetic: one textured picture, moved, scaled, shaken or
dimmed by a known amount, so each number is checked against what was done.

Run with:  python -m pytest tests/test_motion.py -q
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import motion as M  # noqa: E402

FPS = 8.0
W, H = 96, 72


def texture(seed=0, size=400):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size // 8, size // 8), dtype=np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def view(img, cx, cy, scale=1.0, gain=1.0):
    """What a camera centred on (cx, cy) sees; a larger scale is a tighter shot."""
    w, h = W / scale, H / scale
    m = np.float32([[scale, 0, W / 2 - scale * cx], [0, scale, H / 2 - scale * cy]])
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def clip(camera, n=17):
    img = texture()
    return [view(img, *camera(k)) for k in range(n)]


def read(frames):
    return M.as_dict(M.summarise(M.pairs(frames, FPS))), M.describe(M.summarise(M.pairs(frames, FPS)))


def test_a_locked_off_shot_is_still():
    m, line = read(clip(lambda k: (200, 200)))
    assert m["pan"] < M.STILL and abs(m["zoom"]) < M.ZOOMING and m["subject"] < M.MOVING
    assert line == "still camera, nothing moves"


def test_a_pan_has_a_direction_and_a_speed():
    # The camera moves 2 px a frame to the right: 2 * 8 / 96 widths a second.
    m, line = read(clip(lambda k: (150 + 2 * k, 200)))
    assert abs(m["pan_x"] - 2 * FPS / W) < 0.03 and abs(m["pan_y"]) < 0.02
    assert line.startswith("pan right")
    left, line_left = read(clip(lambda k: (250 - 2 * k, 200)))
    assert left["pan_x"] < -0.1 and line_left.startswith("pan left")
    slow, line_slow = read(clip(lambda k: (200 + 0.5 * k, 200)))
    assert line_slow.startswith("slow pan right")


def test_a_tilt_is_told_from_a_pan():
    up, line = read(clip(lambda k: (200, 250 - 2 * k)))
    assert up["pan_y"] > 0.1 and abs(up["pan_x"]) < 0.03
    assert line.startswith("tilt up")
    _, down = read(clip(lambda k: (200, 150 + 2 * k)))
    assert down.startswith("tilt down")


def test_a_zoom_has_a_sign():
    zin, line = read(clip(lambda k: (200, 200, 1.0 + 0.02 * k)))
    assert zin["zoom"] > M.ZOOMING and "zoom in" in line and "pan" not in line
    zout, line_out = read(clip(lambda k: (200, 200, 1.4 - 0.02 * k)))
    assert zout["zoom"] < -M.ZOOMING and "zoom out" in line_out


def test_a_shaking_hand_is_not_a_pan():
    rng = np.random.default_rng(4)
    jitter = rng.normal(0, 1.6, (17, 2))
    m, line = read(clip(lambda k: (200 + jitter[k, 0], 200 + jitter[k, 1])))
    assert m["shake"] > M.SHAKY
    assert "handheld" in line and "pan" not in line and "tilt" not in line


def test_a_subject_moving_in_a_still_frame():
    img = texture()
    frames = []
    for k in range(17):
        f = view(img, 200, 200).copy()
        x = 8 + 3 * k
        f[24:48, x:x + 24] = view(texture(seed=9), 200, 200)[24:48, 0:24]
        frames.append(f)
    m, line = read(frames)
    assert m["pan"] < M.STILL * 2
    assert m["energy"] > M.STILL
    assert "still camera" in line


def test_flicker_is_brightness_not_movement():
    img = texture()
    frames = [view(img, 200, 200, 1.0, 1.0 if k % 2 else 0.55) for k in range(17)]
    m, line = read(frames)
    assert m["flicker"] > M.FLICKERING
    assert "flickering" in line


def test_short_and_flat_input():
    assert M.pairs([], FPS).shape == (0, len(M.PAIR))
    assert M.pairs([np.zeros((H, W), np.uint8)], FPS).shape == (0, len(M.PAIR))
    flat = [np.full((H, W), 128, np.uint8)] * 5
    m, line = read(flat)
    assert m["pan"] == 0 and line == "still camera, nothing moves"
    assert M.summarise(np.zeros((0, len(M.PAIR)))).shape == (len(M.NAMES),)


def test_tags_follow_the_line():
    v = M.summarise(M.pairs(clip(lambda k: (150 + 2 * k, 200, 1.0 + 0.02 * k)), FPS))
    assert {"pan", "zoom"} <= set(M.tags(v))
    still = M.summarise(M.pairs(clip(lambda k: (200, 200)), FPS))
    assert M.tags(still) == ["still"]


# ─────────────────────────── words for movement ───────────────────────────

def test_a_query_is_read_for_the_movements_it_names():
    assert M.read_query("slow pan left over water") == ["pan left", "slow"]
    assert M.read_query("Handheld, zooming in!") == ["zoom in", "handheld"]
    assert M.read_query("a red door") == []
    # "pan left" is not also read as a plain pan, and a word inside another is not a match.
    assert "pan" not in M.read_query("pan left")
    assert M.read_query("japan") == []
    assert set(M.vocabulary()) >= {"still", "pan left", "zoom out", "handheld", "flicker"}


def test_sections_are_scored_against_the_movement_asked_for():
    right = M.summarise(M.pairs(clip(lambda k: (150 + 2 * k, 200)), FPS))
    left = M.summarise(M.pairs(clip(lambda k: (250 - 2 * k, 200)), FPS))
    still = M.summarise(M.pairs(clip(lambda k: (200, 200)), FPS))
    zin = M.summarise(M.pairs(clip(lambda k: (200, 200, 1.0 + 0.02 * k)), FPS))
    V = np.stack([right, left, still, zin])
    for text, best in (("pan right", 0), ("panning left", 1), ("locked off tripod", 2),
                       ("zoom in", 3), ("calm", 2)):
        scores, names = M.query_scores(text, V)
        assert int(np.argmax(scores)) == best, (text, scores, names)
        assert 0.0 <= scores.min() and scores.max() <= 1.0
    scores, names = M.query_scores("pan", V)
    assert scores[0] > 0.5 and scores[1] > 0.5 and scores[2] < 0.2
    assert M.query_scores("a red door", V) == (None, [])
