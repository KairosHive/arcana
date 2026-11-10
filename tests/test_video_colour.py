"""
Colour transfer on video: one table for a clip, or several followed over time.
See VIDEO_ROADMAP.md, 3.3.

Run with:  python -m pytest tests/test_video_colour.py -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import video as V, video_colour as VC  # noqa: E402


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("vc"))
    clip = os.path.join(root, "bars.mp4")
    # A moving test card with a tone: many colours, changing every frame.
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=192x108:r=24:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip], check=True)
    rng = np.random.default_rng(0)

    def tinted(name, colour):
        base = rng.integers(0, 255, (120, 160, 1), dtype=np.uint8).repeat(3, 2).astype(np.float32)
        img = np.clip(base * 0.35 + np.array(colour, np.float32) * 0.65, 0, 255).astype(np.uint8)
        p = os.path.join(root, name)
        Image.fromarray(img).save(p)
        return p

    return {"root": root, "clip": clip, "warm": tinted("warm.png", (230, 120, 40)), "cold": tinted("cold.png", (40, 110, 230))}


def mean_colour(path, t0, t1):
    frames = [f.reshape(-1, 3).mean(0) for _, f in V.frames(path, 8.0, side=96, t0=t0, t1=t1, hw=False)]
    return np.mean(frames, axis=0)


def test_the_pooled_picture_holds_frames_from_across_the_clip(media):
    m = VC.mosaic(media["clip"], n=6)
    assert m.size == (6 * 192, 108)                # a small clip is not enlarged
    assert VC.mosaic(media["warm"]).size == (160, 120)


def test_one_table_moves_the_whole_clip_toward_the_reference(media):
    tables = VC.tables_for(media["clip"], [media["warm"]], "lab")
    assert len(tables) == 1 and tables[0][1].shape == (VC.LUT_N, VC.LUT_N, VC.LUT_N, 3)
    out = VC.render(media["clip"], tables, os.path.join(media["root"], "out", "warm.mp4"))
    assert out["frames"] == 72 and abs(out["seconds"] - 3.0) < 0.1
    before, after = mean_colour(media["clip"], 0, 3), mean_colour(out["path"], 0, 3)
    assert after[0] - after[2] > before[0] - before[2] + 40          # redder and less blue than it was
    info = V.probe(out["path"])
    assert info["has_audio"] and (info["width"], info["height"]) == (192, 108)
    # One table for every frame: the clip jumps no more from frame to frame than its source did.
    assert VC.flicker(out["path"]) <= VC.flicker(media["clip"]) * 1.5 + 0.5


def test_several_references_are_passed_through_in_order(media):
    tables = VC.tables_for(media["clip"], [media["cold"], media["warm"]], "lab")
    assert [round(p, 2) for p, _ in tables] == [0.0, 1.0]
    out = VC.render(media["clip"], tables, os.path.join(media["root"], "out", "cold_to_warm.mp4"))
    start, end = mean_colour(out["path"], 0.0, 0.6), mean_colour(out["path"], 2.4, 3.0)
    assert start[2] > start[0] + 20 and end[0] > end[2] + 20          # blue at the start, red at the end
    middle = mean_colour(out["path"], 1.3, 1.7)
    assert abs(middle[0] - middle[2]) < abs(start[0] - start[2])      # and between the two in the middle


def test_tables_mix_between_keys():
    a, b = VC.identity(), VC.identity() * 0.5
    tables = [(0.0, a), (1.0, b)]
    assert np.allclose(VC.table_at(tables, 0.0), a) and np.allclose(VC.table_at(tables, 1.0), b)
    assert np.allclose(VC.table_at(tables, 0.5), a * 0.75)
    assert np.allclose(VC.table_at([(0.0, a)], 0.7), a)
    frame = np.full((4, 4, 3), 200, np.uint8)
    assert np.abs(VC.apply(frame, a).astype(int) - 200).max() <= 1
    assert np.abs(VC.apply(frame, b).astype(int) - 100).max() <= 2


def test_a_reference_clip_is_followed_at_several_moments(media):
    tables = VC.tables_for(media["clip"], [media["clip"]], "lab", keys=4)
    assert [round(p, 2) for p, _ in tables] == [0.0, 0.33, 0.67, 1.0]
    assert len(VC.tables_for(media["clip"], [media["warm"]], "lab", keys=4)) == 1     # a picture has no time to follow


def test_each_table_is_written_for_an_editor(media):
    tables = VC.tables_for(media["clip"], [media["cold"], media["warm"]], "lab")
    cubes = VC.write_cubes(tables, os.path.join(media["root"], "out", "grade"))
    assert [os.path.basename(c) for c in cubes] == ["grade_01.cube", "grade_02.cube"]
    text = open(cubes[0]).read().splitlines()
    assert text[1] == f"LUT_3D_SIZE {VC.LUT_N}" and len(text) == 4 + VC.LUT_N ** 3
    one = VC.write_cubes(tables[:1], os.path.join(media["root"], "out", "single"))
    assert os.path.basename(one[0]) == "single.cube"


# ─────────────────────────── deflicker ───────────────────────────

def test_gains_remove_the_jumps_and_keep_the_course():
    n = 40
    course = np.linspace(100, 140, n)[:, None].repeat(3, 1)                 # the clip slowly brightens
    jitter = np.where(np.arange(n)[:, None] % 2 == 0, 1.15, 0.87)           # and every frame jumps
    g = VC.gains(course * jitter)
    steadied = course * jitter * g
    assert np.abs(np.diff(steadied, axis=0)).mean() < np.abs(np.diff(course * jitter, axis=0)).mean() / 5
    assert abs(steadied[-5:].mean() - steadied[:5].mean()) > 20             # the slow change is still there
    # With the source, a jump the source really had comes back.
    source = course.copy()
    source[20:] *= 1.5                                                      # a cut to a brighter shot
    kept = (source * jitter) * VC.gains(source * jitter, source)
    assert kept[21, 0] / kept[18, 0] > 1.3


def test_a_flickering_clip_is_steadied(media):
    # The test card with every other frame darkened: a per-frame effect's signature.
    flick = os.path.join(media["root"], "flick.mp4")
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error", "-i", media["clip"],
                    "-vf", "geq=lum='lum(X,Y)*if(mod(N,2),0.72,1)':cb='cb(X,Y)':cr='cr(X,Y)'",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", flick], check=True)
    out = VC.deflicker(flick, os.path.join(media["root"], "out", "steady.mp4"), source=media["clip"])
    assert out["frames"] == 72 and out["before"] > 10
    assert out["after"] < out["before"] / 4
    assert V.probe(out["path"])["has_audio"]

