"""
Sequences: clips cut together with their sound, and proxies for files a browser cannot play.
See VIDEO_ROADMAP.md, 3.1, 3.2 and 2.6.

Run with:  python -m pytest tests/test_sequence.py -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import modality, sequence as Q, video as V  # noqa: E402
from test_video import make_clip  # noqa: E402


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("seq"))
    red = make_clip(os.path.join(root, "red.mp4"), [("0xc02020", 4)], size="160x120")
    blue = make_clip(os.path.join(root, "blue.mp4"), [("0x2040c0", 6)], size="120x160", sound=False)   # upright, silent
    from PIL import Image
    still = os.path.join(root, "green.jpg")
    Image.new("RGB", (300, 200), (30, 170, 60)).save(still)
    return {"root": root, "red": red, "blue": blue, "still": still}


def middle(frame):
    return tuple(int(c) for c in frame[frame.shape[0] // 2, frame.shape[1] // 2])


def test_clips_cut_together_at_their_own_ranges(media):
    spec = {"fps": 24, "size": [320, 180], "clips": [
        {"ref": modality.section_ref(media["red"], 1.0, 3.0)},            # two seconds of red
        {"ref": media["blue"], "in": 2.0, "out": 3.0},                    # one of blue, an upright silent clip
        {"ref": media["still"], "seconds": 1.5}]}                         # a still held
    out = Q.render(spec, os.path.join(media["root"], "out", "cut.mp4"))
    assert out["clips"] == 3 and abs(out["seconds"] - 4.5) < 1e-6
    info = V.probe(out["path"])
    assert (info["width"], info["height"]) == (320, 180) and abs(info["duration"] - 4.5) < 0.2
    assert info["has_audio"] is True
    frames = {round(t, 1): f for t, f in V.frames(out["path"], 2.0, side=320, hw=False)}
    r, g, b = middle(frames[1.0])
    assert r > 150 and b < 90
    r, g, b = middle(frames[2.5])
    assert b > 150 and r < 90                                             # the upright clip fills the frame
    r, g, b = middle(frames[3.5])
    assert g > 130 and r < 90


def test_sound_follows_the_clips_and_silence_fills_the_gaps(media):
    spec = {"fps": 24, "size": [160, 120], "clips": [{"ref": media["blue"], "in": 0, "out": 2},
                                                     {"ref": media["red"], "in": 0, "out": 2}]}
    out = Q.render(spec, os.path.join(media["root"], "out", "sound.mp4"))
    quiet = V.audio(out["path"], 16000, t0=0.2, t1=1.8)
    tone = V.audio(out["path"], 16000, t0=2.2, t1=3.8)
    assert np.abs(quiet).max() < 0.01 and np.abs(tone).max() > 0.02
    silent = Q.render({**spec, "sound": False}, os.path.join(media["root"], "out", "mute.mp4"))
    assert V.probe(silent["path"])["has_audio"] is False


def test_speed_changes_a_clips_length(media):
    spec = {"fps": 24, "size": [160, 120], "clips": [{"ref": media["red"], "in": 0, "out": 4, "speed": 2.0}]}
    out = Q.render(spec, os.path.join(media["root"], "out", "fast.mp4"))
    assert abs(out["seconds"] - 2.0) < 1e-6 and abs(V.probe(out["path"])["duration"] - 2.0) < 0.2
    assert Q._atempo(4.0) == "atempo=2.0,atempo=2.0000" and Q._atempo(0.25).count("atempo") == 2


@pytest.mark.parametrize("clips, words", [
    ([], "at least one clip"),
    ([{"ref": "C:/nowhere/x.mp4"}], "no such file"),
    ([{"ref": "RED", "in": 3, "out": 3}], "out point is not after"),
    ([{"ref": "RED", "speed": 20}], "speed is between"),
])
def test_a_bad_sequence_is_refused_in_its_own_words(media, clips, words):
    clips = [{**c, "ref": media["red"] if c["ref"] == "RED" else c["ref"]} for c in clips]
    with pytest.raises(Q.SequenceError, match=words):
        Q.resolve({"clips": clips})


def test_the_edit_list_names_sources_and_timecodes(media):
    spec = {"name": "rough", "fps": 24, "clips": [
        {"ref": modality.section_ref(media["red"], 1.0, 3.0), "note": "the red part"},
        {"ref": media["blue"], "in": 2.0, "out": 4.0, "speed": 2.0}]}
    text = Q.edl(spec)
    lines = text.splitlines()
    assert lines[0] == "TITLE: rough" and lines[1] == "FCM: NON-DROP FRAME"
    first = next(l for l in lines if l.startswith("001"))
    assert first.split()[-4:] == ["00:00:01:00", "00:00:03:00", "01:00:00:00", "01:00:02:00"]
    second = next(l for l in lines if l.startswith("002"))
    # Two seconds of source at double speed take one second of the timeline.
    assert second.split()[-4:] == ["00:00:02:00", "00:00:04:00", "01:00:02:00", "01:00:03:00"]
    assert any(l.startswith("M2") and "048.0" in l for l in lines)
    assert f"* SOURCE FILE: {media['red']}" in lines and "* COMMENT: the red part" in lines
    assert Q._timecode(3661.5, 24) == "01:01:01:12"


# ─────────────────────────── proxies ───────────────────────────

def test_what_a_browser_can_play(media, tmp):
    assert V.probe(media["red"])["codec"] == "h264" and V.probe(media["red"])["pix_fmt"] == "yuv420p"
    assert V.playable(media["red"])
    # The same picture in a container and a pixel format a browser refuses.
    mkv = os.path.join(tmp, "deep.mkv")
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error", "-i", media["red"], "-c:v", "libx264",
                    "-pix_fmt", "yuv422p10le", "-c:a", "copy", mkv], check=True)
    assert V.probe(mkv)["pix_fmt"].startswith("yuv422p10") and not V.playable(mkv)
    assert not V.playable(os.path.join(tmp, "missing.mp4"))


def test_a_proxy_is_made_once_and_plays(media, tmp):
    mkv = os.path.join(tmp, "deep.mkv")
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error", "-i", media["red"], "-c:v", "libx264",
                    "-pix_fmt", "yuv422p10le", "-c:a", "copy", mkv], check=True)
    cache = os.path.join(tmp, "cache")
    made = V.proxy(mkv, cache)
    assert V.playable(made) and abs(V.probe(made)["duration"] - 4.0) < 0.2 and V.probe(made)["has_audio"]
    stamp = os.path.getmtime(made)
    assert V.proxy(mkv, cache) == made and os.path.getmtime(made) == stamp     # kept, not made again
    assert os.listdir(cache) == [os.path.basename(made)]


def test_the_video_route_serves_a_proxy_for_what_cannot_play(media, tmp, monkeypatch):
    from flask import Flask
    from arcana import media as M
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    mkv = os.path.join(tmp, "deep.mkv")
    subprocess.run([V.ffmpeg_exe(), "-y", "-loglevel", "error", "-i", media["red"], "-c:v", "libx264",
                    "-pix_fmt", "yuv422p10le", "-an", mkv], check=True)
    M.register_dataset_files({0: mkv, 1: media["red"]})
    app = Flask("t")
    M.register(app)
    c = app.test_client()
    got = c.get("/video", query_string={"p": mkv})
    assert got.status_code == 200 and got.mimetype == "video/mp4"
    out = os.path.join(tmp, "served.mp4")
    open(out, "wb").write(got.data)
    assert V.playable(out)
    plain = c.get("/video", query_string={"p": media["red"]})
    assert plain.status_code == 200 and len(plain.data) == os.path.getsize(media["red"])     # the original, untouched
