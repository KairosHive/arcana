"""
Reading video: probe, frames at a rate, the sound track, and a real file cut
into its sections. See VIDEO_ROADMAP.md, 2.3.

The clips are made by the same ffmpeg Arcana reads with, so the tests need no
media in the repository.

Run with:  python -m pytest tests/test_video.py -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import sections as S  # noqa: E402
from arcana import video as V  # noqa: E402


def make_clip(path, parts, size="160x120", rate=24, sound=True):
    """A clip of flat-colour parts, [(colour, seconds), ...], with a tone under it."""
    cmd = [V.ffmpeg_exe(), "-y", "-loglevel", "error"]
    for colour, seconds in parts:
        cmd += ["-f", "lavfi", "-i", f"color=c={colour}:s={size}:r={rate}:d={seconds}"]
    total = sum(s for _, s in parts)
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={total}"]
    chain = "".join(f"[{i}:v]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    cmd += ["-filter_complex", chain, "-map", "[v]"]
    if sound:
        cmd += ["-map", f"{len(parts)}:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Ten seconds of blue, fourteen of grey, eight of green."""
    path = str(tmp_path_factory.mktemp("video") / "three_parts.mp4")
    return make_clip(path, [("0x1b4f8f", 10), ("0x9a9a9a", 14), ("0x2f7d3a", 8)])


def test_probe(clip):
    info = V.probe(clip)
    assert abs(info["duration"] - 32.0) < 0.3
    assert (info["width"], info["height"]) == (160, 120)
    assert abs(info["fps"] - 24) < 0.1
    assert info["has_audio"] is True


def test_probe_refuses_what_is_not_video(tmp):
    not_video = os.path.join(tmp, "notes.txt")
    with open(not_video, "w") as f:
        f.write("no frames here")
    with pytest.raises(V.VideoError):
        V.probe(not_video)
    with pytest.raises(V.VideoError):
        V.probe(os.path.join(tmp, "missing.mp4"))


def test_frames_come_at_the_asked_rate_and_size(clip):
    got = list(V.frames(clip, 2.0, side=80))
    assert 62 <= len(got) <= 65
    t, f = got[0]
    assert t == 0.0 and f.shape == (60, 80, 3) and f.dtype == np.uint8
    assert [round(t, 2) for t, _ in got[:3]] == [0.0, 0.5, 1.0]
    # Blue first, green last.
    assert got[1][1][..., 2].mean() > got[1][1][..., 0].mean() + 40
    assert got[-2][1][..., 1].mean() > got[-2][1][..., 2].mean() + 40


def test_a_range_is_read_without_the_rest(clip):
    got = list(V.frames(clip, 4.0, side=80, t0=12.0, t1=14.0))
    assert 7 <= len(got) <= 9
    assert got[0][0] == 12.0
    grey = got[2][1].astype(int)
    assert abs(grey[..., 0].mean() - grey[..., 2].mean()) < 12


def test_frames_between_gives_exactly_n(clip):
    got = V.frames_between(clip, 10.0, 24.0, 8, side=64)
    assert len(got) == 8 and got[0].shape == (48, 64, 3)
    assert V.frame_at(clip, 30.0, side=64).shape == (48, 64, 3)


def test_the_sound_track(clip, tmp):
    wav = V.audio(clip, 16000, t0=2.0, t1=4.0)
    assert abs(len(wav) - 32000) < 1600
    assert wav.dtype == np.float32 and 0.01 < np.abs(wav).max() <= 1.0
    silent = make_clip(os.path.join(tmp, "silent.mp4"), [("red", 1)], sound=False)
    assert V.probe(silent)["has_audio"] is False
    assert len(V.audio(silent)) == 0


def test_a_real_file_of_three_parts_comes_out_as_three_sections(clip):
    info = V.probe(clip)
    sigs = np.stack([V.colour_signature(f) for _, f in V.ticks(clip, info=info)])
    cuts = V.detect_cuts(sigs)
    assert cuts == [19, 47]                       # between ticks at 10 s and at 24 s
    r = S.section_ticks(sigs, cuts=cuts, duration=info["duration"])
    coarse = r["levels"]["coarse"]
    assert [(round(s.t0), round(s.t1)) for s in coarse] == [(0, 10), (10, 24), (24, 32)]
    # Each section then hands an encoder its own frames.
    times = S.frame_times(coarse[1])
    assert len(times) == 32 and 10.0 < times[0] < times[-1] < 24.0


def test_a_slow_fade_is_not_a_cut():
    a, b = np.zeros(8, np.float32), np.zeros(8, np.float32)
    a[0], b[1] = 1.0, 1.0
    fade = [S.unit_rows(((1 - w) * a + w * b)[None])[0] for w in np.linspace(0, 1, 12)]
    assert V.detect_cuts(np.stack([a] * 6 + fade + [b] * 6)) == []
    assert V.detect_cuts(np.stack([a] * 6 + [b] * 6)) == [5]
