# video.py -- reading a video: what it is, its frames at any rate, its sound
#
# Until this module Arcana only wrote video (the film renderer pipes frames to
# ffmpeg). Indexing a video dataset needs the other direction, and needs it in
# three shapes:
#
#   probe(path)                 duration, frame rate, size, whether it has sound
#   frames(path, fps, ...)      decoded frames at a chosen rate, as RGB arrays
#   audio(path, sr, ...)        the sound track, mono, as a float array
#
# All three go through the ffmpeg binary that imageio-ffmpeg ships, which the
# film renderer already depends on, so reading video adds no dependency and
# works in the packaged build. Frames come scaled by ffmpeg before they reach
# Python: a tick for the sectioning is a 224 px frame, and decoding 4K to throw
# most of it away in numpy would cost more than the model that reads it.
#
# Nothing here writes beside the original. See VIDEO_ROADMAP.md, 2.3 and 2.7.

from __future__ import annotations

import os
import re
import subprocess

import numpy as np

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg_exe() -> str:
    """The ffmpeg imageio-ffmpeg ships."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoError(ValueError):
    """A file ffmpeg could not read as video."""


# ─────────────────────────── probe ───────────────────────────

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO = re.compile(r"Stream #\S+.*?Video:.*?(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR = re.compile(r"(\d+(?:\.\d+)?)\s*tbr")
_ROTATE = re.compile(r"rotat\w*\s*(?:of|:)\s*(-?\d+)")


def probe(path: str) -> dict:
    """
    What a video file is: {"duration", "fps", "width", "height", "has_audio"}.

    imageio-ffmpeg ships ffmpeg without ffprobe, so this reads the header ffmpeg
    prints for `-i`. Width and height are as displayed: a phone clip stored
    sideways with a rotation flag is reported upright, which is how ffmpeg
    hands its frames over.
    """
    if not os.path.isfile(path):
        raise VideoError(f"No such file: {path}")
    done = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True,
                          creationflags=_NO_WINDOW)
    text = done.stderr.decode("utf-8", "replace")
    video = _VIDEO.search(text)
    if not video:
        raise VideoError(f"{os.path.basename(path)} holds no video ffmpeg can read.")
    line = text[video.start():text.find("\n", video.start())]
    width, height = int(video.group(1)), int(video.group(2))
    rot = _ROTATE.search(text)
    if rot and abs(int(rot.group(1))) % 180 == 90:
        width, height = height, width
    fps = _FPS.search(line) or _TBR.search(line)
    dur = _DURATION.search(text)
    duration = (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))) if dur else 0.0
    codec = re.search(r"Video:\s*(\w+)", line)
    pix = re.search(r"\b(yuvj?4\d\dp\d*\w*|gbrp\w*|rgb\w*|bgr\w*|gray\w*|nv12|p010\w*)\b", line)
    return {"duration": duration, "fps": float(fps.group(1)) if fps else 0.0,
            "width": width, "height": height,
            "has_audio": bool(re.search(r"Stream #\S+.*?Audio:", text)),
            "codec": codec.group(1).lower() if codec else "", "pix_fmt": pix.group(1).lower() if pix else ""}


# ─────────────────────────── playing in a browser ───────────────────────────
# A browser plays H.264, VP8, VP9 and AV1 at 8 bits in 4:2:0, in mp4, mov or
# webm. A camera's 10-bit 4:2:2 file, HEVC, ProRes, or anything in mkv or avi
# shows a black player. For those, a proxy is made once and kept in the cache:
# H.264 at up to 720p with AAC sound, small and playable everywhere. The
# original is only read.

BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_PIX = {"yuv420p", "yuvj420p"}
PROXY_SIDE = 1280


def playable(path: str, info: dict | None = None) -> bool:
    """Whether a browser can play this file as it is."""
    try:
        info = info or probe(path)
    except VideoError:
        return False
    return (os.path.splitext(path)[1].lower() in BROWSER_CONTAINERS and info.get("codec") in BROWSER_CODECS
            and (not info.get("pix_fmt") or info["pix_fmt"] in BROWSER_PIX))


def proxy_path(path: str, cache_dir: str) -> str:
    """Where a file's proxy lives: named by the file's path, size and time, so an edited file gets a new one."""
    import hashlib
    st = os.stat(path)
    key = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return os.path.join(cache_dir, hashlib.sha1(key).hexdigest()[:20] + ".mp4")


_PROXY_LOCKS: dict = {}


def proxy(path: str, cache_dir: str) -> str:
    """The proxy of a file, made now if it is not in the cache. Returns its path."""
    import tempfile
    import threading
    out = proxy_path(path, cache_dir)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    lock = _PROXY_LOCKS.setdefault(out, threading.Lock())
    with lock:                                  # two cards asking for one file make it once
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        os.makedirs(cache_dir, exist_ok=True)
        info = probe(path)
        w, h = _fit(info["width"], info["height"], PROXY_SIDE)
        part = out + ".part.mp4"
        errors = tempfile.TemporaryFile()
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
               "-vf", f"scale={w}:{h}:flags=bilinear", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-g", "30", "-c:a", "aac", "-b:a", "128k", "-ac", "2",
               "-movflags", "+faststart", part]
        try:
            done = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=errors, creationflags=_NO_WINDOW)
            if done.returncode or not os.path.exists(part):
                errors.seek(0)
                raise VideoError("Could not make a playable copy of " + os.path.basename(path) + ": "
                                 + errors.read()[-300:].decode("utf-8", "replace").strip())
            os.replace(part, out)
        finally:
            errors.close()
            if os.path.exists(part):
                os.remove(part)
    return out


# ─────────────────────────── frames ───────────────────────────

def _fit(width: int, height: int, side: int | None) -> tuple[int, int]:
    """The frame size with its longest side at `side`, both sides even, never enlarged."""
    if not side or max(width, height) <= side:
        w, h = width, height
    elif width >= height:
        w, h = side, round(height * side / width)
    else:
        w, h = round(width * side / height), side
    return max(2, w - w % 2), max(2, h - h % 2)


_HW: dict = {}


def hardware_decode() -> bool:
    """
    Whether to ask ffmpeg to decode on the NVIDIA card.

    Measured on a 1080p 60 fps camera file (55 seconds, read at two frames a
    second): 10.1 s in software, 6.1 s with -hwaccel cuda. Decoding is most of
    what a tick costs on such footage, so it is worth asking. ARCANA_VIDEO_HW=0
    turns it off. Any file the card cannot decode is read again in software.
    """
    if "on" not in _HW:
        on = os.environ.get("ARCANA_VIDEO_HW", "1") != "0"
        if on:
            try:
                try:
                    from . import gpu as _gpu
                except ImportError:                       # pragma: no cover
                    import gpu as _gpu
                on = bool(_gpu.hardware().get("nvidia"))
            except Exception:
                on = False
        _HW["on"] = on
    return _HW["on"]


def frames(path: str, fps: float, *, side: int | None = 224, t0: float = 0.0,
           t1: float | None = None, info: dict | None = None, hw: bool | None = None):
    """
    Frames of a video at `fps`, as (seconds, RGB uint8 array), from t0 to t1.

    The first frame is the one at t0, and frame k is the one nearest t0 + k/fps.
    Seeking is done by ffmpeg before decoding, so reading ten seconds out of the
    middle of an hour costs ten seconds of decoding.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    info = info or probe(path)
    hw = hardware_decode() if hw is None else hw
    n = 0
    try:
        for item in _frames(path, fps, side, t0, t1, info, hw):
            n += 1
            yield item
    except VideoError:
        if not hw or n:
            raise
        n = -1
    if hw and n <= 0:
        # The card refused the file (a codec or a pixel format it lacks): read it in software.
        yield from _frames(path, fps, side, t0, t1, info, False)


def _frames(path, fps, side, t0, t1, info, hw):
    w, h = _fit(info["width"], info["height"], side)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hw:
        cmd += ["-hwaccel", "cuda"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    if t1 is not None:
        cmd += ["-t", f"{max(0.0, t1 - t0):.3f}"]
    cmd += ["-i", path, "-an", "-sn", "-vf", f"fps={fps:.6f}:round=down,scale={w}:{h}:flags=bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    # ffmpeg's messages go to a file, never a pipe. Nothing reads them while
    # frames are being read, and a decoder that complains on every frame (the
    # NVIDIA one does, on some files) fills a pipe and then waits for ever for
    # someone to empty it: the read below would never return.
    import tempfile
    errors = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors, creationflags=_NO_WINDOW)
    size = w * h * 3
    k = 0
    err = ""
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield t0 + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        try:
            errors.seek(0)
            err = errors.read()[-4000:].decode("utf-8", "replace").strip()
        finally:
            errors.close()
    if k == 0 and err:
        raise VideoError(f"Could not read frames from {os.path.basename(path)}: {err.splitlines()[-1]}")


def frames_between(path: str, t0: float, t1: float, n: int, *, side: int | None = 448,
                   info: dict | None = None) -> list[np.ndarray]:
    """`n` frames spread evenly over a section, for an encoder that takes a clip."""
    info = info or probe(path)
    if info["duration"]:
        t1 = min(t1, info["duration"])
        t0 = min(t0, max(0.0, t1 - 0.05))
    span = max(t1 - t0, 1e-3)
    out = [f for _, f in frames(path, n / span, side=side, t0=t0, t1=t1, info=info)]
    if not out:
        # A range shorter than one sampling step: take the frame at its start.
        one = frame_at(path, t0, side=side, info=info)
        out = [one] if one is not None else []
    return out[:n]


def frame_at(path: str, t: float, *, side: int | None = 448, info: dict | None = None) -> np.ndarray | None:
    """One frame, for a thumbnail or a keyframe."""
    for _, f in frames(path, 1.0, side=side, t0=max(0.0, t), t1=max(0.0, t) + 1.0, info=info):
        return f
    return None


# ─────────────────────────── sound ───────────────────────────

def audio(path: str, sr: int = 16000, *, t0: float = 0.0, t1: float | None = None) -> np.ndarray:
    """The sound track between t0 and t1, mono float32 in [-1, 1]. Empty when the file has none."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    if t1 is not None:
        cmd += ["-t", f"{max(0.0, t1 - t0):.3f}"]
    cmd += ["-i", path, "-vn", "-sn", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
    done = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    return np.frombuffer(done.stdout[:len(done.stdout) - len(done.stdout) % 4], np.float32).copy()


# ─────────────────────────── ticks ───────────────────────────

def ticks(path: str, tick_seconds: float = 0.5, *, side: int = 224, info: dict | None = None):
    """The frames the sectioning reads: one every `tick_seconds`, small."""
    yield from frames(path, 1.0 / tick_seconds, side=side, info=info)


def colour_signature(frame: np.ndarray, bins: int = 4) -> np.ndarray:
    """
    A frame as a small colour histogram, unit length.

    This is the cut detector's eye, and a stand-in for the encoder wherever a
    test or a preview has to section a video without loading one. It sees
    colour and nothing else, which is enough for a hard cut.
    """
    q = (frame.reshape(-1, 3).astype(np.int32) * bins) // 256
    idx = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    hist = np.bincount(idx, minlength=bins ** 3).astype(np.float32)
    return hist / max(float(np.linalg.norm(hist)), 1e-8)


def detect_cuts(signatures, threshold: float = 0.35) -> list[int]:
    """
    Hard cuts between ticks, as boundary indices (i sits between tick i and i+1).

    A cut is a jump in the colour signature that stands out from the changes
    around it. A pan or a fade moves the signature a little every tick and is
    left to the sectioning; a cut moves it once.
    """
    S = np.asarray(signatures, dtype=np.float32)
    if len(S) < 2:
        return []
    jump = 1.0 - np.sum(S[1:] * S[:-1], axis=1)
    out = []
    for i, j in enumerate(jump):
        near = np.concatenate([jump[max(0, i - 3):i], jump[i + 1:i + 4]])
        if j >= threshold and (len(near) == 0 or j >= 3.0 * float(near.mean())):
            out.append(i)
    return out
