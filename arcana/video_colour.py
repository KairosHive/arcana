# video_colour.py -- a colour transfer on moving pictures, without flicker
#
# A colour transfer run on each frame alone flickers: every frame is fitted to
# the reference on its own, so two frames that differ by a leaf get slightly
# different colours. This module fits the transfer once and applies it as a
# table. See VIDEO_ROADMAP.md, 3.3.
#
# Static. Frames are sampled across the clip and laid side by side as one
# picture, so the fit sees the whole clip's colours. The transfer runs once on
# that picture (any of the existing methods), a 3-D table is fitted from the
# before and after, and the same table recolours every frame. One table for
# the clip is stable by construction.
#
# Dynamic. The reference is a video, or an ordered set of pictures. A table is
# fitted at each of several moments of the reference, and every frame of the
# clip uses the tables on either side of its own moment, mixed by how far
# along it is. The clip's colour then follows the reference's colour over
# time. Mixing two tables is mixing two colours for each entry, so the change
# from frame to frame is as smooth as the reference's own.
#
# The clip's sound is kept, and the table at each key is written beside the
# video as a .cube file for an editor.

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

try:
    from . import color_transfer as _ct
    from . import lut as _lut
    from . import modality as _modality
    from . import transfer_methods as _tm
    from . import video as _video
except ImportError:                                   # pragma: no cover
    import color_transfer as _ct
    import lut as _lut
    import modality as _modality
    import transfer_methods as _tm
    import video as _video

SAMPLES = 8            # frames laid side by side for a fit
TILE = 256             # their longest side
LUT_N = 33
MAX_SIDE = 1920


def _range(ref: str) -> tuple[str, float, float]:
    file, t0, t1 = _modality.split_ref(ref)
    info = _video.probe(file)
    t0 = t0 or 0.0
    t1 = min(t1 if t1 is not None else info["duration"], info["duration"] or 1e9)
    return file, t0, max(t1, t0 + 0.04)


def mosaic(ref: str, t0: float | None = None, t1: float | None = None, n: int = SAMPLES) -> Image.Image:
    """Frames spread over a clip (or a part of it), side by side as one picture: its colours, pooled."""
    if _modality.of_path(ref) != _modality.VIDEO:
        im = Image.open(_modality.file_of(ref)).convert("RGB")
        im.thumbnail((TILE * 2, TILE * 2))
        return im
    file, a, b = _range(ref)
    a = a if t0 is None else t0
    b = b if t1 is None else t1
    frames = _video.frames_between(file, a, b, n, side=TILE)
    if not frames:
        raise ValueError(f"No frames could be read from {os.path.basename(file)}.")
    return Image.fromarray(np.concatenate(frames, axis=1))


def fit(content: Image.Image, reference: Image.Image, method: str = "lab", strength: float = 1.0) -> np.ndarray:
    """The table that takes `content`'s colours toward `reference`'s, by one of the transfer methods."""
    styled, _ = _tm.transfer(method, content, reference, strength=strength)
    styled = styled.convert("RGB").resize(content.size)
    return _ct._fit_lut3d(content, styled, LUT_N)


def identity() -> np.ndarray:
    n = LUT_N
    return (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1) / (n - 1) * 255.0).astype(np.float64)


def tables_for(target: str, reference, method: str = "lab", strength: float = 1.0, keys: int = 1):
    """
    [(position 0..1, table), ...] for a target clip.

    One table for a static transfer. With `keys` above 1 and a reference that
    is a video, one table for each of that many moments of the reference; with
    a list of references, one table for each, in order.
    """
    content = mosaic(target)
    if isinstance(reference, (list, tuple)) and len(reference) > 1:
        refs = [mosaic(r) for r in reference]
        return [(i / (len(refs) - 1), fit(content, r, method, strength)) for i, r in enumerate(refs)]
    ref = reference[0] if isinstance(reference, (list, tuple)) else reference
    if keys <= 1 or _modality.of_path(ref) != _modality.VIDEO:
        return [(0.0, fit(content, mosaic(ref), method, strength))]
    file, a, b = _range(ref)
    span = (b - a) / keys
    out = []
    for k in range(keys):
        part = mosaic(ref, a + k * span, a + (k + 1) * span, n=max(2, SAMPLES // 2))
        out.append((k / (keys - 1), fit(content, part, method, strength)))
    return out


def table_at(tables, position: float) -> np.ndarray:
    """The table for a moment between keys: the two around it, mixed."""
    if len(tables) == 1 or position <= tables[0][0]:
        return tables[0][1]
    for (p0, t0), (p1, t1) in zip(tables, tables[1:]):
        if position <= p1:
            w = (position - p0) / max(p1 - p0, 1e-9)
            return t0 * (1.0 - w) + t1 * w
    return tables[-1][1]


def apply(frame: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.asarray(_ct._apply_lut3d(Image.fromarray(frame), table), np.uint8)


def render(target: str, tables, path: str, progress=None) -> dict:
    """Recolour a clip with its tables and write it, with its sound. Returns {"path", "frames", "fps", "seconds"}."""
    file, a, b = _range(target)
    info = _video.probe(file)
    fps = min(info["fps"] or 30.0, 60.0)
    w, h = _video._fit(info["width"], info["height"], MAX_SIDE)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cmd = [_video.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-"]
    if info["has_audio"]:
        cmd += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", file, "-map", "0:v", "-map", "1:a", "-c:a", "aac",
                "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    errors = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    total = max(1, int(round((b - a) * fps)))
    n = 0
    single = tables[0][1] if len(tables) == 1 else None
    try:
        for _, frame in _video.frames(file, fps, side=max(w, h), t0=a, t1=b, info=info):
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = np.asarray(Image.fromarray(frame).resize((w, h)))
            table = single if single is not None else table_at(tables, n / max(total - 1, 1))
            proc.stdin.write(apply(frame, table).tobytes())
            n += 1
            if progress is not None and n % 15 == 0:
                progress(min(0.99, n / total), f"Recolouring frame {n:,} of {total:,}")
    finally:
        proc.stdin.close()
        proc.wait()
        errors.seek(0)
        err = errors.read()[-300:].decode("utf-8", "replace")
        errors.close()
    if proc.returncode or not n:
        raise ValueError("The recoloured clip could not be written: " + (err.strip() or "no frames were read."))
    return {"path": path, "frames": n, "fps": fps, "seconds": n / fps, "size": [w, h]}


def write_cubes(tables, stem: str, title: str = "arcana") -> list[str]:
    """Each key's table as a .cube file beside the video: one for a static transfer, several for a dynamic one."""
    out = []
    for i, (_, table) in enumerate(tables):
        # .cube wants red fastest; the table is indexed [r][g][b].
        rows = (table / 255.0).transpose(2, 1, 0, 3).reshape(-1, 3)
        name = stem + (".cube" if len(tables) == 1 else f"_{i + 1:02d}.cube")
        out.append(str(_lut.write_cube(name, rows, table.shape[0], title)))
    return out


def flicker(path: str, t0: float = 0.0, t1: float | None = None) -> float:
    """
    How much a clip's overall colour jumps from frame to frame, as the mean
    absolute change of the frame's average colour, in levels of 255. A measure
    for comparing a recoloured clip with its source.
    """
    info = _video.probe(path)
    means = [f.reshape(-1, 3).mean(0) for _, f in _video.frames(path, info["fps"] or 30.0, side=96, t0=t0, t1=t1,
                                                                 info=info, hw=False)]
    if len(means) < 2:
        return 0.0
    return float(np.abs(np.diff(np.stack(means), axis=0)).mean())


# ─────────────────────────── deflicker ───────────────────────────
# A per-frame effect (a diffusion look, a transfer fitted on each frame, an
# upscaler) leaves the picture brighter on one frame and darker on the next.
# The cure here follows Bonneel et al.'s idea in its simplest form: the
# processed clip may change over time the way its source did, and no faster.
#
# Each frame's average colour is taken. Its slow course over the clip is what
# the effect meant; the quick jumps around that course are the flicker. Where
# a source is given, the source's own quick changes (a cut, a flash that was
# really filmed) are put back, so only what the effect added is removed. Each
# frame is then scaled, channel by channel, to its target average.

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """A Gaussian over time, on each column, with the ends held."""
    if window <= 1 or len(x) < 3:
        return x.copy()
    sigma = window / 3.0
    r = int(max(1, round(window)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)


def frame_means(path: str, fps: float, t0: float = 0.0, t1: float | None = None) -> np.ndarray:
    info = _video.probe(path)
    got = [f.reshape(-1, 3).mean(0) for _, f in _video.frames(path, fps, side=128, t0=t0, t1=t1, info=info, hw=False)]
    return np.stack(got).astype(np.float64) if got else np.zeros((0, 3))


def gains(processed: np.ndarray, source: np.ndarray | None = None, window: int = 9) -> np.ndarray:
    """The factor each frame's channels are scaled by: (frames, 3)."""
    p = np.maximum(processed, 1.0)
    target = _smooth(p, window)
    if source is not None and len(source) >= len(p):
        s = np.maximum(source[:len(p)], 1.0)
        target = target * (s / np.maximum(_smooth(s, window), 1.0))      # what the source itself did, kept
    return np.clip(target / p, 0.5, 2.0)


def deflicker(path: str, out: str, source: str | None = None, window: int = 9, progress=None) -> dict:
    """
    Write `path` again with its frame-to-frame colour jumps removed.

    `source` is the clip the effect was applied to, a file or a section. With
    it, changes the source really had are kept. Returns {"path", "frames",
    "before", "after"}, the last two being the flicker measure of each.
    """
    info = _video.probe(path)
    fps = min(info["fps"] or 30.0, 60.0)
    means = frame_means(path, fps)
    src = None
    if source:
        file, a, b = _range(source)
        src = frame_means(file, fps, a, b)
    g = gains(means, src, window)
    w, h = _video._fit(info["width"], info["height"], MAX_SIDE)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [_video.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-"]
    if info["has_audio"]:
        cmd += ["-i", path, "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    errors = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    n = 0
    try:
        for _, frame in _video.frames(path, fps, side=max(w, h), info=info, hw=False):
            if n >= len(g):
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = np.asarray(Image.fromarray(frame).resize((w, h)))
            fixed = np.clip(frame.astype(np.float32) * g[n].astype(np.float32), 0, 255).astype(np.uint8)
            proc.stdin.write(fixed.tobytes())
            n += 1
            if progress is not None and n % 15 == 0:
                progress(min(0.99, n / max(len(g), 1)), f"Steadying frame {n:,} of {len(g):,}")
    finally:
        proc.stdin.close()
        proc.wait()
        errors.close()
    if proc.returncode or not n:
        raise ValueError("The steadied clip could not be written.")
    return {"path": out, "frames": n, "before": flicker(path), "after": flicker(out)}

