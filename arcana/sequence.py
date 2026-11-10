# sequence.py -- clips cut together, with their sound, and an edit list for an editor
#
# The film engine morphs still pictures through a token space. A sequence is
# the other thing a timeline is for: sections of real video placed one after
# another, each with its own in and out point, rendered to one file with the
# sound they came with. It is what a search result is dropped into, and what
# Generate Story becomes when its dataset is video. See VIDEO_ROADMAP.md, 3.1
# and 3.2.
#
#   {"name": "rough cut", "fps": 30, "size": [1280, 720], "sound": true,
#    "clips": [{"ref": "C:/footage/walk.mp4#t=12.000,22.500"},
#              {"ref": "C:/footage/sea.mov", "in": 3.0, "out": 8.0, "speed": 0.5, "note": "the wave"},
#              {"ref": "C:/photos/sign.jpg", "seconds": 2.0}]}
#
# A clip's ref is a file or a section of one. `in` and `out` trim it further,
# in seconds of the source. A still picture is held for `seconds`. Every clip
# is scaled to fill the frame and cropped, so sources of any shape cut together.
#
# The render goes through ffmpeg in one pass. The edit list is CMX 3600, the
# plain-text format every editor imports: it names the original files and
# their timecodes, so the cut can be reopened in Resolve or Premiere at full
# quality, and Arcana's render stays a preview of it.

from __future__ import annotations

import os
import subprocess
import tempfile

try:
    from . import modality as _modality
    from . import video as _video
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import video as _video

MAX_CLIPS = 200
STILL_SECONDS = 2.0


class SequenceError(ValueError):
    """A sequence that cannot be made, said in the terms it was written in."""


def resolve(spec: dict) -> list[dict]:
    """
    Each clip with its file, its kind and its range in the source worked out:
    {"file", "kind", "start", "length", "speed", "seconds" (as played), "has_audio", "note"}.
    """
    clips = spec.get("clips") or []
    if not clips:
        raise SequenceError("A sequence needs at least one clip.")
    if len(clips) > MAX_CLIPS:
        raise SequenceError(f"A sequence holds at most {MAX_CLIPS} clips.")
    out = []
    for n, c in enumerate(clips, 1):
        ref = str(c.get("ref") or "")
        file, t0, t1 = _modality.split_ref(ref)
        kind = _modality.of_path(file)
        if not os.path.isfile(file):
            raise SequenceError(f"Clip {n}: no such file, {file}.")
        if kind == _modality.IMAGE:
            length = float(c.get("seconds") or STILL_SECONDS)
            out.append({"file": file, "kind": kind, "start": 0.0, "length": length, "speed": 1.0,
                        "seconds": length, "has_audio": False, "note": c.get("note") or ""})
            continue
        if kind != _modality.VIDEO:
            raise SequenceError(f"Clip {n}: {os.path.basename(file)} is not a video or a picture.")
        info = _video.probe(file)
        start = float(c["in"]) if c.get("in") is not None else (t0 or 0.0)
        end = float(c["out"]) if c.get("out") is not None else (t1 if t1 is not None else info["duration"])
        end = min(end, info["duration"] or end)
        if end - start < 0.04:
            raise SequenceError(f"Clip {n}: its out point is not after its in point.")
        speed = float(c.get("speed") or 1.0)
        if not 0.1 <= speed <= 8:
            raise SequenceError(f"Clip {n}: speed is between 0.1 and 8.")
        out.append({"file": file, "kind": kind, "start": start, "length": end - start, "speed": speed,
                    "seconds": (end - start) / speed, "has_audio": bool(info["has_audio"]), "fps": info["fps"],
                    "note": c.get("note") or ""})
    return out


def _atempo(speed: float) -> str:
    """ffmpeg's atempo takes 0.5 to 2 at a time, so a larger change is a chain of them."""
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.4f}")
    return ",".join(parts)


def render(spec: dict, path: str, progress=None) -> dict:
    """Render a sequence to an mp4. Returns {"path", "seconds", "clips", "size", "fps", "sound"}."""
    clips = resolve(spec)
    fps = float(spec.get("fps", 30))
    w, h = (int(x) for x in spec.get("size", (1280, 720)))
    w, h = w - w % 2, h - h % 2
    if not (64 <= w <= 3840 and 64 <= h <= 2160) or not (1 <= fps <= 120):
        raise SequenceError("size is [width, height] from 64 to 3840, and fps from 1 to 120.")
    sound = bool(spec.get("sound", True))
    cmd = [_video.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    chains, pads = [], []
    for i, c in enumerate(clips):
        if c["kind"] == _modality.IMAGE:
            cmd += ["-loop", "1", "-t", f"{c['length']:.3f}", "-i", c["file"]]
        else:
            cmd += ["-ss", f"{c['start']:.3f}", "-t", f"{c['length']:.3f}", "-i", c["file"]]
        # Fill the frame and crop the middle, at one frame rate and pixel shape for all.
        v = (f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,"
             f"setpts=(PTS-STARTPTS)/{c['speed']:.4f},fps={fps:.4f},format=yuv420p[v{i}]")
        chains.append(v)
        if sound:
            if c["has_audio"]:
                chains.append(f"[{i}:a]asetpts=PTS-STARTPTS,{_atempo(c['speed'])},aresample=48000,"
                              f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                              f"apad,atrim=0:{c['seconds']:.3f}[a{i}]")
            else:
                chains.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{c['seconds']:.3f}[a{i}]")
            pads.append(f"[v{i}][a{i}]")
        else:
            pads.append(f"[v{i}]")
    chains.append("".join(pads) + f"concat=n={len(clips)}:v=1:a={1 if sound else 0}[v]" + ("[a]" if sound else ""))
    cmd += ["-filter_complex", ";".join(chains), "-map", "[v]"]
    if sound:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if progress is not None:
        progress(0.05, f"Cutting {len(clips)} clips together")
    errors = tempfile.TemporaryFile()
    try:
        done = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=errors,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode:
            errors.seek(0)
            raise SequenceError("ffmpeg could not cut the sequence: "
                                + errors.read()[-400:].decode("utf-8", "replace").strip())
    finally:
        errors.close()
    if progress is not None:
        progress(1.0, "Done")
    return {"path": path, "seconds": sum(c["seconds"] for c in clips), "clips": len(clips),
            "size": [w, h], "fps": fps, "sound": sound}


# ─────────────────────────── the edit list ───────────────────────────

def _timecode(seconds: float, fps: float) -> str:
    base = max(1, int(round(fps)))
    frames = int(round(seconds * base))
    return f"{frames // (3600 * base):02d}:{frames // (60 * base) % 60:02d}:{frames // base % 60:02d}:{frames % base:02d}"


def edl(spec: dict, title: str | None = None) -> str:
    """
    The sequence as a CMX 3600 edit list, for Resolve, Premiere or any editor.

    Each event names its source file and gives the source in and out and the
    place on the record timeline. The record starts at one hour, as editors
    expect. A speed change is written as an M2 line.
    """
    clips = resolve(spec)
    fps = float(spec.get("fps", 30))
    lines = [f"TITLE: {title or spec.get('name') or 'Arcana sequence'}", "FCM: NON-DROP FRAME", ""]
    at = 3600.0
    for n, c in enumerate(clips, 1):
        reel = f"AX{n:03d}"[:8]
        src_in, src_out = c["start"], c["start"] + c["length"]
        lines.append(f"{n:03d}  {reel:<8} V     C        {_timecode(src_in, fps)} {_timecode(src_out, fps)} "
                     f"{_timecode(at, fps)} {_timecode(at + c['seconds'], fps)}")
        if c["speed"] != 1.0:
            lines.append(f"M2   {reel:<8}       {fps * c['speed']:05.1f}                {_timecode(src_in, fps)}")
        lines.append(f"* FROM CLIP NAME: {os.path.basename(c['file'])}")
        lines.append(f"* SOURCE FILE: {c['file']}")
        if c["note"]:
            lines.append(f"* COMMENT: {c['note']}")
        lines.append("")
        at += c["seconds"]
    return "\n".join(lines)
