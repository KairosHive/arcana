# pattern.py -- procedural editing: a rule places the frames
#
# A pattern is a pure function from an output frame number to what that frame
# shows. It reads numbers and a seed and nothing else, so it renders the same
# every time, needs no model, and can be written as text by a person or an
# agent. See VIDEO_ROADMAP.md, 3.5.
#
#   {"fps": 30, "seconds": 8, "size": [640, 360], "seed": 7, "base": "A",
#    "sources": {"A": "walk.mp4", "B": "water.mp4#t=4.000,12.000", "I": "sign.jpg", "K": "#000000"},
#    "voices": [
#      {"pulse": {"every": 2}, "cut": ["A", "B"]},                      interlace in time, frame by frame
#      {"pulse": {"euclid": [5, 8], "step": 4}, "flash": "I", "frames": 1},
#      {"pulse": {"every": 45}, "pair": "I", "amount": 0.2}             a complementary pair
#    ],
#    "space": {"kind": "lines", "size": 2, "shift": 1, "with": "B"},    interlace in space
#    "time": {"B": "stepping"}}
#
# Voices. Each has a pulse, which says on which frames it fires, and one action:
#   cut     move the base picture to the next source of a list, on each pulse
#   flash   show a source over everything, for `frames` frames from each pulse
#   pair    add an image to the pulse's frame and subtract it from the next, so
#           the two average back to the footage: unseen at speed, there on pause
#
# Pulses. {"every": p, "offset": o} fires every p frames (p may be fractional,
# which is how two voices drift against each other); {"on": a, "off": b} is a
# duty cycle; {"euclid": [k, n], "step": s} spreads k hits over n steps of s
# frames; {"hits": "x..x.x..", "step": s} is written by hand; {"chance": p}
# fires at random from the seed.
#
# Source time. "running" keeps a clip advancing while it is hidden, "stepping"
# advances it one frame each time it is shown, "freeze" holds its first frame.
#
# Curves. A curve is a wave with its own period, in frames, so it runs against
# the voices' rhythms instead of on them. It moves something continuous:
#
#   "curves": [
#     {"wave": {"period": 97, "shape": "sine"}, "hue": 0.5},          the colour wheel, half a turn each way
#     {"wave": {"period": 41}, "saturation": 0.8, "brightness": 0.2},
#     {"wave": {"period": 13, "shape": "square"}, "invert": 1.0},
#     {"wave": {"period": 61}, "shift": 14},                           red and blue pulled apart, in pixels
#     {"wave": {"period": 150, "shape": "triangle"}, "blend": "B", "amount": 0.8}   a dissolve into B
#   ]
#
# Shapes are sine, triangle, saw and square; "phase" is a share of the period.
# hue, saturation, brightness and shift swing both ways around the picture as
# it is; invert and blend go from none to their full amount and back.

from __future__ import annotations

import os

import numpy as np

try:
    from . import modality as _modality
    from . import video as _video
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import video as _video

MAX_SECONDS = 600
MAX_SOURCE_SECONDS = 120


class PatternError(ValueError):
    """A pattern that cannot be rendered, said in the terms it was written in."""


# ─────────────────────────── pulses ───────────────────────────

def _euclid(k: int, n: int) -> list[bool]:
    """k hits spread as evenly as they go over n steps (Bjorklund's rhythm, by rounding)."""
    if n <= 0 or k <= 0:
        return [False] * max(n, 0)
    k = min(k, n)
    return [(i * k) % n < k for i in range(n)]


def pulse_frames(pulse: dict, n_frames: int, seed: int = 0) -> np.ndarray:
    """A boolean per output frame: does this pulse fire on it."""
    out = np.zeros(n_frames, dtype=bool)
    if not isinstance(pulse, dict) or not pulse:
        raise PatternError('A voice needs a pulse, such as {"every": 3}.')
    step = max(1, int(pulse.get("step", 1)))
    offset = float(pulse.get("offset", 0))
    if "every" in pulse:
        p = float(pulse["every"])
        if p <= 0:
            raise PatternError('"every" is a number of frames above 0.')
        # A fractional period fires on the frame its beat falls in, so 3.02 runs
        # one frame late every 150 beats: that is the drift.
        beats = np.floor((np.arange(n_frames) - offset) / p)
        out[1:] = beats[1:] != beats[:-1]
        out[0] = offset <= 0 and (offset % p == 0 or offset == 0)
        return out
    if "on" in pulse:
        on, off = max(1, int(pulse["on"])), max(0, int(pulse.get("off", 0)))
        phase = (np.arange(n_frames) - int(offset)) % (on + off)
        return phase < on
    if "euclid" in pulse or "hits" in pulse:
        if "euclid" in pulse:
            k, n = (int(x) for x in pulse["euclid"])
            cycle = _euclid(k, n)
        else:
            cycle = [c in "xX1" for c in str(pulse["hits"]) if not c.isspace()]
        if not cycle:
            raise PatternError("That rhythm has no steps.")
        idx = np.arange(n_frames) - int(offset)
        on_step = (idx % step == 0) & (idx >= 0)
        which = (idx // step) % len(cycle)
        return on_step & np.array(cycle)[which]
    if "chance" in pulse:
        rng = np.random.default_rng(int(seed) + int(pulse.get("salt", 0)))
        return rng.random(n_frames) < float(pulse["chance"])
    raise PatternError('A pulse is one of "every", "on", "euclid", "hits" or "chance".')


# ─────────────────────────── curves ───────────────────────────

SHAPES = ("sine", "triangle", "saw", "square")
MOVES = ("hue", "saturation", "brightness", "invert", "shift", "blend")


def wave(spec: dict, n_frames: int) -> np.ndarray:
    """A wave's value on each frame, from -1 to 1."""
    if not isinstance(spec, dict) or "period" not in spec:
        raise PatternError('A curve needs a wave, such as {"period": 97}.')
    period = float(spec["period"])
    if period <= 0:
        raise PatternError("A wave's period is a number of frames above 0.")
    shape = spec.get("shape", "sine")
    if shape not in SHAPES:
        raise PatternError("A wave's shape is sine, triangle, saw or square.")
    x = (np.arange(n_frames) / period + float(spec.get("phase", 0.0))) % 1.0
    if shape == "sine":
        return np.sin(2 * np.pi * x).astype(np.float32)
    if shape == "triangle":
        return (1.0 - 4.0 * np.abs(x - 0.5)).astype(np.float32)
    if shape == "saw":
        return (2.0 * x - 1.0).astype(np.float32)
    return np.where(x < 0.5, 1.0, -1.0).astype(np.float32)


def _colour(frame: np.ndarray, hue: float, sat: float, bright: float) -> np.ndarray:
    """Turn the colour wheel by `hue` half-turns and scale saturation and brightness."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    if hue:
        hsv[..., 0] = (hsv[..., 0] + hue * 90.0) % 180.0          # OpenCV's hue runs 0 to 180
    if sat:
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat), 0, 255)
    if bright:
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + bright), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _shift(frame: np.ndarray, px: int) -> np.ndarray:
    """Pull red one way and blue the other by `px` pixels."""
    if not px:
        return frame
    out = frame.copy()
    out[..., 0] = np.roll(frame[..., 0], px, axis=1)
    out[..., 2] = np.roll(frame[..., 2], -px, axis=1)
    return out


# ─────────────────────────── the plan ───────────────────────────

def plan(spec: dict) -> dict:
    """
    What each output frame shows, as arrays over frames: the base source, the
    source flashed over it (or ""), and the image added or subtracted for a
    complementary pair. Rendering is this plus pixels; tests read it directly.
    """
    fps = float(spec.get("fps", 30))
    seconds = float(spec.get("seconds", 8))
    if not (1 <= fps <= 120):
        raise PatternError("fps is between 1 and 120.")
    if not (0 < seconds <= MAX_SECONDS):
        raise PatternError(f"seconds is above 0 and at most {MAX_SECONDS}.")
    n = int(round(fps * seconds))
    sources = spec.get("sources") or {}
    if not sources:
        raise PatternError("A pattern needs at least one source.")
    seed = int(spec.get("seed", 0))

    def known(name, where):
        if name not in sources:
            raise PatternError(f"{where} names the source {name!r}, and the sources are {', '.join(sources)}.")
        return name

    base = [known(spec.get("base") or next(iter(sources)), "base")] * n
    flash = [""] * n
    pair = [("", 0.0)] * n
    fires = []
    for v, voice in enumerate(spec.get("voices") or []):
        hits = pulse_frames(voice.get("pulse"), n, seed + 101 * v)
        fires.append(hits)
        if "cut" in voice:
            order = [known(s, "cut") for s in voice["cut"]]
            if not order:
                raise PatternError("cut needs a list of sources.")
            at = -1
            for f in range(n):
                if hits[f]:
                    at += 1
                if at >= 0:
                    base[f] = order[at % len(order)]
        elif "flash" in voice:
            src = known(voice["flash"], "flash")
            length = max(1, int(voice.get("frames", 1)))
            for f in np.flatnonzero(hits):
                for g in range(f, min(n, f + length)):
                    flash[g] = src
        elif "pair" in voice:
            src = known(voice["pair"], "pair")
            amount = float(voice.get("amount", 0.2))
            for f in np.flatnonzero(hits):
                if f + 1 < n:
                    pair[f] = (src, amount)
                    pair[f + 1] = (src, -amount)
        else:
            raise PatternError('A voice does one of "cut", "flash" or "pair".')
    curves = []
    for c in spec.get("curves") or []:
        moves = {k: c[k] for k in MOVES if k in c}
        if not moves:
            raise PatternError("A curve moves one of " + ", ".join(MOVES) + ".")
        if "blend" in moves:
            known(moves["blend"], "blend")
        curves.append({"values": wave(c.get("wave"), n), "moves": moves, "amount": float(c.get("amount", 1.0))})
    space = spec.get("space") or None
    if space:
        known(space.get("with", ""), "space.with")
        if space.get("kind") not in ("lines", "columns", "checker", "blocks"):
            raise PatternError("space.kind is lines, columns, checker or blocks.")
    return {"n": n, "fps": fps, "base": base, "flash": flash, "pair": pair, "fires": fires, "space": space,
            "curves": curves}


# ─────────────────────────── sources ───────────────────────────

class Source:
    """One source, decoded to frames at the pattern's size and rate."""

    def __init__(self, ref: str, size, fps: float, limit_frames: int):
        w, h = size
        self.still = True
        ref = str(ref)
        if ref.startswith("#") and len(ref) == 7:
            rgb = [int(ref[i:i + 2], 16) for i in (1, 3, 5)]
            self.frames = np.tile(np.array(rgb, np.uint8), (1, h, w, 1))
            return
        kind = _modality.of_path(ref)
        if kind == _modality.VIDEO:
            file, t0, t1 = _modality.split_ref(ref)
            info = _video.probe(file)
            t0 = t0 or 0.0
            t1 = min(t1 if t1 is not None else info["duration"], t0 + MAX_SOURCE_SECONDS,
                     t0 + limit_frames / fps + 1.0)
            got = [self._fit(f, w, h) for _, f in _video.frames(file, fps, side=max(w, h), t0=t0, t1=t1, info=info)]
            if not got:
                raise PatternError(f"No frames could be read from {os.path.basename(file)}.")
            self.frames = np.stack(got)
            self.still = len(got) == 1
            return
        if kind == _modality.IMAGE:
            from PIL import Image
            with Image.open(ref) as im:
                self.frames = self._fit(np.asarray(im.convert("RGB")), w, h)[None]
            return
        raise PatternError(f"{ref!r} is not a video, a picture or a colour like #000000.")

    @staticmethod
    def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """Fill the pattern's frame: scale to cover it, then crop the middle."""
        import cv2
        fh, fw = frame.shape[:2]
        s = max(w / fw, h / fh)
        nw, nh = max(w, round(fw * s)), max(h, round(fh * s))
        big = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        x, y = (nw - w) // 2, (nh - h) // 2
        return big[y:y + h, x:x + w]

    def at(self, i: int) -> np.ndarray:
        return self.frames[i % len(self.frames)]


def _mask(space: dict, h: int, w: int, frame: int) -> np.ndarray:
    """Where the second source shows, for interlacing in space."""
    size = max(1, int(space.get("size", 2)))
    shift = int(space.get("shift", 0)) * frame
    ys, xs = np.mgrid[0:h, 0:w]
    kind = space["kind"]
    if kind == "lines":
        return ((ys + shift) // size) % 2 == 1
    if kind == "columns":
        return ((xs + shift) // size) % 2 == 1
    return (((ys + shift) // size) + ((xs + shift) // size)) % 2 == 1      # checker and blocks


def render(spec: dict, sources: dict | None = None):
    """Yield the pattern's frames as RGB uint8 arrays. `sources` can supply decoded Source objects."""
    p = plan(spec)
    w, h = (int(x) for x in spec.get("size", (640, 360)))
    w, h = w - w % 2, h - h % 2
    if not (64 <= w <= 3840 and 64 <= h <= 2160):
        raise PatternError("size is [width, height], each between 64 and 3840.")
    srcs = sources or {name: Source(ref, (w, h), p["fps"], p["n"]) for name, ref in spec["sources"].items()}
    policy = spec.get("time") or {}
    shown = {name: 0 for name in srcs}

    def pick(name, f):
        mode = policy.get(name, "running")
        if mode == "freeze":
            return srcs[name].at(0)
        if mode == "stepping":
            i = shown[name]
            shown[name] += 1
            return srcs[name].at(i)
        return srcs[name].at(f)

    for f in range(p["n"]):
        frame = pick(p["base"][f], f)
        if p["space"]:
            other = pick(p["space"]["with"], f)
            frame = np.where(_mask(p["space"], h, w, f)[..., None], other, frame)
        for c in p["curves"]:
            v = float(c["values"][f])
            m = c["moves"]
            if "blend" in m:
                a = (v + 1.0) / 2.0 * c["amount"]
                other = pick(m["blend"], f).astype(np.float32)
                frame = (frame.astype(np.float32) * (1.0 - a) + other * a).astype(np.uint8)
            if "hue" in m or "saturation" in m or "brightness" in m:
                frame = _colour(frame, v * float(m.get("hue", 0)), v * float(m.get("saturation", 0)),
                                v * float(m.get("brightness", 0)))
            if "shift" in m:
                frame = _shift(frame, int(round(v * float(m["shift"]))))
            if "invert" in m:
                a = (v + 1.0) / 2.0 * float(m["invert"])
                frame = (frame.astype(np.float32) * (1.0 - a) + (255.0 - frame.astype(np.float32)) * a).astype(np.uint8)
        if p["flash"][f]:
            frame = pick(p["flash"][f], f)
        name, amount = p["pair"][f]
        if name:
            # The picture's detail around its own average. Adding and subtracting
            # the average as well would make the whole frame brighter and darker
            # in turn, fifteen times a second, which is a flicker and no secret.
            img = srcs[name].at(0).astype(np.float32)
            img = img - img.mean(axis=(0, 1), keepdims=True)
            frame = np.clip(frame.astype(np.float32) + amount * img, 0, 255).astype(np.uint8)
        yield np.ascontiguousarray(frame)


# ─────────────────────────── the flash meter ───────────────────────────

def flash_report(luminance, fps: float) -> dict:
    """
    Flashes per second, as WCAG 2.3.1 counts them: a flash is a pair of opposing
    changes in relative luminance of 10% or more of full white, where the
    darker state is below 0.80. `luminance` is the mean relative luminance of
    each frame, 0 to 1. Returns the worst one-second count, whether it is over
    the limit of three, and when.
    """
    L = np.asarray(luminance, dtype=np.float32)
    if len(L) < 3:
        return {"max_per_second": 0, "over": False, "at": None, "flashes": 0}
    # Follow the luminance as a run of rises and falls of at least 10%. Each
    # entry of `turns` is the frame where the light changed direction by that
    # much, so the directions alternate, and every second one completes a flash.
    turns = []
    anchor, direction = float(L[0]), 0
    for i in range(1, len(L)):
        v = float(L[i])
        d = v - anchor
        if direction <= 0 and d >= 0.10 and min(anchor, v) < 0.80:
            turns.append(i)
            direction, anchor = 1, v
        elif direction >= 0 and d <= -0.10 and min(anchor, v) < 0.80:
            turns.append(i)
            direction, anchor = -1, v
        elif (direction > 0 and v > anchor) or (direction < 0 and v < anchor):
            anchor = v                      # still going the same way: follow it
    ends = turns[1::2]
    worst, at = 0, None
    window = int(round(fps))
    for i, e in enumerate(ends):
        count = sum(1 for x in ends[i:] if x < e + window)
        if count > worst:
            worst, at = count, e / fps
    return {"max_per_second": int(worst), "over": worst > 3, "at": None if at is None else round(float(at), 2),
            "flashes": len(ends)}


def luminance_of(frame: np.ndarray) -> float:
    """Mean relative luminance of an sRGB frame, 0 to 1."""
    x = frame.reshape(-1, 3).astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return float((lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())


# ─────────────────────────── writing it ───────────────────────────

def write(spec: dict, path: str, progress=None) -> dict:
    """Render a pattern to an mp4. Returns {"path", "frames", "fps", "flash"}."""
    import subprocess
    p = plan(spec)
    w, h = (int(x) for x in spec.get("size", (640, 360)))
    w, h = w - w % 2, h - h % 2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import tempfile
    errors = tempfile.TemporaryFile()          # a file, not a pipe: see video.frames
    proc = subprocess.Popen(
        [_video.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{p['fps']:.4f}", "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
         # Every frame a keyframe: a pattern is made of single frames, and a
         # player has to be able to stop on any of them.
         "-g", "1", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE, stderr=errors, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lum = []
    try:
        for f, frame in enumerate(render(spec)):
            proc.stdin.write(frame.tobytes())
            lum.append(luminance_of(frame[::4, ::4]))
            if progress is not None and f % 30 == 0:
                progress(f / p["n"], f"Rendering frame {f:,} of {p['n']:,}")
    finally:
        proc.stdin.close()
        proc.wait()
        errors.seek(0)
        err = errors.read()[-2000:].decode("utf-8", "replace")
        errors.close()
    if proc.returncode:
        raise PatternError("ffmpeg could not write the pattern: " + err.strip()[-200:])
    return {"path": path, "frames": p["n"], "fps": p["fps"], "flash": flash_report(lum, p["fps"])}


PRESETS: dict = {
    "interlace in time": {"voices": [{"pulse": {"every": 1}, "cut": ["A", "B"]}]},
    "three against four": {"voices": [{"pulse": {"every": 3}, "cut": ["A", "B"]},
                                      {"pulse": {"every": 4}, "flash": "B", "frames": 1}]},
    "line interlace, drifting": {"space": {"kind": "lines", "size": 2, "shift": 1, "with": "B"}},
    "euclidean flashes": {"voices": [{"pulse": {"euclid": [5, 8], "step": 4}, "flash": "B", "frames": 1}]},
    "phasing": {"voices": [{"pulse": {"every": 6}, "cut": ["A", "B"]},
                           {"pulse": {"every": 6.1}, "flash": "B", "frames": 2}]},
    "hidden image": {"voices": [{"pulse": {"every": 2}, "pair": "B", "amount": 0.25}]},
}
