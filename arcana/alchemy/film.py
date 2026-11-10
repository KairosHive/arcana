"""Films: pictures morphed through a token space on a timeline, with any number of an Alchemy step keyed
in time.

A film is a plain dict, kept as JSON:

    {"name": "Rites", "fps": 30, "size": 768, "aspect": "3:2", "space": "DINOv2", "loop": true,
     "length": 120.0, "even": true,
     "pictures": [{"t": 0, "ref": "japan:12", "ease": "linear"}, {"t": 6, "ref": "japan:7"}, ...],
     "looks":    [{"t": 0, "ease": "smooth", "settings": {...a step's settings, as alchemy preview...}}],
     "lanes":    {"finish.saturation": [{"t": 0, "v": 0}, {"t": 120, "v": -100, "ease": "in"}],
                  "knobs.picture": [...], "mask.1.x": [...], "zoom": [...]}}

The picture lane. The film passes through each picture at its key's time, on a closed Catmull-Rom
curve through their token grids (open when `loop` is false), so it stops at none of them. A key's
`ease` shapes the passage to the next picture: linear, smooth, in (slow start), out (slow end) or hold
(stay, then cut). With `loop`, the last picture runs back to the first by `length`; without, the film
ends at the last key. Moving keys closer together is how a morph speeds up. `even` spaces each passage
so the picture changes by the same amount every frame (as latent-lab's films do), which is worked out
once per render on a half-size grid.

The look lane. Each snapshot holds a whole step (method, knobs, materials, mask, finish), taken from
the Image subtab. Between two snapshots every number moves from one to the other with the first one's
ease; what is not a number (a method, a model, a mask preset) changes at the second snapshot's time.

Lanes. A lane keys one number in time and wins over the snapshots wherever it has keys: a knob
(`knobs.<key>`), a finish slider (`finish.<key>`), a mask refinement (`mask.feather`, `mask.grow`,
`mask.follow`, `mask.ragged`, `mask.amount`), a layer's coverage (`mask.<n>.coverage`) or a number
of a geometric layer (`mask.<n>.x` for a spot, `.angle` for a gradient, `.y` and `.height` for a
band...), a material's share (`weights.<n>`) and the material zoom (`zoom`). Keys hold `t`, `v` and
the `ease` towards the next key.

Films inside a film. A key on the picture lane can hold a kept film, `{"t": 12, "ref": "film:rites",
"in": 0, "out": 8, "speed": 1}`: the film morphs into that film's frame at `in`, plays it from `in` to
`out` at `speed`, and morphs on from its frame at `out`. A film made of one picture's derivatives
becomes one passage of another film this way.

Clips. A key can hold a video the same way, by its handle, its file or a section of its file:
`{"t": 4, "ref": "clips@video:12", "in": 12.0, "out": 22.5, "speed": 1}`. `in` and `out` are seconds of
the source file, and default to the section's own range. The film morphs into the clip's frame at
`in`, plays the clip, and morphs on from its frame at `out`, so a film passes from a still into a
moving shot and back. Looks, lanes, masks and finish apply to a clip's frames as they do to a morph.

Looks on a clip. `"temporal"` says how a look's method meets a playing clip. `frame` runs it on every
frame, which suits methods that give the same answer for nearly the same picture. `keys` runs it on
frames of the clip every `key_every` seconds (0.5 by default) and carries each result to the
frames between along the clip's movement, mixed from the key before and the key after, so a method
that draws from noise does not flicker and runs a few times a second of clip instead of on every
frame. `auto`, the default, takes `keys` for the generative methods and `frame` for the others.

Sound. A clip is heard while it plays, at its speed, unless its key says `"mute": true`; `"gain"` is
in decibels. The `audio` list places other sounds: `{"t": 0, "ref": "field@audio:212", "gain": -6,
"fade": [0.5, 2.0], "in": 0, "out": 30}`, a sound or the track of a video, from `t`, faded in and out.
`"sound": false` renders a silent film.

The amount lane. `amount` sets how much of the look's method covers the morph, 0 to 1, so a texture
can be laid over the film progressively while the finish lanes keep their own course.

Masks over time. A geometric layer moves through its numbers. A layer read by a model (the subject,
words, depth) is read on the key pictures, once each, and blended along the morph, so it moves with
what it found and no model runs on a frame. A field (light, colour, edges, noise) is read on each
frame of the morph, so it follows the picture as it changes. While a clip plays, a layer read by a
model is read on frames of the clip every half second and carried to the frames between them along
the clip's own movement (optical flow), so the mask stays on a subject that moves. Between two looks whose masks share
their layers, the masks move by their numbers (a spot's place, a coverage, the feather). Between two
looks whose masks differ in their layers (the subject, then grown noise), both are built on each
frame and cross-fade with the look's ease, while lanes keep moving the numbers of either.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

import numpy as np

import threading

from . import mask as mk
from . import spaces
from .inputs import load_frame, resolve, to_image

try:
    from .. import modality as _modality
    from .. import video as _video
except ImportError:                          # pragma: no cover
    import modality as _modality
    import video as _video

EASES = ("linear", "smooth", "in", "out", "hold")
NUMBER_PATHS = ("zoom", "amount")
FILM_REF = "film:"
DEPTH = 2                                    # a film in a film in a film, and no deeper
TRACK_EVERY = 0.5                            # seconds of a clip between two readings of a model's mask
TRACK_SIDE = 192                             # the long side the movement between frames is read at
TEMPORAL = ("auto", "frame", "keys")
KEY_MOVE = 0.06                              # a key interval is halved while the clip moves more than this
KEY_LEAST = 0.125                            # share of the frame across it, down to this many seconds
REFINE_KEYS = ("feather", "grow", "follow", "ragged", "amount")
# The numbers of each geometric layer a lane can move, in the order its source line writes them.
MOVING = {"spot": ("x", "y", "r"), "gradient": ("angle",), "band": ("y", "height"),
          "stripes": ("count", "angle"), "depth band": ("near", "far"), "noise": ("size",)}
SIDES = {"top": 0.0, "right": 90.0, "bottom": 180.0, "left": 270.0}


# ─────────────────────────── the spec ───────────────────────────

def is_film(ref) -> bool:
    return str(ref).startswith(FILM_REF)


def is_clip(ref) -> bool:
    """A video on the picture lane: a video dataset's handle, a video file or a section of one."""
    ref = str(ref).strip().strip('"')
    return not is_film(ref) and _modality.VIDEO in (_modality.of_handle(ref), _modality.of_path(ref))


def plays(ref) -> bool:
    """Whether a key plays for a while (a kept film or a clip) instead of being passed through."""
    return is_film(ref) or is_clip(ref)


def span(key: dict) -> float:
    """How long a key plays: nothing for a picture, (out - in) / speed for a film or a clip."""
    if not plays(key.get("ref", "")) or key.get("out") is None:
        return 0.0
    return (float(key["out"]) - float(key["in"])) / float(key["speed"])


def clip_source(ref) -> tuple[str, float | None, float | None]:
    """(file, start, end) of a clip's reference: the video file, and its section's range when the
    reference names a section."""
    from ..ops import archive

    item = archive.resolve_item(str(ref).strip().strip('"'), "person")
    file, t0, t1 = _modality.split_ref(item.path)
    if not os.path.isfile(file):
        raise ValueError(f"{ref}'s file is gone: {file}")
    return file, t0, t1


_probed: dict = {}


def clip_info(file: str) -> dict:
    """A video's length, rate, size and whether it has sound, read once per file and change."""
    key = (file, os.path.getmtime(file))
    if key not in _probed:
        _probed[key] = _video.probe(file)
    return _probed[key]


def _clip_key(p: dict) -> None:
    """A clip key's in, out and speed filled in and checked, in seconds of its file."""
    file, t0, t1 = clip_source(p["ref"])
    length = float(clip_info(file)["duration"] or 0.0)
    name = os.path.basename(file)
    p["in"] = float(t0 or 0.0) if p.get("in") in (None, "") else float(p["in"])
    p["out"] = float(t1 if t1 is not None else length) if p.get("out") in (None, "") else float(p["out"])
    p["out"] = min(p["out"], length) if length and p["out"] <= length + 0.25 else p["out"]
    p["speed"] = float(p.get("speed") or 1.0)
    if not 0.1 <= p["speed"] <= 8:
        raise ValueError("A clip plays at 0.1 to 8 times its speed.")
    if not 0 <= p["in"] < p["out"] <= length + 1e-6:
        raise ValueError(f"The clip {name} runs 0 to {length:.2f} s: its in and out lie in that range, "
                         f"in before out.")


def held_key(p: dict, depth: int = 0) -> None:
    """The in, out and speed of a key that plays, a kept film or a clip, filled in and checked."""
    if is_film(p["ref"]):
        _film_key(p, depth)
    else:
        _clip_key(p)


def _film_key(p: dict, depth: int) -> None:
    """A film key's in, out and speed filled in and checked."""
    name = str(p["ref"])[len(FILM_REF):].strip()
    if depth >= DEPTH:
        raise ValueError(f"The film {name!r} sits too deep: films nest {DEPTH} deep at most.")
    inner = normalised(load(name), depth + 1)
    p["in"] = float(p.get("in") or 0.0)
    p["out"] = float(inner["length"] if p.get("out") in (None, "") else p["out"])
    p["speed"] = float(p.get("speed") or 1.0)
    if not 0.1 <= p["speed"] <= 8:
        raise ValueError("A film plays inside another at 0.1 to 8 times its speed.")
    if not 0 <= p["in"] < p["out"] <= inner["length"] + 1e-6:
        raise ValueError(f"The film {name} runs 0 to {inner['length']:.2f} s: its in and out lie in that "
                         f"range, in before out.")


def normalised(spec: dict, depth: int = 0) -> dict:
    """The film with its defaults filled in and its keys in time order, or a ValueError naming what is
    wrong."""
    f = dict(spec or {})
    f.setdefault("name", "film")
    f["fps"] = int(f.get("fps") or 30)
    f["size"] = int(f.get("size") or 768)
    f.setdefault("aspect", "auto")
    f.setdefault("space", "DINOv2")
    f["loop"] = bool(f.get("loop", True))
    f["even"] = bool(f.get("even", True))
    if f["space"] not in spaces.SPACES:
        raise ValueError(f"A film's space is one of {', '.join(spaces.SPACES)}.")
    if not 6 <= f["fps"] <= 60:
        raise ValueError("A film runs at 6 to 60 frames a second.")
    if not 256 <= f["size"] <= 2048:
        raise ValueError("A film's long side is 256 to 2048 pixels; Real-ESRGAN enlarges past that at render.")
    pics = sorted((dict(p) for p in f.get("pictures") or []), key=lambda p: float(p["t"]))
    if len(pics) < 2:
        raise ValueError("A film needs at least two pictures on its picture lane.")
    for p in pics:
        p["t"] = float(p["t"])
        p.setdefault("ease", "linear")
        if p["ease"] not in EASES:
            raise ValueError(f"An ease is one of {', '.join(EASES)}, got {p['ease']!r}.")
        if is_film(p["ref"]):
            if slug(str(p["ref"])[len(FILM_REF):]) == slug(f["name"]):
                raise ValueError("A film cannot hold itself.")
            _film_key(p, depth)
        elif is_clip(p["ref"]):
            _clip_key(p)
    if len({p["t"] for p in pics}) != len(pics):
        raise ValueError("Two pictures share a time on the picture lane.")
    for a, b in zip(pics, pics[1:]):
        if a["t"] + span(a) >= b["t"]:
            what = "film" if is_film(a["ref"]) else "clip"
            raise ValueError(f"The {what} at {a['t']:.2f} s plays until {a['t'] + span(a):.2f} s, past the next "
                             f"key at {b['t']:.2f} s. Move that key later or shorten the {what}'s out.")
    f["pictures"] = pics
    last = pics[-1]["t"] + span(pics[-1])
    length = float(f.get("length") or 0)
    if f["loop"]:
        # The run back to the first picture takes as long as the average passage, unless given.
        length = length if length > last else last + (pics[-1]["t"] - pics[0]["t"]) / (len(pics) - 1)
    else:
        length = last
    f["length"] = length
    f["looks"] = sorted((dict(k) for k in f.get("looks") or []), key=lambda k: float(k["t"]))
    for k in f["looks"]:
        k["t"] = float(k["t"])
        k.setdefault("ease", "smooth")
        k.setdefault("settings", {})
    lanes = {}
    for path, keys in (f.get("lanes") or {}).items():
        check_path(path)
        keys = sorted((dict(k) for k in keys), key=lambda k: float(k["t"]))
        for k in keys:
            k["t"], k["v"] = float(k["t"]), float(k["v"])
            k.setdefault("ease", "linear")
            if k["ease"] not in EASES:
                raise ValueError(f"An ease is one of {', '.join(EASES)}, got {k['ease']!r}.")
        if keys:
            lanes[path] = keys
    f["lanes"] = lanes
    f["temporal"] = str(f.get("temporal") or "auto")
    if f["temporal"] not in TEMPORAL:
        raise ValueError(f"A film's temporal is one of {', '.join(TEMPORAL)}.")
    f["key_every"] = float(f.get("key_every") or TRACK_EVERY)
    if not 0.1 <= f["key_every"] <= 5:
        raise ValueError("A look is read on a clip every 0.1 to 5 seconds.")
    f["sound"] = bool(f.get("sound", True))
    sounds = sorted((dict(a) for a in f.get("audio") or []), key=lambda a: float(a.get("t") or 0))
    for a in sounds:
        a["t"] = float(a.get("t") or 0.0)
        a["gain"] = float(a.get("gain") or 0.0)
        fade = list(a.get("fade") or [0.0, 0.0])
        a["fade"] = [max(0.0, float(x)) for x in (fade + [0.0, 0.0])[:2]]
        if not str(a.get("ref") or "").strip():
            raise ValueError("A sound on the audio lane names its file or handle with ref.")
        if not -60 <= a["gain"] <= 24:
            raise ValueError("A sound's gain is -60 to 24 decibels.")
    if sounds:
        f["audio"] = sounds
    else:
        f.pop("audio", None)
    return f


def check_path(path: str) -> None:
    ok = (path in NUMBER_PATHS or re.fullmatch(r"(knobs|finish)\.[a-z_]+", path)
          or re.fullmatch(r"weights\.[123]", path)
          or re.fullmatch(r"mask\.(" + "|".join(REFINE_KEYS) + ")", path)
          or re.fullmatch(r"mask\.[1-6]\.[a-z]+", path))
    if not ok:
        raise ValueError(f"No lane called {path!r}. Lanes key knobs.<key>, finish.<key>, weights.<n>, "
                         f"zoom, mask.<{'|'.join(REFINE_KEYS)}>, mask.<n>.coverage or a geometric "
                         f"layer's number, such as mask.1.x.")


# ─────────────────────────── curves ───────────────────────────

def ease(name: str, s: float) -> float:
    s = min(max(float(s), 0.0), 1.0)
    if name == "smooth":
        return s * s * (3 - 2 * s)
    if name == "in":
        return s * s
    if name == "out":
        return 1 - (1 - s) * (1 - s)
    if name == "hold":
        return 0.0
    return s


def lane_value(keys: list, t: float) -> float:
    """A lane's number at time t: held before its first key and after its last."""
    if t <= keys[0]["t"]:
        return keys[0]["v"]
    for a, b in zip(keys, keys[1:]):
        if t < b["t"]:
            s = ease(a["ease"], (t - a["t"]) / max(b["t"] - a["t"], 1e-9))
            return a["v"] + (b["v"] - a["v"]) * s
    return keys[-1]["v"]


def points(film: dict) -> list[dict]:
    """The stops the morph passes through, in time order: one per picture, two per film (its frame at
    `in` and its frame at `out`), each with the ease of the passage that leaves it. Between a film's two
    stops the film plays."""
    out = []
    for i, p in enumerate(film["pictures"]):
        if plays(p["ref"]):
            out.append({"t": p["t"], "ease": "linear", "key": i, "at": p["in"], "plays": True})
            out.append({"t": p["t"] + span(p), "ease": p["ease"], "key": i, "at": p["out"], "plays": False})
        else:
            out.append({"t": p["t"], "ease": p["ease"], "key": i, "at": None, "plays": False})
    return out


def passage(film: dict, t: float) -> tuple[int, float]:
    """Which passage of the picture lane time t falls in, and how far along it (eased): (k, s), the film
    going from stop k to stop k+1 (wrapping to the first on a loop). Stops are the pictures, and a film
    on the lane gives two."""
    pics = points(film)
    n = len(pics)
    t = t % film["length"] if film["loop"] else min(max(t, pics[0]["t"]), pics[-1]["t"])
    if t < pics[0]["t"]:                     # a loop's run back wraps across zero
        t += film["length"]
    for k in range(n):
        start = pics[k]["t"]
        end = pics[k + 1]["t"] if k + 1 < n else (pics[0]["t"] + film["length"] if film["loop"] else start)
        if t < end or k == n - 1:
            span = max(end - start, 1e-9)
            return k, ease(pics[k]["ease"], (t - start) / span) if end > start else 0.0
    return n - 1, 0.0


# ─────────────────────────── the look at a moment ───────────────────────────

def _mix(a, b, s):
    """Every number of `a` moved toward `b` by s; anything else stays `a`'s (it changes at b's key)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * s
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: (_mix(a[k], b[k], s) if k in b else a[k]) for k in a} | {k: v for k, v in b.items() if k not in a}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_mix(x, y, s) for x, y in zip(a, b)]
    return a


def layer_numbers(line: str):
    """(kind, names, values) of a geometric layer's numbers, or None for any other layer."""
    raw = str(line).strip()
    raw = mk.PRESET[raw].source if raw in mk.PRESET else raw
    name, _, arg = raw.partition(":")
    name = name.strip().lower()
    if name not in MOVING:
        return None
    if name == "gradient":
        side = arg.strip() or "left"
        return name, MOVING[name], [SIDES[side] if side in SIDES else float(side)]
    if name == "noise":
        kind, _, size = arg.partition(",")
        return name, MOVING[name], [float(size or 40)], kind.strip()
    return name, MOVING[name], [float(x) for x in arg.split(",")]


def layer_line(kind: str, values, extra: str = "") -> str:
    if kind == "noise":
        return f"noise:{extra},{values[0]:g}"
    return f"{kind}:" + ",".join(f"{v:.4g}" for v in values)


def look_at(film: dict, t: float) -> dict:
    """The step's settings at time t: the snapshots interpolated, then every lane laid over them."""
    looks = film["looks"]
    if not looks:
        s = {"method": "none"}
    elif t <= looks[0]["t"]:
        s = json.loads(json.dumps(looks[0]["settings"]))
    else:
        s = json.loads(json.dumps(looks[-1]["settings"]))
        for a, b in zip(looks, looks[1:]):
            if t < b["t"]:
                f = ease(a["ease"], (t - a["t"]) / max(b["t"] - a["t"], 1e-9))
                s = _mix(json.loads(json.dumps(a["settings"])), b["settings"], f)
                ma, mb = _mask_dict(a["settings"].get("mask")), _mask_dict(b["settings"].get("mask"))
                if _mask_form(ma) != _mask_form(mb):
                    # Two masks of different layers: each is built on the frame and the two cross-fade.
                    s["mask"], s["mask_to"], s["mask_mix"] = ma, mb, f
                elif ma["layers"]:
                    s["mask"] = _mask_between(ma, mb, f)
                break
    for path, keys in film["lanes"].items():
        v = lane_value(keys, t)
        head, _, rest = path.partition(".")
        if path in NUMBER_PATHS:
            s[path] = v
        elif head in ("knobs", "finish"):
            s.setdefault(head, {})[rest] = v
        elif head == "weights":
            n = len([m for m in (s.get("materials") or []) if m])
            w = list(s.get("weights") or [1.0] * n)
            i = int(rest) - 1
            if i < len(w):
                w[i] = v
            s["weights"] = w
        elif head == "mask":
            for which in ("mask", "mask_to"):
                if which == "mask" or s.get("mask_to") is not None:
                    s[which] = _mask_lane(_mask_dict(s.get(which)), rest, v)
    return s


def _mask_dict(m) -> dict:
    return {k: (list(x) if isinstance(x, list) else x) for k, x in mk.MaskSpec.of(m).__dict__.items()}


def _mask_form(m: dict):
    """What a mask is made of, its numbers aside: two masks of one form move from one to the other by
    their numbers, two of different forms cross-fade."""
    return ([str(layer_numbers(x)[0]) if layer_numbers(x) else str(x).strip() for x in m["layers"]],
            list(m["combine"]), bool(m["invert"]))


def _mask_between(a: dict, b: dict, f: float) -> dict:
    """Two masks of one form at f between them: refinements, coverages and the numbers of geometric
    layers (a spot's place, a gradient's angle) all moved."""
    def cover(m):
        cov = list(m["coverage"]) + [None] * (len(m["layers"]) - len(m["coverage"]))
        return [mk._coverage_of(x, c) for x, c in zip(m["layers"], cov)]

    out = _mix({**a, "coverage": cover(a)}, {**b, "coverage": cover(b)}, f)
    layers = []
    for x, y in zip(a["layers"], b["layers"]):
        ga, gb = layer_numbers(x), layer_numbers(y)
        if ga and gb and len(ga[2]) == len(gb[2]):
            layers.append(layer_line(ga[0], [p + (q - p) * f for p, q in zip(ga[2], gb[2])],
                                     ga[3] if len(ga) > 3 else ""))
        else:
            layers.append(x)
    out["layers"] = layers
    out["seed"] = a["seed"]                  # a seed is a name for noise, not a number to move
    return out


def _mask_lane(m: dict, rest: str, v: float) -> dict:
    """A mask with one lane's number laid over it: a refinement, a layer's coverage or a geometric
    layer's number. A lane naming a layer the mask lacks leaves it as it is."""
    if rest in REFINE_KEYS:
        m[rest] = v
        return m
    idx, _, num = rest.partition(".")
    i = int(idx) - 1
    layers = list(m.get("layers") or [])
    if i >= len(layers):
        return m
    if num == "coverage":
        cov = list(m.get("coverage") or [])
        cov += [None] * (len(layers) - len(cov))
        cov[i] = v
        m["coverage"] = cov
        return m
    got = layer_numbers(layers[i])
    if got is not None and num in got[1]:
        values = list(got[2])
        values[got[1].index(num)] = v
        layers[i] = layer_line(got[0], values, got[3] if len(got) > 3 else "")
        m["layers"] = layers
    return m


# ─────────────────────────── the frame ───────────────────────────

class Clip:
    """A video as a source of frames for a film: `frame(t)` is the frame at t seconds of the file,
    covering W x H, floats in 0..1. Frames asked for in order are read from one running decoder, so a
    render decodes each clip once; a jump (scrubbing) starts the decoder again at that time."""

    def __init__(self, file: str, W: int, H: int):
        self.file = file
        self.W, self.H = int(W), int(H)
        self.info = clip_info(file)
        self.fps = min(max(float(self.info["fps"] or 30.0), 1.0), 60.0)
        k = max(self.W / self.info["width"], self.H / self.info["height"])
        long = max(self.info["width"], self.info["height"])
        self.side = None if k >= 1 else int(np.ceil(long * k)) + 2
        self.spec = {"clip": file, "mtime": os.path.getmtime(file)}
        self._gen, self._next, self._last = None, 0, None
        self._lock = threading.Lock()

    def _close(self):
        if self._gen is not None:
            self._gen.close()
            self._gen = None

    def frame(self, t: float):
        i = max(0, int(round(float(t) * self.fps)))
        with self._lock:
            if self._last is not None and self._last[0] == i:
                return self._last[1]
            if self._gen is None or i < self._next or i - self._next > self.fps * 2:
                self._close()
                self._gen = _video.frames(self.file, self.fps, side=self.side, t0=i / self.fps, info=self.info)
                self._next = i
            got = None
            while self._next <= i:
                try:
                    _, got = next(self._gen)
                except StopIteration:
                    self._gen = None
                    break
                self._next += 1
            if got is None:
                # Past the last frame the file holds: the last one read stands in.
                if self._last is not None:
                    return self._last[1]
                got = _video.frame_at(self.file, max(0.0, float(self.info["duration"] or 0.0) - 0.5),
                                      side=self.side, info=self.info)
                if got is None:
                    raise ValueError(f"No frame could be read from {os.path.basename(self.file)}.")
            from PIL import Image

            from . import pixels as cx

            out = np.asarray(cx.cover(Image.fromarray(got), self.W, self.H), np.float32) / 255.0
            self._last = (i, out)
            return out

    def still(self, t: float):
        """The frame at t read on its own, which leaves the running decoder where it is: for the
        frames a mask is read on, ahead of and behind the one being drawn."""
        from PIL import Image

        from . import pixels as cx

        last = max(0.0, float(self.info["duration"] or 0.0) - 1.0 / self.fps)
        got = _video.frame_at(self.file, min(max(0.0, float(t)), last), side=self.side, info=self.info)
        if got is None:
            return self.frame(t)
        return np.asarray(cx.cover(Image.fromarray(got), self.W, self.H), np.float32) / 255.0

    def __del__(self):                       # pragma: no cover
        try:
            self._close()
        except Exception:
            pass


class Film:
    """A film made ready to draw: its token space, the grids of its stops and their mask maps. A film
    held on the picture lane is made ready too, at this film's size and shape, and plays from it."""

    def __init__(self, spec: dict, size: int | None = None, depth: int = 0, aspect: str | None = None):
        self.spec = normalised({**spec, "aspect": aspect} if aspect else spec, depth)
        self.depth = depth
        self.size = int(size or self.spec["size"])
        self.space_name = self.spec["space"]
        self.sp = spaces.space(self.space_name)
        keys = self.spec["pictures"]
        self.paths = ["" if plays(p["ref"]) else resolve(p["ref"]) for p in keys]
        fw, fh = self._aspect()
        long = max(1, round(self.size / 16))
        self.rows, self.cols = self.sp.grid(fw, fh, long)
        self.W, self.H = self.cols * 16, self.rows * 16
        self.inner: dict = {}                # key index -> the Film it holds
        for i, p in enumerate(keys):
            if is_film(p["ref"]):
                f = Film(load(str(p["ref"])[len(FILM_REF):]), self.size, depth + 1, aspect=f"{fw}:{fh}")
                if f.spec["even"]:
                    f.even()
                self.inner[i] = f
            elif is_clip(p["ref"]):
                self.inner[i] = Clip(clip_source(p["ref"])[0], self.W, self.H)
        self.stops = points(self.spec)
        self.stills = [self._still(q) for q in self.stops]
        from .tokens import encode

        with spaces.lock():
            self.grids = [encode(self.sp, self.space_name, self._key(q, self.W, self.H), c, (self.rows, self.cols))
                          for q, c in zip(self.stops, self.stills)]
        self.maps: dict = {}
        self.table = None                    # even pacing: per passage, eased s -> evened s

    def _key(self, q: dict, w: int, h: int) -> str:
        """The cache key of a stop's frame at a size."""
        p = self.spec["pictures"][q["key"]]
        if q["at"] is None:
            return f"{self.paths[q['key']]}@{w}x{h}"
        inner = self.inner[q["key"]]
        sig = hashlib.blake2b(json.dumps(inner.spec, sort_keys=True).encode(), digest_size=6).hexdigest()
        return f"{p['ref']}#{sig}@{q['at']:.3f}@{w}x{h}"

    def _fit(self, c, w: int, h: int):
        if c.shape[:2] == (h, w):
            return c
        from PIL import Image

        return np.asarray(to_image(c).resize((w, h), Image.LANCZOS), np.float32) / 255.0

    def _still(self, q: dict, w: int | None = None, h: int | None = None):
        """A stop's frame: its picture, or the held film's frame at the stop's time."""
        w, h = w or self.W, h or self.H
        if q["at"] is None:
            return load_frame(self.paths[q["key"]], w, h)
        return self._fit(self.inner[q["key"]].frame(q["at"]), w, h)

    def _aspect(self):
        a = str(self.spec["aspect"])
        if a == "auto":
            return _first_size(self.spec, self.depth)
        w, _, h = a.partition(":")
        return float(w), float(h)

    def tokens(self, k: int, s: float):
        """The Catmull-Rom curve through the grids at passage k, s along it."""
        n, g = len(self.grids), self.grids
        if self.spec["loop"]:
            p0, p1, p2, p3 = (g[(k + d) % n] for d in (-1, 0, 1, 2))
        else:
            p0, p1, p2, p3 = (g[min(max(k + d, 0), n - 1)] for d in (-1, 0, 1, 2))
        z = 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s
                   + (3 * p1 - p0 - 3 * p2 + p3) * s ** 3)
        if hasattr(self.sp, "settle"):
            z = self.sp.settle(z, [p1, p2], [1 - s, s])
        return z

    def morph(self, t: float):
        """The picture at time t: the morph decoded, or the held film playing, [H, W, 3] floats in 0..1."""
        k, s = passage(self.spec, t)
        q = self.stops[k]
        if q["plays"]:
            nxt = self.stops[k + 1]
            return self._fit(self.inner[q["key"]].frame(q["at"] + (nxt["at"] - q["at"]) * s), self.W, self.H)
        if self.table is not None:
            s = float(np.interp(s, *self.table[k]))
        with spaces.lock():
            z = self.tokens(k, s)
            out = self.sp.decode(self.sp.shell(z.reshape(1, self.rows * self.cols, -1)), (self.rows, self.cols))[0]
        # A held film plays its own frames, which the decoder draws a little differently from the grid the
        # morph reaches. What the decoder misses at that stop is added in as the morph nears it, so the
        # morph lands on the very frame the film then plays, and leaves from it the same way.
        n = len(self.stops)
        j = (k + 1) % n if self.spec["loop"] else min(k + 1, n - 1)
        if self.stops[j]["at"] is not None and j != k:
            out = out + s * self._residual(j)
        if q["at"] is not None:
            out = out + (1 - s) * self._residual(k)
        return np.clip(out, 0, 1) if (self.stops[j]["at"] is not None or q["at"] is not None) else out

    def _residual(self, i: int):
        """A film stop's frame less the decoder's drawing of its grid."""
        if not hasattr(self, "_residuals"):
            self._residuals = {}
        if i not in self._residuals:
            with spaces.lock():
                drawn = self.sp.decode(self.sp.shell(self.grids[i].reshape(1, self.rows * self.cols, -1)),
                                       (self.rows, self.cols))[0]
            self._residuals[i] = np.asarray(self.stills[i], np.float32) - np.asarray(drawn, np.float32)
        return self._residuals[i]

    def asker(self, t: float):
        """Answers a mask's question about the frame at t from the two stops it lies between."""
        k, s = passage(self.spec, t)
        n = len(self.stops)
        j = (k + 1) % n if self.spec["loop"] else min(k + 1, n - 1)

        def ask(kind, arg=""):
            if self.stops[k]["plays"] and isinstance(self.inner.get(self.stops[k]["key"]), Clip):
                got = self._tracked(k, s, kind, arg)
                if got is not None:
                    return got
            a, b = self._map(k, kind, arg), self._map(j, kind, arg)
            return (1 - s) * a + s * b

        return ask

    def _tracked(self, k: int, s: float, kind, arg):
        """A model's map on the frame of a playing clip: read on the clip every TRACK_EVERY seconds,
        and each of the two readings around this frame moved onto it along the clip's movement. None
        when the movement cannot be read (no OpenCV), which leaves the two ends to cross-fade."""
        q = self.stops[k]
        clip = self.inner[q["key"]]
        try:
            tau, ta, tb, u = self._around(k, s, TRACK_EVERY)
            now = _grey(clip.frame(tau))
            return _mixed(_carry(*self._reading(q["key"], clip, ta, kind, arg), now),
                          _carry(*self._reading(q["key"], clip, tb, kind, arg), now), u)
        except ImportError:
            return None

    def _around(self, k: int, s: float, every: float):
        """Where a playing clip is at s along its passage, the two key times around that moment and how
        far between them it lies: (tau, before, after, u). Keys sit `every` seconds apart, and closer
        where the clip moves fast: an interval across which the picture moves more than KEY_MOVE of
        the frame is halved, down to KEY_LEAST seconds, since a result carried further than that no
        longer lands where it belongs."""
        q, nxt = self.stops[k], self.stops[k + 1]
        clip = self.inner[q["key"]]
        lo, hi = float(q["at"]), float(nxt["at"])
        tau = lo + (hi - lo) * s
        grids = self.__dict__.setdefault("_key_times", {})
        name = (q["key"], lo, hi, round(every, 3))
        if name not in grids:
            n = max(1, int(np.ceil((hi - lo) / every - 1e-9)))
            times = [lo + (hi - lo) * j / n for j in range(n + 1)]
            greys = {x: _grey(clip.still(x)) for x in times}
            for _ in range(3):
                out = [times[0]]
                for a, b in zip(times, times[1:]):
                    if b - a > KEY_LEAST * 1.5 and _moved(greys[a], greys[b]) > KEY_MOVE:
                        mid = (a + b) / 2
                        greys[mid] = _grey(clip.still(mid))
                        out.append(mid)
                    out.append(b)
                if len(out) == len(times):
                    break
                times = out
            grids[name] = times
        times = grids[name]
        i = min(max(int(np.searchsorted(times, tau, side="right")) - 1, 0), len(times) - 2)
        ta, tb = times[i], times[i + 1]
        return tau, ta, tb, min(max((tau - ta) / max(tb - ta, 1e-9), 0.0), 1.0)

    def _reading(self, key: int, clip: "Clip", at: float, kind, arg):
        """(the model's map, the grey frame it was read on) at a moment of a clip, kept."""
        name = ("clip", key, round(at, 3), kind, str(arg))
        if name not in self.maps:
            from .vision import vision

            frame = clip.still(at)
            img = to_image(frame)
            got = np.asarray(vision().ask(f"{clip.file}@{at:.3f}@{self.W}x{self.H}", lambda: img, kind, arg),
                             np.float32)
            self.maps[name] = (got, _grey(frame))
        return self.maps[name]

    def _map(self, i, kind, arg):
        key = (i, kind, str(arg))
        if key not in self.maps:
            from .vision import vision

            img = to_image(self.stills[i])
            self.maps[key] = np.asarray(vision().ask(self._key(self.stops[i], self.W, self.H), lambda: img, kind, arg),
                                        np.float32)
        return self.maps[key]

    def frame(self, t: float):
        """The finished frame at time t: the morph, then the look's step, then its finish."""
        from .pipeline import step_on

        c = self.morph(t)
        look = look_at(self.spec, t)
        k, s = passage(self.spec, t)
        if self.by_keys(k, look):
            got = self._propagated(k, s, c)
            if got is not None:
                return got
        out, _ = step_on(c, f"film:{id(self)}:{t:.4f}", look, asker=self.asker(t))
        return np.clip(out, 0, 1)

    def by_keys(self, k: int, look: dict) -> bool:
        """Whether the look at this moment is run on key frames of a playing clip and carried between
        them, instead of on every frame."""
        q = self.stops[k]
        if not q["plays"] or not isinstance(self.inner.get(q["key"]), Clip):
            return False
        method = str(look.get("method") or "none")
        how = self.spec["temporal"]
        return method != "none" and (how == "keys" or (how == "auto" and method.startswith("generative.")))

    def _propagated(self, k: int, s: float, c):
        """The frame of a playing clip with its look carried from the key frames around it. The look
        runs on the clip's frame at each key time. Each result is moved onto this frame along the
        clip's movement, from the key before and the key after, and the two are mixed by how near
        each is. Something that enters the shot between two keys comes in with the later key's look.
        Adding the look's change to the frame itself was tried first, and let such a thing show
        through unstyled (a car crossing a quilted street). None when movement cannot be read."""
        clip = self.inner[self.stops[k]["key"]]
        try:
            tau, ta, tb, u = self._around(k, s, self.spec["key_every"])
            now = _grey(c)
            return np.clip(_mixed(_carry(*self._styled(k, clip, ta), now),
                                  _carry(*self._styled(k, clip, tb), now), u), 0, 1)
        except ImportError:
            return None

    def _styled(self, k: int, clip: "Clip", at: float):
        """(the look's result, the grey frame it was run on) at a key time of a clip, kept while the look there
        stays the same. The last few are kept: a render needs two at a time."""
        from .pipeline import step_on

        q, nxt = self.stops[k], self.stops[k + 1]
        lo, hi = float(q["at"]), float(nxt["at"])
        t = q["t"] + (nxt["t"] - q["t"]) * ((at - lo) / max(hi - lo, 1e-9))
        look = look_at(self.spec, t)
        sig = hashlib.blake2b(json.dumps(look, sort_keys=True, default=str).encode(), digest_size=8).hexdigest()
        name = (q["key"], round(at, 3), sig)
        kept = self.__dict__.setdefault("_keys_styled", {})
        if name not in kept:
            frame = clip.still(at)
            out, _ = step_on(frame, f"film:{id(self)}:key:{q['key']}:{at:.3f}:{sig}", look, asker=self.asker(t))
            kept[name] = (np.clip(out, 0, 1).astype(np.float32), _grey(frame))
            while len(kept) > 6:
                kept.pop(next(iter(kept)))
        return kept[name]

    def moves(self, samples: int = 12) -> list:
        """How much the picture changes between samples along each passage, read on a half-size grid:
        one array per passage leaving a stop, None where a held film plays."""
        rows, cols = max(1, self.rows // 2), max(1, self.cols // 2)
        from .tokens import encode

        w, h = cols * 16, rows * 16
        with spaces.lock():
            small = [encode(self.sp, self.space_name, self._key(q, w, h),
                            load_frame(self.paths[q["key"]], w, h) if q["at"] is None else self._fit(c, w, h),
                            (rows, cols)) for q, c in zip(self.stops, self.stills)]
        full, self.grids = self.grids, small
        out = []
        try:
            n = len(self.stops) if self.spec["loop"] else len(self.stops) - 1
            for k in range(n):
                if self.stops[k]["plays"]:
                    out.append(None)
                    continue
                ss = np.linspace(0, 1, samples + 1)
                with spaces.lock():
                    looks = [self.sp.decode(self.sp.shell(self.tokens(k, float(s)).reshape(1, rows * cols, -1)),
                                            (rows, cols))[0] for s in ss]
                out.append(np.array([0.0] + [float(np.abs(b - a).mean()) for a, b in zip(looks, looks[1:])]))
        finally:
            self.grids = full
        return out

    def changes(self, samples: int = 12) -> list[float]:
        """How much the picture changes along each passage in all (0 where a held film plays)."""
        return [0.0 if m is None else float(m.sum()) for m in self.moves(samples)]

    def even(self, samples: int = 12):
        """Even pacing: each passage re-timed so the picture changes by the same amount every frame,
        read on a half-size grid. A film playing on the lane keeps its own pace."""
        same = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        ss = np.linspace(0, 1, samples + 1)
        table = []
        for moved in self.moves(samples):
            if moved is None:
                table.append(same)
                continue
            gone = np.cumsum(moved)
            gone = gone / gone[-1] if gone[-1] > 0 else ss
            table.append((gone, ss))         # np.interp(s, gone, ss): equal change -> where on the curve
        table += [same] * (len(self.stops) - len(table))
        self.table = table
        return table


def _grey(frame) -> np.ndarray:
    """A frame small and grey, for reading movement: uint8, TRACK_SIDE on its long side."""
    from PIL import Image

    h, w = frame.shape[:2]
    k = TRACK_SIDE / max(h, w)
    size = (max(8, round(w * k)), max(8, round(h * k)))
    return np.asarray(to_image(frame).convert("L").resize(size, Image.BILINEAR))


def _moved(a: np.ndarray, b: np.ndarray) -> float:
    """How far the picture moves between two grey frames, as a share of the frame's long side: the
    middle of the flow's length over the frame."""
    import cv2

    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)
    return float(np.median(np.hypot(flow[..., 0], flow[..., 1]))) / max(a.shape)


def _mixed(a, b, u: float) -> np.ndarray:
    """Two carried results mixed by how near each key is and by how well each landed, place by place:
    where one key's frame does not match this frame once moved (something crossed, or turned), the
    other key answers there, so the mix does not show the two as ghosts of each other."""
    (ra, fa), (rb, fb) = a, b
    wa, wb = (1 - u) * fa + 1e-3, u * fb + 1e-3
    if ra.ndim == 3:
        wa, wb = wa[..., None], wb[..., None]
    return (wa * ra + wb * rb) / (wa + wb)


def _carry(reading: np.ndarray, was: np.ndarray, now: np.ndarray):
    """(the reading moved onto `now`, how well it landed): the second is 0..1 at each place, from how
    closely the frame it was read on matches this frame once moved the same way."""
    import cv2

    H, W = reading.shape[:2]
    if was.shape != now.shape or np.array_equal(was, now):
        return reading, np.ones((H, W), np.float32)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(now, was, None)
    h, w = now.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    landed = cv2.remap(was, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    miss = cv2.GaussianBlur(np.abs(landed.astype(np.float32) - now.astype(np.float32)), (0, 0), 2.0)
    fit = cv2.resize(np.exp(-miss / 10.0), (W, H), interpolation=cv2.INTER_LINEAR)
    fx = cv2.resize(flow[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * (W / w)
    fy = cv2.resize(flow[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * (H / h)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = cv2.remap(np.ascontiguousarray(reading, np.float32), xs + fx, ys + fy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out, fit.astype(np.float32)


def _carried(reading: np.ndarray, was: np.ndarray, now: np.ndarray) -> np.ndarray:
    """A map read on the frame `was`, moved onto the frame `now`: each place of `now` takes the map's
    value from where that place was in `was`, found by dense optical flow between the two."""
    import cv2

    if was.shape != now.shape or np.array_equal(was, now):
        return reading
    # DIS flow. Farneback lost a patterned square that moved 15 px between two 192 px frames (it
    # read +3 px where the square had moved -15), and DIS read -14.4.
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(now, was, None)
    H, W = reading.shape[:2]
    h, w = now.shape
    fx = cv2.resize(flow[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * (W / w)
    fy = cv2.resize(flow[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * (H / h)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(np.ascontiguousarray(reading, np.float32), xs + fx, ys + fy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _first_size(spec: dict, depth: int = 0):
    """The size of a film's first picture, looking inside a held film when the lane starts with one."""
    from .inputs import original_size

    for p in spec["pictures"]:
        if is_clip(p["ref"]):
            info = clip_info(clip_source(p["ref"])[0])
            return float(info["width"]), float(info["height"])
        if not is_film(p["ref"]):
            return original_size(resolve(p["ref"]))
        if depth < DEPTH:
            inner = normalised(load(str(p["ref"])[len(FILM_REF):]), depth + 1)
            a = str(inner["aspect"])
            if a != "auto":
                w, _, h = a.partition(":")
                return float(w), float(h)
            return _first_size(inner, depth + 1)
    return 3.0, 2.0


def stills_of(spec: dict, depth: int = 0) -> list[str]:
    """Every picture a film draws on, the films it holds opened: what a render derives from."""
    out = []
    for p in normalised(spec, depth)["pictures"]:
        if is_film(p["ref"]):
            if depth < DEPTH:
                out += stills_of(load(str(p["ref"])[len(FILM_REF):]), depth + 1)
        else:
            out.append(p["ref"])
    return out


# ─────────────────────────── rendering ───────────────────────────

def ffmpeg_exe() -> str:
    """The ffmpeg imageio-ffmpeg ships, so a film renders without a system install."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def render(spec: dict, out_path: str, progress=None, size: int | None = None, fps: int | None = None,
           target: int | None = None, crf: int = 16) -> dict:
    """Every frame of the film written to an H.264 mp4 at `out_path`. `target` enlarges each frame to
    that long side with Real-ESRGAN. Reports progress(fraction, message) and stops cleanly when
    progress.cancelled() turns true. Returns what was made."""
    film = Film(spec, size)
    spec = film.spec
    fps = int(fps or spec["fps"])
    frames = max(1, round(spec["length"] * fps))
    if progress:
        progress(0.0, f"{len(film.stops)} stops, {frames} frames at {film.W} x {film.H}")
    if spec["even"]:
        film.even()
    out_w, out_h = film.W, film.H
    if target and target > max(film.W, film.H):
        k = target / max(film.W, film.H)
        out_w, out_h = int(round(film.W * k / 2)) * 2, int(round(film.H * k / 2)) * 2
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    made = 0
    try:
        for i in range(frames):
            if progress is not None and getattr(progress, "cancelled", lambda: False)():
                break
            image = film.frame(i / fps)
            if (out_w, out_h) != (film.W, film.H):
                from .pipeline import _enlarged

                pic = _enlarged(to_image(image), out_w, out_h)
            else:
                pic = to_image(image)
            writer.stdin.write(pic.tobytes())
            made += 1
            if progress and (i % max(1, frames // 200) == 0 or i == frames - 1):
                progress((i + 1) / frames, f"frame {i + 1} of {frames}")
    finally:
        writer.stdin.close()
        writer.wait()
    if made == 0:
        raise ValueError("The render stopped before its first frame.")
    heard = add_sound(spec, out_path, made / fps)
    return {"path": out_path, "frames": made, "of": frames, "fps": fps, "size": [out_w, out_h],
            "drawn": [film.W, film.H], "seconds": round(made / fps, 2), "pictures": stills_of(spec),
            "sounds": heard}


# ─────────────────────────── sound ───────────────────────────

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


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _sound_length(file: str) -> float:
    """How long a sound or a video's track lasts, from the header ffmpeg prints."""
    done = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", file], capture_output=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = done.stderr.decode("utf-8", "replace")
    if "Audio:" not in text:
        return 0.0
    m = _DURATION.search(text)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def sounds(spec: dict) -> list[dict]:
    """Every sound the film plays, in film time: {"file", "start", "length" (of the source), "speed",
    "at", "gain", "fade"}. The clips on the picture lane that have a track, then the audio lane."""
    f = normalised(spec)
    if not f["sound"]:
        return []
    out = []
    for p in f["pictures"]:
        if is_clip(p["ref"]) and not p.get("mute"):
            file = clip_source(p["ref"])[0]
            if clip_info(file)["has_audio"]:
                out.append({"file": file, "start": p["in"], "length": p["out"] - p["in"], "speed": p["speed"],
                            "at": p["t"], "gain": float(p.get("gain") or 0.0), "fade": [0.03, 0.03]})
    for a in f.get("audio") or []:
        from ..ops import archive

        item = archive.resolve_item(str(a["ref"]).strip().strip('"'), "person")
        file, t0, t1 = _modality.split_ref(item.path)
        if not os.path.isfile(file):
            raise ValueError(f"{a['ref']}'s file is gone: {file}")
        total = _sound_length(file)
        if total <= 0:
            raise ValueError(f"{os.path.basename(file)} holds no sound.")
        start = float(a["in"]) if a.get("in") not in (None, "") else float(t0 or 0.0)
        end = float(a["out"]) if a.get("out") not in (None, "") else float(t1 if t1 is not None else total)
        end = min(end, total)
        if not 0 <= start < end:
            raise ValueError(f"The sound {os.path.basename(file)} runs 0 to {total:.2f} s: its in and out "
                             f"lie in that range, in before out.")
        out.append({"file": file, "start": start, "length": end - start, "speed": 1.0, "at": a["t"],
                    "gain": a["gain"], "fade": a["fade"]})
    return out


def add_sound(spec: dict, video_path: str, seconds: float) -> int:
    """The film's sounds mixed and laid under the rendered video, which keeps its picture as it is.
    Returns how many sounds are heard. A film with none is left silent."""
    plan = [s for s in sounds(spec) if s["at"] < seconds]
    if not plan:
        return 0
    import tempfile

    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", video_path]
    chains, pads = [], []
    for i, s in enumerate(plan, 1):
        cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{s['length']:.3f}", "-i", s["file"]]
        played = min(s["length"] / s["speed"], seconds - s["at"])
        fin, fout = (min(x, played / 2) for x in s["fade"])
        chain = (f"[{i}:a]asetpts=PTS-STARTPTS,{_atempo(s['speed'])},aresample=48000,"
                 f"aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=0:{played:.3f}")
        if fin > 0:
            chain += f",afade=t=in:st=0:d={fin:.3f}"
        if fout > 0:
            chain += f",afade=t=out:st={played - fout:.3f}:d={fout:.3f}"
        chain += f",volume={s['gain']:.2f}dB,adelay={int(round(s['at'] * 1000))}:all=1[a{i}]"
        chains.append(chain)
        pads.append(f"[a{i}]")
    # A silent bed as long as the film: the track then starts at zero and ends with the picture,
    # wherever the first sound comes in.
    chains.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{seconds:.3f}[bed]")
    chains.append("[bed]" + "".join(pads) + f"amix=inputs={len(plan) + 1}:normalize=0:dropout_transition=0:"
                  f"duration=first,atrim=0:{seconds:.3f}[a]")
    mixed = os.path.splitext(video_path)[0] + ".sound.mp4"
    cmd += ["-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[a]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", mixed]
    errors = tempfile.TemporaryFile()
    try:
        done = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=errors,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode:
            errors.seek(0)
            why = errors.read()[-400:].decode("utf-8", "replace").strip()
            if os.path.isfile(mixed):
                os.remove(mixed)
            raise ValueError("ffmpeg could not lay the film's sound: " + why)
    finally:
        errors.close()
    os.replace(mixed, video_path)
    return len(plan)


def length_text(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def passes(spec: dict) -> list[float]:
    """When the film passes each of its pictures, in seconds."""
    return [p["t"] for p in normalised(spec)["pictures"]]


WARPS = {"even": lambda u: u, "faster": lambda u: 1 - (1 - u) ** 2, "slower": lambda u: u ** 2,
         "smooth": lambda u: u * u * (3 - 2 * u)}


def key_changes(film: "Film") -> tuple[list[float], float]:
    """How much the picture changes from each key to the next, and on a loop's run back to the first."""
    per = film.changes()
    gaps, back = [0.0] * (len(film.spec["pictures"]) - 1), 0.0
    for k, c in enumerate(per):
        q = film.stops[k]
        if q["plays"]:
            continue
        if k + 1 < len(film.stops):
            gaps[q["key"]] += c
        else:
            back = c
    return gaps, back


def spaced(spec: dict, shape: str = "even", start: float | None = None, end: float | None = None,
           by: str = "time", changes: tuple | None = None) -> dict:
    """The picture keys re-timed between `start` and `end` by a shape: even, faster (the passages
    shortening toward the end), slower, or smooth (slow at both ends). By time, every passage takes
    as long. By change, each passage takes as long as the picture changes along it, so the morph runs at
    one rate, and the shape then bends that rate. A film held on the lane keeps its length. How a morph
    speeds up without moving twenty keys by hand."""
    f = normalised(spec)
    pics = f["pictures"]
    n = len(pics)
    warp = WARPS.get(shape)
    if warp is None:
        raise ValueError("A spacing is even, faster, slower or smooth.")
    if by not in ("time", "change"):
        raise ValueError("Space the pictures by time or by change.")
    a = pics[0]["t"] if start is None else float(start)
    b = pics[-1]["t"] if end is None else float(end)
    plays = [span(p) for p in pics]
    morph = (b - a) - sum(plays[:-1])
    if morph < 0.1 * (n - 1):
        raise ValueError("The span is too short for the films it holds. Give the film more time.")
    back = 0.0
    if by == "change":
        gaps, back = changes if changes is not None else key_changes(prepared(f))
        gaps = np.asarray(gaps, float)
        floor = 0.05 * (gaps.mean() if gaps.size and gaps.mean() > 0 else 1.0)
        gaps = np.maximum(gaps, floor)       # two near-identical pictures still get a moment
    else:
        gaps = np.ones(n - 1)
    cum = np.concatenate([[0.0], np.cumsum(gaps)]) / gaps.sum()
    at = morph * np.array([warp(float(u)) for u in cum])
    out = dict(spec)
    out["pictures"] = [{**p, "t": round(a + float(m) + sum(plays[:i]), 3)} for i, (p, m) in enumerate(zip(pics, at))]
    if by == "change" and f["loop"] and back > 0:
        # The run back to the first picture goes at the same rate.
        out["length"] = round(out["pictures"][-1]["t"] + plays[-1] + morph * max(back, floor) / gaps.sum(), 3)
    elif "length" in out and f["loop"] and float(out["length"] or 0) <= out["pictures"][-1]["t"] + plays[-1]:
        out.pop("length")
    return out


# ─────────────────────────── kept films ───────────────────────────

_ready: dict = {}


def prepared(spec: dict, size: int | None = None) -> Film:
    """A Film ready to draw, its pictures' grids kept while the pictures, the space, the size and the
    aspect stay the same, so scrubbing a timeline whose looks and lanes change re-encodes nothing."""
    f = normalised(spec)
    held = [[p["ref"], p.get("in"), p.get("out"),
             load(str(p["ref"])[len(FILM_REF):]) if is_film(p["ref"]) else clip_source(p["ref"])[0]]
            for p in f["pictures"] if plays(p["ref"])]
    key = json.dumps([[p["ref"] for p in f["pictures"]], held, f["space"], int(size or f["size"]), f["aspect"],
                      f["loop"]], sort_keys=True)
    film = _ready.get(key)
    if film is None:
        film = Film(f, size)
        _ready.clear()                       # one film at a time: grids for 20 pictures are enough to hold
        _ready[key] = film
    else:
        film.spec = f
        film.stops = points(f)
        film.table = None
    return film


def _paths():
    try:
        from .. import paths
    except ImportError:                      # pragma: no cover
        import paths
    return paths


def films_dir() -> str:
    p = _paths()
    return p.ensure_dir(os.path.join(p.data_dir(), "alchemy", "films"))


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    if not s:
        raise ValueError("A film needs a name.")
    return s[:60]


def film_path(name: str) -> str:
    return os.path.join(films_dir(), slug(name) + ".json")


def load(name: str) -> dict:
    path = film_path(name)
    if not os.path.isfile(path):
        names = ", ".join(listing()) or "none yet"
        raise ValueError(f"No film called {name!r}. Films: {names}.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(spec: dict) -> dict:
    f = normalised(spec)
    path = film_path(f["name"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(f, fh, indent=1)
    os.replace(tmp, path)
    return f


def listing() -> list[str]:
    return sorted(os.path.splitext(n)[0] for n in os.listdir(films_dir()) if n.endswith(".json"))


# ─────────────────────────── templates ───────────────────────────
#
# A template is a film without its pictures: the film's settings, its looks and its lanes, kept with
# the length they were keyed against. Applied to another film, every key stretches by that film's
# length over the template's, so a trend keyed across 2 minutes spans a 40 second film the same way.
# A film with fewer than two pictures has no length yet, and takes the keys at their kept times.

TEMPLATE_SETTINGS = ("fps", "size", "aspect", "space", "loop", "even")


def as_template(spec: dict, name: str) -> dict:
    f = normalised(spec)
    slug(name)
    return {"name": name.strip(), "length": f["length"], **{k: f[k] for k in TEMPLATE_SETTINGS},
            "looks": f["looks"], "lanes": f["lanes"]}


def templated(spec: dict, template: dict) -> dict:
    """The film with the template's settings, looks and lanes in place of its own, keys stretched to its
    length. Its pictures, name and timing stay."""
    out = {**(spec or {}), **{k: template[k] for k in TEMPLATE_SETTINGS if k in template}}
    out.setdefault("pictures", [])
    try:
        length = normalised(out)["length"]
    except ValueError:
        length = None
    kept = float(template.get("length") or 0)
    k = length / kept if length and kept > 0 else 1.0

    def at(t):
        return round(float(t) * k, 3)

    out["looks"] = [{**x, "t": at(x["t"])} for x in template.get("looks") or []]
    out["lanes"] = {path: [{**x, "t": at(x["t"])} for x in keys] for path, keys in (template.get("lanes") or {}).items()}
    for path in out["lanes"]:
        check_path(path)
    return out


def templates_dir() -> str:
    p = _paths()
    return p.ensure_dir(os.path.join(p.data_dir(), "alchemy", "film-templates"))


def template_path(name: str) -> str:
    return os.path.join(templates_dir(), slug(name) + ".json")


def template_listing() -> list[str]:
    return sorted(os.path.splitext(n)[0] for n in os.listdir(templates_dir()) if n.endswith(".json"))


def load_template(name: str) -> dict:
    path = template_path(name)
    if not os.path.isfile(path):
        names = ", ".join(template_listing()) or "none yet"
        raise ValueError(f"No film template called {name!r}. Templates: {names}.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_template(template: dict) -> dict:
    path = template_path(template["name"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=1)
    os.replace(tmp, path)
    return template
