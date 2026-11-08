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

Masks over time. A geometric layer moves through its numbers. A layer read by a model (the subject,
words, depth) is read on the key pictures, once each, and blended along the morph, so it moves with
what it found and no model runs on a frame.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import numpy as np

from . import mask as mk
from . import spaces
from .inputs import load_frame, resolve, to_image

EASES = ("linear", "smooth", "in", "out", "hold")
NUMBER_PATHS = ("zoom",)
REFINE_KEYS = ("feather", "grow", "follow", "ragged", "amount")
# The numbers of each geometric layer a lane can move, in the order its source line writes them.
MOVING = {"spot": ("x", "y", "r"), "gradient": ("angle",), "band": ("y", "height"),
          "stripes": ("count", "angle"), "depth band": ("near", "far"), "noise": ("size",)}
SIDES = {"top": 0.0, "right": 90.0, "bottom": 180.0, "left": 270.0}


# ─────────────────────────── the spec ───────────────────────────

def normalised(spec: dict) -> dict:
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
    if len({p["t"] for p in pics}) != len(pics):
        raise ValueError("Two pictures share a time on the picture lane.")
    f["pictures"] = pics
    last = pics[-1]["t"]
    length = float(f.get("length") or 0)
    if f["loop"]:
        # The run back to the first picture takes as long as the average passage, unless given.
        length = length if length > last else last + (last - pics[0]["t"]) / (len(pics) - 1)
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


def passage(film: dict, t: float) -> tuple[int, float]:
    """Which passage of the picture lane time t falls in, and how far along it (eased): (k, s), the film
    going from picture k to picture k+1 (wrapping to the first on a loop)."""
    pics, n = film["pictures"], len(film["pictures"])
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
                break
    for path, keys in film["lanes"].items():
        v = lane_value(keys, t)
        head, _, rest = path.partition(".")
        if path == "zoom":
            s["zoom"] = v
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
            m = s.setdefault("mask", {})
            if isinstance(m, (str, list)):
                m = s["mask"] = mk.MaskSpec.of(m).__dict__.copy()
            if rest in REFINE_KEYS:
                m[rest] = v
                continue
            idx, _, num = rest.partition(".")
            i = int(idx) - 1
            layers = list(m.get("layers") or [])
            if i >= len(layers):
                continue
            if num == "coverage":
                cov = list(m.get("coverage") or [])
                cov += [None] * (len(layers) - len(cov))
                cov[i] = v
                m["coverage"] = cov
                continue
            got = layer_numbers(layers[i])
            if got is not None and num in got[1]:
                values = list(got[2])
                values[got[1].index(num)] = v
                layers[i] = layer_line(got[0], values, got[3] if len(got) > 3 else "")
                m["layers"] = layers
    return s


# ─────────────────────────── the frame ───────────────────────────

class Film:
    """A film made ready to draw: its token space, the grids of its pictures and their mask maps."""

    def __init__(self, spec: dict, size: int | None = None):
        self.spec = normalised(spec)
        self.size = int(size or self.spec["size"])
        self.space_name = self.spec["space"]
        self.sp = spaces.space(self.space_name)
        self.paths = [resolve(p["ref"]) for p in self.spec["pictures"]]
        fw, fh = self._aspect()
        long = max(1, round(self.size / 16))
        self.rows, self.cols = self.sp.grid(fw, fh, long)
        self.W, self.H = self.cols * 16, self.rows * 16
        from .tokens import encode

        with spaces.lock():
            self.grids = [encode(self.sp, self.space_name, f"{p}@{self.W}x{self.H}", load_frame(p, self.W, self.H),
                                 (self.rows, self.cols)) for p in self.paths]
        self.maps: dict = {}
        self.table = None                    # even pacing: per passage, eased s -> evened s

    def _aspect(self):
        a = str(self.spec["aspect"])
        if a == "auto":
            from .inputs import original_size

            return original_size(self.paths[0])
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
        """The picture at time t: the morph decoded, [H, W, 3] floats in 0..1."""
        k, s = passage(self.spec, t)
        if self.table is not None:
            s = float(np.interp(s, *self.table[k]))
        with spaces.lock():
            z = self.tokens(k, s)
            return self.sp.decode(self.sp.shell(z.reshape(1, self.rows * self.cols, -1)), (self.rows, self.cols))[0]

    def asker(self, t: float):
        """Answers a mask's question about the frame at t from the two pictures it lies between."""
        k, s = passage(self.spec, t)
        n = len(self.paths)
        j = (k + 1) % n if self.spec["loop"] else min(k + 1, n - 1)

        def ask(kind, arg=""):
            a, b = self._map(k, kind, arg), self._map(j, kind, arg)
            return (1 - s) * a + s * b

        return ask

    def _map(self, i, kind, arg):
        key = (i, kind, str(arg))
        if key not in self.maps:
            from .vision import vision

            path = self.paths[i]
            c = load_frame(path, self.W, self.H)
            img = to_image(c)
            self.maps[key] = np.asarray(vision().ask(f"{path}@{self.W}x{self.H}", lambda: img, kind, arg), np.float32)
        return self.maps[key]

    def frame(self, t: float):
        """The finished frame at time t: the morph, then the look's step, then its finish."""
        from .pipeline import step_on

        c = self.morph(t)
        out, _ = step_on(c, f"film:{id(self)}:{t:.4f}", look_at(self.spec, t), asker=self.asker(t))
        return np.clip(out, 0, 1)

    def even(self, samples: int = 12):
        """Even pacing: each passage re-timed so the picture changes by the same amount every frame,
        read on a half-size grid."""
        rows, cols = max(1, self.rows // 2), max(1, self.cols // 2)
        from .tokens import encode

        with spaces.lock():
            small = [encode(self.sp, self.space_name, f"{p}@{cols * 16}x{rows * 16}",
                            load_frame(p, cols * 16, rows * 16), (rows, cols)) for p in self.paths]
        full, self.grids = self.grids, small
        table = []
        try:
            n = len(self.paths) if self.spec["loop"] else len(self.paths) - 1
            for k in range(n):
                ss = np.linspace(0, 1, samples + 1)
                with spaces.lock():
                    looks = [self.sp.decode(self.sp.shell(self.tokens(k, float(s)).reshape(1, rows * cols, -1)),
                                            (rows, cols))[0] for s in ss]
                moved = np.array([0.0] + [float(np.abs(b - a).mean()) for a, b in zip(looks, looks[1:])])
                gone = np.cumsum(moved)
                gone = gone / gone[-1] if gone[-1] > 0 else ss
                table.append((gone, ss))     # np.interp(s, gone, ss): equal change -> where on the curve
            table += [(np.array([0.0, 1.0]), np.array([0.0, 1.0]))] * (len(self.paths) - len(table))
        finally:
            self.grids = full
        self.table = table
        return table


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
        progress(0.0, f"{len(film.paths)} pictures, {frames} frames at {film.W} x {film.H}")
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
    return {"path": out_path, "frames": made, "of": frames, "fps": fps, "size": [out_w, out_h],
            "drawn": [film.W, film.H], "seconds": round(made / fps, 2), "pictures": [p["ref"] for p in spec["pictures"]]}


def length_text(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def passes(spec: dict) -> list[float]:
    """When the film passes each of its pictures, in seconds."""
    return [p["t"] for p in normalised(spec)["pictures"]]


def spaced(spec: dict, shape: str = "even", start: float | None = None, end: float | None = None) -> dict:
    """The picture keys re-timed between `start` and `end` by a shape: even, faster (the passages
    shortening toward the end), slower, or smooth (slow at both ends). How a morph speeds up without
    moving twenty keys by hand."""
    f = normalised(spec)
    pics = f["pictures"]
    n = len(pics)
    a = pics[0]["t"] if start is None else float(start)
    b = pics[-1]["t"] if end is None else float(end)
    u = np.linspace(0, 1, n)
    warp = {"even": u, "faster": 1 - (1 - u) ** 2, "slower": u ** 2,
            "smooth": u * u * (3 - 2 * u)}.get(shape)
    if warp is None:
        raise ValueError("A spacing is even, faster, slower or smooth.")
    out = dict(spec)
    out["pictures"] = [{**p, "t": round(a + (b - a) * float(w), 3)} for p, w in zip(pics, warp)]
    return out



# ─────────────────────────── kept films ───────────────────────────

_ready: dict = {}


def prepared(spec: dict, size: int | None = None) -> Film:
    """A Film ready to draw, its pictures' grids kept while the pictures, the space, the size and the
    aspect stay the same, so scrubbing a timeline whose looks and lanes change re-encodes nothing."""
    f = normalised(spec)
    key = json.dumps([[p["ref"] for p in f["pictures"]], f["space"], int(size or f["size"]), f["aspect"], f["loop"]])
    film = _ready.get(key)
    if film is None:
        film = Film(f, size)
        _ready.clear()                       # one film at a time: grids for 20 pictures are enough to hold
        _ready[key] = film
    else:
        film.spec = f
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
