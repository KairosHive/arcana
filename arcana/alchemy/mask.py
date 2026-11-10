"""Where an Alchemy step acts: a mask built from presets, prompts and pictures, combined and refined.

A mask is up to three layers met by "and", "or" or "but not", then refined. Each layer is one source,
written as a short line:

    highlights                 a preset by name (see PRESETS)
    prompt:red coat, scarf     words: things are cut out by Grounding DINO and SAM 2, and when no
                               thing is found, CLIPSeg's soft map of the words is used instead
    thing:the coat             words, as countable things only (Grounding DINO + SAM 2)
    stuff:wet asphalt          words, as stuff or qualities only (CLIPSeg)
    colour:#c87a3c             nearness to a colour
    spot:0.5,0.4,0.25          a round spot at x, y (0..1 of the frame), radius
    gradient:left              from a side (top, bottom, left, right) or an angle in degrees
    depth band:0.3,0.6         a band of depth, near is 1
    picture:japan:1423         another picture's light, laid over the frame
    material light             the first material's own light (or "material dark")

Two kinds of source. A FIELD (light, edges, colour, depth, noise and the like) ranks every place,
and `coverage` keeps the share of the frame asked for, as a quantile of this picture, so 0.3 is the
brightest (or nearest, or most alike) 30% whatever the exposure. A REGION (subject, things, stuff)
is a cut-out already and is taken as it is.

Refinements, in order: `grow` moves the edge out or in, `follow` snaps it to the picture's own
contours, `ragged` breaks it up with grown noise, `feather` softens it, `invert` takes the rest, and
`amount` scales the whole mask. Pixel sizes are set for a 768 px frame and scale with the frame, so a
preview and a full-size render cut the same mask.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import pixels as cx

REF = 768.0                       # the frame size pixel knobs are given for
LUM = np.array([0.299, 0.587, 0.114], np.float32)
COMBINE = ("and", "or", "but not")


# ─────────────────────────── helpers, from latent-lab's masks.py ───────────────────────────

def blur(a, sigma):
    """A Gaussian for masks: separable convolution on the CPU (a few ms at working size), with
    wide blurs run on a shrunk copy. Masks are smooth fields; they don't need the FFT blur."""
    if sigma <= 0.3:
        return a
    h, w = a.shape
    if sigma > 6 and min(h, w) > 64:
        f = int(min(sigma // 3, 32))
        small = cx._resize(a.astype(np.float32), max(8, w // f), max(8, h // f), Image.BOX)
        return cx._resize(blur(small, sigma / f), w, h, Image.BICUBIC)
    r = min(int(math.ceil(3.0 * sigma)), h - 1, w - 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    with torch.no_grad():
        t = torch.from_numpy(np.array(a, np.float32))[None, None]
        t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def rank(a, sample=20_000, seed=0):
    """A field passed through its own distribution: 0.3 means 'above 30% of this picture'."""
    flat = a.ravel()
    if flat.size > sample:
        flat = flat[np.random.default_rng(seed).choice(flat.size, sample, replace=False)]
    ref = np.sort(flat)
    return np.interp(a, ref, np.linspace(0.0, 1.0, ref.size)).astype(np.float32)


def growth(h, w, scale, seed):
    """Fractal noise whose blobs are `scale`-sized, the same pattern at any resolution."""
    rng = np.random.default_rng(seed)
    cell = max(float(scale), 1.0) * 4.0
    f = int(max(1, min(16, (cell / 16.0) // 2)))
    sh, sw = max(8, h // f), max(8, w // f)
    out = np.zeros((sh, sw), np.float32)
    amp, total = 1.0, 0.0
    for _ in range(5):
        gh = max(2, int(round(h / cell)) + 1)
        gw = max(2, int(round(w / cell)) + 1)
        out += amp * cx._resize(rng.standard_normal((gh, gw)).astype(np.float32), sw, sh, Image.BICUBIC)
        total += amp
        amp *= 0.5
        cell /= 2.0
    out /= total
    return out if (sh, sw) == (h, w) else cx._resize(out, w, h, Image.BICUBIC)


def guided(p, guide, radius, eps=1e-3):
    """Edge-aware smoothing of `p`, steered by `guide` (He & Sun's fast guided filter)."""
    h, w = p.shape
    f = int(max(1, min(8, radius // 3)))
    if f > 1 and min(h, w) // f >= 16:
        ps = cx._resize(p.astype(np.float32), w // f, h // f, Image.BOX)
        gs = cx._resize(guide.astype(np.float32), w // f, h // f, Image.BOX)
    else:
        f, ps, gs = 1, p, guide
    s = max(float(radius) / f, 1.0) / 2.0
    mi, mp = blur(gs, s), blur(ps, s)
    var = blur(gs * gs, s) - mi * mi
    cov = blur(gs * ps, s) - mi * mp
    a = cov / (var + eps)
    b = mp - a * mi
    a, b = blur(a, s), blur(b, s)
    if f > 1:
        a = cx._resize(a, w, h, Image.BILINEAR)
        b = cx._resize(b, w, h, Image.BILINEAR)
    return a * guide + b


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def grow(m, px):
    """A region grown outward (px > 0) or shrunk inward (px < 0), corners rounding as they go."""
    if abs(px) < 0.5:
        return m
    b = blur(m, abs(px) / 2.0)
    t = 0.023 if px > 0 else 0.977
    return smoothstep((b - t) / 0.04 + 0.5)


# ─────────────────────────── the presets ───────────────────────────
# A preset is one layer's source with the coverage it starts at. Most are fields or prompts under a
# friendly name, so the library grows without new code.

@dataclass(frozen=True)
class Preset:
    name: str
    group: str
    source: str
    doc: str
    coverage: float = 0.4


PRESETS = [
    # light and colour
    Preset("highlights", "light and colour", "f:light", "the brightest places", 0.25),
    Preset("light", "light and colour", "f:light", "the lighter half", 0.5),
    Preset("midtones", "light and colour", "f:midtones", "neither the darkest nor the lightest", 0.4),
    Preset("shadows", "light and colour", "f:shadow", "the darkest places", 0.3),
    Preset("edges", "light and colour", "f:edges", "where the picture's light changes sharply", 0.25),
    Preset("flat areas", "light and colour", "f:flat", "smooth, even places", 0.4),
    Preset("saturated", "light and colour", "f:saturated", "the most colourful places", 0.3),
    Preset("muted", "light and colour", "f:muted", "the greyest places", 0.4),
    Preset("warm", "light and colour", "f:warm", "reds, oranges and yellows", 0.35),
    Preset("cool", "light and colour", "f:cool", "blues and cyans", 0.35),
    # depth and focus
    Preset("near", "depth and focus", "f:near", "what is nearest the camera", 0.4),
    Preset("far", "depth and focus", "f:far", "what is furthest away", 0.4),
    Preset("middle distance", "depth and focus", "depth band:0.35,0.65", "a band of middle depth", 1.0),
    Preset("in focus", "depth and focus", "f:focus", "the sharpest places", 0.35),
    Preset("blurred", "depth and focus", "f:blurred", "out-of-focus places", 0.4),
    # things and stuff, by words
    Preset("subject", "things and stuff", "subject", "whatever the picture is of", 1.0),
    Preset("background", "things and stuff", "background", "everything but the main subject", 1.0),
    Preset("people", "things and stuff", "thing:person", "each person, cut out", 1.0),
    Preset("faces", "things and stuff", "thing:face", "each face", 1.0),
    Preset("clothes", "things and stuff", "thing:clothing, coat, dress, shirt", "what people wear", 1.0),
    Preset("sky", "things and stuff", "stuff:sky", "the sky", 1.0),
    Preset("water", "things and stuff", "stuff:water, sea, lake, river", "water of any kind", 1.0),
    Preset("foliage", "things and stuff", "stuff:leaves, foliage, trees, grass", "leaves and plants", 1.0),
    Preset("ground", "things and stuff", "stuff:ground, floor, road, sand", "what is underfoot", 1.0),
    Preset("rock and stone", "things and stuff", "stuff:rock, stone, wall", "rock, stone and walls", 1.0),
    Preset("buildings", "things and stuff", "thing:building, house", "buildings", 1.0),
    # geometry
    Preset("from the top", "geometry", "gradient:top", "fading in from the top edge", 0.5),
    Preset("from the bottom", "geometry", "gradient:bottom", "fading in from the bottom edge", 0.5),
    Preset("from the left", "geometry", "gradient:left", "fading in from the left edge", 0.5),
    Preset("from the right", "geometry", "gradient:right", "fading in from the right edge", 0.5),
    Preset("centre spot", "geometry", "spot:0.5,0.5,0.3", "a round spot in the middle", 1.0),
    Preset("vignette", "geometry", "f:vignette", "the edges and corners of the frame", 0.45),
    Preset("horizon band", "geometry", "band:0.5,0.18", "a horizontal band across the middle", 1.0),
    Preset("stripes", "geometry", "stripes:8,0", "parallel stripes", 0.5),
    # procedural
    Preset("growth", "procedural", "noise:growth,40", "fractal blooms, like lichen spreading", 0.45),
    Preset("cells", "procedural", "noise:cells,60", "a cellular pattern, like scales or cracked mud", 0.5),
    Preset("cracks", "procedural", "noise:cracks,50", "a network of thin cracks", 0.2),
    Preset("flow lines", "procedural", "noise:flow,50", "wavering lines, like wood grain or water", 0.4),
    # from pictures
    Preset("material light", "from pictures", "material light", "the first material's lighter places", 0.5),
    Preset("material dark", "from pictures", "material dark", "the first material's darker places", 0.5),
]
PRESET = {p.name: p for p in PRESETS}
GROUPS = list(dict.fromkeys(p.group for p in PRESETS))


# ─────────────────────────── a mask, as asked ───────────────────────────

@dataclass
class MaskSpec:
    layers: list = field(default_factory=list)       # sources, as lines (see the module's doc)
    combine: list = field(default_factory=list)      # how each layer after the first meets the ones before
    coverage: list = field(default_factory=list)     # per layer, for fields; None takes the preset's
    invert: bool = False
    grow: float = 0.0                                # px at 768, out (>0) or in (<0)
    follow: float = 0.0                              # 0..1: snap the edge to the picture's contours
    ragged: float = 0.0                              # 0..1: break the edge with grown noise
    feather: float = 4.0                             # px at 768
    amount: float = 1.0                              # 0..1: how strongly the mask lets the step in
    seed: int = 0

    @classmethod
    def of(cls, d) -> "MaskSpec":
        if d is None:
            return cls()
        if isinstance(d, MaskSpec):
            return d
        if isinstance(d, str):
            return cls(layers=[d])
        if isinstance(d, (list, tuple)):
            return cls(layers=list(d))
        known = {k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def empty(self) -> bool:
        return not [s for s in self.layers if str(s).strip() and str(s).strip() != "none"]


class Scene:
    """What a mask reads: the picture at the frame's size, the first material, and how to ask the
    vision models about the picture (its key names it, so their answers are kept)."""

    def __init__(self, picture, key, material=None, image=None, load=None, asker=None):
        self.c = picture                      # [H, W, 3] floats in 0..1
        self.key = key
        self.t = material                     # the first material at the frame, or None
        self._image = image                   # the picture as a PIL image, for the models
        self._load = load                     # loads another picture by reference at a size
        self._asker = asker                   # answers ask() in place of the models (a film frame)
        self.H, self.W = picture.shape[:2]
        self.lum = picture @ LUM

    def image(self):
        if self._image is None:
            self._image = Image.fromarray((np.clip(self.c, 0, 1) * 255).astype(np.uint8))
        return self._image

    def ask(self, kind, arg=""):
        """A model's map of this picture: its subject, a named thing or stuff, its depth. A film frame
        gives an `asker`, which blends the maps of the key pictures it lies between, so a mask moves
        with the morph and no model runs on a frame."""
        if self._asker is not None:
            got = self._asker(kind, arg)
        else:
            from .vision import vision

            got = vision().ask(self.key, self.image, kind, arg)
        if got.shape != (self.H, self.W):
            got = cx._resize(got, self.W, self.H, Image.BILINEAR)
        return got


def _field(name, sc: Scene, seed):
    """A named field over the frame: higher is more of what the name says."""
    L, c, H, W = sc.lum, sc.c, sc.H, sc.W
    s = max(H, W) / REF
    if name == "light":
        return blur(L, 1.5 * s)
    if name == "shadow":
        return -blur(L, 1.5 * s)
    if name == "midtones":
        return -np.abs(rank(blur(L, 1.5 * s)) - 0.5)
    if name in ("edges", "flat"):
        b = blur(L, 1.2 * s)
        gy, gx = np.gradient(b)
        e = blur(np.hypot(gx, gy), 3.0 * s)
        return e if name == "edges" else -e
    if name in ("saturated", "muted"):
        sat = c.max(-1) - c.min(-1)
        sat = blur(sat, 2.0 * s)
        return sat if name == "saturated" else -sat
    if name in ("warm", "cool"):
        w = blur(c[..., 0] - c[..., 2], 2.0 * s) * (0.3 + (c.max(-1) - c.min(-1)))
        return w if name == "warm" else -w
    if name in ("near", "far"):
        d = sc.ask("depth")
        return d if name == "near" else -d
    if name in ("focus", "blurred"):
        hp = L - blur(L, 1.5 * s)
        e = blur(hp * hp, 6.0 * s)
        return e if name == "focus" else -e
    if name == "vignette":
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        return np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2))
    raise ValueError(f"no field called {name!r}")


def _noise(kind, size, sc: Scene, seed):
    H, W = sc.H, sc.W
    size = float(size) * max(H, W) / REF
    rng = np.random.default_rng(seed)
    if kind == "growth":
        return growth(H, W, size, seed)
    if kind == "cells":
        n = max(4, int(H * W / (size * size)))
        pts = rng.random((n, 2)) * [H, W]
        yy, xx = np.mgrid[0:H:2, 0:W:2].astype(np.float32)
        q = np.stack([yy.ravel(), xx.ravel()], 1)
        d = np.full(len(q), np.inf, np.float32)
        d2 = np.full(len(q), np.inf, np.float32)
        for i in range(0, n, 256):
            dist = np.sqrt(((q[:, None, :] - pts[None, i:i + 256, :]) ** 2).sum(-1)).astype(np.float32)
            both = np.concatenate([d[:, None], d2[:, None], dist], 1)
            both.sort(1)
            d, d2 = both[:, 0], both[:, 1]
        f = (d2 - d).reshape(yy.shape)          # small on the cell walls
        return cx._resize(-f, W, H, Image.BILINEAR)
    if kind == "cracks":
        g = growth(H, W, size, seed)
        return -np.abs(g)                       # ridges where the noise crosses zero
    if kind == "flow":
        g = growth(H, W, size * 2.5, seed)
        y, _ = np.mgrid[0:H, 0:W].astype(np.float32)
        return np.sin((y / max(size, 1.0)) * 2.0 + g * 9.0)
    raise ValueError(f"no noise called {kind!r}")


def _numbers(arg, n):
    try:
        vals = [float(x) for x in re.split(r"[,\s]+", arg.strip()) if x]
    except ValueError:
        raise ValueError(f"{arg!r} should be {n} numbers separated by commas") from None
    if len(vals) < n:
        raise ValueError(f"{arg!r} should be {n} numbers separated by commas")
    return vals


def source(line, sc: Scene, seed=0):
    """One layer's source as (map, is_region): a field to be ranked, or a region taken as it is."""
    raw = str(line).strip()
    if raw in PRESET:
        raw = PRESET[raw].source            # a preset is its source; resolved once, never again
    name, _, arg = raw.partition(":")
    name, arg = name.strip().lower(), arg.strip()
    H, W = sc.H, sc.W
    if name == "f":
        return _field(arg, sc, seed), False
    if name in ("subject", "background"):
        m = sc.ask("subject")
        return (m if name == "subject" else 1.0 - m), True
    if name == "thing":
        return sc.ask("thing", arg), True
    if name == "stuff":
        return sc.ask("stuff", arg), True
    if name in ("prompt", "words"):
        m = sc.ask("thing", arg)
        if float(m.max(initial=0.0)) < 0.5:
            m = sc.ask("stuff", arg)
        return m, True
    if name == "colour" or name == "color":
        hexv = arg.lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", hexv):
            raise ValueError(f"colour:{arg} should be a hex colour like #c87a3c")
        rgb = np.array([int(hexv[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
        d = np.sqrt((((sc.c - rgb) * [0.3, 0.59, 0.11]) ** 2).sum(-1) * 3)
        return -blur(d, 1.5 * max(H, W) / REF), False
    if name == "spot":
        x, y, r = _numbers(arg, 3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xx / W - x, (yy / H - y) * H / W) / max(r, 1e-3)
        return smoothstep(1.2 - d), True
    if name == "gradient":
        side = arg or "left"
        angle = {"top": 0, "right": 90, "bottom": 180, "left": 270}.get(side)
        angle = math.radians(float(side) if angle is None else angle)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return -((yy / H - 0.5) * math.cos(angle) - (xx / W - 0.5) * math.sin(angle)), False
    if name == "band":
        y, h = _numbers(arg, 2)
        yy = np.mgrid[0:H, 0:W][0].astype(np.float32) / H
        return smoothstep(1.0 - np.abs(yy - y) / max(h / 2, 1e-3) + 0.3), True
    if name == "stripes":
        n, ang = _numbers(arg, 2)
        a = math.radians(ang)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (xx * math.cos(a) + yy * math.sin(a)) / max(H, W)
        return np.sin(u * n * 2 * math.pi), False
    if name == "depth band":
        lo, hi = _numbers(arg, 2)
        d = sc.ask("depth")
        mid, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-3)
        return smoothstep(1.0 - np.abs(d - mid) / half + 0.25), True
    if name == "noise":
        kind, _, size = arg.partition(",")
        return _noise(kind.strip(), float(size or 40), sc, seed), False
    if name in ("material light", "material dark"):
        if sc.t is None:
            raise ValueError(f"'{name}' needs a material")
        l = blur(sc.t @ LUM, 1.5 * max(H, W) / REF)
        return (l if name == "material light" else -l), False
    if name == "picture":
        if sc._load is None:
            raise ValueError("picture: masks need a picture loader")
        return blur(sc._load(arg, W, H) @ LUM, 1.0), False
    raise ValueError(f"No mask source called {raw!r}. Presets: " + ", ".join(p.name for p in PRESETS)
                     + "; or prompt:<words>, thing:, stuff:, colour:#hex, spot:x,y,r, gradient:<side>, "
                       "band:y,height, stripes:n,angle, depth band:lo,hi, noise:<growth|cells|cracks|flow>,size, "
                       "picture:<handle>, material light, material dark.")


def _coverage_of(line, given):
    if given is not None:
        return float(given)
    p = PRESET.get(str(line).strip())
    return p.coverage if p else 0.4


def build(spec, sc: Scene):
    """The mask [H, W] in 0..1 for this scene, or None when no layer is asked for."""
    spec = MaskSpec.of(spec)
    if spec.empty:
        return None
    s = max(sc.H, sc.W) / REF
    layers = [l for l in spec.layers if str(l).strip() and str(l).strip() != "none"]
    m = None
    for i, line in enumerate(layers):
        got, region = source(line, sc, int(spec.seed) + i)
        cov = spec.coverage[i] if i < len(spec.coverage) else None
        if region:
            layer = np.clip(got, 0, 1).astype(np.float32)
        else:
            cov = min(max(_coverage_of(line, cov), 0.0), 1.0)
            layer = smoothstep((rank(got) - (1.0 - cov)) / 0.04 + 0.5) if cov < 1.0 else np.ones_like(got)
        if m is None:
            m = layer
        else:
            how = spec.combine[i - 1] if i - 1 < len(spec.combine) else "and"
            if how == "or":
                m = np.maximum(m, layer)
            elif how == "but not":
                m = m * (1.0 - layer)
            else:
                m = m * layer
    if spec.grow:
        m = grow(m, spec.grow * s)
    if spec.follow > 0:
        snapped = np.clip(guided(m, sc.lum, max(8.0 * s, 4.0)), 0, 1)
        m = (1 - spec.follow) * m + spec.follow * snapped
    if spec.ragged > 0:
        n = growth(sc.H, sc.W, 6.0 * s, int(spec.seed) + 99)
        n = n / (np.abs(n).max() + 1e-6)
        m = smoothstep((m - 0.5 + spec.ragged * 0.6 * n) / 0.08 + 0.5)
    if spec.feather > 0:
        m = blur(m, spec.feather * s)
    if spec.invert:
        m = 1.0 - m
    return np.clip(m * float(spec.amount), 0.0, 1.0).astype(np.float32)


def shown(mask, picture):
    """The mask drawn over the picture: the picture greyed and dimmed outside, gold inside."""
    gold = np.array([0.79, 0.64, 0.33], np.float32)
    grey = (picture @ LUM)[..., None] * 0.45
    m = mask[..., None]
    return np.clip(grey * (1 - m) + (0.55 * picture + 0.45 * gold) * m, 0, 1)
