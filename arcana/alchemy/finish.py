"""The finish: a basic Lightroom toolkit, applied after the method, live.

Light, colour, presence and effects, in the order Lightroom applies them, on the picture as floats.
Every control is 0 at rest and the finish at rest changes nothing. `scope` says whether the finish
touches the whole picture or only what the mask selects.

    light     exposure (stops), contrast, highlights, shadows, whites, blacks
    colour    temperature, tint, vibrance, saturation
    presence  texture (fine detail), clarity (local contrast), dehaze
    effects   vignette (amount, midpoint, feather), grain (amount, size)

Ranges follow Lightroom's: -100..100 for most, exposure -5..5 stops. Sizes (clarity's reach, grain's
size) are given for a 768 px frame and scale with the frame, so a preview and a full-size render
look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from . import mask as mk

REF = 768.0
LUM = mk.LUM


@dataclass
class Finish:
    exposure: float = 0.0       # stops, -5..5
    contrast: float = 0.0       # -100..100
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0    # -100 cooler .. 100 warmer
    tint: float = 0.0           # -100 green .. 100 magenta
    vibrance: float = 0.0
    saturation: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vignette: float = 0.0       # -100 darker edges .. 100 lighter edges
    vignette_midpoint: float = 50.0   # 0..100
    vignette_feather: float = 50.0    # 0..100
    grain: float = 0.0          # 0..100
    grain_size: float = 25.0    # 0..100
    scope: str = "whole"        # whole | mask

    @classmethod
    def of(cls, d) -> "Finish":
        if isinstance(d, Finish):
            return d
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in names})

    @property
    def rest(self) -> bool:
        base = Finish()
        return all(getattr(self, f.name) == getattr(base, f.name) for f in fields(self) if f.name != "scope")


SPEC = [  # (key, group, label, lo, hi, step, doc) for the panel and `alchemy methods`
    ("exposure", "light", "exposure", -5, 5, 0.05, "brighter or darker overall, in stops"),
    ("contrast", "light", "contrast", -100, 100, 1, "more or less separation between light and dark"),
    ("highlights", "light", "highlights", -100, 100, 1, "recover (less) or brighten (more) the bright areas"),
    ("shadows", "light", "shadows", -100, 100, 1, "open up (more) or deepen (less) the dark areas"),
    ("whites", "light", "whites", -100, 100, 1, "move the white point"),
    ("blacks", "light", "blacks", -100, 100, 1, "move the black point"),
    ("temperature", "colour", "temperature", -100, 100, 1, "cooler (less) or warmer (more)"),
    ("tint", "colour", "tint", -100, 100, 1, "greener (less) or more magenta (more)"),
    ("vibrance", "colour", "vibrance", -100, 100, 1, "saturate the muted colours more than the strong ones"),
    ("saturation", "colour", "saturation", -100, 100, 1, "every colour stronger or weaker"),
    ("texture", "presence", "texture", -100, 100, 1, "fine detail: crisper (more) or smoother (less)"),
    ("clarity", "presence", "clarity", -100, 100, 1, "local contrast in the midtones: punch or softness"),
    ("dehaze", "presence", "dehaze", -100, 100, 1, "cut through haze (more) or add it (less)"),
    ("vignette", "effects", "vignette", -100, 100, 1, "darker (less) or lighter (more) edges"),
    ("vignette_midpoint", "effects", "vignette midpoint", 0, 100, 1, "how far in the vignette reaches"),
    ("vignette_feather", "effects", "vignette feather", 0, 100, 1, "how softly the vignette fades"),
    ("grain", "effects", "grain", 0, 100, 1, "film grain"),
    ("grain_size", "effects", "grain size", 0, 100, 1, "how coarse the grain is"),
]


def _to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _set_lum(img, new_lum, old_lum):
    """The picture with its luminance replaced, colours kept (ratio in linear light)."""
    return img * (new_lum / np.maximum(old_lum, 1e-4))[..., None]


def apply(img, f, mask=None, seed=0):
    """The finish on [H, W, 3] floats in 0..1; `mask` [H, W] limits it when scope is "mask"."""
    f = Finish.of(f)
    if f.rest:
        return img
    H, W = img.shape[:2]
    s = max(H, W) / REF
    x = _to_linear(np.clip(img, 0, 1).astype(np.float32))

    # ---- light ----
    if f.exposure:
        x = x * (2.0 ** f.exposure)
    L = x @ LUM
    Lg = _to_srgb(L)                                          # perceptual lightness for the tone tools
    new = Lg.copy()
    if f.highlights or f.shadows:
        base = mk.blur(Lg, 12.0 * s)                          # local, so highlights are regions, not pixels
        hi = mk.smoothstep((base - 0.45) / 0.45)
        sh = mk.smoothstep((0.55 - base) / 0.45)
        new = new + (f.highlights / 100.0) * 0.35 * hi * (1 - new if f.highlights > 0 else new)
        new = new + (f.shadows / 100.0) * 0.45 * sh * (1 - new if f.shadows > 0 else new)
    if f.whites or f.blacks:
        wp = 1.0 - 0.25 * (f.whites / 100.0)
        bp = -0.12 * (f.blacks / 100.0)
        new = (new - bp) / max(wp - bp, 1e-3)
    if f.contrast:
        k = f.contrast / 100.0
        mid = 0.5
        curved = mid + (new - mid) * (1 + k) if k < 0 else \
            np.clip(new, 0, 1) + k * (mk.smoothstep(np.clip(new, 0, 1)) - np.clip(new, 0, 1)) * 1.6
        new = curved
    # ---- presence ----
    if f.clarity:
        base = mk.blur(new, 20.0 * s)
        mids = 1 - np.abs(np.clip(new, 0, 1) * 2 - 1) ** 2
        new = new + (f.clarity / 100.0) * 0.8 * (new - base) * mids
    if f.texture:
        base = mk.blur(new, 2.0 * s)
        new = new + (f.texture / 100.0) * 1.2 * (new - base)
    if f.dehaze:
        k = f.dehaze / 100.0
        dark = mk.blur(np.clip(_to_srgb(x).min(-1), 0, 1), 15.0 * s)
        air = float(np.quantile(dark, 0.99))
        if k > 0:
            t = np.clip(1 - 0.85 * k * dark / max(air, 1e-3), 0.25, 1)
            new = (new - air * (1 - t)) / t
        else:
            new = new * (1 + 0.5 * k) - 0.5 * k * air
    x = _set_lum(x, _to_linear(np.clip(new, 0, 1.5)), np.maximum(L, 1e-6))

    # ---- colour ----
    if f.temperature or f.tint:
        t, g = f.temperature / 100.0, f.tint / 100.0
        gain = np.array([1 + 0.25 * t + 0.08 * g, 1 - 0.18 * g, 1 - 0.25 * t + 0.08 * g], np.float32)
        before = x @ LUM
        x = x * gain
        x = _set_lum(x, before, x @ LUM)
    if f.vibrance or f.saturation:
        lum = (x @ LUM)[..., None]
        chroma = x - lum
        sat = np.clip((x.max(-1) - x.min(-1)) / np.maximum(x.max(-1), 1e-4), 0, 1)[..., None]
        k = 1 + f.saturation / 100.0 + (f.vibrance / 100.0) * (1 - sat) * 1.2
        x = lum + chroma * np.maximum(k, 0)
    out = _to_srgb(x)

    # ---- effects ----
    if f.vignette:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
        mid = 0.2 + 0.6 * (f.vignette_midpoint / 100.0)
        soft = 0.05 + 0.6 * (f.vignette_feather / 100.0)
        v = mk.smoothstep((r - mid) / soft + 0.5)[..., None]
        k = f.vignette / 100.0
        out = out * (1 + k * v) if k < 0 else out + k * v * (1 - out)
    if f.grain:
        rng = np.random.default_rng(seed)
        size = (0.5 + 2.5 * f.grain_size / 100.0) * s
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = mk.blur(n, size)
        n = n / (n.std() + 1e-6)
        lum = out @ LUM
        out = out + ((f.grain / 100.0) * 0.08 * n * (1 - np.abs(lum * 2 - 1) * 0.6))[..., None]

    out = np.clip(out, 0, 1).astype(np.float32)
    if f.scope == "mask" and mask is not None:
        m = mask[..., None]
        out = m * out + (1 - m) * img
    return out
