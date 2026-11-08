"""Every Alchemy method, declared once: its family, its knobs, what it needs and how it runs.

Four families, by the space each works in:

    pixels      the material's own pixels laid into the picture: nothing invented, live
    tokens      the pictures as token grids, mixed by arithmetic and drawn back: live
    colour      the material's colours on the picture (arcana's colour transfer)
    generative  a diffusion model redraws the picture with the material in view (arcana's style transfer)

Each method reads a picture [T] and up to three materials [1] [2] [3] with weights. Every method takes
the mask: tokens weigh each token by it, and the others are laid into the picture through it. The
tab builds its controls from these declarations and `alchemy methods` prints them, so a knob added
here appears everywhere with nothing else to edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import pixels as cx
from . import tokens as tk

PIX_REF = 384.0   # the frame the pixel methods' pixel knobs were tuned in (latent-lab's)


@dataclass(frozen=True)
class Knob:
    key: str
    kind: str                    # float, int, choice, bool, angle, text
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    label: str = ""
    doc: str = ""
    options: tuple = ()
    when: tuple = ()             # (key, op, value) clauses that must all hold for the knob to apply
    px: bool = False             # a pixel size, scaled with the frame
    section: str = "main"        # main (shown first, the dominance knob leading), more, or looks

    def public(self):
        return {k: v for k, v in self.__dict__.items() if v not in ((), None, "") or k in ("default",)}


@dataclass(frozen=True)
class Method:
    id: str
    family: str
    label: str
    doc: str
    knobs: tuple
    run: object
    materials: tuple = (1, 3)    # how many materials it takes (min, max)
    picture: bool = True         # whether it reads a picture [T]
    cost: str = "live"           # live | seconds
    needs: tuple = ()            # gpu, diffusers, modflows

    def public(self):
        return dict(id=self.id, family=self.family, label=self.label, doc=self.doc, cost=self.cost,
                    materials=list(self.materials), picture=self.picture, needs=list(self.needs),
                    knobs=[k.public() for k in self.knobs])

    def defaults(self):
        return {k.key: k.default for k in self.knobs}


def K(key, kind, default, lo=None, hi=None, step=None, label="", doc="", options=(), when=(), px=False,
      section="main"):
    return Knob(key, kind, default, lo, hi, step, label or key.replace("_", " "), doc, tuple(options),
                tuple(tuple(w) for w in when), px, section)


def M(*a, **k):
    """K for a secondary knob, shown under "more"."""
    return K(*a, section="more", **k)


SECTIONS = ("main", "more", "looks")


# ─────────────────────────── pixels ───────────────────────────

def _pixel(fn, px_keys=()):
    def run(step, k):
        p = dict(k)
        s = max(step.H, step.W) / PIX_REF
        for key in px_keys:
            if key == "patch":
                p[key] = max(3, int(round(p[key] * s)) | 1)
            elif key == "block":
                p[key] = max(8, int(round(p[key] * s)))
            else:
                p[key] = p[key] * s
        c, t = step.c, step.materials[0]
        out = np.clip(fn(c, t, p), 0, 1)
        a = float(p.get("amount", 1.0))
        return (c + a * (out - c) if a < 1.0 else out).astype(np.float32)
    return run


AMOUNT = K("amount", "float", 1.0, 0, 1, 0.01, "amount", "how much of the result shows over the picture. "
           "0 keeps the picture, 1 shows the result whole")


PIXELS = [
    Method("pixels.displace", "pixels", "displace", "the material's pixels pushed along the picture's slopes, shaded by its light",
           (AMOUNT,
            K("amp", "float", 0.05, 0, 0.12, 0.002, "push distance", "how far the material is pushed along the slopes"),
            M("reach", "float", 3, 1, 10, 0.25, "smoothing", "how much the picture is softened before its slopes are read", px=True),
            K("shade", "float", 0.7, 0, 1.5, 0.05, "shading", "how strongly the picture's light and dark shade the result")),
           _pixel(lambda c, t, p: cx.displace(c, t, p["amp"], p["reach"], p["shade"]), ("reach",))),
    Method("pixels.relief", "pixels", "relief", "the material lit by the picture's relief, from a chosen side",
           (AMOUNT,
            K("angle", "angle", 0.9, 0, 6.283, 0.05, "light direction", "where the light comes from"),
            K("ambient", "float", 0.55, 0.15, 1.1, 0.05, "ambient light", "light that falls everywhere: higher flattens"),
            M("reach", "float", 2.5, 1, 8, 0.25, "smoothing", "how much the picture is softened first", px=True)),
           _pixel(lambda c, t, p: cx.relief(c, t, p["angle"], p["ambient"], p["reach"]), ("reach",))),
    Method("pixels.grain", "pixels", "grain", "the material's colour and grain under the picture's light",
           (AMOUNT,
            K("share", "float", 0.25, 0, 1, 0.02, "picture light", "how much of the picture's own light returns"),),
           _pixel(lambda c, t, p: cx.chroma(t, c, p["share"]))),
    Method("pixels.chroma", "pixels", "chroma", "the material's colours on the picture's light and forms",
           (AMOUNT,
            K("share", "float", 0.2, 0, 1, 0.02, "material light", "how much of the material's own light returns"),),
           _pixel(lambda c, t, p: cx.chroma(c, t, p["share"]))),
    Method("pixels.guided", "pixels", "guided", "the material's tones poured into the picture's forms by a guided filter",
           (AMOUNT,
            M("radius", "float", 14, 2, 60, 2, "radius", "how big an area each local fit covers", px=True),
            M("eps", "float", 0.015, 0.002, 0.08, 0.002, "edge threshold", "how strong an edge must be to stop the material"),
            K("detail", "float", 0.35, 0, 1, 0.05, "material detail", "how much of the material's grain comes back")),
           _pixel(lambda c, t, p: cx.guided(c, t, p["radius"], p["eps"], p["detail"]), ("radius",))),
    Method("pixels.blend", "pixels", "traded edges", "at each pixel the stronger edge wins, the picture's or the material's",
           (AMOUNT,
            K("wmix", "float", 0.65, 0, 1, 0.02, "picture edges", "0 the material's edges win everywhere, 1 the picture's"),),
           _pixel(lambda c, t, p: cx.blend(c, t, p["wmix"]))),
    Method("pixels.bands", "pixels", "hybrid image", "the picture's broad light under the material's fine grain",
           (AMOUNT,
            K("split", "float", 6, 0.5, 24, 0.25, "split scale", "where the picture hands over to the material", px=True),
            K("grain", "float", 1.0, 0, 2, 0.05, "material strength", "how strongly the material's fine grain comes in")),
           _pixel(lambda c, t, p: cx.bands(c, t, p["split"], p["grain"]), ("split",))),
    Method("pixels.veins", "pixels", "veins", "the material's cracks and lines engraved into the picture",
           (AMOUNT,
            K("depth", "float", 1.1, 0, 1.8, 0.05, "engraving depth", "how deeply the cracks are cut"),
            M("reach", "float", 3, 1, 8, 0.25, "crack width", "how wide a line has to be to count", px=True)),
           _pixel(lambda c, t, p: cx.veins(c, t, p["depth"], p["reach"]), ("reach",))),
    Method("pixels.flow", "pixels", "flow", "the material combed along the picture's contours",
           (AMOUNT,
            K("length", "float", 0.06, 0, 0.12, 0.002, "stroke length", "how far the material is combed"),
            M("reach", "float", 3, 1, 8, 0.25, "smoothing", "how much the picture is softened first", px=True),
            K("smear", "float", 0.25, 0, 1, 0.02, "smear", "0 combs sharply, 1 smears softly"),
            M("steps", "int", 8, 3, 16, 1, "steps", "how many small moves each stroke takes")),
           _pixel(lambda c, t, p: cx.flow(c, t, p["length"], p["reach"], p["smear"], int(p["steps"])), ("reach",))),
    Method("pixels.waterline", "pixels", "waterline", "the material rising from the picture's darkest tones",
           (AMOUNT,
            K("level", "float", 0.55, 0, 1, 0.02, "waterline", "how high the material rises"),
            M("feather", "float", 0.08, 0.01, 0.3, 0.01, "edge softness", "how gradually it fades at the waterline")),
           _pixel(lambda c, t, p: cx.mask(c, t, p["level"], p["feather"]))),
    Method("pixels.quilt", "pixels", "quilt", "the picture rebuilt from patches of the material that match its light",
           (AMOUNT,
            K("alpha", "float", 0.85, 0.2, 0.98, 0.02, "match picture", "how strictly each patch matches the picture"),
            K("block", "int", 36, 16, 96, 4, "patch size", "the size of the material pieces", px=True),
            M("candidates", "int", 400, 100, 1000, 50, "candidates", "how many pieces are tried for each spot"),
            M("seed", "int", 0, 0, 9999, 1, "seed", "another random draw")),
           _pixel(lambda c, t, p: cx.quilt(c, t, p["alpha"], int(p["block"]), int(p["candidates"]), int(p["seed"])),
                  ("block",)), cost="seconds"),
    Method("pixels.analogy", "pixels", "analogy", "every pixel taken from the place in the material whose neighbourhood matches",
           (AMOUNT,
            K("patch", "int", 7, 3, 15, 2, "patch size", "how big a neighbourhood each pixel compares", px=True),
            K("melt", "float", 1.0, 0, 3, 0.25, "melt", "0 copies the material's pixels, higher softens them", px=True),
            M("rounds", "int", 8, 2, 12, 1, "search rounds", "how long the match search runs"),
            M("seed", "int", 0, 0, 9999, 1, "seed", "another random draw")),
           _pixel(lambda c, t, p: cx.analogy(c, t, int(p["patch"]), p["melt"], int(p["rounds"]), int(p["seed"])),
                  ("patch", "melt")), cost="seconds"),
]

# ─────────────────────────── tokens ───────────────────────────

MODEL = K("model", "choice", "DINOv2", label="model", options=tuple(tk.spaces.SPACES),
          doc="DINOv2 fuses materials into one new surface; FLUX.2 VAE keeps the photographs' own grain")
LOOK_KNOBS = tuple(K(n, "float", 0.0, -3, 3, 0.05, n, f"the model's own {n} axis, in units of its spread",
                     section="looks") for n in tk.LOOKS)
OPS = tuple(tk.OPERATIONS)

TOKENS = [
    Method("tokens.mix", "tokens", "mix", "the materials laid into the picture's tokens, by arithmetic",
           (K("picture", "float", 0.0, 0, 1, 0.01, "picture underneath", "how much of the picture's own tokens "
              "stay. 0 lets the materials lead, 1 keeps the picture"),
            MODEL,
            K("carry", "choice", "tokens", label="carries", options=tuple(tk.CARRIES), doc="what each material brings"),
            K("push", "float", 1.0, 0, 3, 0.01, "push", "how far toward what the material carries, above 1 past it",
              when=(("carry", "!=", "tokens"),)),
            K("operation", "choice", "replace", label="operation", options=OPS,
              doc="the arithmetic joining the picture's token with the materials'"),
            K("split_size", "float", 1.5, 0.5, 6, 0.1, "split size", "where broad forms end and fine detail begins, in tokens",
              when=(("operation", "==", "split"),)),
            M("where", "choice", "regions", label="where", options=tuple(tk.MIXES),
              doc="where each material goes, with two or more"),
            M("order", "choice", "light", label="order", options=tuple(tk.ORDERS), when=(("where", "==", "regions"),),
              doc="what decides which material goes where"),
            M("side", "angle", 0.0, 0, 6.283, 0.05, "side", "which side the first material comes from",
              when=(("where", "==", "regions"), ("order", "==", "across"))),
            M("seam", "float", 0.35, 0, 1, 0.01, "seam width", "how gradually one material passes into the next",
              when=(("where", "==", "regions"),)),
            M("firmness", "float", 0.6, 0, 1, 0.01, "firmness", "how firmly each place takes the material it resembles",
              when=(("where", "==", "nearest"),)))
           + LOOK_KNOBS, tk.mix),
    Method("tokens.mosh", "tokens", "mosh", "tokens moved out of their places and drawn back: datamoshing",
           (K("hold", "float", 0.0, 0, 1, 0.01, "picture underneath", "how much of the picture's own tokens "
              "stay. 0 lets the moved tokens lead, 1 keeps the picture"),
            MODEL,
            K("way", "choice", "rebuild", label="way", options=tuple(tk.MOSHES), doc="how the tokens are moved"),
            K("source", "choice", "materials", label="pieces from", options=("materials", "picture"),
              when=(("way", "==", "rebuild"),), doc="the materials' pieces, or the picture's own"),
            K("pieces", "choice", 1, label="pieces", options=(1, 2, 4), when=(("way", "==", "rebuild"),),
              doc="how many tokens across each piece"),
            K("wildness", "int", 1, 1, 12, 1, "wildness", "choose among this many closest matches at random",
              when=(("way", "==", "rebuild"),)),
            K("by", "choice", "noise", label="pulled by", options=("light", "depth", "noise"), when=(("way", "==", "drag"),),
              doc="what decides how far each token is pulled"),
            K("length", "float", 8.0, 0, 24, 0.25, "length", "how far the furthest tokens go, in tokens",
              when=(("way", "==", "drag"),)),
            K("side", "angle", 3.1416, 0, 6.283, 0.05, "toward", "which way the smear runs", when=(("way", "==", "drag"),)),
            K("span", "float", 0.5, 0, 1, 0.01, "span", "the share of tokens that are sorted", when=(("way", "==", "sort"),)),
            K("along", "choice", "columns", label="along", options=("rows", "columns"), when=(("way", "==", "sort"),)),
            M("operation", "choice", "replace", label="operation", options=OPS,
              doc="the arithmetic joining the picture's token with the moved one"),
            M("split_size", "float", 1.5, 0.5, 6, 0.1, "split size", "in tokens", when=(("operation", "==", "split"),)),
            M("seed", "int", 0, 0, 9999, 1, "seed", "other random choices"))
           + LOOK_KNOBS, tk.mosh, materials=(0, 3)),
    Method("tokens.weave", "tokens", "weave", "new textures made from the materials alone",
           (MODEL,
            K("pattern", "choice", "blend", label="pattern", options=tuple(tk.WEAVES), doc="how the materials make one texture"),
            K("walk", "float", 0.5, 0, 1, 0.01, "walk", "where between the materials", when=(("pattern", "==", "blend"),)),
            K("patch", "float", 8.0, 2, 32, 0.5, "patch size", "in tokens", when=(("pattern", "==", "patchwork"),)),
            K("balance", "float", 0.5, 0, 1, 0.01, "balance", "lower gives the first material more",
              when=(("pattern", "==", "patchwork"),)),
            K("seam", "float", 0.35, 0, 1, 0.01, "seam width", "from a clean edge to a long morph",
              when=(("pattern", "==", "patchwork"),)),
            K("pieces", "choice", 2, label="pieces", options=(1, 2, 4), when=(("pattern", "in", ("layout", "remix")),)),
            K("wildness", "int", 1, 1, 12, 1, "wildness", "choose among this many closest pieces",
              when=(("pattern", "==", "layout"),)),
            K("reach", "float", 12.0, 0, 48, 0.5, "reach", "how far a piece may travel, in tokens",
              when=(("pattern", "==", "remix"),)),
            K("hold", "float", 0.0, 0, 1, 0.01, "hold the first", "how much of the first material stays",
              when=(("pattern", "in", ("layout", "remix")),)),
            M("seed", "int", 0, 0, 9999, 1, "seed", "another pattern"))
           + LOOK_KNOBS, tk.weave, picture=False),
]

# ─────────────────────────── colour and generative: arcana's own ───────────────────────────

def _arcana(name):
    import importlib

    try:
        return importlib.import_module("arcana." + name)
    except ImportError:                                   # pragma: no cover
        return importlib.import_module(name)


def _pil(a):
    from .inputs import to_image

    return to_image(a)


def _arr(im, W, H):
    from PIL import Image

    return np.asarray(im.convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255.0


def _colour_lab(step, k):
    lt = _arcana("lab_transfer")
    return _arr(lt.lab_color_transfer_pil(_pil(step.c), _pil(step.materials[0]), float(k["strength"])), step.W, step.H)


def _colour_modflows(step, k):
    ct = _arcana("color_transfer")
    out = ct.transfer_colors(_pil(step.c), _pil(step.materials[0]), strength=float(k["strength"]),
                             steps=int(k["steps"]), max_size=max(step.W, step.H))
    return _arr(out, step.W, step.H)


def _style(method):
    def run(step, k):
        st = _arcana("style_transfer")
        src, tgt = _pil(step.materials[0]), _pil(step.c)
        if method == "controlnet":
            out = st.controlnet_transfer(src, tgt, control=k["control"], scale=float(k["scale"]),
                                         control_scale=float(k["control_scale"]), strength=float(k["strength"]),
                                         steps=int(k["steps"]), seed=int(k["seed"]), prompt=k.get("prompt", ""))
        else:
            out = st.transfer(method, src, tgt, **k)
        return _arr(out, step.W, step.H)
    return run


COLOUR = [
    Method("colour.lab", "colour", "LAB", "the material's colour statistics on the picture, in LAB: instant, any machine",
           (K("strength", "float", 1.0, 0, 1, 0.01, "strength", "how fully the colours move"),),
           _colour_lab, materials=(1, 1)),
    Method("colour.modflows", "colour", "ModFlows", "a learned colour flow: the material's palette on the picture",
           (K("strength", "float", 1.0, 0, 1, 0.01, "strength", "how fully the colours move"),
            M("steps", "int", 8, 2, 50, 1, "steps", "more steps, smoother")),
           _colour_modflows, materials=(1, 1), cost="seconds", needs=("modflows",)),
]

GENERATIVE = [
    Method("generative.texture", "generative", "texture detail", "the picture's pixels with the material's detail at high frequency",
           (K("strength", "float", 0.6, 0, 1, 0.01, "strength", "how much of the material's detail"),
            M("levels", "int", 4, 1, 6, 1, "levels", "how many scales of detail")),
           _style("texture"), materials=(1, 1)),
    Method("generative.img2img", "generative", "img2img", "the picture regenerated, aimed by words CLIP finds for the material",
           (K("strength", "float", 0.45, 0, 1, 0.01, "strength", "how much is regenerated"),
            M("steps", "int", 4, 1, 12, 1, "steps", "diffusion steps"),
            K("prompt", "text", "", label="prompt", doc="words added to the material's")),
           _style("img2img"), materials=(1, 1), cost="seconds", needs=("diffusers",)),
    Method("generative.ip_adapter", "generative", "IP-Adapter", "the picture regenerated with the material itself in view",
           (K("scale", "float", 0.7, 0, 1.5, 0.05, "material weight", "how strongly the material steers"),
            K("strength", "float", 0.45, 0, 1, 0.01, "strength", "how much is regenerated"),
            M("steps", "int", 12, 4, 30, 1, "steps", "diffusion steps"),
            K("prompt", "text", "", label="prompt", doc="words beside the material")),
           _style("ip_adapter"), materials=(1, 1), cost="seconds", needs=("diffusers",)),
    Method("generative.controlnet", "generative", "ControlNet", "the material's look held to a map of the picture",
           (K("control", "choice", "depth", label="map", options=("depth", "canny", "luminance"),
              doc="what of the picture holds the generation in place"),
            K("scale", "float", 0.7, 0, 1.5, 0.05, "material weight", "how strongly the material steers"),
            K("control_scale", "float", 0.8, 0, 1.5, 0.05, "map weight", "how strictly the map is followed"),
            K("strength", "float", 0.7, 0, 1, 0.01, "strength", "how much is regenerated"),
            M("steps", "int", 16, 4, 40, 1, "steps", "diffusion steps"),
            M("seed", "int", 2222, 0, 99999, 1, "seed", "another draw"),
            K("prompt", "text", "", label="prompt", doc="words beside the material")),
           _style("controlnet"), materials=(1, 1), cost="seconds", needs=("diffusers",)),
]

FAMILIES = {"pixels": "the material's own pixels laid into the picture: nothing invented",
            "tokens": "the pictures as token grids, mixed by arithmetic and drawn back",
            "colour": "the material's colours on the picture",
            "generative": "a diffusion model redraws the picture with the material in view"}
METHODS = {m.id: m for m in PIXELS + TOKENS + COLOUR + GENERATIVE}


def get(method_id: str) -> Method:
    m = METHODS.get(method_id)
    if m is None:
        raise ValueError(f"No method called {method_id!r}. Methods: {', '.join(METHODS)}.")
    return m


def applies(knob: Knob, values: dict) -> bool:
    for key, op, val in knob.when:
        v = values.get(key)
        if op == "==" and not str(v) == str(val):
            return False
        if op == "!=" and str(v) == str(val):
            return False
        if op == "in" and v not in val:
            return False
    return True


def settle_knobs(method: Method, given: dict) -> dict:
    """The method's knobs: its defaults, overridden by what was given, each brought into range."""
    out = method.defaults()
    names = {k.key: k for k in method.knobs}
    for key, v in (given or {}).items():
        knob = names.get(key)
        if knob is None:
            continue
        if knob.kind in ("float", "angle"):
            v = float(v)
            v = min(max(v, knob.lo), knob.hi) if knob.lo is not None else v
        elif knob.kind == "int":
            v = int(round(float(v)))
            v = min(max(v, int(knob.lo)), int(knob.hi)) if knob.lo is not None else v
        elif knob.kind == "choice":
            if str(v) not in [str(o) for o in knob.options]:
                raise ValueError(f"{key} is one of {', '.join(str(o) for o in knob.options)}, got {v!r}.")
            v = next(o for o in knob.options if str(o) == str(v))
        elif knob.kind == "bool":
            v = str(v).lower() in ("1", "true", "yes", "on")
        out[key] = v
    return out
