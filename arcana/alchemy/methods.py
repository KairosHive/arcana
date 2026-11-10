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
    source: str = ""             # options read when asked, for things made while the app runs: labels, loras

    def public(self):
        out = {k: v for k, v in self.__dict__.items() if v not in ((), None, "") or k in ("default",)}
        if self.source:
            out["options"] = list(options_of(self))
        return out


def options_of(knob: Knob) -> tuple:
    """A choice knob's options. Labels and LoRAs are made while the app runs, so they are read each time."""
    if knob.source == "labels":
        return _arcana("ip_labels").options()
    if knob.source == "axes":
        return _arcana("ip_labels").axis_options()
    if knob.source == "loras":
        return _arcana("loras").options()
    return knob.options


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
    focused: bool = False        # draws only around a small mask, so the frame may stay at full size
    listed: bool = True          # shown in the tab's menu; an unlisted method still runs for recipes, films, ops

    def public(self):
        return dict(id=self.id, family=self.family, label=self.label, doc=self.doc, cost=self.cost,
                    materials=list(self.materials), picture=self.picture, needs=list(self.needs),
                    knobs=[k.public() for k in self.knobs], listed=self.listed)

    def defaults(self):
        return {k.key: k.default for k in self.knobs}


def K(key, kind, default, lo=None, hi=None, step=None, label="", doc="", options=(), when=(), px=False,
      section="main", source=""):
    return Knob(key, kind, default, lo, hi, step, label or key.replace("_", " "), doc, tuple(options),
                tuple(tuple(w) for w in when), px, section, source)


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
    Method("tokens.mix", "tokens", "materials into the picture", "the materials laid into the picture's DINO or "
           "VAE tokens, by arithmetic: fused into one surface, by regions of light or depth, or carried by transport",
           (K("picture", "float", 0.5, 0, 1, 0.01, "picture underneath", "how much of the picture's own tokens "
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
    Method("tokens.mosh", "tokens", "glitch", "tokens moved out of their places and drawn back: datamoshing",
           (K("hold", "float", 0.5, 0, 1, 0.01, "picture underneath", "how much of the picture's own tokens "
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
    Method("tokens.weave", "tokens", "new texture from the materials", "a hybrid texture made from the materials "
           "alone in DINO's tokens: blended into one surface, in patchwork, laid out or remixed",
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


def _steer(k):
    """The IP label vector a step's label knobs ask for, or None for no label."""
    if k.get("label", "none") in (None, "", "none"):
        return None
    return _arcana("ip_labels").steer(k["label"], k.get("label2"), float(k.get("mix", 0.5)),
                                      float(k.get("push", 2.0)), float(k.get("spread", 0.0)),
                                      int(k.get("seed", 0)), k.get("axis"), float(k.get("axis_amount", 0.0)))


def _label_guide(k):
    """With a label and no material, the picture nearest the label's centre, for what wants a picture
    (the colour the texture LoRA starts from)."""
    if k.get("label", "none") in (None, "", "none"):
        return None
    near = _arcana("ip_labels").load(k["label"]).nearest(1)
    if not near:
        return None
    from PIL import Image, ImageOps

    with Image.open(near[0]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((768, 768))
        return im.copy()


def _style(method):
    def run(step, k):
        st = _arcana("style_transfer")
        tgt = _pil(step.c)
        src = _pil(step.materials[0]) if step.materials else None
        vec = _steer(k) if method in ("controlnet", "ip_adapter") else None
        if src is None and vec is None:
            raise ValueError("This method needs a material [1] or an IP label.")
        toward = float(k.get("toward", 1.0))
        if method == "controlnet":
            second, split = _split(step, k, tgt)
            if second is not None:
                vec = _arcana("ip_labels").steer(k["label"], None, 0.0, float(k.get("push", 2.0)),
                                                 float(k.get("spread", 0.0)), int(k.get("seed", 0)))
            against = _against(k)
            out, _map = st.controlnet_transfer(src, tgt, control=k["control"], scale=float(k["scale"]),
                                               control_scale=float(k["control_scale"]),
                                               strength=float(k["strength"]), steps=int(k["steps"]),
                                               seed=int(k["seed"]), prompt=k.get("prompt", ""),
                                               ip_embed=vec, toward=toward, against=against,
                                               second=second, split=split)
            if k.get("then", "none") not in (None, "", "none"):
                # latent-lab's finishing pass: a second material redraws the surface lightly, the first having
                # built the forms.
                fin = _arcana("ip_labels").steer(k["then"], None, 0.0, float(k.get("push", 2.0)), 0.0,
                                                 int(k.get("seed", 0)))
                out, _map = st.controlnet_transfer(None, out, control=k["control"], scale=float(k["scale"]),
                                                   control_scale=float(k["control_scale"]),
                                                   strength=float(k.get("then_s", 0.45)), steps=int(k["steps"]),
                                                   seed=int(k["seed"]) + 1, prompt=k.get("prompt", ""),
                                                   ip_embed=fin)
        elif method == "ip_adapter":
            out = st.ip_adapter_transfer(src, tgt, scale=float(k["scale"]), strength=float(k["strength"]),
                                         steps=int(k["steps"]), seed=int(k.get("seed", 2222)),
                                         prompt=k.get("prompt", ""), ip_embed=vec, toward=toward)
        else:
            out = st.transfer(method, src, tgt, **k)
        return _arr(out, step.W, step.H)
    return run


def _against(k):
    if k.get("against", "none") in (None, "", "none"):
        return None
    return _arcana("ip_labels").steer(k["against"], None, 0.0, float(k.get("push", 2.0)), 0.0, 0)


def _split(step, k, tgt):
    """(second material's vector, where the first goes) when the step splits two labels over the picture by its
    depth or its light; (None, None) otherwise. The split is a soft quantile: `split_at` is the share of the
    picture the first label takes, nearest or brightest first, over a band of `softness`."""
    if k.get("split", "off") == "off" or k.get("label", "none") in (None, "", "none") or \
            k.get("label2", "none") in (None, "", "none"):
        return None, None
    from PIL import Image

    if k["split"] == "depth":
        field = np.asarray(step.depth(), np.float32)
    else:
        field = np.asarray(step.c, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    w, h = tgt.size
    field = np.asarray(Image.fromarray(field).resize((w, h), Image.BILINEAR), np.float32)
    share = float(k.get("split_at", 0.5))
    cut = float(np.quantile(field, 1 - share))
    band = max(1e-6, float(np.quantile(field, min(1, 1 - share + 0.1)) - np.quantile(field, max(0, 1 - share - 0.1))))
    m = np.clip((field - cut) / band + 0.5, 0, 1)
    second = _arcana("ip_labels").steer(k["label2"], None, 0.0, float(k.get("push", 2.0)), float(k.get("spread", 0.0)),
                                        int(k.get("seed", 0)))
    return second, m


def _without_material(knob):
    """A knob that applies only when there is no material [1] (the editor draws its swatch then)."""
    import dataclasses

    return dataclasses.replace(knob, when=knob.when + (("_materials", "==", 0),))


def _transform_knobs():
    """ControlNet's own moves with labels, from latent-lab's transform: two labels split over the picture,
    a label to steer away from, and a finishing pass."""
    two = (("label", "!=", "none"), ("label2", "!=", "none"))
    return (
        K("split", "choice", "off", label="two materials by", options=("off", "depth", "light"),
          doc="with a second label: off mixes the two everywhere; depth puts the first label on the near part and "
              "the second on the far part, light the first on the light part and the second in the shadows",
          when=two),
        K("split_at", "float", 0.5, 0.1, 0.9, 0.02, "first takes",
          "the share of the picture the first label takes, nearest or lightest first",
          when=two + (("split", "!=", "off"),)),
        M("against", "choice", "none", label="steer away from", options=("none",), source="labels",
          when=(("label", "!=", "none"),),
          doc="a label the drawing moves away from: toward crust, away from water"),
        M("then", "choice", "none", label="finish with", options=("none",), source="labels",
          doc="a label for a second, lighter pass over the result: the first builds the forms, this one the surface. "
              "The result goes back through pixels between the passes, which can break a drawn texture into "
              "confetti (seen on lichen finished with ice); keep it light"),
        M("then_s", "float", 0.45, 0.2, 0.7, 0.05, "finish strength", "how much the finishing pass redraws",
          when=(("then", "!=", "none"),)),
    )


def _label_knobs(seed=True):
    """The knobs that steer a diffusion method by IP labels (ip_labels.py)."""
    on = (("label", "!=", "none"),)
    out = (
        K("label", "choice", "none", label="IP label", options=("none",), source="labels",
          doc="a position in IP-Adapter space measured on a set, words or a picture (Labels, beside the "
              "pictures); none steers by the material [1] alone"),
        K("label2", "choice", "none", label="second label", options=("none",), source="labels", when=on,
          doc="a second label, mixed with the first"),
        K("mix", "float", 0.5, 0, 1, 0.01, "mix", "0 the first label, 1 the second; a film lane on it walks "
          "from one to the other", when=on + (("label2", "!=", "none"),)),
        K("toward", "float", 1.0, 0, 1.5, 0.05, "toward the label",
          "0 keeps the material [1]'s own look, 1 the label's, past 1 beyond it",
          when=on + (("_materials", ">", 0),)),
        # Default 2: label centres from one archive sit at cosine 0.9 to 0.95 of each other and drew
        # alike at 1 (ice, moss and fire on one picture, 2026-11-09); at 2 each drew its own surface.
        M("push", "float", 2.0, 0.5, 3, 0.05, "push",
          "how far the label sits from the average picture: past 1 more of what sets it apart, below 1 "
          "more ordinary", when=on),
        M("spread", "float", 0.25, 0, 1, 0.05, "spread",
          "0 the label's centre, which reads generic; toward 1 one member's own look, picked by the seed",
          when=on),
        K("axis", "choice", "none", label="concept axis", options=("none",), source="axes", when=on,
          doc="a direction fitted between two places (wet against dry): the label is moved along it"),
        K("axis_amount", "float", 0.0, -2.0, 2.0, 0.05, "along the axis",
          "how far along the axis: 1 is the whole distance between its two ends, positive toward its first "
          "end, negative toward its second", when=on + (("axis", "!=", "none"),)),
    )
    if seed:
        out += (M("seed", "int", 2222, 0, 99999, 1, "seed", "another draw"),)
    return out


FOCUS_BELOW = 0.35   # a mask covering less of the frame than this is drawn on a crop around it
FOCUS_PAD = 0.35     # the crop reaches this share of the mask's size past it on every side


def mask_box(mask, below=FOCUS_BELOW, pad=FOCUS_PAD):
    """(x0, y0, x1, y1), the region around a small mask, or None for no mask, an empty one or a large one."""
    if mask is None:
        return None
    on = mask > 0.05
    if not on.any() or on.mean() >= below:
        return None
    ys, xs = np.nonzero(on)
    H, W = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad) + 8, int((x1 - x0) * pad) + 8
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


FOCUS_LONG = 768     # the long side a focused region is drawn at: what SD 1.5 draws without inventing


def _full_resolution(step, box):
    """The region of the working frame `box` covers, read from the photograph itself (the frame is
    the photograph covered and centred, as load_frame makes it). None when there is no file to read."""
    import os
    from PIL import Image, ImageOps

    path = str(getattr(step, "picture_key", "") or "").rsplit("@", 1)[0]
    if not path or not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        ow, oh = im.size
        s = max(step.W / ow, step.H / oh)
        ox, oy = (ow * s - step.W) / 2, (oh * s - step.H) / 2
        x0, y0, x1, y1 = box
        part = im.crop(tuple(int(round(v)) for v in ((x0 + ox) / s, (y0 + oy) / s, (x1 + ox) / s, (y1 + oy) / s)))
    k = min(1.0, FOCUS_LONG / max(part.size))
    return part.resize((max(8, int(part.width * k)), max(8, int(part.height * k))), Image.LANCZOS) if k < 1 else part


def _texture_lora(step, k):
    """latent-lab's transform: the picture redrawn on SD 1.5 under a texture LoRA, its forms held by a
    ControlNet map. A material [1] is optional and steers through IP-Adapter beside the LoRA. The mask
    is laid on after, as for every method, so the new texture appears only inside it.

    With `focus` and a small mask (a face covers 1 to 6% of a portrait), only the region around the
    mask is drawn and laid back, its pixels read from the photograph at full resolution: drawn whole,
    a face got 50 to 150 pixels of the drawing and came back as mush, and cut from the working frame
    it was a 100 px smudge the model could only invent over."""
    st = _arcana("style_transfer")
    src = _pil(step.materials[0]) if step.materials else None
    vec = _steer(k)
    guide = src if src is not None else (_label_guide(k) if vec is not None else None)
    box = mask_box(step.mask) if k.get("focus", True) else None
    x0, y0, x1, y1 = box or (0, 0, step.W, step.H)
    region = (_full_resolution(step, box) if box else None) or _pil(step.c[y0:y1, x0:x1])
    if guide is not None and float(k.get("colour", 0)) > 0:
        # img2img keeps the colours it starts from at any gentle redraw, so a guide steering only
        # through IP-Adapter came back as streaks on the picture's own skin tone. Starting from the
        # guide's palette (LAB statistics) lets each guide's colour arrive. With a label and no
        # material, the label's most central picture is the guide.
        region = _arcana("lab_transfer").lab_color_transfer_pil(region, guide, float(k["colour"]))
    prompt = k.get("prompt", "")
    if k["lora"] != "none":
        # A LoRA trained in Arcana answers its own trigger ("in ice style") and was trained on captions
        # of its pictures; the default prompt names latent-lab's trigger, so it is swapped for the
        # LoRA's typical caption ("a close-up photograph of rippled water, glassy, ..., in ice style").
        lr = _arcana("loras")
        own = lr.trigger(k["lora"])
        if own != lr.TRIGGER and lr.TRIGGER in prompt:
            prompt = prompt.replace(lr.TRIGGER, lr.prompt(k["lora"]) or own)
    out, _map = st.controlnet_transfer(src, region, control=k["control"], scale=float(k["material"]),
                                       control_scale=float(k["structure"]), strength=float(k["strength"]),
                                       steps=int(k["steps"]), guidance=float(k["guidance"]), seed=int(k["seed"]),
                                       prompt=prompt, lora=k["lora"], lora_weight=float(k["lora_w"]),
                                       avoid=k.get("avoid", ""), ip_embed=vec, toward=float(k.get("toward", 1.0)))
    if (x1 - x0) > out.width * 1.25:
        # A region of a full-size photograph is larger than the drawing (diffusion draws at
        # most FOCUS_LONG): Real-ESRGAN takes it up, so it sits as sharp as the photograph around it.
        from .spaces import enlarge

        out = _pil(enlarge(np.asarray(out.convert("RGB"), np.float32) / 255.0))
    drawn = _arr(out, x1 - x0, y1 - y0)
    if box is None:
        return drawn
    whole = step.c.copy()
    whole[y0:y1, x0:x1] = drawn
    return whole


SWATCH_FROM = ("material [1]", "IP label", "words")


def _swatch(step, k):
    """What the editor is shown as the material. A material [1] is shown as it is. Without one, the material
    is drawn by Stable Diffusion 1.5 from a place you choose: an IP label (a position measured on your own
    pictures, a set, the matches for some words or one picture), words, or both, the words then added to the
    label's place. Without any of these there is nothing to show, and the step says so."""
    if step.materials:
        return _pil(step.materials[0])
    vec, words = _steer(k), (k.get("words") or "").strip()
    if vec is None and not words:
        raise ValueError("The material editor needs a material: a picture on [1], an IP label, or words "
                         "(the material's words, such as molten copper).")
    if k.get("swatch_by") == "the pictures (FLUX)":
        return _pictures_swatch(k, words)
    return _arcana("style_transfer").drawn_swatch(vec, words, seed=int(k.get("seed", 0)))


PICTURES_PROMPT = ("A close-up photograph of one continuous material surface with the texture, grain and colours of "
                   "the reference images, filling the frame.")


def _pictures_swatch(k, words=""):
    """The material drawn by FLUX.2 klein looking at the label's own pictures: a few members near its centre,
    unlike each other. With a second label, the pictures come from both in proportion to the mix. Faithful
    to the photographs where the place's drawing (Stable Diffusion at the label's vector) is the archive's
    idea pushed further."""
    from PIL import Image, ImageOps

    from . import editor as ed

    lab = _arcana("ip_labels")
    refs = []
    if k.get("label", "none") not in (None, "", "none"):
        n2 = 0
        if k.get("label2", "none") not in (None, "", "none"):
            n2 = int(round(4 * float(k.get("mix", 0.5))))
            refs += lab.load(k["label2"]).representatives(n2)
        refs = lab.load(k["label"]).representatives(4 - n2) + refs
    pics = []
    for p in refs:
        im = Image.open(p)
        im.draft("RGB", (1024, 1024))
        pics.append(ImageOps.exif_transpose(im).convert("RGB"))
    prompt = PICTURES_PROMPT if not words else (
        f"A close-up photograph of {words}, with the texture, grain and colours of the reference images, filling "
        f"the frame." if pics else f"A close-up photograph of {words}, richly detailed, filling the frame.")
    key = (tuple(refs), prompt, int(k.get("seed", 0)))
    hit = _FLUX_SWATCHES.get(key)
    if hit is None:
        hit = ed.EDITOR.generate(prompt, pics, (768, 768), steps=6, seed=int(k.get("seed", 0)))
        _FLUX_SWATCHES[key] = hit
        while len(_FLUX_SWATCHES) > 16:
            _FLUX_SWATCHES.pop(next(iter(_FLUX_SWATCHES)))
    return hit


_FLUX_SWATCHES: dict = {}


def _edit(step, k):
    """latent-lab's flagship (its atlas): the masked part redrawn by FLUX.2 klein, an editor model shown a
    swatch of the material and told, in words, to put it on the thing's own folds and light. The swatch is
    the material [1]; without one, an IP label's material drawn first by Stable Diffusion 1.5."""
    from . import editor as ed

    swatch = _swatch(step, k)
    mask = step.mask if step.mask is not None else np.ones((step.H, step.W), np.float32)
    what = (k.get("what") or "").strip() or getattr(step, "what", None)
    prompt = (k.get("instruction") or "").strip() or ed.instruction(what)
    out = ed.EDITOR(_pil(step.c), mask, swatch, prompt, steps=6 if k.get("quality") == "full" else 4,
                    strength=float(k["strength"]), seed=int(k.get("seed", 0)), focus=bool(k.get("focus", True)))
    return _arr(out, step.W, step.H)


_LORAS = _arcana("loras").options()


# Texture detail is a pixel method (no model, instant): listed here, under Pixels, as detail transfer. The
# generative.texture id stays for the films and steps saved with it.
PIXELS.append(Method("pixels.detail", "pixels", "detail transfer",
                     "the material's fine detail laid onto the picture's own pixels, scale by scale: no model, instant",
                     (K("amount", "float", 0.6, 0, 1, 0.01, "amount", "how much of the material's detail is laid on"),
                      M("levels", "int", 4, 1, 6, 1, "levels", "how many scales of detail")),
                     lambda step, k: _style("texture")(step, {**k, "strength": k["amount"]}), materials=(1, 1)))

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
           _style("texture"), materials=(1, 1), listed=False),   # in the menu as pixels.detail
    Method("generative.img2img", "generative", "picture toward words",
           "the picture redrawn toward words: the ones CLIP finds for the material [1], or your own in the prompt; "
           "fast",
           (K("strength", "float", 0.45, 0, 1, 0.01, "strength", "how much is regenerated"),
            M("steps", "int", 4, 1, 12, 1, "steps", "diffusion steps"),
            K("prompt", "text", "", label="prompt", doc="words added to the material's")),
           _style("img2img"), materials=(1, 1), cost="seconds", needs=("diffusers",)),
    Method("generative.ip_adapter", "generative", "picture loosely in a material",
           "the picture redrawn with the material [1] or an IP label in view and no map of its forms, so they drift "
           "and reinvent themselves more freely than in \"picture in a material\"",
           (K("scale", "float", 0.7, 0, 1.5, 0.05, "material weight", "how strongly the material or label steers"),
            K("strength", "float", 0.45, 0, 1, 0.01, "strength", "how much is regenerated"),
            M("steps", "int", 12, 4, 30, 1, "steps", "diffusion steps"),
            K("prompt", "text", "", label="prompt", doc="words beside the material")) + _label_knobs(),
           _style("ip_adapter"), materials=(0, 1), cost="seconds", needs=("diffusers",)),
    Method("generative.controlnet", "generative", "picture in a material",
           "the whole picture redrawn in a material [1] or an IP label, its forms held by a map of its depth; "
           "two labels can split the picture by depth or light",
           (K("strength", "float", 0.75, 0, 1, 0.01, "redraw amount", "how much of the picture is redrawn"),
            K("scale", "float", 0.85, 0, 1.5, 0.05, "material weight", "how strongly the material or label steers"),
            K("control_scale", "float", 0.8, 0, 1.5, 0.05, "hold structure", "how strictly the picture's forms are kept"),
            M("control", "choice", "depth", label="map", options=("depth", "canny", "luminance"),
              doc="what of the picture holds the drawing in place"),
            M("steps", "int", 16, 4, 40, 1, "steps", "diffusion steps"),
            M("prompt", "text", "", label="prompt", doc="words beside the material")) + _label_knobs()
           + _transform_knobs(),
           _style("controlnet"), materials=(0, 1), cost="seconds", needs=("diffusers",)),
    Method("generative.lora", "generative", "picture in a learned look",
           "the picture, or a masked part of it, redrawn in the look of a model trained on one of your sets "
           "(a LoRA, made in Sets & labels): the set's own softness and palette rather than one material; its forms "
           "held by a map of its depth",
           (K("strength", "float", 0.75, 0.2, 0.98, 0.02, "redraw amount",
              "how much of the picture is redrawn: low keeps it, high reinvents it"),
            K("lora", "choice", _LORAS[1] if len(_LORAS) > 1 else "none", label="learned look", options=_LORAS,
              source="loras",
              doc="a model trained on one of your sets (a LoRA: set/checkpoint), from Sets & labels, ARCANA_LORA_DIRS "
                  "or ~/latent-lab-models; none draws with the plain model, steered only by the material or label"),
            K("lora_w", "float", 1.4, 0, 2, 0.05, "look strength",
              "how strongly the learned look applies; about 1.0 to 1.8 works best", when=(("lora", "!=", "none"),)),
            M("prompt", "text", "in goofi style", label="prompt",
              doc="words that steer the drawing; the LoRA's trigger words call up its look"),
            M("structure", "float", 1.0, 0, 1.6, 0.05, "hold structure", "how firmly the picture's shapes are kept"),
            K("material", "float", 0.5, 0, 1, 0.05, "material weight",
              "with a material [1] or an IP label, how strongly its look steers beside the LoRA; with "
              "neither it has no effect"),
            K("colour", "float", 0.8, 0, 1, 0.05, "material colour",
              "with a material [1] (or a label's most central picture), how much of its palette the redraw "
              "starts from; at 0 each material arrives only as streaks on the picture's own colours "
              "(measured: 2.5 times less apart)"),
            M("focus", "bool", True, label="draw at the mask",
              doc="with a small mask, such as a face, only the region around it is drawn, at full resolution"),
            M("avoid", "text", "animal, creature", label="avoid",
              doc="what the drawing steers away from; without it, high redraw amounts turn clumps into creatures"),
            M("control", "choice", "depth", label="map", options=("depth", "canny", "luminance"),
              doc="what of the picture holds the drawing in place"),
            M("steps", "int", 20, 4, 40, 1, "steps", "diffusion steps"),
            M("guidance", "float", 7.0, 1, 12, 0.5, "guidance", "how closely the words are followed"),
            M("seed", "int", 0, 0, 99999, 1, "seed", "another draw")) + _label_knobs(seed=False),
           # Out of the menu (2026-11-09): no test found a learned look beating a label or no LoRA at all;
           # LORA_ROADMAP.md is how it comes back. Recipes, films and the ops still run it.
           _texture_lora, materials=(0, 1), cost="seconds", needs=("diffusers",), focused=True, listed=False),
]

GENERATIVE.append(
    Method("generative.edit", "generative", "material on a thing",
           "the masked part redrawn by an editor model (FLUX.2 klein) shown the material: it puts the material "
           "on the thing's own folds and light; name the thing in the mask, such as thing:the dress",
           (K("strength", "float", 1.0, 0.5, 1.0, 0.01, "redraw amount",
              "how much of the masked part is redrawn: below 1 the picture's own shows through more"),
            K("swatch_by", "choice", "the pictures (FLUX)", label="draw the material from", when=(("_materials", "==", 0),),
              options=("the pictures (FLUX)", "the place (SD 1.5)"),
              doc="with no material [1]: FLUX.2 klein looking at the label's own pictures (faithful to your "
                  "photographs), or Stable Diffusion at the label's place in IP space (mixes, push and spread "
                  "move it, beyond what the archive holds)"),
            K("words", "text", "", label="material in words", when=(("_materials", "==", 0),),
              doc="with no material [1]: the material described in words (molten copper, wet slate), drawn by "
                  "Stable Diffusion; with an IP label, added to the label's place"),
            K("what", "text", "", label="what it is",
              doc="what the mask holds, in words (the dress, the cliff); empty reads it from the mask"),
            M("quality", "choice", "full", label="quality", options=("draft", "full"),
              doc="draft is 4 steps, full is 6"),
            M("instruction", "text", "", label="instruction",
              doc="the whole instruction, in place of \"Replace the surface of <what> with the material from the "
                  "reference image ...\""),
            M("focus", "bool", True, label="work around the mask",
              doc="when the mask holds only part of the picture, only its surroundings are drawn, at full "
                  "resolution: faster, and a small thing gets more pixels"),
            M("seed", "int", 1, 0, 99999, 1, "seed", "another draw"))
           + tuple(_without_material(kn) for kn in _label_knobs(seed=False) if kn.key != "toward"),
           _edit, materials=(0, 1), cost="seconds", needs=("diffusers", "gpu"), focused=True))

FAMILIES = {"pixels": "the material's own pixels laid into the picture: nothing invented",
            "tokens": "the pictures as token grids, mixed by arithmetic and drawn back",
            "colour": "the material's colours on the picture",
            "generative": "a diffusion model redraws a thing in the picture, or the whole picture, in a material "
                          "[1], an IP label, or a look learned from one of your sets"}
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
        if op == ">" and not float(v or 0) > float(val):
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
            opts = options_of(knob)
            if str(v) not in [str(o) for o in opts]:
                raise ValueError(f"{key} is one of {', '.join(str(o) for o in opts)}, got {v!r}.")
            v = next(o for o in opts if str(o) == str(v))
        elif knob.kind == "bool":
            v = str(v).lower() in ("1", "true", "yes", "on")
        out[key] = v
    return out
