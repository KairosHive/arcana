"""One Alchemy step, from what was asked to the picture made.

    spec = {"picture": "japan:1423", "materials": ["japan:88"], "weights": [1.0],
            "lead": "picture", "method": "tokens.mix", "knobs": {"model": "FLUX.2 VAE", "carry": "transport"},
            "mask": {"layers": ["sky"]}, "finish": {"exposure": 0.3}, "zoom": 1.0, "crop": 0,
            "recipe": "water light", "enlarge": False}
    image, info = run(spec, long_side=768)

The stages, in order: a recipe (if named) fills the settings it holds; the method is looked up and
its knobs settled; the pictures are read at the frame (with `lead` "material", the first material
and the picture swap places, so the material leads and the picture is hidden in it); the mask is
built; the method runs; its result is laid into the picture through the mask; the finish is
applied; Real-ESRGAN enlarges it four times if asked. `view` "mask" draws the mask instead, and
"before" the picture as it was.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import finish as fn
from . import mask as mk
from . import methods as mt
from . import recipes as rc
from .inputs import crop_of, frame, load_frame, resolve, to_image

TOKEN_MULTIPLE = 16
DRAWN_KEEP = 6                       # steps kept before their finish, so a finish change is instant
_drawn: collections.OrderedDict = collections.OrderedDict()


@dataclass
class Step:
    """What a method reads: the picture and materials at the frame, their keys, the mask."""
    W: int
    H: int
    c: np.ndarray
    picture_key: str
    materials: list
    material_keys: list
    weights: list
    mask: np.ndarray | None
    depth: object = None


def resolved(spec: dict) -> dict:
    """The spec with its recipe's settings filled in under what was given explicitly."""
    spec = dict(spec or {})
    if spec.get("recipe"):
        spec = rc.apply(rc.get(spec["recipe"]), spec)
    spec.setdefault("method", "tokens.mix")
    return spec


def run(spec: dict, long_side: int = 768, view: str = "result"):
    t0 = time.perf_counter()
    spec = resolved(spec)
    method = mt.get(spec["method"])
    knobs = mt.settle_knobs(method, spec.get("knobs") or {})
    pic_ref = spec.get("picture")
    mat_refs = [m for m in (spec.get("materials") or []) if m][:3]
    lo, hi = method.materials
    if view == "result" and len(mat_refs) < lo:
        raise ValueError(f"{method.id} needs at least {lo} material{'s' if lo != 1 else ''}. Give them as [1], [2], [3].")
    if (method.picture or view != "result") and not pic_ref:
        raise ValueError(f"{method.id} needs a picture [T]." if view == "result" else "The mask and before views need a picture [T].")
    if not pic_ref and not mat_refs:
        raise ValueError("A step needs a picture [T] or a material.")
    if spec.get("lead") == "material" and method.picture and mat_refs:
        pic_ref, mat_refs = mat_refs[0], [pic_ref] + mat_refs[1:]
    pic_path = resolve(pic_ref) if pic_ref else None
    mat_paths = [resolve(r) for r in mat_refs[:hi]]
    base = pic_path or mat_paths[0]
    multiple = TOKEN_MULTIPLE if method.family == "tokens" else 8
    W, H = frame(base, int(long_side), multiple)
    zoom, crop = float(spec.get("zoom", 1.0) or 1.0), int(spec.get("crop", 0) or 0)
    c = load_frame(pic_path, W, H) if pic_path else np.zeros((H, W, 3), np.float32) + 0.5
    mats = [crop_of(p, W, H, zoom, crop) for p in mat_paths]
    pkey = f"{pic_path}@{W}x{H}"
    mkeys = [f"{p}@{W}x{H}@z{zoom}@s{crop}" for p in mat_paths]

    scene = mk.Scene(c, pkey, mats[0] if mats else None, load=lambda ref, w, h: load_frame(resolve(ref), w, h))
    m = mk.build(spec.get("mask"), scene) if pic_path else None
    if view == "mask":
        out = mk.shown(m if m is not None else np.ones((H, W), np.float32), c)
        return to_image(out), _info(spec, method, W, H, t0, m)
    if view == "before":
        return to_image(c), _info(spec, method, W, H, t0, m)

    # The finish is the last stage, so moving only the finish reuses the step drawn before it.
    key = json.dumps([{k: v for k, v in spec.items() if k not in ("finish", "enlarge")}, W, H],
                     sort_keys=True, default=str)
    out = _drawn.get(key)
    if out is None:
        step = Step(W, H, c, pkey, mats, mkeys, list(spec.get("weights") or [1.0] * len(mats)), m,
                    depth=lambda: scene.ask("depth"))
        if method.family == "tokens":
            knobs.setdefault("seed", crop)
        out = method.run(step, knobs)
        if out.shape[:2] != (H, W):
            out = mk.cx._resize(out, W, H, mk.Image.BILINEAR) if out.ndim == 2 else \
                np.asarray(to_image(out).resize((W, H), mk.Image.LANCZOS), np.float32) / 255.0
        if m is not None and method.picture:
            out = m[..., None] * out + (1 - m[..., None]) * c
        _drawn[key] = out
        while len(_drawn) > DRAWN_KEEP:
            _drawn.popitem(last=False)
    else:
        _drawn.move_to_end(key)
    out = fn.apply(out, spec.get("finish"), m, seed=crop)
    image = to_image(out)
    if spec.get("enlarge"):
        from .spaces import enlarge

        image = to_image(enlarge(np.asarray(image, np.float32) / 255.0))
    return image, _info(spec, method, image.width, image.height, t0, m)


def mask_view(picture: str, mask, material: str | None = None, long_side: int = 768):
    """A mask on its own, drawn over its picture: (PIL image, share of the picture it lets in)."""
    pic_path = resolve(picture)
    W, H = frame(pic_path, int(long_side), 8)
    c = load_frame(pic_path, W, H)
    mat = crop_of(resolve(material), W, H) if material else None
    scene = mk.Scene(c, f"{pic_path}@{W}x{H}", mat, load=lambda ref, w, h: load_frame(resolve(ref), w, h))
    m = mk.build(mask, scene)
    if m is None:
        m = np.ones((H, W), np.float32)
    return to_image(mk.shown(m, c)), round(float(m.mean()), 3)


def _info(spec, method, W, H, t0, m):
    return {"method": method.id, "family": method.family, "size": [W, H],
            "ms": round((time.perf_counter() - t0) * 1000), "mask_share": None if m is None else round(float(m.mean()), 3),
            "recipe": spec.get("recipe")}
