"""Recipes: named settings for the Alchemy tab, the flagship setups and the person's own.

A recipe holds any subset of a step's settings: the method and its knobs, the lead, the mask, the
finish, the material zoom. Applying one sets only what it holds and leaves the rest as it is, so a
mask recipe and a finish recipe stack, and the pictures chosen stay put. Settings given explicitly
beside a recipe win over the recipe's.

The built-in recipes reproduce the setups that worked best in latent-lab, measured or chosen by eye
on the person's own photographs. The person's own are kept in <data dir>/alchemy/recipes.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    name: str
    group: str
    doc: str
    settings: dict = field(default_factory=dict)
    own: bool = False

    def public(self):
        return {"name": self.name, "group": self.group, "doc": self.doc, "settings": self.settings, "own": self.own}


VAE = "FLUX.2 VAE"
BUILTIN = [
    # tokens: the flagship looks
    Recipe("water light", "tokens", "FLUX.2 VAE transport: the picture repainted in the material's light and colour, "
           "its forms kept", {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "transport", "push": 1.2}}),
    Recipe("material skin", "tokens", "FLUX.2 VAE split: the picture's forms covered in the material's fine detail",
           {"method": "tokens.mix", "knobs": {"model": VAE, "operation": "split", "split_size": 1.5}}),
    Recipe("surface swap", "tokens", "the material's whole surface (principal axes 1 to 24) on the picture's forms",
           {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "surface axes"}}),
    Recipe("tone swap", "tokens", "only the material's tonality (principal axes 1 to 4): its light and colour",
           {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "tone axes"}}),
    Recipe("fused materials", "tokens", "DINOv2 laid in: the materials become one new surface in the picture",
           {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "blend", "carry": "tokens"}}),
    Recipe("laid by light", "tokens", "each material in its own band of the picture's light, seams drawn by the decoder",
           {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "regions", "order": "light", "seam": 0.5}}),
    Recipe("woven superposition", "tokens", "the material added onto the picture: both visible as one",
           {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "blend", "operation": "add"}}),
    Recipe("mosaic mosh", "tokens", "DINO picks the material's pieces, the VAE draws them: a datamosh mosaic",
           {"method": "tokens.mosh", "knobs": {"model": VAE, "way": "rebuild", "source": "materials", "pieces": 2}}),
    Recipe("self mosh", "tokens", "the picture rebuilt from its own look-alike pieces: extra eyes, more heads",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "rebuild", "source": "picture",
                                               "pieces": 2, "wildness": 4}}),
    Recipe("smeared", "tokens", "tokens dragged along a grown noise field, as a datamosh blooms",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "drag", "by": "noise", "length": 10}}),
    Recipe("pixel sort", "tokens", "the brightest runs sorted by light, a token at a time",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "sort", "span": 0.6, "along": "columns"}}),
    Recipe("blended materials", "tokens", "a new texture from the materials alone, one surface at every point",
           {"method": "tokens.weave", "knobs": {"model": "DINOv2", "pattern": "blend"}}),
    Recipe("patchwork", "tokens", "the materials in grown patches, every seam drawn by the decoder",
           {"method": "tokens.weave", "knobs": {"model": "DINOv2", "pattern": "patchwork"}}),
    # a face hidden in a material: the material leads
    Recipe("hidden: traded edges", "hidden", "a face hidden in the material: at each pixel the stronger edge wins",
           {"method": "pixels.blend", "lead": "material", "knobs": {"wmix": 0.35}}),
    Recipe("hidden: hybrid image", "hidden", "the material up close, the face from a distance",
           {"method": "pixels.bands", "lead": "material", "knobs": {"split": 8, "grain": 1.2}}),
    Recipe("hidden: pushed", "hidden", "the material pushed along the face's slopes",
           {"method": "pixels.displace", "lead": "material", "knobs": {"amp": 0.04, "shade": 0.9}}),
    Recipe("hidden: in tokens", "hidden", "the face carried into the material's tokens by transport",
           {"method": "tokens.mix", "lead": "material", "knobs": {"model": VAE, "carry": "transport", "push": 0.7}}),
    # pixels
    Recipe("lit relief", "pixels", "the material lit by the picture's relief, its highlights held",
           {"method": "pixels.relief", "knobs": {"ambient": 0.45}, "finish": {"highlights": -30, "clarity": 20}}),
    Recipe("combed", "pixels", "the material combed along the picture's contours",
           {"method": "pixels.flow", "knobs": {"length": 0.08, "smear": 0.2}}),
    # generative: latent-lab's transform, the person's texture LoRA drawn into the picture's forms
    Recipe("my texture", "generative", "the picture redrawn in your texture LoRA, its forms held by depth; "
           "add a mask recipe to draw it only there", {"method": "generative.lora"}),
    # Chosen on three portraits (2026-11-08): starting from the material's colours set four textures
    # 2.5 times further apart than IP-Adapter alone, and redraw 0.45 kept each face's eyes and expression.
    Recipe("textured face", "generative", "each face redrawn in the material [1]'s colour and texture, the rest of "
           "the portrait untouched, at its full size",
           {"method": "generative.lora",
            "knobs": {"strength": 0.45, "material": 0.8, "colour": 0.8, "lora": "none"},
            "mask": {"layers": ["faces"], "feather": 4}}),
    # where
    Recipe("on the subject", "masks", "only the main subject, with soft edges",
           {"mask": {"layers": ["subject"], "feather": 3, "follow": 0.3}}),
    Recipe("on the sky", "masks", "only the sky", {"mask": {"layers": ["sky"], "feather": 6}}),
    Recipe("growing in", "masks", "the material growing in from the shadows, ragged like lichen",
           {"mask": {"layers": ["shadows", "growth"], "combine": ["or"], "ragged": 0.5, "feather": 2}}),
    # finishes
    Recipe("golden hour", "finish", "warm, low sun: warmer, brighter, highlights held, a soft vignette",
           {"finish": {"temperature": 35, "exposure": 0.2, "contrast": 15, "highlights": -30, "vignette": -20}}),
    Recipe("matte film", "finish", "lifted blacks, softer contrast, muted colour and grain",
           {"finish": {"blacks": 30, "contrast": -15, "saturation": -15, "grain": 25}}),
    Recipe("crisp", "finish", "more texture, clarity and a little dehaze",
           {"finish": {"texture": 30, "clarity": 25, "dehaze": 10}}),
    Recipe("moody", "finish", "darker, cooler, deeper shadows",
           {"finish": {"exposure": -0.4, "temperature": -20, "shadows": -30, "contrast": 20, "vignette": -30}}),
]


def store_path() -> str:
    """The file the person's own recipes are kept in."""
    return _store()


def _store() -> str:
    try:
        from .. import paths
    except ImportError:                                   # pragma: no cover
        import paths
    return os.path.join(paths.ensure_dir(os.path.join(paths.data_dir(), "alchemy")), "recipes.json")


def own() -> list[Recipe]:
    try:
        with open(_store(), encoding="utf-8") as f:
            return [Recipe(d["name"], d.get("group", "mine"), d.get("doc", ""), d.get("settings", {}), True)
                    for d in json.load(f)]
    except (OSError, ValueError, KeyError):
        return []


def every() -> list[Recipe]:
    return BUILTIN + own()


def get(name: str) -> Recipe:
    for r in reversed(every()):          # the person's own win over a built-in of the same name
        if r.name == name:
            return r
    raise ValueError(f"No recipe called {name!r}. Recipes: {', '.join(r.name for r in every())}.")


SETTING_KEYS = ("method", "knobs", "lead", "mask", "finish", "zoom", "crop", "weights")


def apply(recipe: Recipe, spec: dict) -> dict:
    """The spec with the recipe's settings under it: what the spec gives wins, dicts merged key by key."""
    out = {k: v for k, v in spec.items() if k != "recipe"}
    for k, v in recipe.settings.items():
        if k not in SETTING_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = {**v, **(out.get(k) or {})}
        elif out.get(k) in (None, "", []):
            out[k] = v
    out["recipe"] = recipe.name
    return out


def save(name: str, settings: dict, doc: str = "", group: str = "mine") -> Recipe:
    """Keep the person's own recipe, replacing one of the same name."""
    name = name.strip()
    if not name:
        raise ValueError("A recipe needs a name.")
    settings = {k: v for k, v in (settings or {}).items() if k in SETTING_KEYS}
    mine = [r for r in own() if r.name != name] + [Recipe(name, group, doc, settings, True)]
    path = _store()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([{"name": r.name, "group": r.group, "doc": r.doc, "settings": r.settings} for r in mine], f, indent=1)
    os.replace(tmp, path)
    return mine[-1]
