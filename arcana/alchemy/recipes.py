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
    listed: bool = True

    def public(self):
        return {"name": self.name, "group": self.group, "doc": self.doc, "settings": self.settings, "own": self.own}


VAE = "FLUX.2 VAE"
# Grouped by what the recipe does to the picture, in this order, so the menu reads from the most
# targeted change to the most global, then the two kinds of modifier that stack on any of them:
#
#   material on a thing     one named thing remade in a material (the material editor)
#   picture in a material   the whole picture redrawn in a material, its forms kept
#   materials blended       the materials become one new surface
#   hidden image            a face or a picture hidden in the material
#   glitch                  the picture taken apart and rebuilt
#   relief                  the material's own pixels shaped by the picture's light
#   where                   a mask: stacks on any recipe
#   look                    a finish: stacks on any recipe
#
# Each group holds a few recipes that look clearly different. Recipes that drew too close to another
# are retired: kept, so a film or a saved step that names one still opens, but out of the menu.
GROUPS = ("material on a thing", "picture in a material", "materials blended", "hidden image", "glitch",
          "relief", "where", "look")
BUILTIN = [
    # material on a thing: latent-lab's atlas, FLUX.2 klein putting the material [1] on a named thing
    Recipe("on a named thing", "material on a thing", "the material [1] (or an IP label) put on one thing, on its own "
           "folds and light; name it in the mask: thing:the dress, thing:the cliff",
           {"method": "generative.edit", "knobs": {"quality": "full"}}),
    Recipe("on the face", "material on a thing", "each face remade in the material [1], its features kept",
           {"method": "generative.edit", "knobs": {"quality": "full"}, "mask": {"layers": ["faces"], "feather": 2}}),
    Recipe("on the clothes", "material on a thing", "what people wear remade in the material [1], folds and seams kept",
           {"method": "generative.edit", "knobs": {"quality": "full"}, "mask": {"layers": ["clothes"], "feather": 2}}),
    # picture in a material: the whole picture, its forms held
    Recipe("in the material", "picture in a material", "the whole picture redrawn in the material [1] or an IP label, "
           "its forms held by depth", {"method": "generative.controlnet"}),
    Recipe("near and far", "picture in a material", "two IP labels split by the picture's depth: the first on what is "
           "near, the second on what is far",
           {"method": "generative.controlnet", "knobs": {"split": "depth", "split_at": 0.5}}),
    Recipe("lights and shadows", "picture in a material", "two IP labels split by the picture's light: the first in "
           "the light, the second in the shadows",
           {"method": "generative.controlnet", "knobs": {"split": "light", "split_at": 0.5}}),
    Recipe("water light", "picture in a material", "the picture repainted in the material's light and colour, its "
           "forms kept, live", {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "transport", "push": 1.2}}),
    Recipe("material skin", "picture in a material", "the picture's forms covered in the material's fine detail, live",
           {"method": "tokens.mix", "knobs": {"model": VAE, "operation": "split", "split_size": 1.5}}),
    # materials blended
    Recipe("fused materials", "materials blended", "the materials become one new surface in the picture",
           {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "blend", "carry": "tokens"}}),
    Recipe("laid by light", "materials blended", "each material in its own band of the picture's light, seams drawn "
           "by the decoder", {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "regions", "order": "light",
                                                              "seam": 0.5}}),
    Recipe("blended materials", "materials blended", "a hybrid texture from the materials alone in DINO's tokens, "
           "one surface at every point", {"method": "tokens.weave", "knobs": {"model": "DINOv2", "pattern": "blend"}}),
    Recipe("patchwork", "materials blended", "a new texture from the materials alone, in grown patches",
           {"method": "tokens.weave", "knobs": {"model": "DINOv2", "pattern": "patchwork"}}),
    # hidden image: the material leads, the picture is hidden in it
    Recipe("hybrid image", "hidden image", "the material up close, the picture from a distance",
           {"method": "pixels.bands", "lead": "material", "knobs": {"split": 8, "grain": 1.2}}),
    Recipe("traded edges", "hidden image", "the picture hidden in the material: at each pixel the stronger edge wins",
           {"method": "pixels.blend", "lead": "material", "knobs": {"wmix": 0.35}}),
    # glitch
    Recipe("mosaic mosh", "glitch", "DINO picks the material's pieces, the VAE draws them: a datamosh mosaic",
           {"method": "tokens.mosh", "knobs": {"model": VAE, "way": "rebuild", "source": "materials", "pieces": 2}}),
    Recipe("smeared", "glitch", "tokens dragged along a grown noise field, as a datamosh blooms",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "drag", "by": "noise", "length": 10}}),
    Recipe("pixel sort", "glitch", "the brightest runs sorted by light, a token at a time",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "sort", "span": 0.6, "along": "columns"}}),
    # relief
    Recipe("lit relief", "relief", "the material lit by the picture's relief, its highlights held",
           {"method": "pixels.relief", "knobs": {"ambient": 0.45}, "finish": {"highlights": -30, "clarity": 20}}),
    Recipe("combed", "relief", "the material combed along the picture's contours",
           {"method": "pixels.flow", "knobs": {"length": 0.08, "smear": 0.2}}),
    # where (stacks)
    Recipe("on the subject", "where", "only the main subject, with soft edges",
           {"mask": {"layers": ["subject"], "feather": 3, "follow": 0.3}}),
    Recipe("on the sky", "where", "only the sky", {"mask": {"layers": ["sky"], "feather": 6}}),
    Recipe("growing in", "where", "the material growing in from the shadows, ragged like lichen",
           {"mask": {"layers": ["shadows", "growth"], "combine": ["or"], "ragged": 0.5, "feather": 2}}),
    # look (stacks)
    Recipe("golden hour", "look", "warm, low sun: warmer, brighter, highlights held, a soft vignette",
           {"finish": {"temperature": 35, "exposure": 0.2, "contrast": 15, "highlights": -30, "vignette": -20}}),
    Recipe("matte film", "look", "lifted blacks, softer contrast, muted colour and grain",
           {"finish": {"blacks": 30, "contrast": -15, "saturation": -15, "grain": 25}}),
    Recipe("crisp", "look", "more texture, clarity and a little dehaze",
           {"finish": {"texture": 30, "clarity": 25, "dehaze": 10}}),
    Recipe("moody", "look", "darker, cooler, deeper shadows",
           {"finish": {"exposure": -0.4, "temperature": -20, "shadows": -30, "contrast": 20, "vignette": -30}}),
]
# Out of the menu, still opened by name: each drew too close to a recipe kept above, or its method was
# superseded (the texture LoRA by the label and the editor).
RETIRED = [
    # Chosen on three portraits (2026-11-08): starting from the material's colours set four textures
    # 2.5 times further apart than IP-Adapter alone, and redraw 0.45 kept each face's eyes and expression.
    Recipe("textured face", "material on a thing", "a gentler face: each face redrawn in the material [1]'s colour "
           "and texture by Stable Diffusion, the portrait untouched around it",
           {"method": "generative.lora",
            "knobs": {"strength": 0.45, "material": 0.8, "colour": 0.8, "lora": "none"},
            "mask": {"layers": ["faces"], "feather": 4}}, listed=False),
    Recipe("surface swap", "picture in a material", "the material's whole surface (principal axes 1 to 24) on the "
           "picture's forms", {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "surface axes"}}, listed=False),
    Recipe("tone swap", "picture in a material", "only the material's tonality (principal axes 1 to 4)",
           {"method": "tokens.mix", "knobs": {"model": VAE, "carry": "tone axes"}}, listed=False),
    Recipe("woven superposition", "materials blended", "the material added onto the picture: both visible as one",
           {"method": "tokens.mix", "knobs": {"model": "DINOv2", "where": "blend", "operation": "add"}}, listed=False),
    Recipe("self mosh", "glitch", "the picture rebuilt from its own look-alike pieces: extra eyes, more heads",
           {"method": "tokens.mosh", "knobs": {"model": "DINOv2", "way": "rebuild", "source": "picture",
                                               "pieces": 2, "wildness": 4}}, listed=False),
    Recipe("hidden: traded edges", "hidden image", "a face hidden in the material: at each pixel the stronger edge wins",
           {"method": "pixels.blend", "lead": "material", "knobs": {"wmix": 0.35}}, listed=False),
    Recipe("hidden: hybrid image", "hidden image", "the material up close, the face from a distance",
           {"method": "pixels.bands", "lead": "material", "knobs": {"split": 8, "grain": 1.2}}, listed=False),
    Recipe("hidden: pushed", "hidden image", "the material pushed along the face's slopes",
           {"method": "pixels.displace", "lead": "material", "knobs": {"amp": 0.04, "shade": 0.9}}, listed=False),
    Recipe("hidden: in tokens", "hidden image", "the face carried into the material's tokens by transport",
           {"method": "tokens.mix", "lead": "material", "knobs": {"model": VAE, "carry": "transport", "push": 0.7}},
           listed=False),
    Recipe("my texture", "material on a thing", "the picture redrawn in a look learned from one of your sets",
           {"method": "generative.lora"}, listed=False),
    Recipe("material on a thing", "material on a thing", "the material [1] put on one named thing",
           {"method": "generative.edit", "knobs": {"quality": "full"}}, listed=False),
    Recipe("material on the face", "material on a thing", "each face remade in the material [1]",
           {"method": "generative.edit", "knobs": {"quality": "full"}, "mask": {"layers": ["faces"], "feather": 2}},
           listed=False),
    Recipe("material on the clothes", "material on a thing", "what people wear remade in the material [1]",
           {"method": "generative.edit", "knobs": {"quality": "full"}, "mask": {"layers": ["clothes"], "feather": 2}},
           listed=False),
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
    """Every recipe a name can open: the menu's, the retired ones, the person's own."""
    return BUILTIN + RETIRED + own()


def menu() -> list[Recipe]:
    """The recipes the tab offers, grouped in GROUPS' order, the person's own after them."""
    order = {g: i for i, g in enumerate(GROUPS)}
    return sorted([r for r in BUILTIN if r.listed], key=lambda r: order.get(r.group, len(order))) + own()


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
