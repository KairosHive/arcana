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
#   camera movement         the picture exposed while the camera moves, no material needed
#   double exposure         a second picture, the material [1], exposed with its own movement over the first
#   where                   a mask: stacks on any recipe
#   look                    a finish: stacks on any recipe
#
# Each group holds a few recipes that look clearly different. Recipes that drew too close to another
# are retired: kept, so a film or a saved step that names one still opens, but out of the menu.
GROUPS = ("material on a thing", "picture in a material", "materials blended", "hidden image", "glitch",
          "relief", "camera movement", "double exposure", "where", "look")
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
    # camera movement: chosen by eye on the person's own photographs (see motion.py)
    Recipe("vertical sweep, violet glow", "camera movement", "a vertical sweep with the picture's detail kept "
           "and a soft glow, in cool violet, as an ICM marina",
           {"method": "camera.sweep", "knobs": {"sweep": 0.12, "direction": 1.5708, "detail": 0.8, "glow": 0.3},
            "finish": {"temperature": -40, "tint": 25, "dehaze": -15}}),
    Recipe("stop, move, stop", "camera movement", "held still, moved down, held again: two sharp impressions "
           "joined by the movement",
           {"method": "camera.sweep", "knobs": {"sweep": 0.12, "direction": 1.5708, "hold_start": 0.45,
                                                "hold_end": 0.2, "centred": False}}),
    Recipe("zoom burst", "camera movement", "the camera zoomed in during the exposure, the lights drawn out "
           "as rays",
           {"method": "camera.turn", "knobs": {"rotation": 0.0, "zoom": 0.35, "trails": 0.6}}),
    Recipe("swirl", "camera movement", "the camera turned about the centre: everything circles it",
           {"method": "camera.turn", "knobs": {"rotation": 22.0}}),
    Recipe("crossing strokes", "camera movement", "three sweeps fanned across directions over a held sharp "
           "impression, warm and soft: painterly, its forms still there",
           {"method": "camera.layers", "knobs": {"sweep": 0.08, "direction": 0.6, "strokes": 3, "spread": 1.2,
                                                 "hold_start": 0.25, "centred": False},
            "finish": {"temperature": 25, "clarity": -30, "saturation": -20, "contrast": -10}}),
    Recipe("edges hold", "camera movement", "a sideways sweep in which strong edges stay and the spaces "
           "between them flow",
           {"method": "camera.flow", "knobs": {"sweep": 0.12, "direction": 0.0, "anchor": 0.95}}),
    Recipe("light streaks over the dark", "camera movement", "the picture's lights drawn down across a still "
           "dark subject, as a flash closing a long exposure",
           {"method": "camera.flow", "knobs": {"sweep": 0.14, "direction": 1.5708, "anchor": 0.0, "light": 1.0}}),
    Recipe("carved strokes", "camera movement", "short strokes that follow the picture's own contours, its "
           "edges held: the figure stays, the rest turns to carving",
           {"method": "camera.flow", "knobs": {"sweep": 0.04, "direction": 1.5708, "follow": 1.0, "anchor": 0.5,
                                               "detail": 0.3}}),
    Recipe("painted water", "camera movement", "strokes that follow the water's own lines, its light leading, "
           "with a soft glow",
           {"method": "camera.flow", "knobs": {"sweep": 0.12, "direction": 0.0, "anchor": 0.0, "follow": 0.8,
                                               "light": 0.7, "glow": 0.25}}),
    Recipe("held in fog", "camera movement", "the subject held sharp while the world around softens, drags "
           "down and hazes, as figures in a misted long exposure",
           {"method": "camera.sweep", "knobs": {"sweep": 0.08, "direction": 1.5708, "defocus": 4, "glow": 0.5,
                                                "mask_use": "holds still"},
            "mask": {"layers": ["subject"], "feather": 14},
            "finish": {"temperature": 22, "dehaze": -55, "contrast": -10, "vignette": -35, "exposure": 0.15}}),
    Recipe("the world turns about them", "camera movement", "the subject held sharp while everything else "
           "turns about them, softly out of focus; set the pivot on them",
           {"method": "camera.turn", "knobs": {"rotation": 25.0, "pivot_x": 0.45, "pivot_y": 0.4, "defocus": 2,
                                               "mask_use": "holds still"},
            "mask": {"layers": ["subject"], "feather": 16}, "finish": {"temperature": 30, "dehaze": -25, "contrast": -10, "vignette": -40, "saturation": -10}}),
    Recipe("held among strokes", "camera movement", "people held sharp inside three crossing sweeps, warm and "
           "soft",
           {"method": "camera.layers", "knobs": {"strokes": 3, "spread": 1.2, "sweep": 0.1, "direction": 0.6,
                                                 "hold_start": 0.0, "centred": True, "mask_use": "holds still"},
            "mask": {"layers": ["people"], "feather": 10}, "finish": {"temperature": 25, "clarity": -30, "saturation": -20, "contrast": -10}}),
    Recipe("strokes all over", "camera movement", "three crossing sweeps over the whole frame and no sharp "
           "impression: the picture dissolved into warm brushwork",
           {"method": "camera.layers", "knobs": {"strokes": 3, "spread": 1.2, "sweep": 0.1, "direction": 0.6,
                                                 "hold_start": 0.0, "centred": True}, "finish": {"temperature": 25, "clarity": -30, "saturation": -20, "contrast": -10}}),
    # double exposure: chosen by eye on the person's own photographs; each takes a material [1]
    Recipe("streaming light", "double exposure", "the material swept sideways and screened over the whole "
           "picture, its forms held under a stream of light",
           {"method": "camera.double", "knobs": {"blend": "screen", "second": 0.7, "sweep": 0.25, "direction": 0.0,
                                                 "soft_second": 2}, "finish": {"temperature": 25, "clarity": -30, "saturation": -20, "contrast": -10}}),
    Recipe("falling through", "double exposure", "the material falls down through the picture, darken keeping "
           "the darker of the two at each place: the picture's lines stay, the material runs between them",
           {"method": "camera.double", "knobs": {"blend": "darken", "second": 0.8, "sweep": 0.15,
                                                 "direction": 1.5708}}),
    Recipe("a field of colour", "double exposure", "the material softened into a field of colour and light "
           "behind the picture, lightened in and swept sideways",
           {"method": "camera.double", "knobs": {"blend": "lighten", "second": 0.8, "sweep": 0.2, "direction": 0.0,
                                                 "soft_second": 12}}),
    Recipe("into the light", "double exposure", "the material swept sideways into the picture's lighter half "
           "only, as a sky turning to water",
           {"method": "camera.double", "knobs": {"blend": "screen", "second": 0.9, "sweep": 0.2, "direction": 0.0},
            "mask": {"layers": ["light"], "coverage": [0.45], "feather": 40}}),
    Recipe("in shadow", "double exposure", "the material multiplied into the picture and drawn down: both "
           "darken into one muted relief",
           {"method": "camera.double", "knobs": {"blend": "multiply", "second": 0.8, "sweep": 0.1,
                                                 "direction": 1.5708}}),
    Recipe("falling light", "double exposure", "the material's light falling down over the picture, slightly "
           "softened and screened in",
           {"method": "camera.double", "knobs": {"blend": "screen", "second": 0.8, "sweep": 0.15,
                                                 "direction": 1.5708, "soft_second": 1}}),
    Recipe("turning double", "double exposure", "the material turned about the centre and screened over the "
           "picture, which turns a little with it",
           {"method": "camera.double", "knobs": {"blend": "screen", "second": 0.7, "sweep": 0.0, "rotation": 20.0,
                                                 "picture_moves": 0.3}}),
    Recipe("two veils", "double exposure", "the picture sweeps across and the material sweeps down, averaged "
           "into one woven veil",
           {"method": "camera.double", "knobs": {"blend": "average", "second": 1.0, "sweep": 0.2,
                                                 "direction": 1.5708, "picture_moves": 1.0, "across": 1.5708,
                                                 "glow": 0.3}}),
    Recipe("horizon of two", "double exposure", "the material, swept sideways, fills the top of the frame and "
           "fades into the picture below",
           {"method": "camera.double", "knobs": {"blend": "screen", "second": 1.0, "sweep": 0.2, "direction": 0.0},
            "mask": {"layers": ["from the top"], "coverage": [0.55], "feather": 60}}),
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
