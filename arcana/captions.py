# captions.py -- a caption for each training picture, as a photographer would say it
#
# Ported from latent-lab's caption.py. A category word is a blunt caption: one
# archive's "rock" held bare grey faces and stone crusted with orange lichen,
# and a LoRA trained on that word drew the base model's idea of rock. Captioned
# instead "lichen and mineral stains on wet rock, orange and teal", the same
# weights drew what the photographs hold. The caption was the bottleneck.
#
# CLIP scores each picture against vocabularies kept apart (material, surface,
# condition), so a material only competes with materials. Colour is named from
# the pixels: clustered in CIELAB and matched to a named palette by perceptual
# distance, with the tone the picture sits in and its most saturated accent.
#
# Arcana's own CLIP ViT-H/14 does the scoring (its image and text towers), so
# nothing new is downloaded.

from __future__ import annotations

import numpy as np

AXES = {
    "material": (
        "a close-up photograph of {}",
        [
            "still water", "rippled water", "a breaking wave", "sea foam", "froth and bubbles",
            "a shallow stream over stones", "a reflection on water", "ice", "frost", "melting snow",
            "bare rock", "a cracked rock face", "granite", "limestone", "slate", "sandstone", "basalt",
            "marble", "quartz", "a mineral vein", "pebbles", "gravel", "shingle", "sand", "silt", "dried mud",
            "clay", "moss", "lichen on stone", "algae", "seaweed", "a barnacle-covered rock", "shells",
            "tree bark", "birch bark", "pine bark", "driftwood", "weathered timber", "a cut log",
            "fallen leaves", "foliage", "grass", "roots", "lace-like branches",
            "rust on iron", "corroded metal", "oxidised copper", "peeling paint", "flaking enamel",
            "cracked plaster", "weathered concrete", "brick", "ceramic tile", "frosted glass",
            "torn paper", "worn fabric", "rope", "fishing net", "tangled cable",
            "smoke", "mist", "cloud", "a sunlit surface", "a sheet of light",
            "fire", "flames", "embers", "a face", "skin", "a landscape", "a street", "a building",
        ],
    ),
    "surface": (
        "a {} surface",
        [
            "cracked", "deeply fissured", "finely veined", "layered", "striated", "banded",
            "mottled", "speckled", "freckled", "pitted", "pockmarked", "eroded", "weathered",
            "flaking", "peeling", "blistered", "encrusted", "crystalline", "granular", "grainy",
            "pebbled", "smooth", "polished", "glassy", "rippled", "undulating", "swirling",
            "marbled", "braided", "fibrous", "stringy", "dendritic", "branching", "web-like",
            "honeycombed", "cellular", "bubbled", "foamy", "furrowed", "ridged", "scaled",
        ],
    ),
    "condition": (
        "{}",
        [
            "wet and glistening", "soaked", "submerged under clear water", "bone dry", "dusty",
            "sunlit with hard shadows", "in flat overcast light", "backlit", "in deep shade",
            "glossy and reflective", "matte", "frosted", "sharply detailed", "softly blurred",
        ],
    ),
}

PALETTE = {
    "black": (16, 16, 18), "charcoal": (54, 54, 58), "slate grey": (98, 104, 110),
    "ash grey": (150, 150, 148), "pale grey": (196, 197, 196), "bone white": (232, 228, 218),
    "cream": (238, 226, 198), "sand": (214, 192, 152), "khaki": (176, 160, 116),
    "ochre": (190, 142, 62), "mustard": (206, 166, 44), "gold": (212, 175, 55),
    "amber": (196, 128, 36), "copper": (168, 94, 52), "rust": (150, 76, 38),
    "burnt orange": (186, 86, 30), "terracotta": (178, 102, 76), "brick red": (142, 58, 44),
    "crimson": (156, 34, 48), "maroon": (104, 34, 44), "plum": (104, 56, 88),
    "mauve": (150, 118, 142), "rose": (198, 136, 142), "blush": (222, 180, 174),
    "lavender": (176, 168, 206), "violet": (112, 84, 156), "indigo": (62, 60, 124),
    "navy": (34, 46, 80), "cobalt": (44, 86, 158), "steel blue": (92, 124, 156),
    "pale blue": (168, 196, 214), "teal": (36, 118, 124), "turquoise": (56, 166, 165),
    "aquamarine": (134, 204, 196), "sea green": (72, 140, 118), "emerald": (44, 124, 84),
    "moss green": (96, 118, 68), "olive": (118, 118, 62), "sage": (158, 170, 138),
    "acid green": (160, 194, 48), "lime": (190, 210, 86), "umber": (96, 72, 52),
    "chocolate": (72, 52, 40), "warm brown": (122, 88, 60),
    # Weathered things are mostly muted: without this band every dusty, sun-bleached surface was
    # named "ash grey" or "chocolate".
    "taupe": (138, 124, 110), "greige": (166, 158, 144), "stone": (182, 176, 162),
    "oatmeal": (206, 196, 176), "putty": (190, 178, 158), "fawn": (190, 164, 134),
    "driftwood": (156, 146, 132), "mushroom": (146, 132, 124), "pewter": (124, 124, 126),
    "dusty rose": (168, 128, 124), "clay": (158, 110, 90), "tobacco": (134, 104, 68),
    "bronze": (140, 110, 70), "pale olive": (160, 158, 120), "olive drab": (108, 106, 78),
    "celadon": (176, 192, 170), "eucalyptus": (140, 158, 142), "seafoam": (186, 206, 196),
    "dusty blue": (134, 156, 172), "ice blue": (198, 214, 220), "muted teal": (88, 128, 128),
    "charcoal blue": (62, 72, 84), "deep teal": (40, 72, 78),
}


def to_lab(rgb):
    """sRGB in [0, 1] to CIELAB, so the nearest colour is what the eye calls nearest."""
    rgb = np.asarray(rgb, np.float64)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


NAMED = {name: to_lab(np.array(v) / 255.0) for name, v in PALETTE.items()}


def colours_of(image, keep: int = 2) -> list[str]:
    """The tone a picture sits in and its accent, by clustering its pixels in CIELAB.

    The darkest quarter is dropped first (shadow carries little colour and named vivid orange rock
    "charcoal"), and clusters are weighed by area times chroma, so a dull background does not
    outvote the subject and a speck does not win either.
    """
    from PIL import Image

    px = np.asarray(image.convert("RGB").resize((64, 64), Image.BICUBIC), np.float64).reshape(-1, 3) / 255.0
    lab = to_lab(px)
    lab = lab[lab[:, 0] >= np.percentile(lab[:, 0], 25)]
    rng = np.random.default_rng(0)
    centres = lab[rng.choice(len(lab), 6, replace=False)]
    for _ in range(25):
        owner = np.argmin(((lab[:, None] - centres[None]) ** 2).sum(-1), 1)
        moved = np.stack([lab[owner == i].mean(0) if (owner == i).any() else centres[i] for i in range(len(centres))])
        if np.allclose(moved, centres, atol=1e-3):
            break
        centres = moved
    share = np.bincount(owner, minlength=len(centres)) / len(lab)
    chroma = np.linalg.norm(centres[:, 1:], axis=1)
    worth = share * (chroma + 6.0)

    def name_of(i):
        return min(NAMED, key=lambda n: float(((centres[i] - NAMED[n]) ** 2).sum()))

    order = list(np.argsort(-worth))
    out = [name_of(int(order[0]))]
    accent = [i for i in range(len(centres)) if share[i] >= 0.05 and name_of(int(i)) not in out]
    if accent:
        best = max(accent, key=lambda i: chroma[i])
        if chroma[best] >= 14.0:
            out.append(name_of(int(best)))
    for i in order:
        if len(out) >= keep:
            break
        n = name_of(int(i))
        if n not in out and share[i] >= 0.07:
            out.append(n)
    return out


def phrase(material, surfaces, colours, condition, trigger) -> str:
    bits = [f"a close-up photograph of {material}"]
    if surfaces:
        bits.append(" and ".join(surfaces))
    if colours:
        bits.append("in " + " and ".join(colours))
    if condition:
        bits.append(condition)
    if trigger:
        bits.append(trigger)
    return ", ".join(bits)


_BANKS: dict = {}


def _banks(model_id: str):
    """One unit matrix of text features per vocabulary, encoded once."""
    if model_id not in _BANKS:
        import torch

        from . import core

        text, tok = core.load_clip_text(model_id=model_id)
        banks = {}
        with torch.no_grad():
            for axis, (template, terms) in AXES.items():
                t = tok([template.format(x) for x in terms], padding=True, return_tensors="pt")
                f = text(**{k: v.to(text.device) for k, v in t.items()}).text_embeds.float().cpu().numpy()
                banks[axis] = f / np.linalg.norm(f, axis=1, keepdims=True)
        _BANKS[model_id] = banks
    return _BANKS[model_id]


def typical(captions: dict[str, str], trigger: str) -> str:
    """The set in one caption: its commonest material, surfaces and colours, then the trigger. What
    a drawing prompts with to call up the set as a whole."""
    from collections import Counter

    material, surfaces, colours, condition = Counter(), Counter(), Counter(), Counter()
    for c in captions.values():
        parts = [x.strip() for x in c.split(",")]
        if not parts or not parts[0].startswith("a close-up photograph of "):
            continue
        material[parts[0][len("a close-up photograph of "):]] += 1
        for x in parts[1:]:
            if x == trigger:
                continue
            if x.startswith("in ") and all(y.strip() in PALETTE for y in x[3:].split(" and ")):
                colours.update(y.strip() for y in x[3:].split(" and "))
            elif any(x == t for t in AXES["condition"][1]):
                condition[x] += 1
            else:
                surfaces.update(y.strip() for y in x.split(" and "))
    if not material:
        return trigger
    return phrase(material.most_common(1)[0][0], [s for s, _ in surfaces.most_common(2)],
                  [c for c, _ in colours.most_common(2)], condition.most_common(1)[0][0] if condition else "", trigger)


def caption_all(paths: list[str], trigger: str, progress=None) -> dict[str, str]:
    """A caption per picture: material, two surfaces, colours, a condition, then the trigger."""
    import torch
    from PIL import Image, ImageOps

    from . import db as _db

    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    banks = _banks(model_id)
    model, processor = _db.load_clip(model_id=model_id)
    out = {}
    for start in range(0, len(paths), 16):
        chunk, ims = [], []
        for p in paths[start:start + 16]:
            try:
                with Image.open(p) as im:
                    im.draft("RGB", (448, 448))
                    ims.append(ImageOps.exif_transpose(im).convert("RGB"))
                chunk.append(p)
            except OSError:
                out[p] = trigger
        if not chunk:
            continue
        px = processor(images=ims, return_tensors="pt").pixel_values.to(model.device, next(model.parameters()).dtype)
        with torch.no_grad():
            got = model.get_image_features(px)
        feats = (got if torch.is_tensor(got) else got.pooler_output).float().cpu().numpy()
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        for p, im, f in zip(chunk, ims, feats):
            material = AXES["material"][1][int(np.argmax(banks["material"] @ f))]
            s = banks["surface"] @ f
            surfaces = [AXES["surface"][1][int(i)] for i in np.argsort(-s)[:2]]
            condition = AXES["condition"][1][int(np.argmax(banks["condition"] @ f))]
            out[p] = phrase(material, surfaces, colours_of(im), condition, trigger)
        if progress:
            progress(0.02 + 0.06 * min(1.0, (start + 16) / len(paths)),
                     f"captioning {min(start + 16, len(paths))} of {len(paths)}")
    return out
