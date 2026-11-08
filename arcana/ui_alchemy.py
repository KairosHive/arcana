# ui_alchemy.py -- the Alchemy tab's vocabulary, and the controls as op arguments
#
# The page draws the controls; this module says what they are (the method
# catalogue, the mask presets, the finish spec, the recipes) and turns one
# snapshot of them into the arguments `alchemy preview`, `alchemy make` and
# `alchemy recipe save` take. tests/test_alchemy.py checks that translation.

from __future__ import annotations

try:
    from . import ui_style as _ui
    from .alchemy import finish as _finish
    from .alchemy import mask as _mask
    from .alchemy import methods as _methods
    from .alchemy import recipes as _recipes
except ImportError:                                   # pragma: no cover
    import ui_style as _ui
    from alchemy import finish as _finish
    from alchemy import mask as _mask
    from alchemy import methods as _methods
    from alchemy import recipes as _recipes

ROLES = ("T", "1", "2", "3")
ROLE_COLOUR = {"T": _ui.ROLE_TARGET, "1": _ui.ACCENT, "2": "#b08d57", "3": "#7c8f8a"}
LAYERS = 3
WORDS, CUSTOM = "__words", "__custom"
FAMILY_LABEL = {"pixels": "Pixels", "tokens": "Tokens", "colour": "Colour", "generative": "Generative"}
REFINE = [  # (key, label, lo, hi, step, default, doc)
    ("feather", "feather", 0, 40, 1, 4.0, "how soft the edge is, in pixels at 768"),
    ("grow", "grow", -40, 40, 1, 0.0, "the edge moved out (more) or in (less), in pixels at 768"),
    ("follow", "follow edges", 0, 1, 0.05, 0.0, "how closely the edge snaps to the picture's contours"),
    ("ragged", "ragged", 0, 1, 0.05, 0.0, "the edge broken by grown noise"),
    ("amount", "amount", 0, 1, 0.05, 1.0, "how strongly the step shows through the mask"),
]
FIN_DEFAULT = {k: getattr(_finish.Finish(), k) for k, *_ in _finish.SPEC}


# ─────────────────────────── the catalogue, for the page ───────────────────────────

def catalog() -> dict:
    """Everything the Alchemy tab needs to draw itself, in one object."""
    methods = []
    for m in _methods.METHODS.values():
        d = m.public()
        d["knobs"] = [{**k.public(), "section": k.section, "when": [list(w) for w in k.when]}
                      for k in m.knobs]
        methods.append(d)
    presets = [{"name": p.name, "group": p.group, "doc": p.doc, "coverage": p.coverage} for p in _mask.PRESETS]
    return {
        "families": [{"id": f, "label": FAMILY_LABEL[f], "doc": doc} for f, doc in _methods.FAMILIES.items()],
        "methods": methods,
        "maskGroups": list(_mask.GROUPS),
        "maskPresets": presets,
        "refine": [{"key": k, "label": lab, "lo": lo, "hi": hi, "step": st, "default": d, "doc": doc}
                   for k, lab, lo, hi, st, d, doc in REFINE],
        "finish": [{"key": k, "group": g, "label": lab, "lo": lo, "hi": hi, "step": st, "doc": doc,
                    "default": FIN_DEFAULT[k]} for k, g, lab, lo, hi, st, doc in _finish.SPEC],
        "recipes": recipes(),
        "roles": [{"role": r, "colour": ROLE_COLOUR[r], "label": "the picture" if r == "T" else f"material {r}"}
                  for r in ROLES],
        "words": WORDS, "custom": CUSTOM,
    }


def recipes() -> list[dict]:
    return [r.public() for r in _recipes.every()]


def recipe_controls(name: str) -> dict:
    """A recipe as the page's control values: what Apply sets, and nothing else."""
    r = _recipes.get(name)
    s = r.settings
    out: dict = {"doc": r.doc}
    if s.get("method"):
        out["method"] = s["method"]
        out["family"] = _methods.get(s["method"]).family
        out["knobs"] = dict(s.get("knobs") or {})
    for key in ("lead", "zoom"):
        if key in s:
            out[key] = s[key]
    fin = s.get("finish")
    if fin:
        out["finish"] = {k: v for k, v in fin.items() if k != "scope"}
        out["scope"] = fin.get("scope", "whole")
    mask = s.get("mask")
    if mask is not None:
        spec = _mask.MaskSpec.of(mask)
        layers = []
        for i in range(LAYERS):
            line = spec.layers[i] if i < len(spec.layers) else "none"
            if line in _mask.PRESET or line == "none":
                layer = {"choice": line, "words": ""}
            elif str(line).startswith("prompt:"):
                layer = {"choice": WORDS, "words": line[7:]}
            else:
                layer = {"choice": CUSTOM, "words": line}
            layer["combine"] = spec.combine[i - 1] if (i > 0 and i - 1 < len(spec.combine)) else "and"
            layer["coverage"] = (spec.coverage[i] if hasattr(spec, "coverage") and i < len(spec.coverage or [])
                                 and spec.coverage[i] is not None else None)
            layers.append(layer)
        out["mask"] = {"layers": layers, "invert": bool(spec.invert),
                       "refine": {key: getattr(spec, key) for key, *_ in REFINE}}
    return out


# ─────────────────────────── controls -> op arguments ───────────────────────────

def _source(choice, words):
    if choice in (None, "none") or str(choice).startswith("__group"):
        return None
    if choice == WORDS:
        return f"prompt:{words.strip()}" if (words or "").strip() else None
    if choice == CUSTOM:
        return (words or "").strip() or None
    return choice


def settings(v: dict, with_pictures=True) -> dict:
    """
    The op arguments the controls describe.

    `v` is the flat control state. Knobs come either as parallel `knob_ids` and
    `knobs` lists (the old page's shape, which the tests use) or as one
    `knobs` dict keyed by knob; finish values likewise as `fin_ids`/`fins` or a
    `finish` dict. Layers are layer1..3, words1..3, cov1..3 and comb2..3.
    """
    method = _methods.get(v["method"])
    names = {k.key: k for k in method.knobs}
    knobs = {}
    given = v.get("knobs")
    if isinstance(given, dict):
        pairs = given.items()
    else:
        pairs = [(kid["key"] if isinstance(kid, dict) else kid, val)
                 for kid, val in zip(v.get("knob_ids") or [], given or [])]
    for key, val in pairs:
        k = names.get(key)
        if k is not None and val is not None:
            knobs[k.key] = bool(val) if k.kind == "bool" else val
    layers, combine, coverage = [], [], []
    for i in range(1, LAYERS + 1):
        src = _source(v.get(f"layer{i}"), v.get(f"words{i}"))
        if src is None:
            continue
        if layers:
            combine.append(v.get(f"comb{i}") or "and")
        layers.append(src)
        cov = v.get(f"cov{i}")
        coverage.append(float(cov if cov is not None else 0.5))
    sets = []
    if layers:
        sets += [f"mask.{key}={v[key]}" for key, *_ in REFINE if v.get(key) is not None]
        if v.get("invert"):
            sets.append("mask.invert=true")
    fin_given = v.get("finish")
    if isinstance(fin_given, dict):
        fin_pairs = fin_given.items()
    else:
        fin_pairs = [(fid["key"] if isinstance(fid, dict) else fid, val)
                     for fid, val in zip(v.get("fin_ids") or [], v.get("fins") or [])]
    finish = {key: val for key, val in fin_pairs
              if key in FIN_DEFAULT and val is not None and val != FIN_DEFAULT.get(key)}
    if finish and v.get("scope") == "mask":
        finish["scope"] = "mask"
    out = {"method": method.id, "knobs": knobs, "lead": v.get("lead") or "picture",
           "zoom": float(v.get("zoom") or 1.0), "crop": int(v.get("crop") or 0),
           "mask": layers, "combine": combine, "coverage": coverage, "set": sets, "finish": finish}
    if with_pictures:
        roles = v.get("roles") or {}
        mats, weights = [], []
        for i, w in (("1", v.get("w1")), ("2", v.get("w2")), ("3", v.get("w3"))):
            if roles.get(i):
                mats.append(roles[i])
                weights.append(float(w if w is not None else 1.0))
        out.update(picture=roles.get("T"), material=mats, weight=weights)
    return out
