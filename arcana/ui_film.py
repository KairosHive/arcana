# ui_film.py -- the Video subtab's film arithmetic, for the page
#
# A film is a spec the page edits (pictures, looks and lanes on a timeline).
# What needs the engine's knowledge stays here: which settings a lane can key
# and their ranges, a setting's value at a moment, the normalised film and its
# length, spacing the pictures, the look a step snapshot becomes.

from __future__ import annotations

import json
import os

try:
    from . import ui_alchemy as _ia
    from .alchemy import finish as _finish
    from .alchemy import methods as _methods
except ImportError:                                   # pragma: no cover
    import ui_alchemy as _ia
    from alchemy import finish as _finish
    from alchemy import methods as _methods

EASES = ("linear", "smooth", "in", "out", "hold")
GAP = 4.0                                             # seconds between pictures added one after another
REFINE_RANGE = {"feather": (0, 40, 1), "grow": (-40, 40, 1), "follow": (0, 1, 0.05),
                "ragged": (0, 1, 0.05), "amount": (0, 1, 0.05)}
NUMBER_RANGE = {"x": (0, 1, 0.01), "y": (0, 1, 0.01), "r": (0.02, 1, 0.01), "angle": (0, 360, 1),
                "height": (0.02, 1, 0.01), "count": (1, 40, 1), "near": (0, 1, 0.01), "far": (0, 1, 0.01),
                "size": (4, 200, 1), "coverage": (0.02, 1, 0.01)}
EMPTY_FILM = {"name": "untitled film", "fps": 30, "size": 640, "aspect": "auto", "space": "DINOv2",
              "loop": True, "even": True, "pictures": [], "looks": [], "lanes": {}}
SPACES = {"DINOv2": "one new surface: the decoder redraws, so materials fuse",
          "FLUX.2 VAE": "the photographs' own grain: a mix reads as layers"}
SHAPES = [("even", "even", "every passage as long"),
          ("faster", "speeding up", "each passage shorter than the one before"),
          ("slower", "slowing down", "each passage longer than the one before"),
          ("smooth", "slow at both ends", "slow to start and to end")]


def _film():
    try:
        from .alchemy import film
    except ImportError:                               # pragma: no cover
        from alchemy import film
    return film


def full(spec: dict | None) -> dict:
    return {**EMPTY_FILM, **(spec or {})}


def lane_meta(spec: dict) -> tuple[list, dict]:
    """The settings a lane can key, and each one's label and range: (options, {path: meta})."""
    film = _film()
    meta: dict = {}
    methods = {k.get("settings", {}).get("method") for k in spec.get("looks") or []}
    for mid in sorted(m for m in methods if m and m != "none"):
        try:
            method = _methods.get(mid)
        except ValueError:
            continue
        for k in method.knobs:
            if k.kind in ("float", "int", "angle") and k.lo is not None:
                meta[f"knobs.{k.key}"] = {"label": f"{method.label} · {k.label}", "lo": k.lo, "hi": k.hi,
                                          "step": k.step or 0.01, "group": "method"}
    for key, group, label, lo, hi, step, doc in _finish.SPEC:
        meta[f"finish.{key}"] = {"label": f"finish · {label}", "lo": lo, "hi": hi, "step": step, "group": "finish"}
    for key, (lo, hi, step) in REFINE_RANGE.items():
        meta[f"mask.{key}"] = {"label": f"mask · {key}", "lo": lo, "hi": hi, "step": step, "group": "mask"}
    most_layers, most_materials = [], 0
    for k in spec.get("looks") or []:
        s = k.get("settings") or {}
        m = s.get("mask") or {}
        layers = m.get("layers") if isinstance(m, dict) else ([m] if isinstance(m, str) else m) or []
        if len(layers or []) > len(most_layers):
            most_layers = list(layers)
        most_materials = max(most_materials, len([x for x in s.get("materials") or [] if x]))
    for i, line in enumerate(most_layers, 1):
        meta[f"mask.{i}.coverage"] = {"label": f"mask {i} · coverage", "lo": 0.02, "hi": 1, "step": 0.01,
                                      "group": "mask"}
        got = film.layer_numbers(line)
        if got:
            for name in got[1]:
                lo, hi, step = NUMBER_RANGE[name]
                meta[f"mask.{i}.{name}"] = {"label": f"mask {i} · {got[0]} {name}", "lo": lo, "hi": hi,
                                            "step": step, "group": "mask"}
    for i in range(1, most_materials + 1):
        meta[f"weights.{i}"] = {"label": f"share of [{i}]", "lo": 0, "hi": 1, "step": 0.05, "group": "pictures"}
    meta["zoom"] = {"label": "material zoom", "lo": 0.5, "hi": 4, "step": 0.25, "group": "pictures"}
    for path, keys in (spec.get("lanes") or {}).items():
        if path not in meta:
            vs = [float(k["v"]) for k in keys] or [0.0]
            meta[path] = {"label": path, "lo": min(vs + [0.0]), "hi": max(vs + [1.0]), "step": 0.01,
                          "group": "other"}
    options = [{"label": v["label"], "value": p, "group": v["group"]} for p, v in meta.items()
               if p not in (spec.get("lanes") or {})]
    return options, meta


def value_at(spec: dict, path: str, t: float, lo: float):
    """A setting's value at time t, from the looks, to start a new lane where the film already is."""
    film = _film()
    s = film.look_at(spec, t)
    head, _, rest = path.partition(".")
    try:
        if path == "zoom":
            return float(s.get("zoom", 1.0))
        if head in ("knobs", "finish"):
            v = (s.get(head) or {}).get(rest)
            if v is None and head == "finish":
                return float(getattr(_finish.Finish(), rest))
            if v is None:
                for k in s.get("method") and _methods.get(s["method"]).knobs or ():
                    if k.key == rest:
                        return float(k.default)
            return float(v) if v is not None else lo
        if head == "weights":
            w = s.get("weights") or []
            i = int(rest) - 1
            return float(w[i]) if i < len(w) else 1.0
        if head == "mask":
            m = s.get("mask") or {}
            if rest in REFINE_RANGE:
                return float(m.get(rest, {"feather": 4.0, "amount": 1.0}.get(rest, 0.0)))
            idx, _, num = rest.partition(".")
            layers = m.get("layers") or []
            line = layers[int(idx) - 1] if int(idx) - 1 < len(layers) else ""
            if num == "coverage":
                cov = (m.get("coverage") or [None] * len(layers))
                return (float(cov[int(idx) - 1]) if int(idx) - 1 < len(cov) and cov[int(idx) - 1] is not None
                        else 0.4)
            got = film.layer_numbers(line)
            if got and num in got[1]:
                return float(got[2][got[1].index(num)])
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    return lo


def normal(spec):
    """The film with its defaults, or (None, the reason) when it is not a film yet."""
    try:
        return _film().normalised(spec), None
    except (ValueError, KeyError, TypeError) as e:
        return None, str(e)


def meta(spec: dict) -> dict:
    """What the timeline needs: length, the lanes on offer, the pictures and looks by time."""
    s = full(spec)
    options, lanes = lane_meta(s)
    f, why = normal(s)
    try:
        from .alchemy.inputs import resolve
    except ImportError:                               # pragma: no cover
        from alchemy.inputs import resolve
    pictures = []
    for p in sorted(s.get("pictures") or [], key=lambda p: float(p["t"])):
        try:
            pictures.append({"t": float(p["t"]), "path": resolve(p["ref"]), "name": os.path.basename(str(p["ref"]))})
        except ValueError:
            pictures.append({"t": float(p["t"]), "path": "", "name": str(p["ref"])})
    looks = []
    for k in s.get("looks") or []:
        st = k.get("settings") or {}
        m = st.get("method") or "none"
        label = "morph only" if m == "none" else (_methods.get(m).label if m in _methods.METHODS else m)
        looks.append({"t": float(k["t"]), "label": label})
    n_pics = len(s.get("pictures") or [])
    length = f["length"] if f else max([float(p["t"]) for p in s.get("pictures") or []] + [GAP])
    return {"length": length, "lanes": lanes, "laneOptions": options, "pictures": pictures, "looks": looks,
            "ok": f is not None, "status": "" if f or n_pics < 2 else why,
            "empty": "" if n_pics >= 2 else
            "Click two or more pictures on the left to start a film. Each lands at the playhead.",
            "normalised": f}


def add_picture(spec: dict, ref: str, t: float) -> tuple[dict, float]:
    """A picture onto the timeline at the playhead, or after the last when the playhead is taken."""
    s = full(spec)
    pics = sorted(s.get("pictures") or [], key=lambda p: float(p["t"]))
    t = float(t or 0.0)
    taken = {round(float(p["t"]), 3) for p in pics}
    if not pics:
        t = 0.0
    elif round(t, 3) in taken:
        t = float(pics[-1]["t"]) + GAP
    s["pictures"] = pics + [{"t": round(t, 3), "ref": ref, "ease": "linear"}]
    if t >= float(s.get("length") or 0):
        s.pop("length", None)
    return s, t


def spaced(spec: dict, shape: str) -> dict:
    return _film().spaced(full(spec), shape)


def add_lane(spec: dict, path: str, t: float) -> dict:
    s = full(spec)
    _, lanes = lane_meta(s)
    m = lanes.get(path) or {"lo": 0, "hi": 1}
    f, _ = normal(s)
    t = float(t or 0.0)
    v = value_at(f, path, t, m["lo"]) if f else m["lo"]
    new = dict(s.get("lanes") or {})
    new[path] = [{"t": round(t, 3), "v": v, "ease": "linear"}]
    return {**s, "lanes": new}


def look_from_controls(controls: dict) -> dict:
    """The Image subtab's step, as a look for the film's look lane."""
    try:
        from .ops.alchemy import AlchemyPreviewArgs, settings_of
    except ImportError:                               # pragma: no cover
        from ops.alchemy import AlchemyPreviewArgs, settings_of
    args = _ia.settings(controls)
    look = settings_of(AlchemyPreviewArgs(**{k: val for k, val in args.items()
                                             if k not in ("picture", "material", "weight")}))
    look["materials"] = list(args.get("material") or [])
    if args.get("weight"):
        look["weights"] = list(args["weight"])
    return look


def with_look(spec: dict, t: float, look: dict) -> dict:
    s = full(spec)
    t = round(float(t or 0.0), 3)
    looks = [k for k in s.get("looks") or [] if round(float(k["t"]), 3) != t]
    s["looks"] = sorted(looks + [{"t": t, "ease": "smooth", "settings": look}], key=lambda k: float(k["t"]))
    return s
