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


REFINE_DOC = {"feather": "how soft the mask's edge is, in pixels at 768",
              "grow": "the mask's edge moved out (more) or in (less), in pixels at 768",
              "follow": "how closely the mask's edge snaps to the picture's contours",
              "ragged": "the mask's edge broken by grown noise",
              "amount": "how strongly the step shows through the mask: 0 shows the morph alone"}
NUMBER_DOC = {"x": "where the spot's centre sits across the frame, 0 left to 1 right",
              "y": "where it sits down the frame, 0 top to 1 bottom",
              "r": "the spot's radius, as a share of the frame",
              "angle": "the direction, in degrees", "height": "the band's height, as a share of the frame",
              "count": "how many stripes", "near": "the nearest depth the band takes, 0 to 1",
              "far": "the farthest depth it takes", "size": "the size of the pattern, in pixels at 768",
              "coverage": "how much of the picture the layer takes, 0 to 1"}
GROUP_TITLE = {"method": "The look's method", "mask": "The look's mask", "pictures": "The materials",
               "finish": "The finish", "other": "Other"}


def lane_meta(spec: dict) -> tuple[list, dict]:
    """
    What a lane can animate, with each one's label, range and what it controls:
    (options with their group title, {path: meta}). Mask lanes are offered only
    when a look has a mask, and they name the layer they move.
    """
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
                                          "step": k.step or 0.01, "group": "method", "doc": k.doc}
    most_layers, most_materials = [], 0
    for k in spec.get("looks") or []:
        st = k.get("settings") or {}
        m = st.get("mask") or {}
        layers = m.get("layers") if isinstance(m, dict) else ([m] if isinstance(m, str) else m)
        if len(layers or []) > len(most_layers):
            most_layers = list(layers)
        most_materials = max(most_materials, len([x for x in st.get("materials") or [] if x]))
    if most_layers:
        for key, (lo, hi, step) in REFINE_RANGE.items():
            meta[f"mask.{key}"] = {"label": f"mask · {key}", "lo": lo, "hi": hi, "step": step, "group": "mask",
                                   "doc": REFINE_DOC[key]}
        for i, line in enumerate(most_layers, 1):
            meta[f"mask.{i}.coverage"] = {"label": f"mask layer {i} ({line}) · coverage", "lo": 0.02, "hi": 1,
                                          "step": 0.01, "group": "mask", "doc": NUMBER_DOC["coverage"]}
            got = film.layer_numbers(line)
            if got:
                for name in got[1]:
                    lo, hi, step = NUMBER_RANGE[name]
                    meta[f"mask.{i}.{name}"] = {"label": f"mask layer {i} ({line}) · {name}", "lo": lo, "hi": hi,
                                                "step": step, "group": "mask", "doc": NUMBER_DOC[name]}
    for i in range(1, most_materials + 1):
        meta[f"weights.{i}"] = {"label": f"share of material [{i}]", "lo": 0, "hi": 1, "step": 0.05,
                                "group": "pictures", "doc": f"how much of the mix material [{i}] takes"}
    if most_materials:
        meta["zoom"] = {"label": "material zoom", "lo": 0.5, "hi": 4, "step": 0.25, "group": "pictures",
                        "doc": "how large the materials are drawn: 2 shows half of each, twice as large"}
    if methods - {None, "none"}:
        meta["amount"] = {"label": "look · how much shows", "lo": 0, "hi": 1, "step": 0.05, "group": "method",
                          "doc": "how much of the look's method covers the morph: 0 the morph alone, 1 the whole "
                                 "look. A slow rise lays a texture over the film bit by bit"}
    for key, group, label, lo, hi, step, doc in _finish.SPEC:
        meta[f"finish.{key}"] = {"label": f"finish · {label}", "lo": lo, "hi": hi, "step": step, "group": "finish",
                                 "doc": doc}
    for path, keys in (spec.get("lanes") or {}).items():
        if path not in meta:
            vs = [float(k["v"]) for k in keys] or [0.0]
            meta[path] = {"label": path, "lo": min(vs + [0.0]), "hi": max(vs + [1.0]), "step": 0.01, "group": "other",
                          "doc": "a lane the looks no longer have: it moves nothing until a look has it again"}
    taken = set(spec.get("lanes") or {})
    options = []
    for group, title in GROUP_TITLE.items():
        options += [{"label": v["label"], "value": p, "group": title} for p, v in meta.items()
                    if v["group"] == group and p not in taken]
    return options, meta


def look_index(looks: list, t: float):
    """The look in force at time t: the last one at or before it, or the first when t is before them all."""
    if not looks:
        return None
    before = [k for k, look in enumerate(looks) if float(look["t"]) <= t + 1e-6]
    return before[-1] if before else 0


def look_summary(spec: dict, t: float) -> dict:
    """The look at the playhead, in words: its method, mask, materials and changed finish."""
    looks = sorted((spec or {}).get("looks") or [], key=lambda k: float(k["t"]))
    i = look_index(looks, float(t or 0.0))
    if i is None:
        return {"index": None, "count": 0}
    st = looks[i].get("settings") or {}
    m = st.get("method") or "none"
    name = "morph only" if m == "none" else (_methods.get(m).label if m in _methods.METHODS else m)
    mask = st.get("mask") or {}
    layers = [str(x) for x in (mask.get("layers") or [])] if isinstance(mask, dict) else []
    combine = (mask.get("combine") or []) if isinstance(mask, dict) else []
    line = layers[0] if layers else ""
    for k, x in enumerate(layers[1:]):
        line += f" {combine[k] if k < len(combine) else 'and'} {x}"
    changed = [f"{k} {v:g}" for k, v in (st.get("finish") or {}).items()
               if isinstance(v, (int, float)) and v != getattr(_finish.Finish(), k, v)]
    return {"index": i, "count": len(looks), "t": float(looks[i]["t"]), "method": name,
            "model": (st.get("knobs") or {}).get("model"), "mask": line,
            "materials": [x for x in st.get("materials") or [] if x], "finish": changed, "settings": st}


def replace_look(spec: dict, t: float, look: dict) -> dict:
    """The look at the playhead now holds these settings."""
    s = full(spec)
    looks = sorted(s.get("looks") or [], key=lambda k: float(k["t"]))
    i = look_index(looks, float(t or 0.0))
    if i is None:
        raise ValueError("There is no look here to replace. Add Image's step here instead.")
    looks[i] = {**looks[i], "settings": look}
    return {**s, "looks": looks}


def remove_look(spec: dict, t: float) -> tuple[dict, float]:
    s = full(spec)
    looks = sorted(s.get("looks") or [], key=lambda k: float(k["t"]))
    i = look_index(looks, float(t or 0.0))
    if i is None:
        raise ValueError("There is no look here to remove.")
    gone = looks.pop(i)
    return {**s, "looks": looks}, float(gone["t"])


def value_at(spec: dict, path: str, t: float, lo: float):
    """A setting's value at time t, from the looks, to start a new lane where the film already is."""
    film = _film()
    s = film.look_at(spec, t)
    head, _, rest = path.partition(".")
    try:
        if path == "zoom":
            return float(s.get("zoom", 1.0))
        if path == "amount":
            return float(s.get("amount", 1.0))
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
    film = _film()
    pictures = []
    for p in sorted(s.get("pictures") or [], key=lambda p: float(p["t"])):
        if film.plays(p.get("ref", "")):
            clip = film.is_clip(p["ref"])
            name, path = str(p["ref"])[len(film.FILM_REF):], ""
            q = dict(p)
            try:
                film.held_key(q, 0)
                dur = film.span(q)
                if clip:
                    file = film.clip_source(p["ref"])[0]
                    name = os.path.basename(file)
                    # The page shows the clip by its frame at `in`.
                    path = film._modality.section_ref(file, q["in"], q["out"])
            except (ValueError, KeyError, TypeError):
                dur = 0.0
            pictures.append({"t": float(p["t"]), "path": path, "name": name, "film": "" if clip else name,
                             "clip": clip, "dur": dur, "in": q.get("in"), "out": q.get("out"),
                             "speed": q.get("speed")})
            continue
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
    length = f["length"] if f else max([p["t"] + p.get("dur", 0.0) for p in pictures] + [GAP])
    return {"length": length, "lanes": lanes, "laneOptions": options, "pictures": pictures, "looks": looks,
            "ok": f is not None, "status": "" if f or n_pics < 2 else why,
            "empty": "" if n_pics >= 2 else
            "Click two or more pictures on the left to start a film, each at the playhead, or drag them "
            "onto the timeline where you want them.",
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
    film = _film()
    key = {"t": round(t, 3), "ref": ref, "ease": "linear"}
    if film.plays(ref):
        film.held_key(key, 0)
    if pics:
        # Where a film would run into the next key, or a picture lands while a film plays, the key goes
        # after everything.
        end = max(float(p["t"]) + _span(p) for p in pics)
        nxt = [float(p["t"]) for p in pics if float(p["t"]) > t]
        inside = any(float(p["t"]) < t <= float(p["t"]) + _span(p) for p in pics)
        if inside or (nxt and t + film.span(key) >= min(nxt)):
            t = end + GAP
            key["t"] = round(t, 3)
    s["pictures"] = pics + [key]
    if t >= float(s.get("length") or 0):
        s.pop("length", None)
    return s, t


def _span(p: dict) -> float:
    film = _film()
    if not film.plays(p.get("ref", "")):
        return 0.0
    q = dict(p)
    try:
        film.held_key(q, 0)
        return film.span(q)
    except (ValueError, KeyError, TypeError):
        return 0.0


def place_series(spec: dict, refs: list, t: float, every: float = GAP, back: bool = False) -> tuple[dict, float]:
    """Pictures onto the picture lane one after another from time t, `every` seconds apart, moving any
    key already after t later by the room they take. With `back`, the first picture comes again at the
    end, so the series closes on where it began. Returns the film and the time after the series."""
    s = full(spec)
    refs = [r for r in refs if r]
    if not refs:
        raise ValueError("Give at least one picture to place.")
    if back and len(refs) > 1:
        refs = refs + [refs[0]]
    t = round(max(float(t or 0.0), 0.0), 3)
    every = max(float(every or GAP), 0.2)
    pics = sorted(s.get("pictures") or [], key=lambda p: float(p["t"]))
    room = every * len(refs)
    kept = [p for p in pics if float(p["t"]) < t - 1e-6]
    later = [{**p, "t": round(float(p["t"]) + room, 3)} for p in pics if float(p["t"]) >= t - 1e-6]
    if kept and float(kept[-1]["t"]) + _span(kept[-1]) >= t:
        t = round(float(kept[-1]["t"]) + _span(kept[-1]) + every, 3)     # after the film playing there
    new = [{"t": round(t + i * every, 3), "ref": r, "ease": "linear"} for i, r in enumerate(refs)]
    s["pictures"] = kept + new + later
    s.pop("length", None)
    return s, round(t + room, 3)


LANE_SHAPES = [("rise", "rise", "from the first value to the second"),
               ("fall", "fall", "from the second value back to the first"),
               ("swell", "swell", "up to the second value at the middle, and back"),
               ("dip", "dip", "down to the first value at the middle, and back"),
               ("wave", "wave", "back and forth, a number of times"),
               ("steps", "steps", "in a number of steps, each held, then a cut")]


def shape_lane(spec: dict, path: str, shape: str, lo: float, hi: float, start: float, end: float,
               count: int = 3, ease: str = "smooth") -> dict:
    """A lane's keys between start and end replaced by a whole curve: rise, fall, swell, dip, wave or
    steps between the values lo and hi. Keys outside that span stay."""
    _film().check_path(path)
    start, end = float(start), float(end)
    if end - start < 0.05:
        raise ValueError("A shape needs a span: end after start.")
    count = max(1, min(int(count or 1), 64))
    lo, hi = float(lo), float(hi)
    if shape == "rise":
        pts = [(0, lo), (1, hi)]
    elif shape == "fall":
        pts = [(0, hi), (1, lo)]
    elif shape == "swell":
        pts = [(0, lo), (0.5, hi), (1, lo)]
    elif shape == "dip":
        pts = [(0, hi), (0.5, lo), (1, hi)]
    elif shape == "wave":
        pts = [(i / (2 * count), lo if i % 2 == 0 else hi) for i in range(2 * count + 1)]
    elif shape == "steps":
        n = max(count, 2)
        pts = [(i / n, lo + (hi - lo) * i / (n - 1)) for i in range(n)] + [(1, hi)]
    else:
        raise ValueError(f"A lane's shape is one of {', '.join(k for k, _, _ in LANE_SHAPES)}.")
    curve = "hold" if shape == "steps" else ease
    keys = [{"t": round(start + (end - start) * u, 3), "v": round(v, 4), "ease": curve} for u, v in pts]
    keys[-1]["ease"] = "linear"
    s = full(spec)
    lanes = dict(s.get("lanes") or {})
    old = [k for k in lanes.get(path) or [] if not start - 1e-6 <= float(k["t"]) <= end + 1e-6]
    lanes[path] = sorted(old + keys, key=lambda k: float(k["t"]))
    return {**s, "lanes": lanes}


def spaced(spec: dict, shape: str, by: str = "time") -> dict:
    return _film().spaced(full(spec), shape, by=by)


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
