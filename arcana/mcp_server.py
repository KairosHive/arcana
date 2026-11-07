# mcp_server.py — Arcana's archive, offered to an agent over MCP
#
# An MCP client (Claude Code, Claude Desktop, or any client for a local model)
# starts this as its own process and talks to it over stdin/stdout. The agent
# brings the judgment -- breaking a theme into angles, choosing, cutting -- and
# this brings the archive: search over the person's own pictures, run here, on
# this machine, against indexes the app already built. See AGENT_PLAN.md.
#
# Four rules hold everywhere in this file:
#
#   * Read only. Phase 1 tools look; nothing writes, and no tool can ever touch
#     an original picture.
#   * Pictures are named by handle, "dataset:key" (japan:1423), never by path.
#     Shorter for the model to repeat, and stable within an index.
#   * Blind unless the person says otherwise. Whether thumbnails may leave the
#     machine is decided in agent.json in the data dir, by the person. No tool
#     can change it, so a model cannot decide for itself to start sending
#     pictures. While blind, every result says so.
#
# stdout carries the protocol, and Arcana's modules print progress freely. The
# MCP SDK's stdio transport (2.x) takes care of that: while it serves, file
# descriptor 1 points at stderr and the protocol runs on a private duplicate,
# so a stray print() -- or a native library writing to fd 1 -- misses the wire.
# tests/test_mcp_server.py checks it over a real pipe.

from __future__ import annotations

import base64
import json
import os
import threading
from typing import Annotated

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

try:
    from . import core, sheets
    from . import paths as _paths
    from . import models as _models
except ImportError:                                   # pragma: no cover
    import core
    import sheets
    import paths as _paths
    import models as _models


CONFIG_NAME = "agent.json"
AUDIO_SUFFIX = "@audio"

# Tile sizes by purpose: see sheets.py for what each costs.
SCAN_TILE = 160
INSPECT_SIDE = 1024
SHEET_COLS = 6
MAX_SURVEY_CLUSTERS = 16
SURVEY_ROWS_PER_SHEET = 8


# ─────────────────────────── settings ───────────────────────────

def settings() -> dict:
    """
    The person's choices for the agent, read fresh on every call.

    Read each time rather than at start-up, so turning pixels on or off applies
    to the next tool call without restarting the client.
    """
    path = os.path.join(_paths.data_dir(), CONFIG_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"pixels": bool(data.get("pixels", False)), "path": path}


def _mode_line(cfg: dict) -> str:
    if cfg["pixels"]:
        return "Mode: pixels. Thumbnails are included."
    return ("Mode: blind. No pictures leave this machine, so results are text only: "
            "cluster names, palettes, match strength. The person can allow thumbnails "
            f"by setting \"pixels\": true in {cfg['path']}.")


# ─────────────────────────── handles ───────────────────────────

def dataset_id(name: str, modality: str) -> str:
    return name if modality == "image" else name + AUDIO_SUFFIX


def split_dataset_id(ds_id: str) -> tuple[str, str]:
    if ds_id.endswith(AUDIO_SUFFIX):
        return ds_id[:-len(AUDIO_SUFFIX)], "audio"
    return ds_id, "image"


def handle(ds_id: str, key: int) -> str:
    return f"{ds_id}:{int(key)}"


def parse_handle(h: str) -> tuple[str, int]:
    ds_id, sep, key = (h or "").strip().rpartition(":")
    if not sep or not ds_id or not key.isdigit():
        raise ValueError(f"{h!r} is not a picture handle; handles look like 'japan:1423'.")
    return ds_id, int(key)


# ─────────────────────────── datasets ───────────────────────────

_ROWS: dict = {}
_ROWS_LOCK = threading.Lock()


def open_dataset(ds_id: str) -> core.Dataset:
    name, modality = split_dataset_id(ds_id)
    known = {(d["name"], d["modality"]): d for d in core.find_datasets()}
    found = known.get((name, modality))
    if found is None:
        names = ", ".join(dataset_id(n, m) for n, m in known) or "none"
        raise ValueError(f"No dataset called {ds_id!r}. Known datasets: {names}.")
    layouts = found["layouts"]
    n_dim = 2 if 2 in layouts or not layouts else layouts[0]
    return core.open_dataset(name, modality, n_dim)


def rows_by_key(ds: core.Dataset) -> dict:
    """Frame row for each key: row i of the latent frame is the i-th key of idx2path."""
    with _ROWS_LOCK:
        hit = _ROWS.get(id(ds))
        if hit is None:
            hit = {k: i for i, k in enumerate(ds.idx2path.keys())}
            _ROWS.clear()            # one dataset's map at a time is plenty
            _ROWS[id(ds)] = hit
        return hit


def facts(ds: core.Dataset, key: int) -> dict:
    """What the frame knows about one picture: cluster, stars, colour tag, date."""
    out = {}
    if ds.frame is None:
        return out
    row = rows_by_key(ds).get(key)
    if row is None or row >= len(ds.frame):
        return out
    rec = ds.frame.iloc[row]
    if "label" in rec and str(rec["label"]) not in ("", "nan"):
        out["cluster"] = str(rec["label"])
    if "rating" in rec and rec["rating"] == rec["rating"]:          # not NaN
        out["stars"] = int(rec["rating"])
    if "colour" in rec and str(rec["colour"]) not in ("", "nan"):
        out["colour"] = str(rec["colour"])
    if "captured" in rec and rec["captured"] == rec["captured"]:
        try:
            out["captured"] = str(rec["captured"])[:10]
        except Exception:
            pass
    return out


_PALETTES: dict = {}


def palettes(ds: core.Dataset, keys, n: int = 4) -> dict:
    """
    The dominant colours of each picture as hex, from the palette features the
    dataset already has. Empty when it has none: computing them here would mean
    decoding every picture, which is indexing's job.
    """
    try:
        from . import palette as _palette
    except ImportError:                               # pragma: no cover
        import palette as _palette
    path = os.path.join(_paths.subdir("databases"), f"features_{ds.name}_palette.npz")
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _PALETTES.get(path)
    if cached is None or cached[0] != stamp:
        data = np.load(path)
        if "dominant" not in data or "ids" not in data:
            return {}
        cached = (stamp, {int(k): i for i, k in enumerate(data["ids"])}, data["dominant"])
        _PALETTES.clear()
        _PALETTES[path] = cached
    _, row_of, dominant = cached
    out = {}
    for k in keys:
        i = row_of.get(int(k))
        if i is None:
            continue
        pal = dominant[i][:n]
        rgb = _palette.dominant_colors_to_rgb(pal)
        out[int(k)] = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb.tolist()]
    return out


def missing_share(ds: core.Dataset, sample: int = 200) -> float:
    """
    The share of a dataset's files no longer where the index says, from an
    evenly spaced sample -- the same check relocate.dataset_health makes, on the
    index already in memory. Search still works on a moved dataset, since it
    needs only the vectors; thumbnails and inspect do not.
    """
    paths = list(ds.idx2path.values())
    if not paths:
        return 0.0
    step = max(1, len(paths) // sample)
    picked = paths[::step][:sample]
    return sum(not os.path.exists(p) for p in picked) / len(picked)


def describe_dataset(ds_id: str) -> str:
    ds = open_dataset(ds_id)
    n = len(ds.idx2path)
    parts = [f"{ds_id}", ds.modality, f"{n:,} {'pictures' if ds.modality == 'image' else 'sounds'}"]
    if ds.modality == "image":
        try:
            info = _models.get(core.text_model_for_dim(int(ds.index.ndim)))
            parts.append(f"encoder {info.label.split(' — ')[0] if info else ds.index.ndim}")
        except RuntimeError:
            parts.append(f"unknown {ds.index.ndim}-d encoder")
    if ds.frame is None:
        parts.append("no map yet (search works; clusters do not)")
    elif "label" in ds.frame.columns:
        parts.append(f"{ds.frame['label'].nunique()} clusters")
    db = _paths.subdir("databases")
    feats = [f for f in ("palette", "style")
             if os.path.exists(os.path.join(db, f"features_{ds.name}_{f}.npz"))]
    parts.append("features: " + (", ".join(feats) if feats else "none"))
    if ds.frame is not None and core.has_marks(ds.frame):
        marks = []
        if "rating" in ds.frame.columns and ds.frame["rating"].notna().any():
            marks.append("stars")
        if "colour" in ds.frame.columns and (ds.frame["colour"].astype(str) != "").any():
            marks.append("colour tags")
        if "captured" in ds.frame.columns and ds.frame["captured"].notna().any():
            when = ds.frame["captured"].dropna()
            marks.append(f"dates {str(when.min())[:10]} to {str(when.max())[:10]}")
        parts.append("marks: " + ", ".join(marks))
    gone = missing_share(ds)
    if gone:
        parts.append(f"FILES MOVED: about {gone:.0%} are not where the index says. Search "
                     f"still works, but thumbnails and inspect cannot show them. Tell the "
                     f"person; Arcana's Relocate (or arcana-relocate) fixes it")
    return " · ".join(parts)


# ─────────────────────────── output ───────────────────────────

def _image(jpeg: bytes) -> ImageContent:
    return ImageContent(type="image", data=base64.b64encode(jpeg).decode("ascii"),
                        mime_type="image/jpeg")


def _text(s: str) -> TextContent:
    return TextContent(type="text", text=s)


def _hit_line(num: int, ds_id: str, ds: core.Dataset, key: int, path: str,
              metric: str | None, pal: dict | None) -> str:
    f = facts(ds, key)
    bits = [f"#{num}", handle(ds_id, key)]
    if metric:
        bits.append(metric)
    if "stars" in f:
        bits.append("★" * f["stars"] if f["stars"] > 0 else "0★")
    if "colour" in f:
        bits.append(f["colour"])
    if "captured" in f:
        bits.append(f["captured"])
    if "cluster" in f:
        bits.append(f"[{f['cluster']}]")
    bits.append(os.path.basename(path))
    if not os.path.exists(path):
        bits.append("(file missing)")
    if pal and key in pal:
        bits.append(" ".join(pal[key]))
    return "  ".join(bits)


# ─────────────────────────── the tools' work ───────────────────────────
# Plain functions, so tests can call them without an MCP client. The async
# wrappers below run them on a worker thread: a search can take seconds the
# first time it loads an encoder, and the protocol loop must keep answering.

class Facet(BaseModel):
    """One visual idea, said a few ways."""
    name: str = Field(description="A short name for this angle, e.g. 'water as weight'.")
    phrases: list[str] = Field(
        min_length=1, max_length=6,
        description="1-6 concrete visual phrasings of the same idea: things a camera "
                    "could see ('rain on a dark street at night', 'wet asphalt reflecting "
                    "lights'), not abstractions ('melancholy').")
    negatives: list[str] = Field(
        default_factory=list, max_length=4,
        description="Things that keep intruding and should be pushed away, e.g. 'umbrella', "
                    "'people'.")
    negative_weight: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="How hard negatives push. 0.3-0.5 works; 1.0 tends to return noise.")


class Filters(BaseModel):
    """The person's own decisions about their pictures, from ON1 / Lightroom sidecars."""
    min_stars: int = Field(default=0, ge=-1, le=5,
                           description="Only pictures rated at least this. -1 means never rated.")
    colours: list[str] = Field(default_factory=list,
                               description="Colour tags to keep, e.g. ['Red']. '(none)' means untagged.")
    date_from: str | None = Field(default=None, description="YYYY-MM-DD, inclusive.")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD, inclusive.")


def do_list_datasets() -> list:
    cfg = settings()
    found = core.find_datasets()
    if not found:
        return [_text("No datasets yet. The person builds one from Arcana's Datasets tab.\n"
                      + _mode_line(cfg))]
    lines = []
    for d in found:
        ds_id = dataset_id(d["name"], d["modality"])
        try:
            lines.append(describe_dataset(ds_id))
        except Exception as e:                        # one bad dataset must not hide the rest
            lines.append(f"{ds_id} · could not be opened: {e}")
    return [_text("\n".join(lines) + "\n\n" + _mode_line(cfg))]


def do_survey(ds_id: str, per_cluster: int = 4) -> list:
    cfg = settings()
    ds = open_dataset(ds_id)
    if ds.frame is None or "label" not in ds.frame.columns:
        return [_text(f"{ds_id} has no map yet, so there are no clusters to survey. "
                      f"search still works.\n" + _mode_line(cfg))]

    keys = list(ds.idx2path.keys())
    labels = ds.frame["label"].astype(str).tolist()[:len(keys)]
    groups: dict = {}
    for k, lab in zip(keys, labels):
        groups.setdefault(lab, []).append(k)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    shown, rest = ordered[:MAX_SURVEY_CLUSTERS], ordered[MAX_SURVEY_CLUSTERS:]
    total = len(keys)

    # Exemplars: the members nearest their cluster's centre, which are what the
    # cluster is "about" rather than its outliers.
    picks = []
    for lab, members in shown:
        vecs = core.unit_rows(np.stack([ds.index.get(k) for k in members]))
        centre = core.unit(vecs.mean(axis=0))
        order = np.argsort(-(vecs @ centre))[:per_cluster]
        picks.append((lab, len(members), [members[i] for i in order]))

    pal = {} if cfg["pixels"] else palettes(ds, [k for _, _, ks in picks for k in ks])
    lines = [f"{ds_id}: {len(ordered)} clusters over {total:,} items, largest first."]
    num = 0
    numbered = []
    for lab, size, ks in picks:
        lines.append(f"\n{lab} — {size:,} ({100 * size / total:.0f}%)")
        for k in ks:
            num += 1
            numbered.append((num, k))
            lines.append("  " + _hit_line(num, ds_id, ds, k, ds.idx2path[k], None, pal))
    if rest:
        lines.append("\nAlso: " + ", ".join(f"{lab} ({len(m)})" for lab, m in rest))
    lines.append("\n" + _mode_line(cfg))

    out = [_text("\n".join(lines))]
    if cfg["pixels"]:
        # One cluster per row; long surveys split so no sheet is so tall the
        # client scales it down past legibility.
        for start in range(0, len(picks), SURVEY_ROWS_PER_SHEET):
            chunk = numbered[sum(len(ks) for _, _, ks in picks[:start]):
                             sum(len(ks) for _, _, ks in picks[:start + SURVEY_ROWS_PER_SHEET])]
            out.append(_image(sheets.contact_sheet(
                [ds.idx2path[k] for _, k in chunk], [n for n, _ in chunk],
                tile=SCAN_TILE, cols=per_cluster)))
    return out


def do_search(ds_id: str, facets: list[Facet], k: int = 12, diversity: float = 0.3,
              filters: Filters | None = None) -> list:
    cfg = settings()
    ds = open_dataset(ds_id)
    f = filters or Filters()
    allowed = core.allowed_keys(ds, f.min_stars, f.colours or None, f.date_from, f.date_to)

    out = []
    shown: set = set()
    num = 0
    summary = []
    for facet in facets:
        q, pos = core.query_vectors(facet.phrases, ds.index, ds.modality,
                                    negatives=facet.negatives,
                                    negative_weight=facet.negative_weight)
        r = core.rank(ds, q, k, allowed=allowed, exclude=shown, diversity=diversity)
        p = core.confidence(ds, pos, [h.key for h in r.hits])
        nums = list(range(num + 1, num + 1 + len(r.hits)))
        num += len(r.hits)
        if p is None:
            verdict = "unrated" if r.hits else "none"
            head = f"## {facet.name} — {verdict}"
        else:
            # Judge the five best matches, not the first five shown: with
            # diversity on, the page interleaves weaker picks for variety.
            best = p[np.argsort([-h.score for h in r.hits])]
            verdict = core.strength(best)
            head = f"## {facet.name} — {verdict}"
            if r.hits:
                head += f" (best five average p {float(np.mean(best[:5])):.2f})"
        summary.append(f"{facet.name}: {verdict}")
        pal = {} if cfg["pixels"] else palettes(ds, [h.key for h in r.hits])
        lines = [head]
        if verdict in ("weak", "none"):
            lines.append("The archive has little for this. Say so rather than presenting these as a match.")
            # A weak idea can still have one real instance -- the one cat in an
            # archive of temples -- and on a moodboard one can be enough. A lone
            # high score can also be a false positive, hence "check".
            lone = [f"#{n}" for n, x in zip(nums, p if p is not None else []) if x >= core.STRONG_P]
            if lone:
                lines.append(f"Though {', '.join(lone)} score{'s' if len(lone) == 1 else ''} "
                             f"well on {'its' if len(lone) == 1 else 'their'} own: look before using.")
        for i, h in enumerate(r.hits):
            shown.add(h.key)
            metric = f"p {p[i]:.2f}" if p is not None else f"z {h.z:.1f}"
            lines.append(_hit_line(nums[i], ds_id, ds, h.key, h.path, metric, pal))
        if r.skipped_duplicates:
            lines.append(f"({r.skipped_duplicates} near-duplicates skipped)")
        if allowed is not None:
            lines.append(f"(filters left {r.considered:,} of {len(ds.idx2path):,})")
        out.append(_text("\n".join(lines)))
        if cfg["pixels"] and r.hits:
            out.append(_image(sheets.contact_sheet([h.path for h in r.hits], nums,
                                                   tile=SCAN_TILE, cols=SHEET_COLS,
                                                   title=facet.name)))
    out.append(_text("Summary: " + "; ".join(summary)
                     + ". p is how sure CLIP is that a picture shows the facet's phrases "
                       "rather than something generic.\n" + _mode_line(cfg)))
    return out


def do_more_like(handles: list[str], k: int = 12, mode: str = "semantic",
                 diversity: float = 0.3, exclude: list[str] | None = None) -> list:
    cfg = settings()
    parsed = [parse_handle(h) for h in handles]
    ds_ids = {d for d, _ in parsed}
    if len(ds_ids) != 1:
        raise ValueError("more_like takes pictures from one dataset at a time.")
    ds_id = ds_ids.pop()
    ds = open_dataset(ds_id)
    seeds = [key for _, key in parsed]
    missing = [handle(ds_id, s) for s in seeds if s not in ds.idx2path]
    if missing:
        raise ValueError(f"Not in {ds_id}: {', '.join(missing)}")
    skip = set(seeds) | {parse_handle(h)[1] for h in (exclude or [])}

    if mode == "semantic":
        r = core.rank(ds, core.example_vector(ds.index, seeds), k, exclude=skip,
                      diversity=diversity)
        found = [(h.key, h.path, f"sim {h.score:.2f}") for h in r.hits]
        note = ("Ranked by what the pictures show (CLIP); sim is picture-to-picture "
                "similarity, 1 being identical.")
    elif mode in ("palette", "style"):
        try:
            from . import db as _db
        except ImportError:                           # pragma: no cover
            import db as _db
        query = ds.idx2path[seeds[0]]
        fn = _db.search_by_palette if mode == "palette" else _db.search_by_style
        kwargs = {"method": "emd"} if mode == "palette" else {"method": "gram"}
        try:
            ranked = fn(query, ds.name, ds.idx2path, top_k=k + len(skip) + 8, **kwargs)
        except FileNotFoundError:
            raise ValueError(f"{ds_id} has no {mode} features. The person can add them "
                             f"with Rework on the Datasets tab.")
        key_of = {p: kk for kk, p in ds.idx2path.items()}
        found = []
        for p, _sim in ranked:
            kk = key_of.get(p)
            if kk is None or kk in skip:
                continue
            found.append((kk, p, None))
            if len(found) >= k:
                break
        note = (f"Ranked by {'colour palette' if mode == 'palette' else 'texture and grain'} "
                f"of the first picture only.")
    else:
        raise ValueError("mode is one of: semantic, palette, style.")

    pal = {} if cfg["pixels"] else palettes(ds, [kk for kk, _, _ in found])
    lines = [f"More like {', '.join(handles)}. {note}"]
    for i, (kk, p, z) in enumerate(found, 1):
        lines.append(_hit_line(i, ds_id, ds, kk, p, z, pal))
    lines.append(_mode_line(cfg))
    out = [_text("\n".join(lines))]
    if cfg["pixels"] and found:
        out.append(_image(sheets.contact_sheet([p for _, p, _ in found], tile=SCAN_TILE,
                                               cols=SHEET_COLS)))
    return out


def _swatch(hexes: list[str], weights: list[float] | None) -> np.ndarray:
    """A palette as a picture (BGR), each colour's band as wide as its weight."""
    cols = []
    for h in hexes:
        h = h.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"'{h}' is not a hex colour like #1f2a36.")
        cols.append((int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)))     # BGR
    w = np.asarray(weights if weights else [1.0] * len(cols), dtype=np.float64)
    if len(w) != len(cols) or (w <= 0).any():
        raise ValueError("weights, when given, are one positive number per colour.")
    widths = np.maximum(1, np.round(256 * w / w.sum())).astype(int)
    return np.concatenate([np.tile(np.array(c, np.uint8), (128, n, 1))
                           for c, n in zip(cols, widths)], axis=1)


def do_find_by_palette(ds_id: str, colours: list[str], weights: list[float] | None = None,
                       k: int = 12) -> list:
    cfg = settings()
    ds = open_dataset(ds_id)
    try:
        from . import db as _db
    except ImportError:                               # pragma: no cover
        import db as _db
    try:
        ranked = _db.search_by_palette(_swatch(colours, weights), ds.name, ds.idx2path,
                                       method="emd", n_colors=max(len(colours), 4), top_k=k)
    except FileNotFoundError:
        raise ValueError(f"{ds_id} has no palette features. The person can add them with "
                         f"Rework on the Datasets tab.")
    key_of = {p: kk for kk, p in ds.idx2path.items()}
    found = [(key_of[p], p) for p, _ in ranked if p in key_of]
    pal = {} if cfg["pixels"] else palettes(ds, [kk for kk, _ in found])
    lines = [f"Pictures closest to {' '.join(colours)} in {ds_id}, best first."]
    for i, (kk, p) in enumerate(found, 1):
        lines.append(_hit_line(i, ds_id, ds, kk, p, None, pal))
    lines.append(_mode_line(cfg))
    out = [_text("\n".join(lines))]
    if cfg["pixels"] and found:
        out.append(_image(sheets.contact_sheet([p for _, p in found], tile=SCAN_TILE,
                                               cols=SHEET_COLS)))
    return out


def do_inspect(handles: list[str]) -> list:
    cfg = settings()
    out = []
    for h in handles:
        ds_id, key = parse_handle(h)
        ds = open_dataset(ds_id)
        path = ds.idx2path.get(key)
        if path is None:
            out.append(_text(f"{h}: not in {ds_id}."))
            continue
        f = facts(ds, key)
        lines = [f"{h}  {os.path.basename(path)}"]
        lines += [f"  {name}: {val}" for name, val in f.items()]
        try:
            from PIL import Image as _PIL
            try:
                from . import refselect as _refselect
            except ImportError:                       # pragma: no cover
                import refselect as _refselect
            with _PIL.open(path) as im:
                lines.append(f"  size: {im.width} x {im.height}")
                strip = _refselect.palette_strip(im.convert("RGB"), n=6)
            lines.append("  palette: " + " ".join(f"{hx} {share:.0%}" for hx, share in strip))
        except Exception as e:
            lines.append(f"  could not be read: {e}")
            out.append(_text("\n".join(lines)))
            continue
        if not cfg["pixels"]:
            # In blind mode this is the agent's only view of the picture, so a
            # failure is reported rather than swallowed. The description reads
            # the picture, not the index: any CLIP will do, and the dataset's
            # own encoder is merely preferred when it is a known one.
            try:
                try:
                    from . import style_transfer as _st
                except ImportError:                   # pragma: no cover
                    import style_transfer as _st
                try:
                    model_id = core.text_model_for_dim(int(ds.index.ndim))
                except RuntimeError:
                    model_id = None
                words = _st.describe(path, model_id=model_id)
                lines.append(f"  looks like: {words}" if words
                             else "  looks like: (no description available)")
            except Exception as e:
                lines.append(f"  looks like: (description failed: {type(e).__name__}: {e})")
        out.append(_text("\n".join(lines)))
        if cfg["pixels"]:
            jpeg = sheets.fitted_jpeg(path, INSPECT_SIDE)
            if jpeg:
                out.append(_image(jpeg))
    out.append(_text(_mode_line(cfg)))
    return out


# ─────────────────────────── the server ───────────────────────────

INSTRUCTIONS = """\
Arcana searches the person's own image archive, on their machine, by what the
pictures show. Nothing here edits or moves their files.

Work like this: list_datasets, then survey a dataset to see what it actually
holds before planning, then search. Refer to pictures by handle (japan:1423).

CLIP, which ranks the pictures, sees concrete visual things: objects, places,
light, weather, colour, texture, composition, photographic technique. It cannot
see abstractions, so translate them ('melancholy' -> 'empty bench in rain,
overcast, muted colours'). Give each idea several phrasings, and use negatives
for things that keep intruding.

Every search facet comes back marked strong, fair or weak. Weak means the
archive has little for that idea: say so plainly rather than presenting the
results as a match.

Every result states whether you can see thumbnails (pixels) or not (blind).
Only the person can change that.
"""

def _call(fn, *args):
    """
    Run a tool's work, turning the failures we expect into messages for the model.

    The SDK shows the model a ToolError's text. Any other exception counts as a
    crash and reaches it only as "Error executing tool search", which is how
    "No dataset called 'nope'. Known datasets: ..." would otherwise be lost.
    """
    try:
        return fn(*args)
    except (ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e


mcp = MCPServer("arcana", instructions=INSTRUCTIONS)
_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

# Tools are plain functions: the SDK runs them on a worker thread, so a search
# that spends seconds loading an encoder the first time does not stall the
# protocol loop. Each one hands straight to its do_* counterpart above.

PerFacet = Annotated[int, Field(ge=1, le=24, description="Pictures per facet.")]
Diversity = Annotated[float, Field(
    ge=0.0, le=1.0,
    description="0 = closest matches only; higher spreads results across different "
                "takes on the idea.")]


@mcp.tool(annotations=_READ, structured_output=False)
def list_datasets() -> list:
    """The person's datasets: size, encoder, clusters, which features and marks exist."""
    return _call(do_list_datasets)


@mcp.tool(annotations=_READ, structured_output=False)
def survey(dataset: str,
           per_cluster: Annotated[int, Field(ge=1, le=8)] = 4) -> list:
    """
    What a dataset holds: its clusters, largest first, each with its most typical
    pictures. Do this before planning a search, so the plan fits the archive.
    """
    return _call(do_survey, dataset, per_cluster)


@mcp.tool(annotations=_READ, structured_output=False)
def search(dataset: str,
           facets: Annotated[list[Facet], Field(min_length=1, max_length=8)],
           k: PerFacet = 12,
           diversity: Diversity = 0.3,
           filters: Filters | None = None) -> list:
    """
    Search a dataset for several visual ideas at once, one facet per idea.

    Each facet returns up to k pictures, near-duplicates removed, never repeating
    a picture an earlier facet already showed. Each picture has p: how sure CLIP
    is that it shows the facet's phrases rather than something generic. Each
    facet is marked strong, fair or weak.
    """
    return _call(do_search, dataset, facets, k, diversity, filters)


@mcp.tool(annotations=_READ, structured_output=False)
def more_like(handles: Annotated[list[str], Field(min_length=1, max_length=8)],
              k: PerFacet = 12,
              mode: Annotated[str, Field(
                  description="semantic (what the pictures show), palette (their "
                              "colours) or style (texture and grain).")] = "semantic",
              diversity: Diversity = 0.3,
              exclude: Annotated[list[str], Field(
                  description="Handles not to return again.")] = []) -> list:
    """
    Pictures like these: search by example. Semantic mode blends several
    examples; palette and style modes use the first one.
    """
    return _call(do_more_like, handles, k, mode, diversity, exclude)


@mcp.tool(annotations=_READ, structured_output=False)
def find_by_palette(dataset: str,
                    colours: Annotated[list[str], Field(
                        min_length=1, max_length=8,
                        description="Hex colours, e.g. ['#1f2a36', '#b5651d'].")],
                    weights: Annotated[list[float] | None, Field(
                        description="Optional share of each colour.")] = None,
                    k: PerFacet = 12) -> list:
    """Pictures whose colours match a palette you choose."""
    return _call(do_find_by_palette, dataset, colours, weights, k)


@mcp.tool(annotations=_READ, structured_output=False)
def inspect(handles: Annotated[list[str], Field(min_length=1, max_length=4)]) -> list:
    """
    A close look at up to four pictures: size, palette, marks, cluster, and the
    picture itself in pixels mode (or a CLIP description of it in blind mode).
    """
    return _call(do_inspect, handles)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
