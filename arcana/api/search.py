# api/search.py -- the map, prompt search, story mode and the moodboard's similarity search
#
#   GET  /api/map?dataset=            every point: x, y, [z], cluster, path, marks
#   GET  /api/marks?dataset=          filter options, the stars x colours matrix, the timeline
#   POST /api/search                  top-k by prompt, grouped into twins, with map rows
#   POST /api/story                   top-1 per line, with the path through the map
#   POST /api/story/poetry            the diffusion pass over a story, as a job
#   POST /api/story/save              originals, poetry pictures and story.txt to output/stories
#   POST /api/save                    chosen results to output/search/<folder>
#   POST /api/similar                 palette and style similarity from the [R] picture
#
# Reads only: the datasets are immutable once built. Saves write under the
# output tree through paths.safe_join, never anywhere the page names.

from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
from io import BytesIO

import numpy as np
import pandas as pd
from flask import jsonify, request

from . import bp, body, call, BadRequest, parse_dataset

try:
    from .. import core as _core
    from .. import on1 as _on1
    from .. import media as _media
    from .. import paths as _paths
    from .. import jobs as _jobs
    from .. import gpu as _gpu
    from .. import ops as _ops
    from ..cvio import imread_unicode, imwrite_unicode
except ImportError:                                   # pragma: no cover
    import core as _core
    import on1 as _on1
    import media as _media
    import paths as _paths
    import jobs as _jobs
    import gpu as _gpu
    import ops as _ops
    from cvio import imread_unicode, imwrite_unicode

COLOUR_SWATCH = {"Red": "#e05252", "Yellow": "#d9b234", "Green": "#43b581",
                 "Blue": "#4a90d9", "Purple": "#a982d9"}
NO_COLOUR = _core.NO_COLOUR
UNRATED = _core.UNRATED
AUDIO_EXTS = _media.AUDIO_EXTS


# ─────────────────────────── loading ───────────────────────────

def load_data(name, n_dim=2, modality="image"):
    """core.load_data, plus making the dataset's files servable to the browser."""
    df = _core.load_data(name, n_dim=n_dim, modality=modality, latent_dir=_media.LATENTS_DIR)
    if "path" in df.columns:
        _media.register_dataset_files(dict(enumerate(df["path"].tolist())),
                                      cache_key=f"latent:{name}:{modality}")
    return df


def load_index(name, modality="image"):
    index, idx2path = _core.load_index(name, modality=modality, db_dir=_media.DB_DIR)
    _media.register_dataset_files(idx2path, cache_key=f"index:{name}:{modality}")
    return index, idx2path


def _value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _marks_lookup(df) -> dict:
    """path -> {stars, colour, captured}, for the badge on a result card."""
    if not any(c in df.columns for c in _core.MARK_COLUMNS):
        return {}
    ratings = df["rating"] if "rating" in df.columns else None
    colours = df["colour"] if "colour" in df.columns else None
    dates = df["captured"] if "captured" in df.columns else None
    out = {}
    for i, path in enumerate(df["path"].tolist()):
        r = _value(ratings.iloc[i]) if ratings is not None else None
        d = _value(dates.iloc[i]) if dates is not None else None
        out[path] = {"stars": None if r is None else int(r),
                     "colour": str(colours.iloc[i] or "") if colours is not None else "",
                     "captured": pd.Timestamp(d).strftime("%d %b %Y") if d is not None else ""}
    return out


def _filter(data: dict):
    stars = int(data.get("stars") or 0)
    colours = data.get("colours") or []
    date_from = data.get("dateFrom") or None
    date_to = data.get("dateTo") or None
    return stars, colours, date_from, date_to, bool(stars or colours or date_from or date_to)


# ─────────────────────────── the map ───────────────────────────

@bp.get("/map")
def _map():
    """
    Every point of a dataset's layout, column-wise, so the page draws it itself.

    Clusters come as an index into a label list, which keeps a 100k-point map
    to a few hundred kilobytes. Row i is the i-th key of the index.
    """
    name, dim, modality = parse_dataset(request.args.get("dataset"))
    df = load_data(name, n_dim=dim, modality=modality)
    labels = df["label"].astype(str).tolist() if "label" in df.columns else [""] * len(df)
    vocab = list(dict.fromkeys(labels))
    at = {l: i for i, l in enumerate(vocab)}
    out = {"dataset": request.args.get("dataset"), "n": int(len(df)), "modality": modality,
           "x": [round(float(v), 4) for v in df["x"].tolist()],
           "y": [round(float(v), 4) for v in df["y"].tolist()],
           "labels": vocab, "cluster": [at[l] for l in labels],
           "paths": df["path"].astype(str).tolist() if "path" in df.columns else [],
           "hasMarks": bool(_core.has_marks(df))}
    if "z" in df.columns:
        out["z"] = [round(float(v), 4) for v in df["z"].tolist()]
    if out["hasMarks"]:
        out["stars"] = [(_value(v) if _value(v) is None else int(v)) for v in df["rating"].tolist()] \
            if "rating" in df.columns else None
        out["colour"] = df["colour"].astype(str).tolist() if "colour" in df.columns else None
        out["captured"] = [(None if _value(v) is None else pd.Timestamp(v).strftime("%Y-%m-%d"))
                           for v in df["captured"].tolist()] if "captured" in df.columns else None
    return jsonify(out)


@bp.post("/map/filter")
def _map_filter():
    """Which rows a marks filter keeps, as indices, and how many that is."""
    data = body()
    name, dim, modality = parse_dataset(data.get("dataset"))
    df = load_data(name, n_dim=dim, modality=modality)
    stars, colours, f_from, f_to, filtering = _filter(data)
    if not filtering:
        return jsonify({"kept": None, "count": int(len(df)), "total": int(len(df))})
    keep = _core.marks_mask(df, stars, colours, f_from, f_to)
    return jsonify({"kept": [int(i) for i in np.flatnonzero(keep.to_numpy())],
                    "count": int(keep.sum()), "total": int(len(df))})


# ─────────────────────────── marks: options, matrix, timeline ───────────────────────────

def _star_options(df) -> list:
    opts = [{"label": "Any rating", "value": 0}]
    if "rating" not in df.columns:
        return opts
    rating = df["rating"].fillna(UNRATED)
    for n in range(1, 6):
        count = int((rating >= n).sum())
        if count:
            opts.append({"label": f"{'★' * n} and up · {count:,}", "value": n})
    never = int(df["rating"].isna().sum())
    if never:
        opts.append({"label": f"Never rated · {never:,}", "value": UNRATED})
    return opts


def _colour_options(df) -> list:
    if "colour" not in df.columns:
        return []
    counts = df["colour"].astype(str).value_counts()
    order = {name: i for i, name in enumerate(_on1.LABELS)}
    tags = [c for c in counts.index if c]
    tags.sort(key=lambda c: (-int(counts[c]), order.get(c, len(order)), c))
    opts = [{"label": f"{c} · {int(counts[c]):,}", "value": c, "colour": COLOUR_SWATCH.get(c, "#8a8278")}
            for c in tags]
    untagged = int(counts.get("", 0))
    if untagged and tags:
        opts.append({"label": f"No tag · {untagged:,}", "value": NO_COLOUR, "colour": "#8a8278"})
    return opts


def _matrix_counts(df):
    """(stars, colour) -> count, plus the row and column vocabularies present."""
    ratings = df["rating"] if "rating" in df.columns else pd.Series(dtype="Int64")
    colours = df["colour"].astype(str) if "colour" in df.columns else pd.Series(dtype=str)
    if not len(colours):
        colours = pd.Series([""] * len(df), index=df.index)
    present = {c for c in colours.unique() if c}
    cols = [c for c in _on1.LABELS if c in present]
    cols += sorted(present - set(cols))
    cols.append(NO_COLOUR)
    rows = [s for s in range(5, -1, -1) if (ratings == s).any()]
    if ratings.isna().any():
        rows.append(UNRATED)
    grid = {}
    for s in rows:
        sel = ratings.isna() if s == UNRATED else (ratings == s)
        for c in cols:
            hit = (colours == "") if c == NO_COLOUR else (colours == c)
            grid[(s, c)] = int((sel & hit).sum())
    return grid, rows, cols


def _matrix(df) -> dict:
    grid, rows, cols = _matrix_counts(df)
    return {"rows": rows, "cols": cols, "cells": [[grid[(s, c)] for c in cols] for s in rows],
            "swatch": {c: COLOUR_SWATCH.get(c, "#8a8278") for c in cols}}


def _timeline(df) -> dict | None:
    if "captured" not in df.columns or not df["captured"].notna().any():
        return None
    dated = df[df["captured"].notna()].copy()
    span_days = (dated["captured"].max() - dated["captured"].min()).days
    unit = "D" if span_days <= 120 else ("W" if span_days <= 730 else "M")
    dated["bin"] = dated["captured"].dt.to_period(unit).dt.start_time
    tags = dated["colour"].astype(str) if "colour" in dated.columns else pd.Series("", index=dated.index)
    dated["tag"] = tags.where(tags != "", "No tag")
    order = [c for c in _on1.LABELS if (dated["tag"] == c).any()] + ["No tag"]
    bins = sorted(dated["bin"].unique())
    series = []
    for tag in order:
        part = dated[dated["tag"] == tag]
        if not len(part):
            continue
        counts = part.groupby("bin").size()
        series.append({"tag": tag, "colour": COLOUR_SWATCH.get(tag, "#8a8278"),
                       "counts": [int(counts.get(b, 0)) for b in bins]})
    return {"unit": unit, "bins": [pd.Timestamp(b).strftime("%Y-%m-%d") for b in bins], "series": series}


@bp.get("/marks")
def _marks():
    name, dim, modality = parse_dataset(request.args.get("dataset"))
    df = load_data(name, n_dim=dim, modality=modality)
    if not _core.has_marks(df):
        return jsonify({"has": False})
    ratings = df["rating"] if "rating" in df.columns else pd.Series(dtype="Int64")
    chips = [{"value": f"{len(df):,}", "caption": "frames"},
             {"value": f"{int((ratings.fillna(0) > 0).sum()):,}", "caption": "rated"},
             {"value": f"{int((df['colour'].astype(str) != '').sum()) if 'colour' in df.columns else 0:,}",
              "caption": "colour tagged"},
             {"value": f"{int((ratings.fillna(0) >= 3).sum()):,}", "caption": "three stars and up"}]
    if "captured" in df.columns and df["captured"].notna().any():
        lo, hi = df["captured"].min(), df["captured"].max()
        chips.append({"value": f"{lo:%b %Y} – {hi:%b %Y}", "caption": "shot between"})
    if "camera" in df.columns:
        cams = df.loc[df["camera"].astype(str) != "", "camera"]
        if len(cams):
            chips.append({"value": str(cams.value_counts().index[0]), "caption": "mostly shot on"})
    dates = None
    if "captured" in df.columns and df["captured"].notna().any():
        dates = {"min": df["captured"].min().strftime("%Y-%m-%d"), "max": df["captured"].max().strftime("%Y-%m-%d")}
    return jsonify({"has": True, "total": int(len(df)),
                    "stars": _star_options(df), "colours": _colour_options(df), "dates": dates,
                    "hasStars": bool("rating" in df.columns and df["rating"].notna().any()),
                    "hasColours": bool("colour" in df.columns and (df["colour"].astype(str) != "").any()),
                    "chips": chips, "matrix": _matrix(df), "timeline": _timeline(df)})


# ─────────────────────────── search ───────────────────────────

def _groups(keys, paths, index, group_on, thresh):
    if group_on and len(keys) > 1:
        groups = _core.cosine_group(keys, paths, index, float(thresh or 0.08))
    else:
        groups = [{"gid": f"g{i}", "keys": [keys[i]], "paths": [paths[i]]} for i in range(len(keys))]
    return groups


def _rows_for(keys, idx2path) -> list[int]:
    key_to_row = {k: i for i, k in enumerate(idx2path.keys())}
    return [key_to_row[k] for k in keys if k in key_to_row]


@bp.post("/search")
def _search():
    data = body()
    name, dim, modality = parse_dataset(data.get("dataset"))
    text = (data.get("text") or "").strip()
    label = (data.get("label") or "").strip()
    if label.startswith("axis:"):
        # A concept axis sorts the dataset: the pictures furthest along it first.
        from .. import ip_labels as _labels
        k = max(1, min(1000, int(data.get("k") or 4)))
        df = load_data(name, n_dim=dim, modality=modality)
        index, idx2path = load_index(name, modality=modality)
        stars, colours, f_from, f_to, filtering = _filter(data)
        allowed = _core.keys_where(idx2path, _core.marks_mask(df, stars, colours, f_from, f_to)) if filtering else None
        try:
            ranked = _labels.rank_along(name, label[5:], k, allowed)
        except ValueError as e:
            raise BadRequest(str(e))
        results = [(key, path, 1.0 - proj) for key, path, proj in ranked]
        keys = [r[0] for r in results]
        paths = [r[1] for r in results]
        groups = _groups(keys, paths, index, bool(data.get("group", True)), data.get("thresh"))
        marks = _marks_lookup(df)
        rows_by_key = {kk: i for i, kk in enumerate(idx2path.keys())}
        dist_by_key = {r[0]: r[2] for r in results}
        for g in groups:
            g["marks"] = [marks.get(p) for p in g["paths"]]
            g["rows"] = [rows_by_key[kk] for kk in g.get("keys", []) if kk in rows_by_key]
            g["scores"] = [round(float(dist_by_key.get(kk, 0.0)), 4) for kk in g.get("keys", [])]
        return jsonify({"groups": groups, "rows": _rows_for(keys, idx2path), "modality": modality, "k": k})
    if label and label != "none":
        # An IP label searched in place of words: its position is a ViT-H image vector, so it
        # searches a ViT-H dataset by picture likeness.
        from .. import ip_labels as _labels
        lab = _labels.load(label)
        text = _labels.point(lab, push=float(data.get("push") or 1.0))
    elif not text:
        raise BadRequest("Describe what you are looking for first.")
    k = max(1, min(1000, int(data.get("k") or 4)))
    df = load_data(name, n_dim=dim, modality=modality)
    index, idx2path = load_index(name, modality=modality)
    stars, colours, f_from, f_to, filtering = _filter(data)
    allowed = None
    if filtering:
        keep = _core.marks_mask(df, stars, colours, f_from, f_to)
        if keep.any():
            allowed = _core.keys_where(idx2path, keep)
    if not isinstance(text, str) and int(index.ndim) != len(text):
        raise BadRequest(f"IP labels live in CLIP ViT-H/14's {len(text)} numbers and this dataset was "
                         f"indexed with a {index.ndim}-number CLIP. Search with a label in a dataset "
                         f"indexed with ViT-H/14.")
    results = _core.search(index, idx2path, text, k, modality=modality, allowed=allowed)
    keys = [r[0] for r in results]
    paths = [r[1] for r in results]
    groups = _groups(keys, paths, index, bool(data.get("group", True)), data.get("thresh"))
    marks = _marks_lookup(df)
    rows_by_key = {k: i for i, k in enumerate(idx2path.keys())}
    dist_by_key = {r[0]: r[2] for r in results}
    for g in groups:
        g["marks"] = [marks.get(p) for p in g["paths"]]
        g["rows"] = [rows_by_key[k] for k in g.get("keys", []) if k in rows_by_key]
        g["scores"] = [round(float(dist_by_key.get(k, 0.0)), 4) for k in g.get("keys", [])]
    return jsonify({"groups": groups, "rows": _rows_for(keys, idx2path), "modality": modality, "k": k})


@bp.post("/story")
def _story():
    data = body()
    name, dim, modality = parse_dataset(data.get("dataset"))
    chunks = [c.strip() for c in (data.get("text") or "").split("\n") if c.strip()]
    if not chunks:
        raise BadRequest("Write one scene per line first.")
    df = load_data(name, n_dim=dim, modality=modality)
    index, idx2path = load_index(name, modality=modality)
    stars, colours, f_from, f_to, filtering = _filter(data)
    allowed = None
    if filtering:
        keep = _core.marks_mask(df, stars, colours, f_from, f_to)
        if keep.any():
            allowed = _core.keys_where(idx2path, keep)
    rows_by_key = {k: i for i, k in enumerate(idx2path.keys())}
    scenes = []
    for chunk in chunks:
        hits = _core.search(index, idx2path, chunk, 1, modality=modality, allowed=allowed)
        if hits:
            key, path, _ = hits[0]
            scenes.append({"text": chunk, "path": path, "row": rows_by_key.get(key)})
    return jsonify({"scenes": scenes, "chunks": chunks, "modality": modality})


@bp.post("/click")
def _click():
    """The card for a point clicked on the map."""
    data = body()
    path = data.get("path")
    if not path:
        raise BadRequest("No path.")
    name, dim, modality = parse_dataset(data.get("dataset"))
    df = load_data(name, n_dim=dim, modality=modality)
    marks = _marks_lookup(df)
    return jsonify({"path": path, "marks": marks.get(path), "modality": modality})


# ─────────────────────────── saving ───────────────────────────

def _safe_output_dir(*parts) -> str:
    cleaned = [str(p) for p in parts if p not in (None, "")] or ["untitled"]
    target = _paths.safe_join(_paths.subdir("output"), *cleaned)
    if target is None:
        raise BadRequest("That folder name is not allowed. Use a plain name without slashes, drive letters, or '..'.")
    return _paths.ensure_dir(target)


@bp.post("/save")
def _save():
    """Copy chosen pictures to output/search/<folder>, named by a hash prefix so twins stay apart."""
    data = body()
    paths = [p for p in (data.get("paths") or []) if isinstance(p, str) and p]
    if not paths:
        raise BadRequest("No images selected.")
    folder = (data.get("folder") or "").strip() or _paths.dated_name(data.get("prompt") or "search")
    save_dir = _safe_output_dir("search", folder)
    n = 0
    for path in paths:
        full = _media.resolve_path(path)
        safe = f"{hashlib.md5(path.encode('utf-8')).hexdigest()[:8]}_{os.path.basename(full)}"
        try:
            if os.path.splitext(full)[1].lower() in AUDIO_EXTS:
                shutil.copy2(full, os.path.join(save_dir, safe))
            else:
                img = imread_unicode(full)
                if img is not None:
                    imwrite_unicode(os.path.join(save_dir, safe), img)
                else:
                    shutil.copy2(full, os.path.join(save_dir, safe))
            n += 1
        except Exception as e:
            print(f"[save] could not save {full}: {e}")
    import json
    import time
    with open(os.path.join(save_dir, "search.json"), "w", encoding="utf-8") as fh:
        json.dump({"queries": [{"text": data.get("prompt") or "", "dataset": data.get("dataset") or "", "k": len(paths),
                                "added": n, "when": time.time()}]}, fh, indent=1)
    _media.register_media_root(_paths.subdir("output"))
    return jsonify({"saved": n, "folder": save_dir, "message": f"{n} files saved successfully to {save_dir}."})


@bp.post("/story/save")
def _story_save():
    data = body()
    scenes = data.get("scenes") or []
    if not scenes:
        return jsonify({"saved": 0, "message": "No story to save."})
    text = data.get("text") or "story"
    folder = (data.get("folder") or "").strip() or _paths.dated_name(text[:40])
    save_dir = _safe_output_dir("stories", folder)
    poetry_dir = _paths.ensure_dir(os.path.join(save_dir, "poetry"))
    original_dir = _paths.ensure_dir(os.path.join(save_dir, "original"))
    n = 0
    for i, item in enumerate(scenes):
        img = imread_unicode(_media.resolve_path(item["path"]))
        if img is not None:
            imwrite_unicode(os.path.join(original_dir, f"{i:02d}_original.jpg"), img)
            n += 1
        poetry = item.get("poetryPath")
        if poetry and os.path.exists(poetry):
            pimg = imread_unicode(poetry)
            if pimg is not None:
                imwrite_unicode(os.path.join(poetry_dir, f"{i:02d}_poetry.jpg"), pimg)
                n += 1
    with open(os.path.join(save_dir, "story.txt"), "w", encoding="utf-8") as f:
        for i, chunk in enumerate(data.get("chunks") or [s["text"] for s in scenes]):
            f.write(f"{i + 1}. {chunk}\n")
    return jsonify({"saved": n, "folder": save_dir,
                    "message": f"Story and {n} images (original + poetry-injected) saved successfully to {save_dir}."})


# ─────────────────────────── inject poetry ───────────────────────────

def _poetry_job(scenes: list[dict], folder: str | None, strength_val):
    def run(handle):
        subfolder = folder or _paths.dated_name("story")
        output_dir = _safe_output_dir("stories", subfolder, "poetry")
        handle.update(fraction=0.0, message="Loading the image model", detail=_gpu.hardware_note())
        strength = 0.72 if strength_val is None else max(0.05, min(1.0, float(strength_val)))
        n = len(scenes)
        out = []
        ctx = _ops.context("person")
        for idx, item in enumerate(scenes):
            handle.raise_if_cancelled()
            prompt = item["text"]
            handle.update(done=idx, total=n, message=f"Scene {idx + 1} of {n}", detail=prompt[:70])

            def _step(frac, msg, _idx=idx, _p=prompt):
                handle.update(fraction=(_idx + (frac or 0)) / float(n), message=f"Scene {_idx + 1} of {n}",
                              detail=f"{msg} · {_p[:50]}")
            ctx.progress = _step
            res = _ops.call("generate", {"prompt": prompt, "init": _media.resolve_path(item["path"]),
                                         "strength": strength, "steps": 4, "seed": 2222 + idx,
                                         "folder": f"stories/{subfolder}/poetry"}, ctx=ctx)
            out.append({"text": prompt, "path": item["path"], "poetryPath": res.path})
        handle.update(fraction=1.0, done=n, total=n, message="Done", detail="")
        _media.register_media_root(_paths.subdir("output"))
        return {"items": out, "output_dir": output_dir}
    return run


@bp.post("/story/poetry")
def _poetry():
    data = body()
    scenes = [s for s in (data.get("scenes") or []) if s.get("path") and s.get("text")]
    if not scenes:
        raise BadRequest("Generate a story first.")
    if _jobs.MANAGER.active():
        raise BadRequest("Something is already running. Wait for it, or cancel it.")
    job = _jobs.MANAGER.submit(_poetry_job(scenes, (data.get("folder") or "").strip() or None,
                                           data.get("strength")), kind="poetry", label="Injecting poetry")
    return jsonify({"job": job})


# ─────────────────────────── the moodboard's similarity search ───────────────────────────

@bp.post("/similar")
def _similar():
    data = body()
    ref = data.get("reference")
    if not ref:
        raise BadRequest("Choose a reference picture [R] in the collection first.")
    name, dim, modality = parse_dataset(data.get("dataset"))
    if modality != "image":
        raise BadRequest("Similarity search only available for images")
    use_palette, use_style = bool(data.get("palette", True)), bool(data.get("style", False))
    if not use_palette and not use_style:
        raise BadRequest("Please enable at least one feature (Color Palette or Style/Texture)")
    try:
        from ..db import search_by_palette, search_by_style
    except ImportError:                               # pragma: no cover
        from db import search_by_palette, search_by_style
    _media.preload_palette_cache(name)
    try:
        index, idx2path = load_index(name, modality=modality)
    except Exception as e:
        raise BadRequest(f"Failed to load index: {e}")
    k = max(1, min(500, int(data.get("k") or 50)))
    n_colors = int(data.get("nColors") or 16)
    palette_method = data.get("paletteMethod") or "emd"
    style_method = data.get("styleMethod") or "gram"
    prompt = (data.get("prompt") or "").strip()
    candidates = None
    if prompt:
        hits = _core.search(index, idx2path, prompt, min(k * 3, len(idx2path)), modality=modality)
        candidates = {r[1] for r in hits}
    scores: dict[str, float] = {}
    try:
        if use_palette:
            for path, score in search_by_palette(ref, name, idx2path, method=palette_method, n_colors=n_colors,
                                                 top_k=len(idx2path)):
                if candidates is None or path in candidates:
                    scores[path] = scores.get(path, 0) + score * (0.5 if use_style else 1.0)
        if use_style:
            try:
                style_hits = search_by_style(ref, name, idx2path, method=style_method, top_k=len(idx2path))
            except Exception:
                style_hits = search_by_style(ref, name, idx2path, method="edge", top_k=len(idx2path))
            for path, score in style_hits:
                if candidates is None or path in candidates:
                    scores[path] = scores.get(path, 0) + score * (0.5 if use_palette else 1.0)
    except FileNotFoundError as e:
        raise BadRequest(f"Features not found: {e}. Rework the dataset with palette and style features.")
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    ranked = [(p, s) for p, s in ranked if os.path.normpath(p) != os.path.normpath(ref)]
    if not ranked:
        return jsonify({"groups": [], "message": "No results found"})
    score_map = {p: s for p, s in ranked}
    path2key = {v: k_ for k_, v in idx2path.items()}
    keys = [path2key[p] for p, _ in ranked if p in path2key]
    paths = [p for p, _ in ranked if p in path2key]
    groups = _groups(keys, paths, index, bool(data.get("group", True)), data.get("thresh"))
    marks = {}
    try:
        marks = _marks_lookup(load_data(name, n_dim=dim, modality=modality))
    except Exception as e:
        print(f"[marks] no badges for the moodboard results: {type(e).__name__}: {e}")
    for g in groups:
        g["scores"] = [round(float(score_map.get(p, 0.0)), 3) for p in g["paths"]]
        g["marks"] = [marks.get(p) for p in g["paths"]]
    return jsonify({"groups": groups, "db": name})


@bp.get("/features")
def _features():
    """Whether a dataset carries palette and style feature blocks, for the similarity card."""
    name = request.args.get("name") or ""
    db_dir = _media.DB_DIR
    return jsonify({"palette": os.path.exists(os.path.join(db_dir, f"features_{name}_palette.npz")),
                    "style": os.path.exists(os.path.join(db_dir, f"features_{name}_style.npz"))})
