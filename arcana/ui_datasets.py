# ui_datasets.py -- what the Datasets tab reads
#
# The page draws the wizard and the lists; this module answers its questions:
# what is in a folder, what to call it, what each encoder costs, which datasets
# exist and in what state, which encoders are downloaded. Every change (create,
# extend, finish, marks, delete, relocate, download) is an op (ops/dataset.py,
# ops/lifecycle.py), and tests/test_parity.py keeps the writing modules out of
# here.

from __future__ import annotations

import os
import pickle
import re

try:
    from . import db as _db
    from . import models as _models
    from . import paths as _paths
    from . import ui_style as _ui
    from .jobs import MANAGER
except ImportError:                                   # pragma: no cover
    import db as _db
    import models as _models
    import paths as _paths
    import ui_style as _ui
    from jobs import MANAGER

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

picker_busy: dict = {"open": False}


# ─────────────────────────── folders and names ───────────────────────────

def next_name(folder: str, current_name: str | None, last_suggested: str | None) -> str | None:
    """
    What the Name field should become when the folder changes, or None to leave
    it alone: an empty field or one still holding our last guess takes the new
    suggestion; anything the person typed is never touched.
    """
    suggested = suggest_name(folder)
    if not suggested:
        return None
    current = (current_name or "").strip()
    if not current or current == (last_suggested or ""):
        return suggested
    return None


folder_vocabulary = _db.folder_vocabulary


def scan_both(folder: str, cap: int = 200_000, keep: int = 40) -> dict:
    """Count images and audio in one walk, keeping a sample of each."""
    n_img = n_aud = 0
    s_img: list[str] = []
    s_aud: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                n_img += 1
                if len(s_img) < keep:
                    s_img.append(os.path.join(root, f))
            elif ext in AUDIO_EXTS:
                n_aud += 1
                if len(s_aud) < keep:
                    s_aud.append(os.path.join(root, f))
            if n_img + n_aud >= cap:
                return {"image": n_img, "audio": n_aud, "sample_image": s_img, "sample_audio": s_aud}
    return {"image": n_img, "audio": n_aud, "sample_image": s_img, "sample_audio": s_aud}


def suggest_name(folder: str) -> str:
    """A dataset name from the folder: lowercased, punctuation to hyphens, at most 48 characters."""
    base = os.path.basename(os.path.normpath(folder or ""))
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in base).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.lower()[:48]


def scan_media(folder: str, modality: str, cap: int = 200_000, keep: int = 40) -> tuple[int, list[str]]:
    exts = IMAGE_EXTS if modality == "image" else AUDIO_EXTS
    n = 0
    sample: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                n += 1
                if len(sample) < keep:
                    sample.append(os.path.join(root, f))
                if n >= cap:
                    return n, sample
    return n, sample


def count_media(folder: str, modality: str, cap: int = 200_000) -> int:
    return scan_media(folder, modality, cap, keep=0)[0]


# ─────────────────────────── encoders ───────────────────────────

def model_choices(modality: str, n_items: int = 0, decode_ms: float | None = None) -> list[dict]:
    """The encoder catalogue for a modality, each with what it costs here."""
    rows = _models.catalogue(modality, n_items=n_items, decode_ms=decode_ms)
    has_gpu = _models.gpu_available()
    out = []
    for r in rows:
        bits = [f"{r['dim']}-d", "ready" if r["downloaded"] else f"{r['download_mb']:,} MB download"]
        if r["estimate"]:
            bits.append(r["estimate"])
        note = [r["blurb"]]
        if n_items and r["estimate"]:
            note.append(f"Indexing {n_items:,} files with this encoder takes {r['estimate']} "
                        f"{'using your GPU' if has_gpu else 'on the CPU'}.")
        if not r["downloaded"]:
            note.append(f"It will be downloaded first ({r['download_mb']:,} MB).")
        out.append({**r, "summary": " · ".join(bits), "note": " ".join(note),
                    "default": r["id"] == _models.default_for(modality).id})
    return out


def model_rows() -> list[dict]:
    status = {}
    for modality in ("image", "audio"):
        try:
            for s in _db.label_cache_status(modality):
                status[s["model_id"]] = s
        except Exception:
            pass
    rows = []
    for m in _models.MODELS:
        downloaded, why_not = _models.verify_model(m.id)
        partial = (not downloaded) and any(w in why_not for w in ("incomplete", "truncated"))
        labels_ready = bool(status.get(m.id, {}).get("ready", False))
        rows.append({**m.as_dict(), "downloaded": downloaded, "partial": partial, "labelsReady": labels_ready,
                     "complete": downloaded and labels_ready, "whyNot": why_not if not downloaded else ""})
    return rows


# ─────────────────────────── datasets ───────────────────────────

_HEALTH_CACHE: dict = {}
_EXTRAS_CACHE: dict = {}


def invalidate_dataset_cache() -> None:
    _HEALTH_CACHE.clear()
    _EXTRAS_CACHE.clear()


def _dataset_extras(d) -> dict:
    """Named labels, a palette block, a style block, and whether the palette is dated."""
    latent = (d.latent_paths or {}).get(2) or next(iter((d.latent_paths or {}).values()), None)
    key = (d.name, d.modality, latent, os.path.getmtime(latent) if latent and os.path.exists(latent) else 0)
    hit = _EXTRAS_CACHE.get(key)
    if hit is not None:
        return hit
    out = {"palette": bool(d.palette_path and os.path.exists(d.palette_path)),
           "style": bool(d.style_path and os.path.exists(d.style_path)),
           "labels": False, "palette_stale": False}
    if out["palette"]:
        try:
            out["palette_stale"] = _db.palette_features_stale(d.name)
        except Exception:
            pass
    if latent and os.path.exists(latent):
        try:
            with open(latent, "rb") as fh:
                df = pickle.load(fh)
            if "label" in getattr(df, "columns", ()):
                out["labels"] = df["label"].dtype == object
        except Exception:
            pass
    _EXTRAS_CACHE[key] = out
    return out


def dataset_rows() -> list[dict]:
    """One record per discovered dataset, with health, extras and what can be done to it."""
    try:
        from .relocate import dataset_health
        from .legacy import discover
    except ImportError:                               # pragma: no cover
        from relocate import dataset_health
        from legacy import discover
    try:
        active = MANAGER.active()
    except Exception:
        active = None
    rows = []
    for d in discover():
        key = (d.name, d.modality, d.index_path,
               os.path.getmtime(d.index_path) if os.path.exists(d.index_path) else 0)
        h = _HEALTH_CACHE.get(key)
        if h is None:
            try:
                h = dataset_health(d.name, d.modality, sample=40)
            except Exception:
                h = {"ok": True, "total": 0, "missing": 0, "root": "", "error": ""}
            _HEALTH_CACHE[key] = h
        ex = _dataset_extras(d)
        has_map = bool(d.latent_paths)
        label = (active or {}).get("label") or ""
        building = bool(active) and label.startswith(("dataset create " + d.name, "dataset extend " + d.name,
                                                      "dataset finish " + d.name, "dataset marks " + d.name))
        if h.get("error"):
            status = "unreadable"
        elif building:
            status = "building"
        elif not has_map:
            status = "unfinished"
        elif h.get("missing"):
            status = "files missing"
        else:
            status = "ready"
        rows.append({"name": d.name, "modality": d.modality, "total": int(h.get("total") or 0),
                     "root": h.get("root") or "", "missing": int(h.get("missing") or 0),
                     "error": h.get("error") or "", "hasMap": has_map, "dims": sorted(d.latent_paths or {}),
                     "labels": ex["labels"], "palette": ex["palette"], "paletteStale": ex.get("palette_stale", False),
                     "style": ex["style"], "status": status, "building": building,
                     "canRework": has_map and not h.get("error") and not h.get("missing"),
                     "canExtend": has_map and not h.get("error"),
                     "canMarks": has_map and d.modality == "image"})
    return rows


def picker_options() -> list[dict]:
    """One entry per (dataset, layout) with both an index and a map, for the Dataset menu."""
    lat_pat = re.compile(r"latent_space_(.+)_(image|audio)_(\d+)d\.pkl$")
    db_pat = re.compile(r"index_(.+)_(image|audio)\.pkl$")
    lat_map: dict = {}
    for fname in _paths.listdir_safe(_paths.subdir("latents")):
        m = lat_pat.match(fname)
        if m:
            lat_map.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3)))
    db_keys = {(m.group(1), m.group(2)) for fname in _paths.listdir_safe(_paths.subdir("databases"))
               if (m := db_pat.match(fname))}
    options = []
    for (name, mod), dims in lat_map.items():
        if (name, mod) in db_keys:
            for d in sorted(dims):
                options.append({"label": f"{name} · {mod} ({d}D)", "value": f"{name}::{d}::{mod}",
                                "name": name, "modality": mod, "dim": d})
    return sorted(options, key=lambda x: x["label"])


def agent_command() -> str:
    """The command that registers this Arcana with Claude Code, for this install."""
    import sys
    if getattr(sys, "frozen", False):
        return f'claude mcp add arcana -- "{sys.executable}" --mcp'
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f'claude mcp add arcana -- uv run --directory "{repo}" arcana-mcp'
