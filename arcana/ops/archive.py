# archive.py -- what every read op shares: settings, handles, datasets, hits
#
# Moved here from mcp_server.py, where these were the private helpers of six
# MCP tools. They are the vocabulary the ops speak: a dataset id (japan, or
# field@audio), a handle (japan:1423), what the frame knows about a picture,
# its palette, and one line per picture in a fixed order.

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from .. import core
    from .. import models as _models
    from .. import paths as _paths
except ImportError:                                   # pragma: no cover
    import core
    import models as _models
    import paths as _paths


CONFIG_NAME = "agent.json"
AUDIO_SUFFIX = "@audio"

# Tile sizes by purpose: see sheets.py for what each costs.
SCAN_TILE = 160
INSPECT_SIDE = 1024
SHEET_COLS = 6


# ─────────────────────────── settings ───────────────────────────

def settings() -> dict:
    """
    The person's choices for the agent, read fresh on every call.

    Read each time rather than at start-up, so turning pixels on or off applies
    to the next call without restarting the client.
    """
    path = os.path.join(_paths.data_dir(), CONFIG_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    collection = data.get("collection")
    return {"pixels": bool(data.get("pixels", False)),
            "collection": collection if isinstance(collection, str) and collection else None,
            "path": path}


def mode_line(cfg: dict | None = None) -> str:
    cfg = cfg or settings()
    if cfg["pixels"]:
        return "Mode: pixels. Thumbnails are included."
    return ("Mode: blind. No pictures leave this machine, so results are text only: "
            "cluster names, palettes, match strength. The person can allow thumbnails "
            f"by setting \"pixels\": true in {cfg['path']}.")


def dataset_arg(index: int = 0):
    """The positional dataset id most ops start with. Completes from the datasets on disk."""
    from .registry import positional
    return positional(index, description="Dataset id, e.g. japan (field@audio for an audio set).",
                      json_schema_extra={"complete": "dataset", "metavar": "DATASET"})


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


def resolve_handles(handles: list[str], one_dataset: bool = True):
    """
    Handles to (dataset id, dataset, keys), refusing ones the dataset lacks.

    Most ops that take pictures take them from one dataset, since a handle's
    key means nothing outside its index.
    """
    parsed = [parse_handle(h) for h in handles]
    ds_ids = list(dict.fromkeys(d for d, _ in parsed))
    if one_dataset and len(ds_ids) != 1:
        raise ValueError("These pictures come from " + ", ".join(ds_ids)
                         + ". Give pictures from one dataset at a time.")
    ds_id = ds_ids[0]
    ds = open_dataset(ds_id)
    keys = [k for _, k in parsed]
    missing = [handle(ds_id, k) for k in keys if k not in ds.idx2path]
    if missing:
        raise ValueError(f"Not in {ds_id}: {', '.join(missing)}.")
    return ds_id, ds, keys


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
    # pd.notna rather than x == x: a frame with nullable integer columns holds
    # pd.NA for an unrated picture, and pd.NA == pd.NA raises instead of
    # answering False the way NaN does.
    if "label" in rec and pd.notna(rec["label"]) and str(rec["label"]) not in ("", "nan"):
        out["cluster"] = str(rec["label"])
    if "rating" in rec and pd.notna(rec["rating"]):
        out["stars"] = int(rec["rating"])
    if "colour" in rec and pd.notna(rec["colour"]) and str(rec["colour"]) not in ("", "nan"):
        out["colour"] = str(rec["colour"])
    if "captured" in rec and pd.notna(rec["captured"]):
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
        from .. import palette as _palette
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


def feature_blocks(name: str) -> list[str]:
    db = _paths.subdir("databases")
    return [f for f in ("palette", "style")
            if os.path.exists(os.path.join(db, f"features_{name}_{f}.npz"))]


def encoder_label(ds: core.Dataset) -> str | None:
    """The encoder's short name (ViT-H/14), or None when the dimension matches none we know."""
    if ds.modality != "image":
        return None
    try:
        info = _models.get(core.text_model_for_dim(int(ds.index.ndim)))
    except RuntimeError:
        return None
    if info is None:
        return str(ds.index.ndim)
    return info.label.split(" — ")[0]


# ─────────────────────────── one dataset, described ───────────────────────────

class DatasetInfo(BaseModel):
    id: str
    modality: str
    count: int = 0
    encoder: str | None = None
    ndim: int | None = None
    clusters: int | None = None
    has_map: bool = True
    features: list[str] = Field(default_factory=list)
    marks: list[str] = Field(default_factory=list)
    moved_share: float = 0.0
    error: str | None = None

    def line(self) -> str:
        if self.error:
            return f"{self.id} · could not be opened: {self.error}"
        noun = "pictures" if self.modality == "image" else "sounds"
        parts = [self.id, self.modality, f"{self.count:,} {noun}"]
        if self.modality == "image":
            parts.append(f"encoder {self.encoder}" if self.encoder
                         else f"unknown {self.ndim}-d encoder")
        if not self.has_map:
            parts.append("no map yet (search works; clusters do not)")
        elif self.clusters is not None:
            parts.append(f"{self.clusters} clusters")
        parts.append("features: " + (", ".join(self.features) if self.features else "none"))
        if self.marks:
            parts.append("marks: " + ", ".join(self.marks))
        if self.moved_share:
            parts.append(f"FILES MOVED: about {self.moved_share:.0%} are not where the index "
                         f"says. Search still works, but thumbnails and inspect cannot show "
                         f"them. Tell the person; Arcana's Relocate (or arcana-relocate) fixes it")
        return " · ".join(parts)


def describe_dataset(ds_id: str) -> DatasetInfo:
    ds = open_dataset(ds_id)
    info = DatasetInfo(id=ds_id, modality=ds.modality, count=len(ds.idx2path),
                       encoder=encoder_label(ds), ndim=int(ds.index.ndim),
                       has_map=ds.frame is not None, features=feature_blocks(ds.name))
    if ds.frame is not None and "label" in ds.frame.columns:
        info.clusters = int(ds.frame["label"].nunique())
    if ds.frame is not None and core.has_marks(ds.frame):
        marks = []
        if "rating" in ds.frame.columns and ds.frame["rating"].notna().any():
            marks.append("stars")
        if "colour" in ds.frame.columns and (ds.frame["colour"].astype(str) != "").any():
            marks.append("colour tags")
        if "captured" in ds.frame.columns and ds.frame["captured"].notna().any():
            when = ds.frame["captured"].dropna()
            marks.append(f"dates {str(when.min())[:10]} to {str(when.max())[:10]}")
        info.marks = marks
    info.moved_share = missing_share(ds)
    return info


# ─────────────────────────── one picture, as a line ───────────────────────────

class Hit(BaseModel):
    """One picture in a result, with what the index knows about it."""
    num: int
    handle: str
    file: str
    path: str
    metric: str | None = None          # "p 0.92", "z 1.4", "sim 0.88"
    score: float | None = None
    stars: int | None = None
    colour: str | None = None
    captured: str | None = None
    cluster: str | None = None
    missing: bool = False
    palette: list[str] = Field(default_factory=list)

    def line(self) -> str:
        bits = [f"#{self.num}", self.handle]
        if self.metric:
            bits.append(self.metric)
        if self.stars is not None:
            bits.append("★" * self.stars if self.stars > 0 else "0★")
        if self.colour:
            bits.append(self.colour)
        if self.captured:
            bits.append(self.captured)
        if self.cluster:
            bits.append(f"[{self.cluster}]")
        bits.append(self.file)
        if self.missing:
            bits.append("(file missing)")
        if self.palette:
            bits.append(" ".join(self.palette))
        return "  ".join(bits)


def make_hit(num: int, ds_id: str, ds: core.Dataset, key: int, path: str,
             metric: str | None = None, score: float | None = None,
             pal: dict | None = None) -> Hit:
    f = facts(ds, key)
    return Hit(num=num, handle=handle(ds_id, key), file=os.path.basename(path), path=path,
               metric=metric, score=score, stars=f.get("stars"), colour=f.get("colour"),
               captured=f.get("captured"), cluster=f.get("cluster"),
               missing=not os.path.exists(path),
               palette=(pal or {}).get(key, []))


def sheet(paths: list[str], numbers: list[int] | None = None, *, tile: int = SCAN_TILE,
          cols: int = SHEET_COLS, title: str | None = None, caption: str = ""):
    """A contact sheet as an Attachment."""
    try:
        from .. import sheets as _sheets
    except ImportError:                               # pragma: no cover
        import sheets as _sheets
    from .registry import Attachment
    data = _sheets.contact_sheet(paths, numbers, tile=tile, cols=cols, title=title)
    return Attachment(caption=caption or title or "contact sheet", data=data)


# ─────────────────────────── items: handles, derived pictures, paths ───────────────────────────

class Item(BaseModel):
    """A picture an op was given, resolved to its file."""
    ref: str                           # what the caller wrote
    path: str
    handle: str | None = None          # japan:1423 or out:3f9c1e
    dataset: str | None = None
    kind: str = "archive"              # archive, derived, generated, file
    derived_from: str | None = None    # a handle, for derived pictures


def is_path(ref: str) -> bool:
    """A file path rather than a handle: handles never hold a slash."""
    return "/" in ref or "\\" in ref


def resolve_item(ref: str, actor: str = "cli") -> Item:
    """
    A picture reference to its file.

    Handles (japan:1423) and ledger handles (out:3f9c1e) work for everyone. A
    plain path works for the person and the terminal, which is how the app's
    collection (a list of paths) reaches the board ops. An agent names
    pictures by handle only (A6).
    """
    try:
        from .. import ledger as _ledger
    except ImportError:                               # pragma: no cover
        import ledger as _ledger
    ref = (ref or "").strip()
    if ref.startswith(_ledger.PREFIX):
        e = _ledger.get(ref)
        if e is None:
            raise ValueError(f"No derived picture called {ref}. `item show` takes out: handles "
                             f"that colour transfer, style transfer and generate return.")
        return Item(ref=ref, path=e["path"], handle=e["handle"], kind=e.get("kind", "derived"),
                    derived_from=(e.get("sources") or [None])[0])
    if is_path(ref):
        if actor == "agent":
            raise ValueError(f"Name pictures by handle (japan:1423), not by path. Got {ref!r}.")
        found = handle_for_path(ref)
        if found:
            return Item(ref=ref, path=ref, handle=found, dataset=split_handle_ds(found))
        e = _ledger.for_path(ref)
        if e:
            return Item(ref=ref, path=ref, handle=e["handle"], kind=e.get("kind", "derived"),
                        derived_from=(e.get("sources") or [None])[0])
        return Item(ref=ref, path=ref, kind="file")
    ds_id, key = parse_handle(ref)
    ds = open_dataset(ds_id)
    p = ds.idx2path.get(key)
    if p is None:
        raise ValueError(f"Not in {ds_id}: {ref}.")
    return Item(ref=ref, path=p, handle=handle(ds_id, key), dataset=ds_id)


def split_handle_ds(h: str) -> str:
    return h.rpartition(":")[0]


_REVERSE: dict = {}


def handle_for_path(p: str, prefer: str | None = None) -> str | None:
    """
    The handle of a file that some dataset indexed, or None.

    Boards saved by the app hold paths only, so showing one to an agent needs
    the way back. Each dataset's path-to-key map is built once and kept while
    its index file is unchanged.
    """
    norm = os.path.normcase(os.path.abspath(p))
    found = core.find_datasets()
    order = sorted(found, key=lambda d: dataset_id(d["name"], d["modality"]) != prefer)
    for d in order:
        ds_id = dataset_id(d["name"], d["modality"])
        ipath = core.index_path(d["name"], d["modality"])
        try:
            stamp = os.path.getmtime(ipath)
        except OSError:
            continue
        cached = _REVERSE.get(ds_id)
        if cached is None or cached[0] != stamp:
            try:
                ds = open_dataset(ds_id)
            except Exception:
                continue
            cached = (stamp, {os.path.normcase(os.path.abspath(v)): k
                              for k, v in ds.idx2path.items()})
            _REVERSE[ds_id] = cached
        key = cached[1].get(norm)
        if key is not None:
            return handle(ds_id, key)
    return None
