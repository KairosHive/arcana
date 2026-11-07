# core.py — search, filters and dataset loading, without the app
#
# These lived in arcana.py, which builds the whole Dash app when it is imported:
# the layout, every callback, the Flask routes. So nothing could search a
# dataset without also constructing a web application -- not a test, not a
# command-line tool, and not the MCP server an agent talks to, which runs as its
# own process and must not start a UI to answer a query.
#
# So the pieces that only need numbers, files and a text encoder are here, and
# arcana.py imports them back under the same names. This module must never
# import dash; tests/test_core.py checks that.
#
# What stays in arcana.py is anything that decides what the browser may see:
# load_data() there wraps the one here and registers the dataset's files with the
# media allowlist. That is a policy of the web server, not of a dataset.

from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
from usearch.index import Index

try:
    from . import paths as _paths
except ImportError:
    import paths as _paths


# ------------- DATASET FILES -------------
# A dataset is two files written by db.py: the usearch index (vectors plus the
# key -> path map) and the latent frame (2-D or 3-D layout, cluster labels, and
# ON1 marks when there are any). The index is enough to search; the frame is
# what the map and the filters need.

def index_path(name: str, modality: str = "image", db_dir: str | None = None) -> str:
    return os.path.join(db_dir or _paths.subdir("databases"), f"index_{name}_{modality}.pkl")


def latent_path(name: str, n_dim: int = 2, modality: str = "image",
                latent_dir: str | None = None) -> str:
    return os.path.join(latent_dir or _paths.subdir("latents"),
                        f"latent_space_{name}_{modality}_{n_dim}d.pkl")


def load_data(name, n_dim=2, modality="image", latent_dir=None):
    """The latent frame for a dataset: one row per item, typed for plotting."""
    df = pd.read_pickle(latent_path(name, n_dim, modality, latent_dir))

    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "path" in df.columns:
        df["path"] = df["path"].astype(str)
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df.reset_index(drop=True)


def load_index(name, modality="image", db_dir=None):
    """The usearch index and its key -> path map."""
    with open(index_path(name, modality, db_dir), "rb") as f:
        idx_blob, idx2path = pickle.load(f)
    return Index.restore(idx_blob), idx2path


@dataclass(frozen=True)
class Dataset:
    """
    One dataset, opened.

    `frame` is None when the index exists but no latent space was ever laid out
    -- an indexing run that stopped after encoding leaves exactly that, and the
    index is still searchable. Treat `frame` as read-only: it is shared by
    everyone who opened the same dataset.
    """
    name: str
    modality: str
    n_dim: int
    index: Index
    idx2path: dict
    frame: pd.DataFrame | None


_OPEN: dict = {}
_OPEN_LOCK = threading.Lock()


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def open_dataset(name: str, modality: str = "image", n_dim: int = 2, *,
                 db_dir: str | None = None, latent_dir: str | None = None) -> Dataset:
    """
    A dataset, loaded once per process and reloaded when its files change.

    The app reads its pickles afresh on every interaction and gets away with it
    because a person clicks slowly. An agent issues a dozen searches in a row;
    restoring an 82k-vector index each time would dominate every one of them.

    The cache is keyed on the files' modification times as well as the name, so
    a dataset reworked or extended by another process is picked up on the next
    call rather than served stale until a restart.
    """
    ipath = index_path(name, modality, db_dir)
    lpath = latent_path(name, n_dim, modality, latent_dir)
    key = (name, modality, n_dim, ipath, lpath)
    stamp = (_mtime(ipath), _mtime(lpath))

    with _OPEN_LOCK:
        hit = _OPEN.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]

    index, idx2path = load_index(name, modality, db_dir)
    frame = load_data(name, n_dim, modality, latent_dir) if stamp[1] is not None else None
    ds = Dataset(name=name, modality=modality, n_dim=n_dim,
                 index=index, idx2path=idx2path, frame=frame)
    with _OPEN_LOCK:
        _OPEN[key] = (stamp, ds)
    return ds


# ------------- MARKS: STARS, COLOUR TAGS, CAPTURE DATES -------------
# Read from ON1 sidecars at index time and carried in the latent frame (see
# db.attach_marks). A dataset indexed before this existed has none of these
# columns, so everything here treats a missing column as "nothing known" and
# leaves the dataset behaving exactly as it did.
MARK_COLUMNS = ("rating", "colour", "captured")

# The colour dropdown needs a value meaning "no tag at all", and "" cannot be
# one: Dash treats an empty string as no selection and silently drops it.
NO_COLOUR = "(none)"

# The star control is a floor -- "three and up" -- so it needs one value that is
# not a floor at all, for the pictures nothing has ever been decided about.
# Negative, so it can never be mistaken for a star count.
UNRATED = -1


def has_marks(df) -> bool:
    """Does this dataset carry anything worth filtering on?"""
    if df is None or not len(df):
        return False
    if "rating" in df.columns and df["rating"].notna().any():
        return True
    if "colour" in df.columns and (df["colour"].astype(str) != "").any():
        return True
    return "captured" in df.columns and df["captured"].notna().any()


def marks_mask(df, stars=0, colours=None, date_from=None, date_to=None):
    """
    Boolean mask over `df` for the active filters. All-True when none are set.

    An unrated file fails a star filter: "three stars and up" is a request for
    pictures somebody chose, and a file ON1 has never opened is not one of them.
    The same file passes a colour filter of ["(none)"], which is the way to ask
    for exactly the unsorted remainder.
    """
    mask = pd.Series(True, index=df.index)

    if stars and "rating" in df.columns:
        if int(stars) == UNRATED:
            mask &= df["rating"].isna()
        else:
            mask &= df["rating"].fillna(UNRATED) >= int(stars)

    if colours and "colour" in df.columns:
        col = df["colour"].astype(str)
        wanted = set(colours)
        picked = pd.Series(False, index=df.index)
        if NO_COLOUR in wanted:
            picked |= col == ""
            wanted.discard(NO_COLOUR)
        if wanted:
            picked |= col.isin(wanted)
        mask &= picked

    if (date_from or date_to) and "captured" in df.columns:
        when = df["captured"]
        # An undated file cannot satisfy a date range, and NaT comparisons are
        # False anyway -- being explicit keeps that from looking accidental.
        dated = when.notna()
        if date_from:
            dated &= when >= pd.Timestamp(date_from)
        if date_to:
            # The picker gives a day; the user means the whole of it.
            dated &= when < pd.Timestamp(date_to) + pd.Timedelta(days=1)
        mask &= dated

    return mask


def keys_where(idx2path: dict, keep) -> set:
    """
    usearch keys the filter left standing.

    The latent frame is built by walking idx2path in insertion order, so row i
    of the frame is the i-th key -- the same correspondence the search-result
    highlighting relies on, and the reason neither can use the key as a row
    label directly.
    """
    flags = list(keep)
    return {k for i, k in enumerate(idx2path.keys()) if i < len(flags) and flags[i]}


def allowed_keys(ds: Dataset, stars=0, colours=None, date_from=None, date_to=None) -> set | None:
    """Keys passing the marks filter, or None when nothing is filtered out."""
    if ds.frame is None or not (stars or colours or date_from or date_to):
        return None
    return keys_where(ds.idx2path, marks_mask(ds.frame, stars, colours, date_from, date_to))


# ------------- NEAR-DUPLICATES -------------

def cosine_group(keys, paths, index: Index, thresh: float = 0.08):
    """
    Group the top-N results by cosine distance threshold (1 - cosine_sim).
    Returns: [{'gid': 'g0', 'keys': [...], 'paths': [...]}...], preserving rank order.
    """
    if len(keys) <= 1:
        return [{"gid": "g0", "keys": keys, "paths": paths}]

    vecs = np.stack([index.get(k) for k in keys]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    sim = vecs @ vecs.T
    dist = 1.0 - sim

    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= thresh:
                union(i, j)

    buckets = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(i)

    ordered = sorted(buckets.values(), key=lambda idxs: min(idxs))
    groups = []
    for gi, idxs in enumerate(ordered):
        groups.append(
            {
                "gid": f"g{gi}",
                "keys": [keys[i] for i in idxs],
                "paths": [paths[i] for i in idxs],
            }
        )
    return groups


# ------------- CLIP TEXT ENCODER (lazy) -------------
# Searching only ever encodes the prompt, so only the text tower is needed:
# 354M parameters instead of the full model's 986M, and ~0.9 s to load instead
# of ~25 s. Loading it lazily also means the app starts (and can show an error)
# without a 4 GB download having to succeed first.
#
# torch and transformers are imported inside the functions that need them, so
# importing this module costs nothing a filter or a dataset listing doesn't use.
CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"   # fallback only

# Keyed by model id. A single-slot cache was wrong for the same reason it was
# wrong in db.load_clip: the first dataset you opened decided which text
# encoder every later search used.
_CLIP_TEXT: dict = {}
_MODEL_LOAD_LOCK = threading.Lock()


def text_model_for_dim(ndim: int) -> str:
    """
    Which encoder produced vectors of this width.

    A prompt has to be encoded by the same model that encoded the pictures, or
    the two live in unrelated spaces. Nothing in a legacy index records which
    model built it -- but the vector width does, because Arcana's image
    encoders have distinct dimensions (512 / 768 / 1024). Reading it off the
    index means every dataset searches correctly with no migration and no extra
    metadata.
    """
    try:
        from . import models as _models
    except ImportError:
        import models as _models
    for m in _models.MODELS:
        if m.modality == "image" and m.dim == ndim:
            return m.id
    raise RuntimeError(
        f"This dataset's vectors are {ndim}-dimensional, which does not match "
        f"any encoder Arcana knows about. It was probably built by a different "
        f"version; re-index the folder to search it."
    )


def load_clip_text(device: str = "cpu", model_id: str | None = None):
    """Load a CLIP text tower on first use. Raises with something actionable."""
    model_id = model_id or CLIP_MODEL_ID
    with _MODEL_LOAD_LOCK:
        if model_id not in _CLIP_TEXT:
            try:
                import torch
                from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
                m = CLIPTextModelWithProjection.from_pretrained(model_id, dtype=torch.float32)
                m.eval().to(device)
                tok = CLIPTokenizerFast.from_pretrained(model_id)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load the text encoder '{model_id}': {e}. "
                    f"The first run needs to download it, so check your internet connection. "
                    f"Set HF_HOME to choose where models are cached."
                ) from e
            _CLIP_TEXT[model_id] = (m, tok)
    return _CLIP_TEXT[model_id]


# --- lazy CLAP (keep FP32 to avoid BN/dtype issues) ---
_CLAP = {"model": None, "proc": None}


def load_clap(device="cpu"):
    with _MODEL_LOAD_LOCK:
        if _CLAP["model"] is None:
            from transformers import ClapModel, ClapProcessor
            m = ClapModel.from_pretrained("laion/clap-htsat-fused")
            m.eval().to(device)
            p = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
            _CLAP.update(model=m, proc=p)
    return _CLAP["model"], _CLAP["proc"]


def encode_query(query: str, index: Index, modality: str = "image") -> np.ndarray:
    """
    A text query as a vector in the same space as `index`.

    no_grad is set here rather than relied on: the app switches gradients off
    globally at import, but a process that never imports the app -- the MCP
    server, a test -- would otherwise build an autograd graph for every query.
    """
    import torch

    with torch.no_grad():
        if modality == "image":
            # Match the encoder to the index. This used to always load ViT-H/14
            # while an index built with ViT-B/32 holds 512-d vectors, so every
            # search against such a dataset died inside usearch with "The number
            # of vector dimensions doesn't match!" and the results panel simply
            # stayed empty. ViT-B/32 is the encoder the panel recommends to anyone
            # without a GPU, so prompt search was broken for exactly the people
            # most likely to choose it.
            clip_model, clip_tok = load_clip_text(
                model_id=text_model_for_dim(int(index.ndim)))
            inputs = clip_tok(query, return_tensors="pt")
            return clip_model(**inputs).text_embeds.detach().cpu().numpy().flatten()

        clap_model, clap_proc = load_clap(device="cpu")
        inputs = clap_proc(text=[query], return_tensors="pt", padding=True)
        # keep FP32
        for k in inputs:
            inputs[k] = inputs[k].to(clap_model.device)
        try:
            emb = clap_model.get_text_features(**inputs)
        except AttributeError:
            emb = clap_model(**inputs).text_embeds
        return emb.squeeze().detach().cpu().numpy().flatten()


def search(index, idx2path, query, n, modality="image", allowed=None):
    """
    Top `n` matches for `query`.

    `allowed`, when given, is the set of usearch keys the stars/colour/date
    filter left standing. The search runs over the whole index and the losers
    are dropped afterwards rather than usearch being asked to skip them: the
    search is already `exact=True`, so every distance is computed either way and
    the only extra cost is sorting a longer list. Asking for `n` and hoping
    enough of them survive is what does not work -- a four-star filter over an
    archive that is 5% four-star returns an empty page perhaps 19 times in 20.
    """
    vec = encode_query(query, index, modality)

    if allowed is None:
        idxs = index.search(vec, n, exact=True)
        return [(idx.key, idx2path[idx.key], idx.distance) for idx in idxs]

    idxs = index.search(vec, len(idx2path), exact=True)
    out = []
    for idx in idxs:
        if idx.key in allowed and idx.key in idx2path:
            out.append((idx.key, idx2path[idx.key], idx.distance))
            if len(out) >= n:
                break
    return out


# ------------- RANKING FOR AN AGENT -------------
# A person shown twenty pictures judges them at a glance. An agent asking on
# their behalf needs three things the app never computed: several phrasings and
# exclusions combined into one query, a page that is not twenty versions of one
# photograph, and some sign of whether the best match is any good.

# A diverse page is chosen from this many times as many candidates as it shows.
CANDIDATE_FACTOR = 8

# Two pictures this similar are one picture as far as curation goes. It is the
# app's default grouping threshold -- cosine distance 0.08 -- seen from the
# other side.
NEAR_DUPLICATE = 0.92

# Is what came back actually there? Ranking always returns something, so the
# question is whether the best matches show the query or are merely the least
# bad. Measuring that against the dataset's own score distribution was tried
# first and fails on common subjects: if a fifth of an archive is temples, a
# temple photograph cannot stand more than about two standard deviations above
# the mean, and "temple" would come back weak in an archive full of them.
#
# So ask CLIP the way it was trained to be asked: zero-shot classification of
# each top picture between the query and a bank of generic descriptions. A
# temple photograph prefers "temple" to "a building" however many temples there
# are; a landscape searched for "spaceship" prefers "a landscape". The share of
# probability the query wins is the picture's confidence, p.
DISTRACTORS = {
    "image": ("a photo", "a person", "people", "a building", "a street", "a room",
              "a landscape", "the sky", "a tree", "a plant", "water", "a mountain",
              "an animal", "a vehicle", "food", "an object", "a texture", "text",
              "a painting", "a blurry photo", "darkness", "light"),
}

# A distractor this close to the query in text space is a synonym, not an
# alternative: "water" must not compete with "water".
SYNONYM = 0.9

# CLIP's trained temperature. LAION's models learn it to the same ceiling.
LOGIT_SCALE = 100.0

# Average p of the best five, for "strong" and "fair". Calibrated in October
# 2026 on two ViT-B/32 datasets, japan (246 photographs) and midjourney (9,359
# images): subjects present scored 0.52-1.00 and absent ones 0.00-0.16, with
# "a street at night" (few in that archive) at 0.29. The z-score this replaced
# put "a spreadsheet of numbers" highest of all in the larger set, at 4.7.
# Not yet checked on the larger encoders.
STRONG_P = 0.5
FAIR_P = 0.2

_DISTRACTOR_VECS: dict = {}


@dataclass(frozen=True)
class Hit:
    key: int
    path: str
    score: float   # cosine similarity to the query
    z: float       # standard deviations above the dataset's mean score for this query


@dataclass(frozen=True)
class Ranking:
    hits: list
    mean: float              # the dataset's mean score for this query
    std: float
    considered: int          # items the filter and exclusions left in play
    skipped_duplicates: int  # near-copies of a chosen picture, passed over


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-8)


def unit_rows(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def query_vectors(phrases, index: Index, modality: str = "image",
                  negatives=(), negative_weight: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Several phrasings of one idea, minus what keeps intruding: (query, positive).

    Each phrase is encoded on its own and the unit vectors averaged, so two
    phrasings count equally however long either is. Negatives are averaged the
    same way and subtracted at `negative_weight`: a full 1.0 tends to push the
    query somewhere no picture lives, and the page fills with noise.

    The positive part alone is returned too. Negatives decide the ranking, but
    whether the idea is present at all is a question about the phrases only.
    """
    phrases = [p for p in (phrases or []) if p and p.strip()]
    if not phrases:
        raise ValueError("a query needs at least one phrase")
    pos = unit(np.mean([unit(encode_query(p, index, modality)) for p in phrases], axis=0))
    q = pos
    negatives = [p for p in (negatives or []) if p and p.strip()]
    if negatives:
        neg = np.mean([unit(encode_query(p, index, modality)) for p in negatives], axis=0)
        q = unit(pos - float(negative_weight) * neg)
    return q, pos


def query_vector(phrases, index: Index, modality: str = "image",
                 negatives=(), negative_weight: float = 0.4) -> np.ndarray:
    return query_vectors(phrases, index, modality, negatives, negative_weight)[0]


def confidence(ds: Dataset, positive, keys) -> np.ndarray | None:
    """
    For each picture, the probability CLIP gives the query over generic
    alternatives (see DISTRACTORS). None where there is no distractor bank for
    the modality.
    """
    bank = DISTRACTORS.get(ds.modality)
    keys = list(keys)
    if not bank or not keys:
        return None
    cache_key = (ds.modality, int(ds.index.ndim))
    d = _DISTRACTOR_VECS.get(cache_key)
    if d is None:
        d = unit_rows(np.stack([encode_query(t, ds.index, ds.modality) for t in bank]))
        _DISTRACTOR_VECS[cache_key] = d
    positive = unit(positive)
    d = d[(d @ positive) < SYNONYM]
    v = unit_rows(np.stack([ds.index.get(k) for k in keys]))
    logits = LOGIT_SCALE * np.concatenate([(v @ positive)[:, None], v @ d.T], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, 0] / e.sum(axis=1)


def example_vector(index: Index, keys) -> np.ndarray:
    """
    The direction several stored pictures share: search by example.

    It reads the vectors already in the index, so it needs no encoder, no model
    download and no GPU.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("search by example needs at least one picture")
    return unit(unit_rows(np.stack([index.get(k) for k in keys])).mean(axis=0))


def rank(ds: Dataset, q, k: int, *, allowed: set | None = None, exclude=(),
         diversity: float = 0.3) -> Ranking:
    """
    The best `k` pictures for the vector `q`, spread out, with how good they are.

    Every item is scored -- search() is exact anyway -- which also yields the
    score distribution for this query over the whole dataset, and each hit's z
    against it: how unusual the picture is for this query, here. Whether the
    query is present at all is a different question; see confidence().

    `diversity` trades relevance for variety by maximal marginal relevance: 0 is
    a plain ranking, 1 ignores the query after the first pick. Both terms are
    scaled to 0..1 over the candidates first. Text-to-image cosines span a few
    hundredths while image-to-image ones span most of the range, and unscaled
    the variety term would swamp relevance at any setting. Near-copies of a
    chosen picture are skipped at every setting, 0 included.
    """
    res = ds.index.search(unit(q), len(ds.idx2path), exact=True)
    keys = np.asarray(res.keys).tolist()
    scores = 1.0 - np.asarray(res.distances, dtype=np.float32)
    mean = float(scores.mean()) if len(scores) else 0.0
    std = float(scores.std()) if len(scores) else 0.0
    std = std if std > 1e-6 else 1e-6

    exclude = set(exclude)
    keep = np.array([(allowed is None or kk in allowed) and kk not in exclude
                     and kk in ds.idx2path for kk in keys], dtype=bool)
    considered = int(keep.sum())
    n_cand = max(int(k) * CANDIDATE_FACTOR, 64)
    cand_keys = [kk for kk, ok in zip(keys, keep) if ok][:n_cand]
    cand_scores = scores[keep][:len(cand_keys)]
    if not cand_keys or k <= 0:
        return Ranking([], mean, std, considered, 0)

    vecs = unit_rows(np.stack([ds.index.get(kk) for kk in cand_keys]))
    spread = float(cand_scores.max() - cand_scores.min())
    rel = ((cand_scores - cand_scores.min()) / spread if spread > 1e-9
           else np.ones_like(cand_scores))
    lam = 1.0 - min(max(float(diversity), 0.0), 1.0)

    chosen: list[int] = []
    skipped = 0
    max_sim = np.full(len(cand_keys), -1.0, dtype=np.float32)
    open_ = np.ones(len(cand_keys), dtype=bool)
    while len(chosen) < k and open_.any():
        if chosen:
            # max_sim is a cosine in -1..1; map it to 0..1 to match `rel`.
            mmr = lam * rel - (1.0 - lam) * (max_sim + 1.0) / 2.0
        else:
            mmr = rel.copy()
        mmr = np.where(open_, mmr, -np.inf)
        i = int(np.argmax(mmr))
        open_[i] = False
        if chosen and max_sim[i] >= NEAR_DUPLICATE:
            skipped += 1
            continue
        chosen.append(i)
        max_sim = np.maximum(max_sim, vecs @ vecs[i])

    hits = [Hit(key=int(cand_keys[i]), path=ds.idx2path[cand_keys[i]],
                score=float(cand_scores[i]), z=(float(cand_scores[i]) - mean) / std)
            for i in chosen]
    return Ranking(hits, mean, std, considered, skipped)


def strength(p) -> str:
    """'strong', 'fair', 'weak', or 'none', from the confidences of a ranking's hits."""
    top = list(p[:5]) if p is not None else []
    if not top:
        return "none"
    m = float(np.mean(top))
    return "strong" if m >= STRONG_P else "fair" if m >= FAIR_P else "weak"


def find_datasets(db_dir: str | None = None, latent_dir: str | None = None) -> list[dict]:
    """
    Every dataset with an index, and which layouts it has.

    The app lists only datasets with both an index and a latent space, because
    it needs the map. Search needs only the index, so an indexing run that
    stopped before the layout is listed here too, with no layouts.
    """
    import re
    db_dir = db_dir or _paths.subdir("databases")
    latent_dir = latent_dir or _paths.subdir("latents")
    db_pat = re.compile(r"index_(.+)_(image|audio)\.pkl$")
    lat_pat = re.compile(r"latent_space_(.+)_(image|audio)_(\d+)d\.pkl$")
    layouts: dict = {}
    for fname in _paths.listdir_safe(latent_dir):
        m = lat_pat.match(fname)
        if m:
            layouts.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3)))
    out = []
    for fname in _paths.listdir_safe(db_dir):
        m = db_pat.match(fname)
        if m:
            name, modality = m.group(1), m.group(2)
            out.append({"name": name, "modality": modality,
                        "layouts": sorted(layouts.get((name, modality), []))})
    return sorted(out, key=lambda d: (d["name"], d["modality"]))
