"""
A small synthetic dataset, laid out as db.py lays out a real one, for tests
that exercise search without downloading a model.

The text encoder is replaced by fixed vectors: each known word is a direction,
and every other phrase -- which includes the generic descriptions
core.confidence() weighs a query against -- is one shared "ordinary picture"
direction. Used by test_mcp_server.py, test_ops_registry.py and test_cli.py.
"""

import os
import pickle

import numpy as np
import pandas as pd

DIM = 64
N_WATER, N_FOREST, N_MIXED, N_NOISE = 5, 5, 3, 30
N = N_WATER + N_FOREST + N_MIXED + N_NOISE


def axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


WORDS = {"water": axis(0), "forest": axis(1), "spaceship": axis(63)}
GENERIC = axis(62)


_FEATURES: list = []


def _palette_features(idx2path):
    """
    Palette features for the synthetic pictures, computed once per session.

    The pictures are the same in every scratch folder, so are their features.
    Each picture is one flat colour, and k-means asked for 32 colours in it
    warns once per picture and can take minutes when it competes for threads,
    so the warning is silenced and the work done a single time.
    """
    if not _FEATURES:
        import warnings
        from arcana.palette import extract_all_palette_features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _FEATURES.extend(extract_all_palette_features(idx2path[k]) for k in range(len(idx2path)))
    return _FEATURES


def write_dataset(tmp, palette_features=False):
    """
    A dataset of water, forest, water-and-forest, and ordinary pictures.

    The ordinary pictures lean towards GENERIC, the way real photographs of
    nothing in particular sit closest to "a photo" or "a landscape" in CLIP:
    that is what lets a query for something absent come back weak. The mixed
    pictures carry enough noise not to be near-copies of each other, so the
    negatives test has three of them to watch.
    """
    from PIL import Image
    from usearch.index import Index

    rng = np.random.default_rng(7)
    photos = os.path.join(tmp, "photos")
    os.makedirs(photos)
    db = os.path.join(tmp, "databases")
    lat = os.path.join(tmp, "latents")
    os.makedirs(db)
    os.makedirs(lat)

    vecs, labels = [], []
    for _ in range(N_WATER):
        vecs.append(axis(0) + 0.05 * rng.standard_normal(DIM)); labels.append("Water")
    for _ in range(N_FOREST):
        vecs.append(axis(1) + 0.05 * rng.standard_normal(DIM)); labels.append("Forest")
    for _ in range(N_MIXED):
        vecs.append(axis(0) + axis(1) + 0.15 * rng.standard_normal(DIM)); labels.append("Mixed")
    for _ in range(N_NOISE):
        vecs.append(GENERIC + 0.3 * rng.standard_normal(DIM)); labels.append("Noise")

    n = len(vecs)
    index = Index(ndim=DIM, metric="cos")
    idx2path = {}
    for k, v in enumerate(vecs):
        index.add(k, np.asarray(v, np.float32))
        p = os.path.join(photos, f"img_{k:03d}.jpg")
        shade = (40 + 5 * k) % 255
        Image.new("RGB", (64, 48), (shade, 90, 200 - shade % 150)).save(p, "JPEG")
        idx2path[k] = p
    with open(os.path.join(db, "index_synth_image.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)

    ratings = [5, 4, None, 3, 2] * (n // 5) + [None] * (n % 5)
    pd.DataFrame({
        "x": np.arange(n, dtype=np.float32), "y": np.zeros(n, np.float32),
        "path": list(idx2path.values()), "label": labels,
        "rating": ratings[:n], "colour": ["Red" if k % 2 else "" for k in range(n)],
        "captured": pd.to_datetime(["2025-06-05"] * n),
    }).to_pickle(os.path.join(lat, "latent_space_synth_image_2d.pkl"))

    if palette_features:
        from arcana import db as _db
        feats = _palette_features(idx2path)
        np.savez_compressed(
            os.path.join(db, "features_synth_palette.npz"),
            ids=np.arange(n, dtype=np.int32),
            histogram=np.stack([f["histogram"] for f in feats]).astype(np.float32),
            dominant=np.stack([f["dominant"] for f in feats]).astype(np.float32),
            moments=np.stack([f["moments"] for f in feats]).astype(np.float32),
            fmt=np.array([_db.PALETTE_FEATURE_FMT], dtype=np.int32))
    return idx2path


def use(tmp, monkeypatch, palette_features=False):
    """Point Arcana at a fresh synthetic dataset in `tmp`, with the fixed encoder."""
    from arcana import core, db as _db
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    # db.py resolves its databases folder once, at import.
    monkeypatch.setattr(_db, "db_dir", os.path.join(tmp, "databases"))
    monkeypatch.setattr(core, "encode_query",
                        lambda q, index, modality="image": WORDS.get(q, GENERIC))
    monkeypatch.setattr(core, "_DISTRACTOR_VECS", {})
    return write_dataset(tmp, palette_features)


def pixels(tmp, on=True):
    """The person's switch for agents, in agent.json."""
    import json
    with open(os.path.join(tmp, "agent.json"), "w", encoding="utf-8") as fh:
        json.dump({"pixels": on}, fh)
