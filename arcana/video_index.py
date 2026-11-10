# video_index.py -- a folder of videos as a searchable index of sections
#
# For each video: read a small frame every half second (a tick), embed the
# ticks with the dataset's CLIP image encoder, find where the footage changes
# (sections.py), and add one vector per section to the index, at every level
# from the whole file down to single moments. See VIDEO_ROADMAP.md, 2.3 to 2.6.
#
# The index it returns has the shape every other dataset has, (usearch index,
# {key: path}), so search, the map, cluster naming and the cards work on it
# unchanged. A key's "path" is the video file plus the section's time range
# (modality.section_ref).
#
# Beside the index it writes two files:
#
#   sections_<name>_video.pkl   one row per key: file, video number, t0, t1,
#                               the levels the section belongs to
#   tree_<name>_video.pkl       per video, the boundary strengths and cuts, so
#                               the sections can be redrawn at other
#                               sensitivities without reading a frame again
#
# A section's vector is the mean of its ticks' vectors. That is the content
# channel in CLIP's space, the one the photo archives share. The omni encoder,
# the dynamics channel and the sound channel of the roadmap add their own
# indexes beside this one.

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

try:
    from . import modality as _modality
    from . import motion as _motion
    from . import paths as _paths
    from . import compat as _compat
    from . import sections as _sections
    from . import video as _video
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import motion as _motion
    import paths as _paths
    import compat as _compat
    import sections as _sections
    import video as _video

LEVEL_ORDER = ("file",) + tuple(name for name, _ in _sections.LEVELS)

# Frames are read four times as often as ticks are taken. The extra frames cost
# almost nothing to decode, since ffmpeg decodes every frame of the file either
# way, and they are what the motion is measured from: at two frames a second a
# shake or a quick pan falls between the samples.
MOTION_PER_TICK = 4


def motion_path(name: str, db_dir: str | None = None) -> str:
    return os.path.join(db_dir or _paths.subdir("databases"),
                        _modality.feature_file(name, "motion", "video"))


def sections_path(name: str, db_dir: str | None = None) -> str:
    return os.path.join(db_dir or _paths.subdir("databases"), f"sections_{name}_video.pkl")


def tree_path(name: str, db_dir: str | None = None) -> str:
    return os.path.join(db_dir or _paths.subdir("databases"), f"tree_{name}_video.pkl")


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _modality.VIDEO_EXTS


# ─────────────────────────── encoding ticks ───────────────────────────

def clip_frames(frames, model_id: str | None = None, batch: int = 64) -> np.ndarray:
    """RGB frames as CLIP image vectors, one row each, unit length."""
    import torch
    try:
        from . import db as _db
    except ImportError:                               # pragma: no cover
        import db as _db
    model, proc = _db.load_clip(model_id=model_id)
    dtype = next(model.parameters()).dtype
    out = []
    for i in range(0, len(frames), batch):
        px = proc(images=list(frames[i:i + batch]), return_tensors="pt").pixel_values
        with torch.no_grad():
            vec = _compat.features(model.get_image_features(px.to(model.device, dtype))).float().cpu().numpy()
        if not np.isfinite(vec).all():
            # Half precision overflowed on this batch: do it again in full.
            with torch.no_grad():
                vec = _compat.features(model.float().get_image_features(px.to(model.device))).cpu().numpy()
            model.to(dtype)
        out.append(vec)
    if not out:
        return np.zeros((0, 0), np.float32)
    return _sections.unit_rows(np.concatenate(out))


# ─────────────────────────── one video ───────────────────────────

def analyse(path: str, encode=None, *, levels=_sections.LEVELS,
            min_seconds: float = _sections.MIN_SECONDS, should_cancel=None) -> dict | None:
    """
    One video as ticks and sections.

    Returns {"info", "X" (tick vectors), "result" (sections.section_ticks),
    "cuts"}, or None when the file holds no frame that can be read. `encode`
    turns a list of RGB frames into vectors; it defaults to the default CLIP.
    """
    encode = encode or clip_frames
    info = _video.probe(path)
    frames, sigs = [], []
    X_parts = []
    rate = MOTION_PER_TICK / _sections.TICK_SECONDS
    before = None
    moves = []
    for k, (_, f) in enumerate(_video.frames(path, rate, side=224, info=info)):
        g = _motion.grey(f)
        if before is not None:
            moves.append(_motion.pair(before, g, 1.0 / rate))
        before = g
        if k % MOTION_PER_TICK:
            continue
        frames.append(f)
        sigs.append(_video.colour_signature(f))
        # Encode as the frames arrive, so an hour of video is never held in memory.
        if len(frames) >= 256:
            X_parts.append(encode(frames))
            frames = []
            if should_cancel is not None and should_cancel():
                raise KeyboardInterrupt("cancelled")
    if frames:
        X_parts.append(encode(frames))
    if not X_parts:
        return None
    X = np.concatenate(X_parts)
    moves = np.stack(moves) if moves else np.zeros((0, len(_motion.PAIR)), np.float32)
    cuts = _video.detect_cuts(np.stack(sigs))
    duration = info["duration"] or len(X) * _sections.TICK_SECONDS
    result = _sections.section_ticks(X, cuts=cuts, levels=levels, min_seconds=min_seconds,
                                     duration=duration)
    return {"info": info, "X": X, "result": result, "cuts": cuts, "moves": moves}


def section_motion(moves: np.ndarray, section) -> np.ndarray:
    """A section's motion vector, from the frame pairs that fall inside it."""
    a, b = section.i0 * MOTION_PER_TICK, section.i1 * MOTION_PER_TICK
    return _motion.summarise(moves[a:max(a, min(b, len(moves)))])


def unique_sections(result: dict) -> list[tuple[_sections.Section, list[str]]]:
    """
    Each distinct time range once, with every level it belongs to.

    A video that never changes is one section at every level, and a coarse
    section that holds no finer change is also a medium and a fine one. Each is
    stored and embedded once, and remembers its levels so a search at any level
    still finds it.
    """
    seen: dict[tuple[int, int], tuple[_sections.Section, list[str]]] = {}
    for s in _sections.all_sections(result):
        key = (s.i0, s.i1)
        if key in seen:
            seen[key][1].append(s.level)
        else:
            seen[key] = (s, [s.level])
    return list(seen.values())


# ─────────────────────────── a folder ───────────────────────────

def build(paths, index_path: str, name: str, *, model_id: str | None = None, progress=None,
          should_cancel=None, encode=None, db_dir: str | None = None):
    """
    Index videos by section. Returns (index, idx2path) like db.build.

    `paths` are video files. An unreadable one is reported and skipped.
    """
    from usearch.index import Index

    def report(done, total, message=""):
        if progress is not None:
            progress(done, total, message)

    encode = encode or (lambda frames: clip_frames(frames, model_id=model_id))
    paths = sorted(p for p in paths if is_video(p))
    if not paths:
        raise SystemExit("No video files found for indexing.")
    report(0, len(paths), f"Found {len(paths):,} videos")

    index = None
    idx2path: dict[int, str] = {}
    rows: list[dict] = []
    motions: list[np.ndarray] = []
    tree: dict = {}
    key = 0
    seconds = 0.0
    for n, path in enumerate(paths):
        if should_cancel is not None and should_cancel():
            raise KeyboardInterrupt("cancelled")
        report(n, len(paths), f"Reading {os.path.basename(path)}")
        try:
            got = analyse(path, encode, should_cancel=should_cancel)
        except (_video.VideoError, OSError) as e:
            print(f"[WARN] skipping {path}: {e}")
            continue
        if got is None:
            print(f"[WARN] no frames in {path}, skipping")
            continue
        X, result, info = got["X"], got["result"], got["info"]
        if index is None:
            index = Index(ndim=int(X.shape[1]), metric="cos", dtype="f32")
        unique = unique_sections(result)
        vectors = _sections.pooled(X, [s for s, _ in unique])
        for (s, levels), vec in zip(unique, vectors):
            ref = _modality.section_ref(path, s.t0, s.t1)
            index.add(key, vec)
            idx2path[key] = ref
            moved = section_motion(got["moves"], s)
            motions.append(moved)
            rows.append({"key": key, "path": ref, "file": path, "video": n,
                         "t0": float(s.t0), "t1": float(s.t1), "seconds": float(s.t1 - s.t0),
                         "levels": ",".join(levels), "level": levels[0],
                         "motion": _motion.describe(moved),
                         "tags": ",".join(_motion.tags(moved))})
            key += 1
        tree[path] = {"strengths": result["strengths"], "cuts": list(got["cuts"]),
                      "duration": float(result["file"].t1), "ticks": int(len(X)),
                      "width": info["width"], "height": info["height"], "fps": info["fps"],
                      "has_audio": info["has_audio"]}
        seconds += float(result["file"].t1)

    if index is None or not idx2path:
        raise SystemExit("No video could be read; nothing to index.")
    report(len(paths), len(paths), f"{len(tree):,} videos, {len(idx2path):,} sections")
    print(f"[INFO] Indexed {len(tree)} videos ({seconds / 60:.1f} minutes) as {len(idx2path)} sections.")

    _paths.ensure_dir(os.path.dirname(os.path.abspath(index_path)))
    with open(index_path, "wb") as f:
        pickle.dump((index.save(), idx2path), f)
    out_dir = db_dir or os.path.dirname(os.path.abspath(index_path))
    pd.DataFrame(rows).to_pickle(sections_path(name, out_dir))
    with open(tree_path(name, out_dir), "wb") as f:
        pickle.dump(tree, f)
    # How each section moves, beside the index like the palette and style blocks.
    np.savez_compressed(motion_path(name, out_dir), ids=np.arange(key, dtype=np.int32),
                        motion=np.stack(motions).astype(np.float32),
                        names=np.array(_motion.NAMES), fmt=np.array([1], dtype=np.int32))
    return index, idx2path


def extend(name: str, root: str, index_path: str, *, model_id: str | None = None, progress=None,
           should_cancel=None, encode=None, db_dir: str | None = None) -> dict:
    """
    Add the videos that appeared under `root` since the dataset was built.

    Only the new files are read. Their sections are appended to the index, the
    sections table, the boundary tree and the motion block, with keys that go
    on from the last one, so nothing already indexed changes its key. Returns
    {"added" (files), "sections", "skipped"}.
    """
    from usearch.index import Index

    out_dir = db_dir or os.path.dirname(os.path.abspath(index_path))
    with open(index_path, "rb") as fh:
        blob, idx2path = pickle.load(fh)
    index = Index.restore(blob)
    sections = pd.read_pickle(sections_path(name, out_dir))
    with open(tree_path(name, out_dir), "rb") as fh:
        tree = pickle.load(fh)
    moved = np.load(motion_path(name, out_dir))
    motion_ids, motion_rows = list(moved["ids"]), list(moved["motion"])

    have = {os.path.normcase(os.path.abspath(p)) for p in tree}
    found = []
    for folder, _, names in os.walk(root):
        for f in names:
            p = os.path.join(folder, f)
            if is_video(p) and os.path.normcase(os.path.abspath(p)) not in have:
                found.append(p)
    found.sort()
    encode = encode or (lambda frames: clip_frames(frames, model_id=model_id))
    key = (max(idx2path) + 1) if idx2path else 0
    video = (int(sections["video"].max()) + 1) if len(sections) else 0
    rows, added, skipped = [], 0, 0
    for n, path in enumerate(found):
        if should_cancel is not None and should_cancel():
            break
        if progress is not None:
            progress(n, len(found), f"Reading {os.path.basename(path)}")
        try:
            got = analyse(path, encode, should_cancel=should_cancel)
        except (_video.VideoError, OSError) as e:
            print(f"[WARN] skipping {path}: {e}")
            got = None
        if got is None:
            skipped += 1
            continue
        X, result, info = got["X"], got["result"], got["info"]
        unique = unique_sections(result)
        for (s, levels), vec in zip(unique, _sections.pooled(X, [s for s, _ in unique])):
            ref = _modality.section_ref(path, s.t0, s.t1)
            index.add(key, vec)
            idx2path[key] = ref
            m = section_motion(got["moves"], s)
            motion_ids.append(key)
            motion_rows.append(m)
            rows.append({"key": key, "path": ref, "file": path, "video": video,
                         "t0": float(s.t0), "t1": float(s.t1), "seconds": float(s.t1 - s.t0),
                         "levels": ",".join(levels), "level": levels[0],
                         "motion": _motion.describe(m), "tags": ",".join(_motion.tags(m))})
            key += 1
        tree[path] = {"strengths": result["strengths"], "cuts": list(got["cuts"]),
                      "duration": float(result["file"].t1), "ticks": int(len(X)),
                      "width": info["width"], "height": info["height"], "fps": info["fps"],
                      "has_audio": info["has_audio"]}
        video += 1
        added += 1
    if added:
        with open(index_path, "wb") as fh:
            pickle.dump((index.save(), idx2path), fh)
        pd.concat([sections, pd.DataFrame(rows)], ignore_index=True).to_pickle(sections_path(name, out_dir))
        with open(tree_path(name, out_dir), "wb") as fh:
            pickle.dump(tree, fh)
        np.savez_compressed(motion_path(name, out_dir), ids=np.asarray(motion_ids, np.int32),
                            motion=np.stack(motion_rows).astype(np.float32),
                            names=np.array(_motion.NAMES), fmt=np.array([1], dtype=np.int32))
        _SECTIONS.clear()
        _MOTION.clear()
    return {"added": added, "sections": len(rows), "skipped": skipped}


# ─────────────────────────── reading it back ───────────────────────────

_SECTIONS: dict = {}


def load_sections(name: str, db_dir: str | None = None) -> pd.DataFrame | None:
    """The sections table of a video dataset, cached while its file is unchanged."""
    path = sections_path(name, db_dir)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    hit = _SECTIONS.get(path)
    if hit is None or hit[0] != stamp:
        hit = (stamp, pd.read_pickle(path))
        _SECTIONS.clear()
        _SECTIONS[path] = hit
    return hit[1]


_MOTION: dict = {}


def load_motion(name: str, db_dir: str | None = None) -> dict | None:
    """{key: motion vector} for a video dataset, cached while its file is unchanged."""
    path = motion_path(name, db_dir)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    hit = _MOTION.get(path)
    if hit is None or hit[0] != stamp:
        data = np.load(path)
        hit = (stamp, {int(k): v for k, v in zip(data["ids"], data["motion"])})
        _MOTION.clear()
        _MOTION[path] = hit
    return hit[1]


def _z(values) -> np.ndarray:
    v = np.asarray(values, np.float32)
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 1e-8 else np.zeros_like(v)


def rank_with_motion(hits: list[tuple], motion: dict, text: str, balance: float):
    """
    Re-rank (key, path, distance) hits by content and movement together.

    `balance` runs from 0, content alone, to 1, movement alone. Each side is
    standardised over the hits before mixing, since a cosine distance and a
    motion score live on different scales. Returns (hits best first, the
    movement words read from `text`); the hits come back untouched when the
    text names no movement.
    """
    if not hits:
        return hits, []
    vectors = np.stack([motion.get(int(h[0]), np.zeros(len(_motion.NAMES), np.float32)) for h in hits])
    moved, names = _motion.query_scores(text, vectors)
    if moved is None:
        return hits, []
    content = _z([-float(h[2]) for h in hits])
    score = (1.0 - balance) * content + balance * _z(moved)
    order = np.argsort(-score, kind="stable")
    return [hits[int(i)] for i in order], names


def keys_at(df: pd.DataFrame, level: str = "auto", min_seconds: float | None = None,
            max_seconds: float | None = None, tags=()) -> set:
    """The keys a search may return: sections of one level, within a length range, moving a given way."""
    keep = pd.Series(True, index=df.index)
    for tag in tags or ():
        if "tags" in df.columns:
            keep &= df["tags"].map(lambda v, tag=tag: tag in str(v).split(","))
    if level and level != "auto":
        keep &= df["levels"].map(lambda v: level in v.split(","))
    if min_seconds:
        keep &= df["seconds"] >= float(min_seconds)
    if max_seconds:
        keep &= df["seconds"] <= float(max_seconds)
    return set(int(k) for k in df.loc[keep, "key"])


def whole_videos(df: pd.DataFrame, hits: list[tuple], n: int):
    """
    Whole videos, ranked by their best section, each with a strip of where it matches.

    `hits` are (key, path, distance) for sections, best first. Returns
    (hits for the whole files, {file path: strip}). A strip is
    {"seconds", "parts": [[t0, t1, score], ...], "best": [t0, t1]}: the video's
    finest sections with a score from 0 to 1 over this search, so the card can
    draw where in the file the match sits and jump there.
    """
    if not hits:
        return [], {}
    by_key = df.set_index("key")
    sims = np.array([-float(h[2]) for h in hits], np.float32)
    lo, hi = float(sims.min()), float(sims.max())
    score = {int(h[0]): (float(s) - lo) / (hi - lo) if hi > lo else 1.0 for h, s in zip(hits, sims)}
    whole = {int(r.video): (int(r.key), r.path, float(r.seconds))
             for r in df.itertuples() if "file" in str(r.levels).split(",")}
    finest = "fine" if df["levels"].str.contains("fine").any() else "file"
    order, best = [], {}
    for h in hits:
        row = by_key.loc[h[0]] if h[0] in by_key.index else None
        if row is None:
            continue
        v = int(row["video"])
        if v not in best:
            best[v] = (h, float(row["t0"]), float(row["t1"]))
            order.append(v)
    out, strips = [], {}
    for v in order[:n]:
        if v not in whole:
            continue
        key, path, seconds = whole[v]
        h, t0, t1 = best[v]
        parts = [[float(r.t0), float(r.t1), round(score.get(int(r.key), 0.0), 3)]
                 for r in df[df["video"] == v].itertuples() if finest in str(r.levels).split(",")]
        parts.sort()
        out.append((key, path, h[2]))
        strips[path] = {"seconds": seconds, "parts": parts, "best": [t0, t1]}
    return out, strips


def best_per_video(df: pd.DataFrame, hits: list[tuple], n: int) -> list[tuple]:
    """
    Auto granularity over a ranked list of (key, path, distance).

    Within each video, a section is kept only when no section it overlaps
    ranks above it, so a match comes back once, at the length that fits it
    best. `hits` are best first and stay in that order.
    """
    by_key = df.set_index("key")
    taken: dict[int, list[tuple[float, float]]] = {}
    out = []
    for hit in hits:
        row = by_key.loc[hit[0]] if hit[0] in by_key.index else None
        if row is None:
            continue
        spans = taken.setdefault(int(row["video"]), [])
        t0, t1 = float(row["t0"]), float(row["t1"])
        if any(t0 < b and a < t1 for a, b in spans):
            continue
        spans.append((t0, t1))
        out.append(hit)
        if len(out) >= n:
            break
    return out
