# spaces.py -- a second index for a dataset, in the space every modality shares
#
# A dataset is built with the encoder of its modality: CLIP for photos and
# video, CLAP for sound. Those spaces do not meet, so a sound cannot find a
# picture. This module gives any dataset one more set of vectors, in the omni
# encoder's space (omni.py), beside the index it already has and without
# touching it. Once two datasets both have one, anything in either can be the
# query for the other: a field recording against the photo archive, a
# photograph against the clips. See VIDEO_ROADMAP.md, 2.5.
#
# The vectors are kept as a plain matrix, keys and rows, in
#
#     databases/space_<name>_<modality>.<space>.npz
#
# and searched by one matrix product. At 768 numbers a row, a hundred thousand
# items is 300 MB and a few milliseconds, and the whole matrix is needed anyway
# for what makes mixed queries fair (see `rank`).

from __future__ import annotations

import os

import numpy as np

try:
    from . import modality as _modality
    from . import omni as _omni
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import modality as _modality
    import omni as _omni
    import paths as _paths

DEFAULT = _omni.SPACE
# A video's sections are embedded twice: what they show, and what they sound
# like. The sound vectors live in a space file of their own, in the same
# shared space, so a search can be asked to listen instead of look.
SOUND = _omni.SPACE + "-sound"
SOUND_SECONDS = 30.0


def space_path(name: str, modality: str, space: str = DEFAULT, db_dir: str | None = None) -> str:
    return os.path.join(db_dir or _paths.subdir("databases"), f"space_{name}_{modality}.{space}.npz")


def has(name: str, modality: str, space: str = DEFAULT, db_dir: str | None = None) -> bool:
    return os.path.exists(space_path(name, modality, space, db_dir))


_LOADED: dict = {}


def load(name: str, modality: str, space: str = DEFAULT, db_dir: str | None = None):
    """(keys, unit vectors) for a dataset in a space, or None when it has none. Cached by file time."""
    path = space_path(name, modality, space, db_dir)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    hit = _LOADED.get(path)
    if hit is None or hit[0] != stamp:
        data = np.load(path)
        hit = (stamp, data["keys"].astype(np.int64), data["vectors"].astype(np.float32))
        _LOADED[path] = hit
    return hit[1], hit[2]


def build(name: str, modality: str, idx2path: dict, *, space: str = DEFAULT, db_dir: str | None = None,
          progress=None, should_cancel=None, encode=None, limit: int = 0, batch: int = 8,
          channel: str = "picture") -> dict:
    """
    Embed every item of a dataset into the shared space.

    Resumes: keys already in the file are kept, so a cancelled run or a dataset
    that has grown costs only what is new. Items that cannot be read are
    reported and left out. Returns {"added", "kept", "skipped", "total", "path"}.
    """
    encode = encode or _omni.encode
    if channel == "sound":
        if modality != _modality.VIDEO:
            raise ValueError("The sound channel is for video datasets. A sound dataset is sound already.")
        space = SOUND
    path = space_path(name, modality, space, db_dir)
    have = load(name, modality, space, db_dir)
    keys = list(have[0]) if have else []
    rows = [v for v in have[1]] if have else []
    done = set(int(k) for k in keys)
    todo = [(int(k), p) for k, p in idx2path.items() if int(k) not in done]
    if limit:
        todo = todo[:limit]
    kept, skipped, added = len(keys), 0, 0
    cache: dict = {}
    # Sections of one video sit together, so each file is decoded once.
    todo.sort(key=lambda kp: _modality.split_ref(kp[1])[0])

    def save():
        _paths.ensure_dir(os.path.dirname(os.path.abspath(path)))
        tmp = path + ".tmp.npz"
        np.savez(tmp, keys=np.asarray(keys, np.int64),
                 vectors=np.stack(rows).astype(np.float32) if rows else np.zeros((0, _omni.DIM), np.float32),
                 space=np.array([space]))
        os.replace(tmp, path)
        _LOADED.pop(path, None)

    for start in range(0, len(todo), batch):
        if should_cancel is not None and should_cancel():
            break
        chunk = todo[start:start + batch]
        if progress is not None:
            progress(start / max(len(todo), 1), f"Embedding into the shared space ({start:,} of {len(todo):,})")
        if channel == "sound":
            items = [_sound_item(p) for _, p in chunk]
        else:
            items = [_video_item(p, cache) if modality == _modality.VIDEO else _omni.item_for(p)
                     for _, p in chunk]
        try:
            vecs = encode(items)
            good = list(zip(chunk, vecs))
        except Exception:
            # One bad file must not cost the batch: go through it one at a time.
            good = []
            for (k, p), it in zip(chunk, items):
                try:
                    good.append(((k, p), encode([it])[0]))
                except Exception as e:                       # noqa: BLE001
                    print(f"[WARN] not embedded: {p}: {e}")
                    skipped += 1
        for (k, _), v in good:
            keys.append(k)
            rows.append(np.asarray(v, np.float32))
            added += 1
        # Saved every few batches, so a long run that stops keeps what it did.
        if (start // batch) % 10 == 9:
            save()
    save()
    if progress is not None:
        progress(1.0, f"{added:,} embedded")
    return {"added": added, "kept": kept, "skipped": skipped, "total": len(idx2path), "path": path}


# A video is decoded once for all its sections. Reading each section's own
# range took about a second a section on 1080p 60 fps camera files, more than
# the model itself; one pass at a low rate and a small size costs the same as
# a single section did.
DECODE_FPS = 4.0
DECODE_MINUTES = 12.0          # longer files are read section by section, to bound memory


def _sound_item(ref: str) -> dict:
    """A section's sound track, up to half a minute of it."""
    file, t0, t1 = _modality.split_ref(ref)
    t0 = t0 or 0.0
    return _omni.sound(file, t0, None if t1 is None else min(t1, t0 + SOUND_SECONDS))


def _video_item(ref: str, cache: dict) -> dict:
    try:
        from . import video as _video
    except ImportError:                               # pragma: no cover
        import video as _video
    file, t0, t1 = _modality.split_ref(ref)
    if cache.get("file") != file:
        cache.clear()
        cache["file"] = file
        try:
            info = _video.probe(file)
            if info["duration"] <= DECODE_MINUTES * 60:
                got = [f for _, f in _video.frames(file, DECODE_FPS, side=_omni.FRAME_SIDE, info=info)]
                cache["frames"] = np.stack(got) if got else None
                cache["duration"] = info["duration"]
        except Exception:
            cache["frames"] = None
    stack = cache.get("frames")
    if stack is None or not len(stack):
        return _omni.clip(ref)
    a = 0 if t0 is None else int(t0 * DECODE_FPS)
    b = len(stack) if t1 is None else max(a + 1, int(np.ceil(t1 * DECODE_FPS)))
    part = stack[a:min(b, len(stack))]
    if not len(part):
        return _omni.clip(ref)
    if len(part) > _omni.FRAMES:
        part = part[np.linspace(0, len(part) - 1, _omni.FRAMES).round().astype(int)]
    return _omni.frames(part)


# ─────────────────────────── ranking ───────────────────────────

def _z(scores: np.ndarray) -> np.ndarray:
    sd = float(scores.std())
    return (scores - float(scores.mean())) / sd if sd > 1e-8 else np.zeros_like(scores)


def rank(keys, vectors, terms, k: int = 12, allowed=None) -> list[tuple[int, float]]:
    """
    The best `k` keys for a mixed query: [(key, score), ...], best first.

    `terms` are (vector, weight) pairs, the weight negative for something to
    move away from. Each term's similarities are standardised over the dataset
    before they are added. In a shared space a text sits closer to every
    picture than a sound does, whatever either says, so raw sums would let one
    kind of term drown the other; standardised, a text and a sound each count
    for their weight.
    """
    if len(keys) == 0 or not terms:
        return []
    total = np.zeros(len(keys), np.float32)
    for vec, weight in terms:
        v = np.asarray(vec, np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        total += float(weight) * _z(vectors @ v)
    order = np.argsort(-total)
    out = []
    for i in order:
        key = int(keys[i])
        if allowed is not None and key not in allowed:
            continue
        out.append((key, float(total[i])))
        if len(out) >= k:
            break
    return out
