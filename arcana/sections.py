# sections.py -- cutting a video where it changes, at several nested scales
#
# A video dataset is searched by section, and a section should last as long as
# what it shows: ten seconds of water, thirty of sky, twelve of someone walking.
# Fixed windows cannot give that. This module finds the sections from the
# footage itself. See VIDEO_ROADMAP.md, 2.3.
#
# The input is a stream of ticks: one feature vector every half second (a CLIP
# frame vector, with motion descriptors beside it when there are any). Nothing
# here decodes a frame or loads a model, so it runs in milliseconds and is
# tested on synthetic streams.
#
#   1. Neighbouring ticks are merged pair by pair, always the two neighbours
#      whose merge costs least (Ward's criterion, restricted to neighbours in
#      time), until the whole video is one segment.
#   2. Every point between two ticks gets a strength: the cost at which its two
#      sides finally merged. A strong boundary is a place where the footage
#      changes and stays changed.
#   3. A threshold on strength gives the sections. Three thresholds give three
#      nested levels, coarse, medium and fine, from the one tree.
#
# A hard cut found by a cut detector raises the strength of the boundary where
# it falls. A gradual change inside one take gets a boundary at the point of
# greatest change, so continuous footage is sectioned as well as edited footage.

from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict

import numpy as np

TICK_SECONDS = 0.5

# Each level keeps the boundaries at least this strong, as a share of the
# video's strongest boundary. Coarse first: the order is also the nesting.
LEVELS: tuple[tuple[str, float], ...] = (("coarse", 0.35), ("medium", 0.12), ("fine", 0.04))

# Below this strength a boundary is noise whatever the video's own range is.
# Strengths are Ward costs divided by the number of ticks, on unit vectors, so
# the number is a squared distance: 0.012 is two halves whose mean vectors sit
# at a cosine of about 0.976. It is what keeps a video that never changes from
# being cut at its own grain.
#
# Measured with CLIP ViT-B/32 ticks on twenty of the person's own videos
# (evals/video_sections.py): single takes of one subject, a talking head or a
# figure in a cave, peaked between 0.003 and 0.008 and were being cut in two at
# the first value, 0.004. Videos with a real change of subject peaked at 0.018
# and above. Another encoder will want its own number.
FLOOR = 0.012

MIN_SECONDS = 2.0
CUT_BOOST = 2.0


@dataclass(frozen=True)
class Section:
    level: str            # coarse, medium, fine, or "file" for the whole video
    index: int            # position among the sections of its level
    i0: int               # first tick
    i1: int               # one past the last tick
    t0: float             # seconds
    t1: float
    parent: int | None    # index of the section one level coarser that holds it

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0

    def as_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────── features ───────────────────────────

def unit_rows(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("ticks are a 2-D array, one row per tick")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-8)


def tick_features(content, motion=None, motion_weight: float = 0.5) -> np.ndarray:
    """
    One row per tick for the sectioning: content vectors, with motion beside them.

    Content rows are unit length. Motion columns are standardised over the video
    and scaled so that all of them together weigh `motion_weight` against the
    content, which keeps a 48-number descriptor from drowning a 1024-number
    embedding or being drowned by it. A video whose content never changes and
    whose motion does (a locked-off shot that starts to shake) is then cut
    where the motion changes.
    """
    X = unit_rows(content)
    if motion is None or motion_weight <= 0:
        return X
    M = np.asarray(motion, dtype=np.float32)
    if M.ndim != 2 or len(M) != len(X):
        raise ValueError("motion has one row per tick, like content")
    sd = M.std(axis=0, keepdims=True)
    M = (M - M.mean(axis=0, keepdims=True)) / np.maximum(sd, 1e-6)
    M = M * (motion_weight / max(1.0, np.sqrt(M.shape[1])))
    # Standardising makes every video's motion look equally eventful. The
    # spread it had before says how eventful it was, so a still video's motion
    # columns are scaled back down to nothing.
    activity = float(np.clip(sd.mean() / 0.05, 0.0, 1.0))
    return np.hstack([X, M * activity]).astype(np.float32)


# ─────────────────────────── the tree ───────────────────────────

def boundary_strengths(X, cuts=(), cut_boost: float = CUT_BOOST) -> np.ndarray:
    """
    The strength of each of the n-1 boundaries between n ticks.

    `cuts` are boundary indices where a cut detector found a hard cut: boundary
    i sits between tick i and tick i+1.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        return np.zeros(0, dtype=np.float32)

    # Segments are a linked list over their first tick. Each holds its size
    # and the sum of its rows, which is all Ward's cost needs.
    size = np.ones(n)
    total = X.copy()
    left = np.arange(-1, n - 1)
    right = np.arange(1, n + 1)
    alive = np.ones(n, dtype=bool)
    stamp = np.zeros(n, dtype=np.int64)        # bumped whenever a segment changes

    def cost(a: int, b: int) -> float:
        d = total[a] / size[a] - total[b] / size[b]
        return float(size[a] * size[b] / (size[a] + size[b]) * d.dot(d))

    heap = [(cost(i, i + 1), i, i + 1, 0, 0) for i in range(n - 1)]
    heapq.heapify(heap)
    strength = np.zeros(n - 1, dtype=np.float64)
    last_tick = np.arange(n)                   # last tick of the segment starting at i
    high = 0.0
    while heap:
        c, a, b, sa, sb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or stamp[a] != sa or stamp[b] != sb or right[a] != b:
            continue
        # Ward's cost can dip after a merge. Carrying the running maximum keeps
        # the strengths in the order the merges happened, so a boundary that
        # outlived another is never weaker than it.
        high = max(high, c)
        strength[last_tick[a]] = high
        size[a] += size[b]
        total[a] += total[b]
        last_tick[a] = last_tick[b]
        alive[b] = False
        right[a] = right[b]
        if right[a] < n:
            left[right[a]] = a
        stamp[a] += 1
        if left[a] >= 0:
            p = left[a]
            heapq.heappush(heap, (cost(p, a), p, a, stamp[p], stamp[a]))
        if right[a] < n:
            q = right[a]
            heapq.heappush(heap, (cost(a, q), a, q, stamp[a], stamp[q]))

    strength /= n
    for c in cuts:
        if 0 <= int(c) < n - 1:
            strength[int(c)] *= cut_boost
    return strength.astype(np.float32)


# ─────────────────────────── sections ───────────────────────────

def _absorb_short(bounds: list[int], strength: np.ndarray, n: int, min_ticks: int,
                  keep: frozenset) -> list[int]:
    """Drop the weaker side of any section shorter than min_ticks, never a boundary in `keep`."""
    bounds = sorted(bounds)
    while True:
        edges = [-1] + bounds + [n - 1]            # a section runs from edge+1 to the next edge
        short = [(edges[k + 1] - edges[k], k) for k in range(len(edges) - 1)
                 if edges[k + 1] - edges[k] < min_ticks]
        if not short:
            return bounds
        removed = False
        for _, k in sorted(short):
            sides = [b for b in (edges[k], edges[k + 1])
                     if b in bounds and b not in keep]
            if sides:
                bounds.remove(min(sides, key=lambda b: strength[b]))
                removed = True
                break
        if not removed:
            return bounds


def section_ticks(X, *, cuts=(), levels=LEVELS, floor: float = FLOOR,
                  min_seconds: float = MIN_SECONDS, tick_seconds: float = TICK_SECONDS,
                  duration: float | None = None, strengths=None) -> dict:
    """
    Sections of one video at every level, nested, plus the whole file.

    Returns {"strengths": array, "levels": {name: [Section, ...]}, "file": Section}.
    A finer level always keeps every boundary of the coarser one, so each
    section has exactly one parent. `strengths` can be passed back in to
    re-section stored ticks at new sensitivities without rebuilding the tree.
    """
    n = len(X)
    end = float(duration) if duration is not None else n * tick_seconds
    s = np.asarray(strengths, dtype=np.float32) if strengths is not None \
        else boundary_strengths(X, cuts)
    whole = Section("file", 0, 0, n, 0.0, end, None)
    out = {"strengths": s, "levels": {}, "file": whole}
    if n == 0:
        return out
    top = float(s.max()) if len(s) else 0.0
    min_ticks = max(1, int(round(min_seconds / tick_seconds)))

    def time_of(i: int) -> float:
        return end if i >= n else i * tick_seconds

    kept: list[int] = []
    coarser: list[Section] = [whole]
    for name, share in levels:
        threshold = max(share * top, floor)
        bounds = sorted(set(kept) | {int(i) for i in np.flatnonzero(s >= threshold)})
        bounds = _absorb_short(bounds, s, n, min_ticks, frozenset(kept))
        edges = [0] + [b + 1 for b in bounds] + [n]
        made = []
        for k in range(len(edges) - 1):
            i0, i1 = edges[k], edges[k + 1]
            parent = next(p.index for p in coarser if p.i0 <= i0 and i1 <= p.i1)
            made.append(Section(name, k, i0, i1, time_of(i0), time_of(i1),
                                None if coarser[0].level == "file" else parent))
        out["levels"][name] = made
        kept, coarser = bounds, made
    return out


def all_sections(result: dict) -> list[Section]:
    """Every section to embed: the file, then each level coarse to fine."""
    out = [result["file"]]
    for made in result["levels"].values():
        out.extend(made)
    return out


def pooled(X, sections) -> np.ndarray:
    """The mean tick vector of each section, unit length: its content vector in the ticks' space."""
    X = np.asarray(X, dtype=np.float32)
    rows = [X[s.i0:s.i1].mean(axis=0) if s.i1 > s.i0 else np.zeros(X.shape[1], np.float32)
            for s in sections]
    return unit_rows(np.stack(rows)) if rows else np.zeros((0, X.shape[1]), np.float32)


def frame_times(section: Section, per_second: float = 4.0, cap: int = 32, least: int = 4) -> list[float]:
    """
    The moments of a section to hand an encoder: spread evenly, four a second
    for a short section, never more than `cap`, the model's frame budget.
    """
    span = max(section.seconds, 1e-6)
    n = int(np.clip(round(span * per_second), least, cap))
    step = span / n
    return [float(section.t0 + (k + 0.5) * step) for k in range(n)]


# ─────────────────────────── searching across levels ───────────────────────────

def best_nested(sections, scores) -> list[int]:
    """
    Auto granularity: among sections that contain one another, keep the best.

    `sections` are from one video, any levels mixed; `scores` are theirs. A
    section is kept when no section it overlaps scores higher, so a ten-second
    shot of water comes back once, at the length that matches best, and its
    parts and its parent stay out of the results. Returns indices into
    `sections`, best first.
    """
    order = sorted(range(len(sections)), key=lambda i: -float(scores[i]))
    kept: list[int] = []
    for i in order:
        a = sections[i]
        if all(a.i1 <= sections[j].i0 or sections[j].i1 <= a.i0 for j in kept):
            kept.append(i)
    return kept
