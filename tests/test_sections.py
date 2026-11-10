"""
Sections: a video cut where it changes, at nested scales, from a stream of ticks.
See VIDEO_ROADMAP.md, 2.3.

Run with:  python -m pytest tests/test_sections.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import sections as S  # noqa: E402

DIM = 64


def direction(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def stream(parts, noise=0.05, seed=0):
    """Ticks for a video made of parts: [(direction or vector, seconds), ...] at two ticks a second."""
    rng = np.random.default_rng(seed)
    rows = []
    for v, seconds in parts:
        v = direction(v) if isinstance(v, int) else np.asarray(v, np.float32)
        for _ in range(int(seconds / S.TICK_SECONDS)):
            rows.append(v + noise * rng.standard_normal(DIM))
    return S.unit_rows(np.asarray(rows, np.float32))


def spans(made):
    return [(round(s.t0, 1), round(s.t1, 1)) for s in made]


def test_three_parts_come_out_as_three_coarse_sections():
    # Ten seconds of water, thirty of sky, twelve of someone walking.
    X = stream([(0, 10), (1, 30), (2, 12)])
    r = S.section_ticks(X)
    assert spans(r["levels"]["coarse"]) == [(0.0, 10.0), (10.0, 40.0), (40.0, 52.0)]
    assert r["file"].t0 == 0.0 and r["file"].t1 == 52.0 and r["file"].level == "file"


def test_a_video_that_never_changes_is_one_section_at_every_level():
    X = stream([(0, 60)])
    r = S.section_ticks(X)
    for name, _ in S.LEVELS:
        assert spans(r["levels"][name]) == [(0.0, 60.0)], name


def test_levels_are_nested_and_every_section_has_one_parent():
    # Two big parts, and inside the second a smaller change.
    a, b = direction(0), direction(1)
    b2 = S.unit_rows((b + 1.2 * direction(2))[None])[0]
    X = stream([(a, 20), (b, 16), (b2, 14)])
    r = S.section_ticks(X)
    coarse, medium, fine = (r["levels"][n] for n in ("coarse", "medium", "fine"))
    assert spans(coarse) == [(0.0, 20.0), (20.0, 50.0)]
    assert (20.0, 36.0) in spans(medium) and (36.0, 50.0) in spans(medium)
    for finer, coarser in ((medium, coarse), (fine, medium)):
        edges = {s.i0 for s in coarser} | {s.i1 for s in coarser}
        assert edges <= {s.i0 for s in finer} | {s.i1 for s in finer}
        for s in finer:
            p = coarser[s.parent]
            assert p.i0 <= s.i0 and s.i1 <= p.i1
    assert all(s.parent is None for s in coarse)


def test_a_gradual_change_gets_a_boundary_where_it_changes_most():
    # A slow tilt from water to sky over ten seconds, inside one take.
    a, b = direction(0), direction(1)
    ramp = [((1 - w) * a + w * b, 0.5) for w in np.linspace(0, 1, 20)]
    X = stream([(a, 15)] + ramp + [(b, 15)])
    coarse = S.section_ticks(X)["levels"]["coarse"]
    assert len(coarse) == 2
    assert 17.0 <= coarse[0].t1 <= 23.0


def test_a_brief_disturbance_is_absorbed():
    # One second of something else in the middle of a long shot.
    X = stream([(0, 20), (1, 1), (0, 20)])
    r = S.section_ticks(X, min_seconds=2.0)
    for name, _ in S.LEVELS:
        assert all(s.seconds >= 2.0 for s in r["levels"][name]), name


def test_a_hard_cut_raises_its_boundary():
    X = stream([(0, 10), (1, 10)], noise=0.3, seed=2)
    plain = S.boundary_strengths(X)
    boosted = S.boundary_strengths(X, cuts=[19])
    assert boosted[19] == plain[19] * S.CUT_BOOST
    assert np.allclose(np.delete(boosted, 19), np.delete(plain, 19))


def test_motion_alone_can_cut_a_video():
    # The picture never changes; the camera starts to shake at ten seconds.
    content = stream([(0, 20)], noise=0.01)
    motion = np.zeros((40, 6), np.float32)
    motion[20:] = 1.0
    motion += 0.02 * np.random.default_rng(1).standard_normal(motion.shape)
    still = S.section_ticks(S.tick_features(content))["levels"]["coarse"]
    moving = S.section_ticks(S.tick_features(content, motion))["levels"]["coarse"]
    assert len(still) == 1
    assert spans(moving) == [(0.0, 10.0), (10.0, 20.0)]
    # Motion that is only noise leaves a still video whole.
    quiet = 0.001 * np.random.default_rng(1).standard_normal((40, 6))
    assert len(S.section_ticks(S.tick_features(content, quiet))["levels"]["coarse"]) == 1


def test_stored_strengths_resection_without_the_ticks():
    X = stream([(0, 10), (1, 30), (2, 12)])
    first = S.section_ticks(X)
    again = S.section_ticks(np.zeros((len(X), 1)), strengths=first["strengths"],
                            levels=(("only", 0.5),))
    assert spans(again["levels"]["only"]) == spans(first["levels"]["coarse"])


def test_short_and_empty_videos():
    assert S.section_ticks(np.zeros((0, DIM)))["levels"] == {}
    one = S.section_ticks(stream([(0, 0.5)]))
    assert spans(one["levels"]["fine"]) == [(0.0, 0.5)]
    # The real duration closes the last section, not a whole number of ticks.
    r = S.section_ticks(stream([(0, 10)]), duration=10.3)
    assert r["file"].t1 == 10.3 and r["levels"]["coarse"][-1].t1 == 10.3


def test_pooled_vectors_follow_their_section():
    X = stream([(0, 10), (1, 30), (2, 12)])
    coarse = S.section_ticks(X)["levels"]["coarse"]
    P = S.pooled(X, coarse)
    assert [int(np.argmax(row)) for row in P] == [0, 1, 2]
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-5)


def test_frame_times_fit_the_budget():
    ten = S.Section("coarse", 0, 0, 20, 0.0, 10.0, None)
    thirty = S.Section("coarse", 1, 20, 80, 10.0, 40.0, None)
    brief = S.Section("fine", 0, 0, 1, 0.0, 0.5, None)
    assert len(S.frame_times(ten)) == 32            # four a second would be forty: capped
    assert len(S.frame_times(thirty)) == 32
    assert len(S.frame_times(brief)) == 4
    t = S.frame_times(thirty)
    assert 10.0 < t[0] < t[-1] < 40.0 and np.allclose(np.diff(t), np.diff(t)[0])


def test_auto_granularity_keeps_the_best_of_nested_sections():
    X = stream([(0, 10), (1, 30), (2, 12)])
    r = S.section_ticks(X)
    every = S.all_sections(r)
    # A query for the water: score each section by how much of it is water.
    P = S.pooled(X, every)
    scores = P @ direction(0)
    kept = S.best_nested(every, scores)
    best = every[kept[0]]
    assert (round(best.t0, 1), round(best.t1, 1)) == (0.0, 10.0)
    # Nothing kept overlaps anything else kept, and the whole file lost to its best part.
    for i in kept:
        for j in kept:
            if i != j:
                assert every[i].i1 <= every[j].i0 or every[j].i1 <= every[i].i0
    assert every[kept[0]].level != "file"
