"""
Tests for the encoder catalogue, the encoder cache, label caches, and the
environment check.

Run with:  python tests/test_models.py
"""

import importlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import models as M  # noqa: E402
from arcana import envcheck  # noqa: E402

VIT_B = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
VIT_L = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
VIT_H = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


# ─────────────────────────── catalogue ───────────────────────────
def test_every_model_is_self_consistent(tmp):
    assert M.MODELS, "the catalogue must not be empty"
    for m in M.MODELS:
        assert m.id and m.label and m.blurb
        assert m.dim > 0 and m.download_mb > 0
        assert m.cpu_ms > 0 and m.gpu_ms > 0
        assert m.gpu_ms < m.cpu_ms, f"{m.id}: a GPU should not be slower"
        assert m.modality in (M.IMAGE, M.AUDIO)
    assert len({m.id for m in M.MODELS}) == len(M.MODELS), "duplicate model ids"


def test_the_legacy_encoders_are_in_the_catalogue(tmp):
    """Existing datasets were built with these; the app must still describe them."""
    assert M.get(M.LEGACY_IMAGE_ID) is not None
    assert M.get(M.LEGACY_AUDIO_ID) is not None
    assert M.get(M.LEGACY_IMAGE_ID).dim == 1024


def test_default_depends_on_whether_there_is_a_gpu(tmp):
    """Without a GPU the biggest model is a multi-hour job, so it must not be default."""
    no_gpu = M.default_for(M.IMAGE, has_gpu=False)
    with_gpu = M.default_for(M.IMAGE, has_gpu=True)
    assert no_gpu.id == VIT_B
    assert with_gpu.id == VIT_H
    assert M.default_for(M.AUDIO, has_gpu=False).modality == M.AUDIO


def test_estimates_include_serial_decode(tmp):
    """
    db.py decodes each file inside the batch loop, one at a time, so decode is
    ADDED to model time rather than hidden behind it.

    An earlier version divided decode by core count, as though it were
    parallel. That quoted 79 seconds for a midjourney run that actually took
    eight minutes, and 4.6 minutes for a 10k CPU job that is really closer to
    half an hour. These bounds exist so that optimism cannot creep back.
    """
    b, h = M.get(VIT_B), M.get(VIT_H)
    d = M.DECODE_MS_DEFAULT

    sec_b = M.estimate_seconds(b, 10_000, has_gpu=False)
    sec_h = M.estimate_seconds(h, 10_000, has_gpu=False)
    assert sec_b == (d + b.cpu_ms) * 10        # decode + model, per item
    assert sec_h == (d + h.cpu_ms) * 10
    assert sec_b > 1_200, "decode must not be divided away"

    # A GPU makes the model nearly free, but decode still has to happen, so the
    # two models converge rather than the job becoming instant.
    gpu_b = M.estimate_seconds(b, 10_000, has_gpu=True)
    gpu_h = M.estimate_seconds(h, 10_000, has_gpu=True)
    assert gpu_b < sec_b and gpu_h < sec_h
    assert abs(gpu_b - gpu_h) / gpu_b < 0.10, "on a GPU, decode should dominate"


def test_a_measured_decode_cost_beats_the_default(tmp):
    """Midjourney PNGs decode in ~31 ms, not the 143 ms of a 24 MP JPEG."""
    b = M.get(VIT_B)
    slow = M.estimate_seconds(b, 1_000, has_gpu=True)
    fast = M.estimate_seconds(b, 1_000, has_gpu=True, decode_ms=31.0)
    assert fast < slow / 3, "a measured cost should change the answer materially"


def test_humanize_reads_like_a_person_wrote_it(tmp):
    assert "seconds" in M.humanize(30)
    assert M.humanize(600).startswith("about") and "minutes" in M.humanize(600)
    assert "hours" in M.humanize(3600 * 5)
    assert "days" in M.humanize(3600 * 50)


def test_catalogue_reports_download_state_and_an_estimate(tmp):
    rows = M.catalogue(M.IMAGE, has_gpu=False, n_items=10_000)
    assert len(rows) == len(M.for_modality(M.IMAGE))
    for r in rows:
        assert "downloaded" in r and isinstance(r["downloaded"], bool)
        assert r["estimate"], "an estimate should be offered when a count is known"
    assert all(not r["estimate"] for r in M.catalogue(M.IMAGE, n_items=0))


# ─────────────────────────── the encoder cache ───────────────────────────
def test_load_clip_cache_is_keyed_by_model_id(tmp):
    """
    The cache used to hold one model regardless of which was asked for, so
    indexing a second dataset with a different encoder silently reused the
    first -- while recording the second in the bundle.

    Tested without loading weights and without stubbing transformers (its lazy
    module machinery defeats attribute patching): a sentinel is planted in the
    cache, and the question is simply whether load_clip hands it back.
    """
    from arcana import db

    sentinel_model = object()
    sentinel_proc = object()
    saved = dict(db._CLIP)
    try:
        db._CLIP.update(model=sentinel_model, proc=sentinel_proc, id=VIT_B)

        # Same id -> the cache is used, nothing is loaded.
        m, p = db.load_clip(device="cpu", model_id=VIT_B)
        assert m is sentinel_model and p is sentinel_proc

        # Default id resolves to the module default, which is ViT-H here, so it
        # must NOT be served from a ViT-B cache either.
        assert db.CLIP_MODEL_ID != VIT_B, "this test assumes the default is not ViT-B"

        # A different id must reject the cache. Proven by asking for a model
        # that cannot possibly load: if the cache were consulted we would get
        # the sentinel back instead of an error.
        try:
            got, _ = db.load_clip(device="cpu",
                                  model_id="arcana-test/definitely-not-a-real-model")
        except Exception:
            pass                      # tried to load -> cache correctly rejected
        else:
            raise AssertionError(
                f"a different model id returned the cached model ({got!r}); "
                "the cache is not keyed by id")

        # The sentinel is still there for its own id.
        m2, _ = db.load_clip(device="cpu", model_id=VIT_B)
        assert m2 is sentinel_model
    finally:
        db._CLIP.clear()
        db._CLIP.update(saved)


def test_load_clip_default_id_is_recorded_in_the_cache(tmp):
    """A cache entry with no id would compare unequal forever and reload endlessly."""
    from arcana import db
    saved = dict(db._CLIP)
    try:
        db._CLIP.update(model=object(), proc=object(), id=db.CLIP_MODEL_ID)
        before = db._CLIP["model"]
        m, _ = db.load_clip(device="cpu")          # no model_id -> the default
        assert m is before, "the default id must match a cache entry stored under it"
    finally:
        db._CLIP.clear()
        db._CLIP.update(saved)


# ─────────────────────────── label caches ───────────────────────────
def _fresh_db(tmp):
    os.environ["ARCANA_DATA_DIR"] = tmp
    from arcana import paths as P
    importlib.reload(P)
    from arcana import db as D
    importlib.reload(D)
    return D


def test_label_cache_status_covers_every_image_model(tmp):
    db = _fresh_db(tmp)
    try:
        rows = db.label_cache_status("image")
        assert {r["model_id"] for r in rows} == {m.id for m in M.for_modality(M.IMAGE)}
        for r in rows:
            assert r["n_labels"] > 0, "the shipped label list should not be empty"
            assert r["ready"] is False, "a fresh data dir has no caches yet"
            assert r["dim"] == M.get(r["model_id"]).dim
    finally:
        os.environ.pop("ARCANA_DATA_DIR", None)


def test_each_cache_path_is_distinct_per_model(tmp):
    """One shared path would hand a 512-d matrix to a 1024-d model."""
    db = _fresh_db(tmp)
    try:
        paths = [r["path"] for r in db.label_cache_status("image")]
        assert len(set(paths)) == len(paths), "label caches must not collide"
    finally:
        os.environ.pop("ARCANA_DATA_DIR", None)


def test_warming_produces_a_matrix_of_the_models_dimension(tmp):
    """
    Cluster naming dots label embeddings with image centroids, so a mismatch
    is a hard failure late in a long build. Uses the smallest model only.
    """
    db = _fresh_db(tmp)
    try:
        r = db.warm_label_cache(VIT_B, "image")
        assert r["dim"] == M.get(VIT_B).dim == 512
        assert r["n_labels"] > 0
        status = {s["model_id"]: s for s in db.label_cache_status("image")}
        assert status[VIT_B]["ready"] is True
        assert status[VIT_H]["ready"] is False, "warming one must not claim the others"
    finally:
        os.environ.pop("ARCANA_DATA_DIR", None)


def test_warm_all_skips_models_that_are_not_downloaded(tmp):
    """It must never silently pull gigabytes."""
    db = _fresh_db(tmp)
    try:
        real = M.is_downloaded
        M.is_downloaded = lambda mid: False
        try:
            done = db.warm_all_label_caches("image", only_downloaded=True)
            assert done == [], "nothing local means nothing to do"
        finally:
            M.is_downloaded = real
    finally:
        os.environ.pop("ARCANA_DATA_DIR", None)


# ─────────────────────────── environment ───────────────────────────
def test_the_installed_environment_matches_the_lock(tmp):
    """
    Drift here is not cosmetic: transformers 5 changed
    CLIPModel.get_image_features to return a BaseModelOutputWithPooling
    instead of a tensor, which breaks indexing at the first image.
    """
    bad = envcheck.drift()
    assert bad == [], "environment has drifted:\n" + envcheck.report()


def test_envcheck_reports_a_planted_mismatch(tmp):
    lock = os.path.join(tmp, "requirements-lock.txt")
    with open(lock, "w", encoding="utf-8") as f:
        f.write("numpy==0.0.1\n")
        f.write("definitely-not-installed==1.2.3\n")
    bad = dict((n, (w, g)) for n, w, g in envcheck.drift(lock))
    assert "numpy" in bad and bad["numpy"][0] == "0.0.1"
    assert bad["definitely-not-installed"][1] == "MISSING"


def test_envcheck_ignores_the_cuda_local_tag(tmp):
    """torch==2.9.1 is satisfied by 2.9.1+cu128; that is not drift."""
    import importlib.metadata as md
    try:
        got = md.version("torch")
    except Exception:
        return
    lock = os.path.join(tmp, "requirements-lock.txt")
    with open(lock, "w", encoding="utf-8") as f:
        f.write(f"torch=={got.split('+')[0]}\n")
    assert envcheck.drift(lock) == []


def test_envcheck_honours_environment_markers(tmp):
    """A pin for another Python version must not be reported as missing."""
    lock = os.path.join(tmp, "requirements-lock.txt")
    with open(lock, "w", encoding="utf-8") as f:
        f.write("nonexistent-pkg==9.9.9 ; python_version < '3.0'\n")
    assert envcheck.drift(lock) == []


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_models_")
        try:
            fn(tmp)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            os.environ.pop("ARCANA_DATA_DIR", None)
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        import traceback
        for name, e in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(e), e, e.__traceback__)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
