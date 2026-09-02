"""
End-to-end tests for `arcana-build-latent` (db.main()).

These exist because the whole test suite previously stopped at the seams: every
component was tested in isolation, and nothing ran main() itself. That let a
plain UnboundLocalError ship -- `feature_paths` was assigned only inside the
`--features palette|style` branch but read unconditionally when writing the
portable bundle, so on the DEFAULT invocation the bundle silently never appeared.

main() is driven with a pre-built index (--reuse_index) and a stubbed label
encoder, so no CLIP/CLAP weights are needed and these run in seconds.

Run with:  python tests/test_db_main.py
"""

import importlib
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM = 1024          # must match db.CLIP_MODEL_ID's dimension
N = 40              # enough for t-SNE's perplexity floor


def _fresh_db(tmp):
    """Import arcana.db with ARCANA_DATA_DIR pointed at a scratch directory."""
    os.environ["ARCANA_DATA_DIR"] = tmp
    from arcana import paths as _paths
    importlib.reload(_paths)
    from arcana import db as _db
    importlib.reload(_db)
    return _db


def _seed(db, tmp, modality="image", n=N):
    """Create media plus the index pickle main() will reuse."""
    from usearch.index import Index
    ext = ".jpg" if modality == "image" else ".wav"
    media = os.path.join(tmp, "media")
    os.makedirs(media, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(media, f"item_{i:04d}{ext}")
        with open(p, "wb") as f:
            f.write(b"DATA" + bytes([i % 251]) * (300 + i))
        paths.append(p)

    dim = DIM if modality == "image" else 512
    index = Index(ndim=dim, metric="cos", dtype="f32")
    idx2path = {}
    rng = np.random.default_rng(0)
    for i, p in enumerate(paths):
        index.add(i, rng.standard_normal(dim).astype(np.float32))
        idx2path[i] = os.path.abspath(p)

    os.makedirs(db.db_dir, exist_ok=True)
    os.makedirs(db.latents_dir, exist_ok=True)
    with open(os.path.join(db.db_dir, f"index_synth_{modality}.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)
    return media, paths


def _run_main(db, argv):
    """Call main() with argv, with the label encoder stubbed out."""
    real_encode = db._encode_label_matrix
    real_argv = sys.argv
    db._encode_label_matrix = lambda *a, **k: ([], np.zeros((0, 1), dtype=np.float32))
    sys.argv = ["arcana-build-latent"] + argv
    try:
        db.main()
    finally:
        db._encode_label_matrix = real_encode
        sys.argv = real_argv


def _cleanup():
    os.environ.pop("ARCANA_DATA_DIR", None)
    from arcana import paths as _paths
    importlib.reload(_paths)


# ─────────────────────────── the regression ───────────────────────────
def test_default_build_writes_a_portable_bundle(tmp):
    """
    The default invocation -- no --features -- must still produce the .arcana
    bundle. This is the exact case the UnboundLocalError broke.
    """
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp)
        _run_main(db, ["--imgs_path", media, "--name", "synth", "--reuse_index", "--k", "3"])

        from arcana.bundle import Bundle, SUFFIX
        out = os.path.join(db._paths.subdir("bundles"), "synth_image" + SUFFIX)
        assert os.path.exists(out), "default build produced no bundle"
        with Bundle.open(out) as b:
            assert len(b) == N
            assert b.verify() == []
            assert b.model.dim == DIM
            assert b.layout is not None and b.layout.shape == (N, 2)
            assert not any(os.path.isabs(i.rel_path) for i in b.items)
    finally:
        _cleanup()


def test_audio_build_writes_a_portable_bundle(tmp):
    """Audio never enters the feature branch either, so it hit the same bug."""
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp, modality="audio")
        _run_main(db, ["--imgs_path", media, "--name", "synth", "--modality", "audio",
                       "--reuse_index", "--k", "3"])

        from arcana.bundle import Bundle, SUFFIX
        out = os.path.join(db._paths.subdir("bundles"), "synth_audio" + SUFFIX)
        assert os.path.exists(out), "audio build produced no bundle"
        with Bundle.open(out) as b:
            assert b.model.modality == "audio"
            assert b.model.dim == 512
            assert len(b) == N
    finally:
        _cleanup()


def test_bundle_records_the_model_that_built_it(tmp):
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp)
        _run_main(db, ["--imgs_path", media, "--name", "synth", "--reuse_index", "--k", "3"])
        from arcana.bundle import Bundle, ModelSpec, ModelMismatch, SUFFIX
        out = os.path.join(db._paths.subdir("bundles"), "synth_image" + SUFFIX)
        with Bundle.open(out) as b:
            assert b.model.id == db.CLIP_MODEL_ID
            b.require_model(ModelSpec(id=db.CLIP_MODEL_ID, dim=DIM, modality="image"))
            try:
                b.require_model(ModelSpec(id="other/model", dim=DIM, modality="image"))
            except ModelMismatch:
                pass
            else:
                raise AssertionError("a mismatched encoder should have been refused")
    finally:
        _cleanup()


def test_legacy_pickles_are_still_written(tmp):
    """The bundle is additive; the .pkl files other code still reads must remain."""
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp)
        _run_main(db, ["--imgs_path", media, "--name", "synth", "--reuse_index", "--k", "3"])
        assert os.path.exists(os.path.join(db.db_dir, "index_synth_image.pkl"))
        assert os.path.exists(os.path.join(db.latents_dir, "latent_space_synth_image_2d.pkl"))
    finally:
        _cleanup()


def test_a_bug_in_the_bundle_step_is_not_downgraded_to_a_warning(tmp):
    """
    The original failure hid behind `except Exception`. Programming errors must
    now propagate; only genuine I/O-ish failures may be warned about.
    """
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp)
        real = db.write_bundle
        db.write_bundle = lambda *a, **k: (_ for _ in ()).throw(TypeError("simulated bug"))
        try:
            _run_main(db, ["--imgs_path", media, "--name", "synth", "--reuse_index", "--k", "3"])
        except TypeError as e:
            assert "simulated bug" in str(e)
        else:
            raise AssertionError("a TypeError in write_bundle must not be swallowed")
        finally:
            db.write_bundle = real
    finally:
        _cleanup()


def test_an_io_failure_in_the_bundle_step_still_only_warns(tmp):
    """...but a disk problem must not throw away a completed index."""
    db = _fresh_db(tmp)
    try:
        media, _ = _seed(db, tmp)
        real = db.write_bundle
        db.write_bundle = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            _run_main(db, ["--imgs_path", media, "--name", "synth", "--reuse_index", "--k", "3"])
        finally:
            db.write_bundle = real
        # the legacy artefacts survived
        assert os.path.exists(os.path.join(db.latents_dir, "latent_space_synth_image_2d.pkl"))
    finally:
        _cleanup()


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_dbmain_")
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
