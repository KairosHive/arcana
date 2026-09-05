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


def test_estimates_add_preprocessing_to_model_time(tmp):
    """
    Preprocessing is ADDED to model time, not hidden behind it.

    Two earlier versions got this wrong in opposite directions. The first
    divided decode by core count as though it were parallel, and quoted 79
    seconds for a job that took eight minutes. The second kept a single 143 ms
    constant after cvio.imread_for_encoder made decoding ~8x cheaper, which
    made every GPU estimate about eight times too pessimistic. Both bounds are
    pinned here.
    """
    b, h = M.get(VIT_B), M.get(VIT_H)
    fixed = M.DECODE_MS_DEFAULT + M.PREPROCESS_MS

    sec_b = M.estimate_seconds(b, 10_000, has_gpu=False)
    sec_h = M.estimate_seconds(h, 10_000, has_gpu=False)
    assert sec_b == (fixed + b.cpu_ms) * 10
    assert sec_h == (fixed + h.cpu_ms) * 10

    # On the CPU the model dominates, so the two encoders are far apart.
    assert sec_h > sec_b * 5, (sec_b, sec_h)

    gpu_b = M.estimate_seconds(b, 10_000, has_gpu=True)
    gpu_h = M.estimate_seconds(h, 10_000, has_gpu=True)
    assert gpu_b < sec_b and gpu_h < sec_h

    # On a GPU preprocessing is what is left, so the heavy model is no longer
    # an order of magnitude worse -- but it is not free either. Measured:
    # 18.3 ms/image for B/32 against 25.2 for H/14.
    assert 1.1 < gpu_h / gpu_b < 2.0, (gpu_b, gpu_h)


def test_a_measured_decode_cost_changes_the_answer(tmp):
    """
    A folder of slow-to-decode files must produce a slower estimate.

    decode_ms is now the decode term only -- reduced-scale decoding put the
    default at 4.5 ms, so a folder that really costs 30 ms per file should push
    the estimate up, where the old 143 ms default meant every measurement
    pushed it down.
    """
    b = M.get(VIT_B)
    default = M.estimate_seconds(b, 1_000, has_gpu=True)
    slow_folder = M.estimate_seconds(b, 1_000, has_gpu=True, decode_ms=30.0)
    fast_folder = M.estimate_seconds(b, 1_000, has_gpu=True, decode_ms=1.0)
    assert slow_folder > default > fast_folder, (fast_folder, default, slow_folder)
    # and it must be a material difference, not a rounding one
    assert slow_folder > default * 1.5, (default, slow_folder)


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



# ─────────────── datasets panel: answering step 2 from the folder ───────────
def test_suggest_name_slugifies_the_folder(tmp):
    from arcana.ui_datasets import suggest_name
    assert suggest_name(r"C:\Users\me\Pictures\Holiday Photos 2026") == "holiday-photos-2026"
    assert suggest_name("/home/a/japan") == "japan"
    assert suggest_name("/home/a/my_pics") == "my_pics"
    # trailing separator must not produce an empty name
    assert suggest_name("/home/a/trip/") == "trip"


def test_suggest_name_collapses_punctuation_runs(tmp):
    from arcana.ui_datasets import suggest_name
    assert suggest_name("/x/a  b -- c") == "a-b-c"


def test_next_name_fills_an_empty_field(tmp):
    from arcana.ui_datasets import next_name
    assert next_name("/x/japan", "", None) == "japan"
    assert next_name("/x/japan", None, None) == "japan"


def test_next_name_replaces_its_own_previous_guess(tmp):
    # Pointing at a second folder must not leave the first folder's name behind:
    # a stale name looks deliberate, which is worse than an empty box.
    from arcana.ui_datasets import next_name
    assert next_name("/x/japan", "audio_dataset", "audio_dataset") == "japan"


def test_next_name_never_overwrites_what_the_user_typed(tmp):
    from arcana.ui_datasets import next_name
    assert next_name("/x/japan", "my careful name", "audio_dataset") is None


def test_next_name_declines_when_the_folder_yields_nothing(tmp):
    from arcana.ui_datasets import next_name
    assert next_name("/", "", None) is None


def test_scan_both_counts_each_kind_in_one_walk(tmp):
    import os
    from arcana.ui_datasets import scan_both
    os.makedirs(os.path.join(tmp, "sub"), exist_ok=True)
    for i in range(3):
        open(os.path.join(tmp, f"a{i}.jpg"), "wb").close()
    for i in range(2):
        open(os.path.join(tmp, "sub", f"b{i}.wav"), "wb").close()
    open(os.path.join(tmp, "notes.txt"), "wb").close()

    got = scan_both(tmp)
    assert got["image"] == 3, got
    assert got["audio"] == 2, got
    assert len(got["sample_image"]) == 3
    assert len(got["sample_audio"]) == 2


def test_scan_both_recurses_and_ignores_other_files(tmp):
    import os
    from arcana.ui_datasets import scan_both
    deep = os.path.join(tmp, "a", "b", "c")
    os.makedirs(deep, exist_ok=True)
    open(os.path.join(deep, "deep.png"), "wb").close()
    open(os.path.join(tmp, "readme.md"), "wb").close()
    got = scan_both(tmp)
    assert got["image"] == 1 and got["audio"] == 0, got


def test_text_encoder_is_chosen_by_index_width(tmp):
    """
    A prompt must be encoded by the model that encoded the pictures.

    search() used to always load ViT-H/14 while an index built with ViT-B/32
    holds 512-d vectors, so searching such a dataset died inside usearch with
    "The number of vector dimensions doesn't match!" and the results panel just
    stayed empty. ViT-B/32 is what the panel recommends to anyone without a
    GPU, so prompt search was broken for the people most likely to pick it.
    """
    from arcana.arcana import text_model_for_dim
    from arcana import models as _models

    for m in _models.MODELS:
        if m.modality == "image":
            assert text_model_for_dim(m.dim) == m.id, m.id


def test_image_encoder_dimensions_stay_distinct(tmp):
    # text_model_for_dim resolves by width, which only works while no two image
    # encoders share one. If a new encoder collides, that lookup silently picks
    # whichever came first in the catalogue.
    from arcana import models as _models
    dims = [m.dim for m in _models.MODELS if m.modality == "image"]
    assert len(dims) == len(set(dims)), dims


def test_unknown_vector_width_is_refused_with_advice(tmp):
    from arcana.arcana import text_model_for_dim
    try:
        text_model_for_dim(999)
    except RuntimeError as e:
        assert "re-index" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected a RuntimeError for an unknown width")


# ─────────────── labelling: a vocabulary from the folder tree ───────────────
def test_folder_vocabulary_uses_subfolder_names(tmp):
    """
    Cluster names are chosen from a fixed 100-word list that knows nothing
    about the pictures -- a library of circuit boards gets named out of a
    vocabulary with no word for circuit board. Most people have already written
    a better vocabulary without realising: their folder tree.
    """
    import os
    from arcana.ui_datasets import folder_vocabulary
    for sub in ("Street Photography", "portraits", "night_shots"):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "a.jpg"), "wb").close()
    # Order is irrelevant -- this is a bag of words to match cluster centroids
    # against -- and os.listdir sorts case-sensitively, so compare as a set.
    got = folder_vocabulary(tmp)
    assert set(got) == {"night shots", "portraits", "Street Photography"}, got


def test_folder_vocabulary_skips_folders_without_media(tmp):
    # An "exports" or "originals" wrapper is not a label.
    import os
    from arcana.ui_datasets import folder_vocabulary
    os.makedirs(os.path.join(tmp, "real"), exist_ok=True)
    open(os.path.join(tmp, "real", "a.jpg"), "wb").close()
    os.makedirs(os.path.join(tmp, "empty"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "docs"), exist_ok=True)
    open(os.path.join(tmp, "docs", "notes.txt"), "wb").close()
    assert folder_vocabulary(tmp) == ["real"]


def test_folder_vocabulary_finds_media_nested_deeply(tmp):
    import os
    from arcana.ui_datasets import folder_vocabulary
    deep = os.path.join(tmp, "trip", "day one", "raw")
    os.makedirs(deep, exist_ok=True)
    open(os.path.join(deep, "a.wav"), "wb").close()
    assert folder_vocabulary(tmp) == ["trip"]


def test_folder_vocabulary_on_a_flat_folder_is_empty(tmp):
    # The caller falls back to the built-in list and says so.
    import os
    from arcana.ui_datasets import folder_vocabulary
    open(os.path.join(tmp, "a.jpg"), "wb").close()
    assert folder_vocabulary(tmp) == []


def test_folder_vocabulary_is_bounded(tmp):
    import os
    from arcana.ui_datasets import folder_vocabulary
    for i in range(80):
        d = os.path.join(tmp, f"folder{i:03d}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "a.jpg"), "wb").close()
    assert len(folder_vocabulary(tmp, limit=60)) == 60


def test_folder_vocabulary_handles_a_missing_folder(tmp):
    import os
    from arcana.ui_datasets import folder_vocabulary
    assert folder_vocabulary(os.path.join(tmp, "nope")) == []
    assert folder_vocabulary("") == []

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


# ------------------------------------------------------------------------------------
# A half-downloaded model must not report itself ready
# ------------------------------------------------------------------------------------
def _safetensors_bytes(n_data: int = 1000) -> bytes:
    """A minimal, well-formed .safetensors payload."""
    import json, struct
    header = {"w": {"dtype": "F32", "shape": [n_data // 4],
                    "data_offsets": [0, n_data]}}
    hb = json.dumps(header).encode()
    return struct.pack("<Q", len(hb)) + hb + b"\x00" * n_data


def test_truncated_safetensors_is_detected(tmp):
    """
    A download cut off part-way leaves a file that exists but is short. The
    safetensors header states exactly how many bytes of tensor data must
    follow, so this is detectable from the file alone -- no network, no
    checksum, and without reading four gigabytes of weights.
    """
    import os
    from arcana import models as M

    whole = _safetensors_bytes()
    good = os.path.join(tmp, "good.safetensors")
    with open(good, "wb") as fh:
        fh.write(whole)
    assert M._safetensors_missing_bytes(good) == 0

    cut = os.path.join(tmp, "cut.safetensors")
    with open(cut, "wb") as fh:
        fh.write(whole[:-400])
    assert M._safetensors_missing_bytes(cut) == 400

    stub = os.path.join(tmp, "stub.safetensors")
    with open(stub, "wb") as fh:
        fh.write(whole[:10])
    assert M._safetensors_missing_bytes(stub) > 0


def test_weights_without_config_is_not_downloaded(monkeypatch):
    """
    The old check asked for ONE weight file and returned True if it existed.

    An interrupted download -- closing the app, a reboot, restarting to run as
    administrator -- can leave the weights present and the tokenizer or config
    absent. The app then reported the encoder ready and failed later, somewhere
    less obvious than the download it actually needed to finish.
    """
    from arcana import models as M
    import huggingface_hub as hh

    def only_weights(repo_id, filename, **kw):
        return "/fake/model.safetensors" if filename.endswith(
            (".safetensors", ".bin")) else None

    monkeypatch.setattr(hh, "try_to_load_from_cache", only_weights)
    monkeypatch.setattr(M.os.path, "exists", lambda p: True)
    monkeypatch.setattr(M, "_VERIFY_CACHE", {})

    ok, why = M.verify_model("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    assert ok is False
    assert "config.json" in why
    assert M.is_downloaded("laion/CLIP-ViT-H-14-laion2B-s32B-b79K") is False


def test_missing_weights_reports_plainly(monkeypatch):
    """Nothing downloaded at all is still reported as nothing downloaded."""
    from arcana import models as M
    import huggingface_hub as hh

    monkeypatch.setattr(hh, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(M, "_VERIFY_CACHE", {})
    ok, why = M.verify_model("laion/clap-htsat-fused")
    assert ok is False and "weights" in why
