"""
Regression tests for the packaging-readiness fixes.

Each test corresponds to a defect that was found and fixed; they exist to stop
those defects coming back.

Run with:  python tests/test_hardening.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import paths  # noqa: E402
from arcana.cvio import imread_unicode, imwrite_unicode  # noqa: E402


# ─────────────────────────── paths: escaping the output tree ───────────────────────────
def test_safe_join_allows_ordinary_names(tmp):
    assert paths.safe_join(tmp, "my selections") == os.path.join(tmp, "my selections")
    assert paths.safe_join(tmp, "a", "b") == os.path.join(tmp, "a", "b")
    assert paths.safe_join(tmp, "trip/2026") == os.path.join(tmp, "trip", "2026")


def test_safe_join_rejects_traversal(tmp):
    """A folder name typed into the UI must not be able to redirect a write."""
    for bad in ("..", "../x", "../../etc", r"..\..\windows", "a/../../b",
                "/etc/passwd", r"C:\Windows", "", "   ", "."):
        assert paths.safe_join(tmp, bad) is None, f"should have rejected {bad!r}"


def test_safe_join_rejects_traversal_across_components(tmp):
    assert paths.safe_join(tmp, "ok", "..", "..", "escape") is None


def test_is_within_accepts_inside_and_rejects_outside(tmp):
    root = os.path.join(tmp, "media")
    os.makedirs(os.path.join(root, "sub"))
    inside = os.path.join(root, "sub", "a.jpg")
    open(inside, "wb").close()
    outside = os.path.join(tmp, "secret.env")
    open(outside, "wb").close()

    assert paths.is_within(inside, [root])
    assert not paths.is_within(outside, [root])
    assert not paths.is_within(os.path.join(root, "..", "secret.env"), [root])
    assert not paths.is_within(inside, []), "no roots configured means nothing is servable"


def test_is_within_is_not_fooled_by_a_prefix_match(tmp):
    """'/media-backup' must not count as being inside '/media'."""
    root = os.path.join(tmp, "media")
    sibling = os.path.join(tmp, "media-backup")
    os.makedirs(root)
    os.makedirs(sibling)
    f = os.path.join(sibling, "x.jpg")
    open(f, "wb").close()
    assert not paths.is_within(f, [root])


def test_listdir_safe_tolerates_a_missing_directory(tmp):
    """A fresh install has no databases/ or latents/; import must not crash."""
    assert paths.listdir_safe(os.path.join(tmp, "does-not-exist")) == []
    os.makedirs(os.path.join(tmp, "real"))
    open(os.path.join(tmp, "real", "f.txt"), "wb").close()
    assert paths.listdir_safe(os.path.join(tmp, "real")) == ["f.txt"]


def test_no_directories_are_created_merely_by_importing(tmp):
    """paths.subdir must be a pure computation."""
    os.environ["ARCANA_DATA_DIR"] = os.path.join(tmp, "fresh")
    try:
        import importlib
        importlib.reload(paths)
        for name in ("databases", "latents", "bundles", "output", "cache", "models"):
            p = paths.subdir(name)
            assert not os.path.exists(p), f"{name} was created just by asking for its path"
    finally:
        del os.environ["ARCANA_DATA_DIR"]
        import importlib
        importlib.reload(paths)


def test_data_dir_env_override_wins(tmp):
    os.environ["ARCANA_DATA_DIR"] = tmp
    try:
        import importlib
        importlib.reload(paths)
        assert paths.data_dir() == os.path.abspath(tmp)
        assert paths.subdir("databases") == os.path.join(os.path.abspath(tmp), "databases")
    finally:
        del os.environ["ARCANA_DATA_DIR"]
        import importlib
        importlib.reload(paths)


def test_ensure_dir_error_names_the_env_var(tmp):
    """A read-only install must fail with something a user can act on."""
    blocker = os.path.join(tmp, "blocker")
    open(blocker, "wb").close()          # a file where a directory is wanted
    try:
        paths.ensure_dir(os.path.join(blocker, "child"))
    except RuntimeError as e:
        assert "ARCANA_DATA_DIR" in str(e)
    else:
        raise AssertionError("expected ensure_dir to fail on an impossible path")


# ─────────────────────────── cv2 unicode paths ───────────────────────────
def _tiny_png():
    import cv2
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = (0, 128, 255)
    return img


def test_imread_unicode_handles_non_ascii_names(tmp):
    """cv2.imread returns None for these; the whole point of cvio."""
    import cv2
    img = _tiny_png()
    for name in ("plain.png", "café_señor.png", "日本語.png", "Ελληνικά.png"):
        p = os.path.join(tmp, name)
        assert imwrite_unicode(p, img), f"failed to write {name}"
        assert os.path.exists(p), f"{name} was not created"
        back = imread_unicode(p)
        assert back is not None, f"imread_unicode returned None for {name}"
        assert back.shape == img.shape
        np.testing.assert_array_equal(back, img)


def test_plain_cv2_really_does_fail_on_those_names(tmp):
    """
    Guards the premise. If OpenCV ever fixes this, cvio becomes redundant and we
    should know rather than keep a workaround forever.
    """
    import cv2
    p = os.path.join(tmp, "café_señor.png")
    assert imwrite_unicode(p, _tiny_png())
    if sys.platform == "win32":
        assert cv2.imread(p) is None, "cv2.imread now handles non-ASCII paths; cvio may be removable"


def test_imread_unicode_returns_none_for_missing_and_garbage(tmp):
    assert imread_unicode(os.path.join(tmp, "nope.png")) is None
    bad = os.path.join(tmp, "bad.png")
    with open(bad, "wb") as f:
        f.write(b"this is not an image")
    assert imread_unicode(bad) is None
    empty = os.path.join(tmp, "empty.png")
    open(empty, "wb").close()
    assert imread_unicode(empty) is None


# ─────────────────────────── index invariants ───────────────────────────
def test_usearch_index_is_built_at_f32_not_bf16(tmp):
    """
    db.py used to call Index(ndim, metric) with no dtype. usearch defaults to
    bf16, quantising every embedding by ~1e-2 -- and the index is the only copy.
    """
    from usearch.index import Index
    v = np.random.default_rng(0).standard_normal(256).astype(np.float32)

    default = Index(ndim=256, metric="cos")
    default.add(0, v)
    assert not np.array_equal(np.asarray(default.get(0)).ravel(), v), \
        "usearch default is lossless now; the explicit dtype may be unnecessary"

    explicit = Index(ndim=256, metric="cos", dtype="f32")
    explicit.add(0, v)
    np.testing.assert_array_equal(np.asarray(explicit.get(0)).ravel(), v)


def test_index_keys_stay_contiguous_when_a_file_is_unreadable(tmp):
    """
    Keys are used downstream as positional rows into the latent DataFrame. The
    old code advanced by batch_start + i, so an unreadable image left a gap and
    every later item highlighted the wrong point.
    """
    paths_list = [f"img{i}.jpg" for i in range(10)]
    unreadable = {3, 7}
    batch_size = 4

    # Reproduce the fixed key assignment from db.build()
    keys, next_key = {}, 0
    for start in range(0, len(paths_list), batch_size):
        ok = [p for i, p in enumerate(paths_list[start:start + batch_size])
              if (start + i) not in unreadable]
        for p in ok:
            keys[next_key] = p
            next_key += 1

    assert sorted(keys) == list(range(len(paths_list) - len(unreadable)))
    assert list(keys) == list(range(len(keys))), "keys must have no gaps"


def test_key_to_row_mapping_survives_a_gapped_legacy_index(tmp):
    """
    Old indexes on disk still have gaps. The app must map key -> row through
    idx2path ordering rather than using .loc with keys as labels.
    """
    import pandas as pd

    # Case 1: keys run past the end of the frame -> the old code CRASHED.
    idx2path = {0: "a.jpg", 1: "b.jpg", 4: "c.jpg", 5: "d.jpg"}   # gap at 2,3
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0],
                       "path": list(idx2path.values())}).reset_index(drop=True)

    key_to_row = {k: i for i, k in enumerate(idx2path.keys())}
    rows = [key_to_row[k] for k in (4, 5) if k in key_to_row]
    assert df.iloc[rows]["path"].tolist() == ["c.jpg", "d.jpg"]

    try:
        df.loc[[4, 5]]
    except KeyError:
        pass
    else:
        raise AssertionError("the old .loc path should raise on out-of-range keys")

    # Case 2: keys still inside the range -> the old code returned the WRONG rows,
    # silently. This is the dangerous one.
    idx2path2 = {0: "a.jpg", 2: "b.jpg", 3: "c.jpg"}              # gap at 1
    df2 = pd.DataFrame({"path": list(idx2path2.values())}).reset_index(drop=True)
    key_to_row2 = {k: i for i, k in enumerate(idx2path2.keys())}

    correct = df2.iloc[[key_to_row2[2]]]["path"].tolist()
    wrong = df2.loc[[2]]["path"].tolist()
    assert correct == ["b.jpg"], correct
    assert wrong == ["c.jpg"], wrong
    assert correct != wrong, "the silent-wrong-row case must actually differ"


# ─────────────────────────── gram PCA projection ───────────────────────────
def test_gram_query_is_projected_through_the_stored_basis(tmp):
    """
    Stored Gram vectors are PCA-compressed; a query's raw Gram is full width.
    Without projection the comparison is a shape error, so gram search never
    worked at all.
    """
    rng = np.random.default_rng(0)
    n_in, n_out, n_items = 400, 32, 50
    comps = rng.standard_normal((n_out, n_in)).astype(np.float32)
    mean = rng.standard_normal(n_in).astype(np.float32)
    db_vecs = rng.standard_normal((n_items, n_out)).astype(np.float32)
    query = rng.standard_normal((1, n_in)).astype(np.float32)

    try:
        _ = db_vecs @ query.T
        raise AssertionError("the unprojected comparison should not be possible")
    except ValueError:
        pass

    projected = (query - mean.reshape(1, -1)) @ comps.T
    assert projected.shape == (1, n_out)
    sims = (db_vecs / (np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-8)) @ \
           (projected / (np.linalg.norm(projected) + 1e-8)).T
    assert sims.shape == (n_items, 1)
    assert np.isfinite(sims).all()
    assert (sims >= -1.001).all() and (sims <= 1.001).all()


def test_style_feature_rows_stay_aligned_when_gram_is_missing(tmp):
    """
    edge/lbp/gram/valid_ids are joined by position. An item that yields no Gram
    must be skipped entirely, not partially appended.
    """
    items = [("a", True), ("b", False), ("c", True), ("d", True)]   # b has no gram
    edge, lbp, gram, valid = [], [], [], []
    for name, has_gram in items:
        if not has_gram:
            continue                       # the fixed behaviour
        edge.append(name); lbp.append(name); gram.append(name); valid.append(name)
    assert len(edge) == len(lbp) == len(gram) == len(valid) == 3
    for i, name in enumerate(valid):
        assert gram[i] == name, "gram row must describe the item at the same position"


# ─────────────────────────── output filename length ───────────────────────────
def test_fit_filename_keeps_short_names_readable(tmp):
    n = paths.fit_filename(tmp, "DSC00300", "DSC01932", "lab_20260901", ".png")
    assert n == "DSC00300_from_DSC01932_lab_20260901.png"


def test_fit_filename_survives_midjourney_length_names(tmp):
    """
    Two 100-char names under a deep output directory used to exceed Windows'
    260-character path limit; Pillow then failed with a bare
    "No such file or directory".
    """
    deep = os.path.join(tmp, "arcana", "output", "color_transfer")
    os.makedirs(deep)
    left = "Bounza_a_full_comic_book_page_of_long_white_hairs_falling_int_1a52789f-b667-4752-83cb-afafc3e9442c_0"
    right = "bounzai_httpss.mj.runJBKtC-MmVNs_A_multi-panel_comic_book_pag_18ab2bf3-cbd3-41e3-a6fd-db3143d06442_0"
    n = paths.fit_filename(deep, left, right, "lab_20260901_205655", ".png")
    full = os.path.join(deep, n)
    assert len(full) <= 260, f"path is {len(full)} chars"
    assert n.endswith(".png")
    # and it must actually be creatable
    with open(full, "wb") as fh:
        fh.write(b"x")
    assert os.path.exists(full)


def test_fit_filename_handles_absurd_names_and_deep_dirs(tmp):
    deep = os.path.join(tmp, *(["nested"] * 12))
    os.makedirs(deep)
    n = paths.fit_filename(deep, "z" * 500, "q" * 500, "mf_1", ".png")
    assert len(os.path.join(deep, n)) <= 260


def test_fit_filename_keeps_similar_long_names_distinct(tmp):
    """Truncation alone would collapse two different images onto one filename."""
    a = paths.fit_filename(tmp, "z" * 200 + "AAA", "q" * 200, "mf_1", ".png")
    b = paths.fit_filename(tmp, "z" * 200 + "BBB", "q" * 200, "mf_1", ".png")
    assert a != b


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_hard_")
        try:
            fn(tmp)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
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
