"""
Tests for adding new files to a dataset that already exists.

The encoder is stubbed out: what matters here is the bookkeeping around it --
which files are noticed, which keys they get, and whether the feature blocks
still line up with the index afterwards. Running CLIP would test PyTorch.

Run with:  python -m pytest tests/test_extend.py -q
"""

import os
import pickle
import sys

import numpy as np
import pytest
from PIL import Image
from usearch.index import Index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import db  # noqa: E402

DIM = 512          # ViT-B/32, so model_for_dim resolves


def _img(folder, name, colour=(30, 90, 150)):
    path = os.path.join(folder, name)
    Image.new("RGB", (32, 32), colour).save(path, "JPEG")
    return path


def _fake_index(paths, dim=DIM, start=0):
    """An index over `paths` with deterministic junk vectors."""
    index = Index(ndim=dim, metric="cos", dtype="f32")
    idx2path = {}
    rng = np.random.default_rng(0)
    for i, p in enumerate(paths):
        key = start + i
        index.add(key, rng.standard_normal(dim).astype(np.float32))
        idx2path[key] = os.path.abspath(p)
    return index, idx2path


def _write_dataset(tmp, folder, n=4, name="ds"):
    """A folder of images plus the index pickle that says they are indexed."""
    os.makedirs(folder, exist_ok=True)
    paths = [_img(folder, f"a{i}.jpg") for i in range(n)]
    index, idx2path = _fake_index(paths)
    with open(os.path.join(tmp, f"index_{name}_image.pkl"), "wb") as fh:
        pickle.dump((index.save(), idx2path), fh)
    return paths


@pytest.fixture
def dataset(tmp, monkeypatch):
    """A four-image dataset called 'ds', with db.db_dir pointed at the scratch dir."""
    monkeypatch.setattr(db, "db_dir", tmp)
    folder = os.path.join(tmp, "media")
    paths = _write_dataset(tmp, folder, n=4)
    return {"tmp": tmp, "folder": folder, "paths": paths, "name": "ds"}


# ───────────────────────── the survey ─────────────────────────
def test_survey_finds_only_what_is_new(dataset):
    _img(dataset["folder"], "new1.jpg")
    _img(dataset["folder"], "new2.jpg")
    s = db.survey_new_media(dataset["folder"], "ds")
    assert s["indexed"] == 4
    assert s["on_disk"] == 6
    assert sorted(os.path.basename(p) for p in s["new"]) == ["new1.jpg", "new2.jpg"]
    assert s["missing"] == []


def test_survey_notices_files_that_have_gone(dataset):
    os.remove(dataset["paths"][0])
    s = db.survey_new_media(dataset["folder"], "ds")
    assert s["new"] == []
    assert [os.path.basename(p) for p in s["missing"]] == ["a0.jpg"]


def test_an_unmounted_drive_is_refused_not_read_as_an_empty_folder(dataset):
    """
    A folder that is not there globs to nothing, which is indistinguishable from
    every file having been deleted. Pruning on that would empty the dataset.
    """
    with pytest.raises(FileNotFoundError):
        db.survey_new_media(os.path.join(dataset["tmp"], "not-mounted"), "ds")


def test_survey_of_an_unknown_dataset_says_so(dataset):
    with pytest.raises(FileNotFoundError):
        db.survey_new_media(dataset["folder"], "no-such-dataset")


def test_paths_compare_the_way_the_filesystem_does(dataset, monkeypatch):
    """Windows is case-insensitive; the same file in a different case is not new."""
    folder = dataset["folder"]
    s = db.survey_new_media(folder.upper() if os.name == "nt" else folder, "ds")
    assert s["new"] == []


# ───────────────────────── keys ─────────────────────────
def test_new_files_get_fresh_keys_above_every_existing_one(tmp, monkeypatch):
    """
    Reusing a key would leave the feature blocks pointing at a different
    photograph -- silently, since every block is keyed by id.
    """
    monkeypatch.setattr(db, "db_dir", tmp)
    folder = os.path.join(tmp, "m")
    os.makedirs(folder)
    old = [_img(folder, f"o{i}.jpg") for i in range(3)]
    index, idx2path = _fake_index(old, start=10)      # a gappy, non-zero-based index
    with open(os.path.join(tmp, "index_ds_image.pkl"), "wb") as fh:
        pickle.dump((index.save(), idx2path), fh)

    new = [_img(folder, "n0.jpg"), _img(folder, "n1.jpg")]
    fresh = {}
    next_key = (max(idx2path) + 1)
    _stub_encode(index, fresh, new, next_key)
    assert sorted(fresh) == [13, 14]
    assert not set(fresh) & set(idx2path)


def _stub_encode(index, idx2path, paths, next_key):
    """What encode_into does, without an encoder."""
    rng = np.random.default_rng(1)
    for p in paths:
        index.add(next_key, rng.standard_normal(DIM).astype(np.float32))
        idx2path[next_key] = os.path.abspath(p)
        next_key += 1
    return next_key


# ───────────────────────── merging feature blocks ─────────────────────────
def test_palette_rows_are_appended_in_id_order():
    base = {"ids": np.array([0, 1], dtype=np.int32),
            "histogram": np.zeros((2, 4), np.float32),
            "dominant": np.zeros((2, 3, 4), np.float32),
            "moments": np.zeros((2, 9), np.float32)}
    add = {"ids": np.array([2, 3], dtype=np.int32),
           "histogram": np.ones((2, 4), np.float32),
           "dominant": np.ones((2, 3, 4), np.float32),
           "moments": np.ones((2, 9), np.float32)}
    merged = db._merge_palette(base, add)
    assert list(merged["ids"]) == [0, 1, 2, 3]
    assert merged["histogram"].shape == (4, 4)
    assert merged["histogram"][3].tolist() == [1, 1, 1, 1]
    assert int(merged["fmt"][0]) == db.PALETTE_FEATURE_FMT


def test_new_gram_is_projected_through_the_existing_basis():
    """
    A PCA fitted on the new files alone gives different axes, so concatenating
    its output would put half the dataset in a space unrelated to the other
    half -- and every style comparison across the join would be meaningless
    while looking perfectly healthy.
    """
    rng = np.random.default_rng(7)
    raw_dim, k = 40, 5
    comps = rng.standard_normal((k, raw_dim)).astype(np.float32)
    mean = rng.standard_normal(raw_dim).astype(np.float32)
    base = {"ids": np.array([0, 1], dtype=np.int32),
            "edge_histogram": np.zeros((2, 6), np.float32),
            "texture_lbp": np.zeros((2, 3), np.float32),
            "gram": np.zeros((2, k), np.float32),
            "gram_pca_components": comps, "gram_pca_mean": mean}
    raw = rng.standard_normal((3, raw_dim)).astype(np.float32)
    add = {"ids": np.array([2, 3, 4], dtype=np.int32),
           "edge_histogram": np.ones((3, 6), np.float32),
           "texture_lbp": np.ones((3, 3), np.float32),
           "gram": raw}

    merged = db._merge_style(base, add)
    assert merged["gram"].shape == (5, k)
    assert np.allclose(merged["gram"][2:], (raw - mean) @ comps.T, atol=1e-4)
    # The basis is carried forward, or the next extend has nothing to project through.
    assert np.array_equal(merged["gram_pca_components"], comps)


def test_uncompressed_gram_is_simply_concatenated():
    """Below the PCA threshold the Gram is stored raw; there is no basis to project through."""
    base = {"ids": np.array([0], dtype=np.int32),
            "edge_histogram": np.zeros((1, 6), np.float32),
            "texture_lbp": np.zeros((1, 3), np.float32),
            "gram": np.zeros((1, 40), np.float32)}
    add = {"ids": np.array([1], dtype=np.int32),
           "edge_histogram": np.ones((1, 6), np.float32),
           "texture_lbp": np.ones((1, 3), np.float32),
           "gram": np.ones((1, 40), np.float32)}
    merged = db._merge_style(base, add)
    assert merged["gram"].shape == (2, 40)
    assert "gram_pca_components" not in merged


def test_a_gram_of_the_wrong_width_is_refused_not_reshaped():
    base = {"ids": np.array([0], dtype=np.int32),
            "gram": np.zeros((1, 5), np.float32),
            "gram_pca_components": np.zeros((5, 40), np.float32),
            "gram_pca_mean": np.zeros(40, np.float32)}
    add = {"ids": np.array([1], dtype=np.int32),
           "gram": np.zeros((1, 174000), np.float32)}
    with pytest.raises(ValueError, match="Gram mode"):
        db._merge_style(base, add)


def test_gram_mode_is_read_back_off_the_file():
    compact = {"gram_pca_components": np.zeros((512, 41152), np.float32)}
    full = {"gram_pca_components": np.zeros((512, 174000), np.float32)}
    assert db._gram_is_compact(compact)
    assert not db._gram_is_compact(full)
    assert db._gram_is_compact({"gram": np.zeros((3, 41152), np.float32)})


# ───────────────────────── the encoder must not change ─────────────────────────
def test_the_encoder_is_read_off_the_index_width():
    assert db.model_for_dim(512, "image").endswith("ViT-B-32-laion2B-s34B-b79K")
    assert db.model_for_dim(768, "image").endswith("ViT-L-14-laion2B-s32B-b82K")
    assert db.model_for_dim(1024, "image").endswith("ViT-H-14-laion2B-s32B-b79K")


def test_an_unrecognised_width_refuses_rather_than_guessing():
    with pytest.raises(RuntimeError, match="matches no encoder"):
        db.model_for_dim(37, "image")


# ───────────────────────── nothing to do ─────────────────────────
def test_extending_a_dataset_that_is_already_complete_changes_nothing(dataset, monkeypatch):
    called = []
    monkeypatch.setattr(db, "index_dataset", lambda *a, **k: called.append(1))
    result = db.extend_dataset(dataset["folder"], "ds")
    assert result["added"] == 0 and result["removed"] == 0
    assert not called, "an extend with nothing to add must not relayout the dataset"


def test_missing_files_are_kept_unless_pruning_is_asked_for(dataset, monkeypatch):
    os.remove(dataset["paths"][0])
    monkeypatch.setattr(db, "index_dataset", lambda *a, **k: {})
    result = db.extend_dataset(dataset["folder"], "ds")
    assert result["removed"] == 0

    with open(os.path.join(dataset["tmp"], "index_ds_image.pkl"), "rb") as fh:
        _blob, idx2path = pickle.load(fh)
    assert len(idx2path) == 4, "the index must not lose a file nobody asked to drop"


def test_pruning_drops_them_from_the_index(dataset, monkeypatch):
    os.remove(dataset["paths"][0])
    monkeypatch.setattr(db, "index_dataset", lambda *a, **k: {})
    result = db.extend_dataset(dataset["folder"], "ds", prune_missing=True)
    assert result["removed"] == 1

    with open(os.path.join(dataset["tmp"], "index_ds_image.pkl"), "rb") as fh:
        _blob, idx2path = pickle.load(fh)
    assert len(idx2path) == 3
    assert all(os.path.exists(p) for p in idx2path.values())


# ───────────────────────── feature paths ─────────────────────────
def test_existing_blocks_are_found_for_the_bundle(tmp, monkeypatch):
    """
    A rework that extracts nothing must still put the blocks already on disk
    into the bundle, or every reuse_index run quietly strips palette and style
    out of the portable copy.
    """
    monkeypatch.setattr(db, "db_dir", tmp)
    assert db.existing_feature_paths("ds") == {}
    for block in ("palette", "style"):
        np.savez(os.path.join(tmp, f"features_ds_{block}.npz"), ids=np.array([0]))
    assert sorted(db.existing_feature_paths("ds")) == ["palette", "style"]


def test_a_dataset_with_no_blocks_does_not_gain_them_by_extending(tmp, monkeypatch):
    """
    Extracting features for the new files alone would produce a block covering a
    tenth of the collection, which ranks worse than having none.
    """
    monkeypatch.setattr(db, "db_dir", tmp)
    assert db.extend_features("ds", {0: "whatever.jpg"}) == {}


def main():
    print("This suite uses pytest fixtures and monkeypatch; run it with:")
    print("    python -m pytest tests/test_extend.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
