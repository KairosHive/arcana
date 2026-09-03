"""
Tests for legacy -> bundle migration.

Builds a synthetic legacy dataset (usearch index pickle + pandas latent pickle +
feature npz), converts it, and checks that nothing was lost or mis-keyed.

Run with:  python tests/test_legacy.py
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana.bundle import Bundle, SUFFIX  # noqa: E402
from arcana import legacy  # noqa: E402

DIM = 1024          # must match legacy.LEGACY_MODELS["image"].dim
N = 12


def _write_legacy(tmp, *, n=N, with_features=True, missing=0, dup=False):
    """Create databases/ and latents/ exactly as the old pipeline did."""
    from usearch.index import Index

    root = os.path.join(tmp, "images", "shoot")
    os.makedirs(root, exist_ok=True)
    db = os.path.join(tmp, "databases")
    lat = os.path.join(tmp, "latents")
    os.makedirs(db, exist_ok=True)
    os.makedirs(lat, exist_ok=True)

    paths, vecs = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        p = os.path.join(root, f"IMG_{i:04d}.jpg")
        # last `missing` files are deliberately not created on disk
        if i < n - missing:
            payload = b"JPEG" + (bytes([i % 251]) * (300 + i))
            if dup and i == 1:
                payload = b"JPEG" + (bytes([0]) * 300)   # identical to i == 0
            with open(p, "wb") as f:
                f.write(payload)
        paths.append(p)
        vecs.append(rng.standard_normal(DIM).astype(np.float32))

    index = Index(ndim=DIM, metric="cos")
    idx2path = {}
    for i, (p, v) in enumerate(zip(paths, vecs)):
        index.add(i, v)
        idx2path[i] = os.path.abspath(p)
    with open(os.path.join(db, "index_synth_image.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)

    df = pd.DataFrame({
        "x": np.arange(n, dtype=np.float32),
        "y": np.arange(n, dtype=np.float32) * -1,
        "path": [os.path.abspath(p) for p in paths],
        "cluster_id": np.arange(n) % 4,
        "label": [f"cluster{i % 4}" for i in range(n)],
    })
    df.to_pickle(os.path.join(lat, "latent_space_synth_image_2d.pkl"))

    if with_features:
        # Deliberately out of order and missing one id, to exercise remapping.
        ids = np.array([i for i in range(n) if i != 3][::-1], dtype=np.int32)
        np.savez_compressed(
            os.path.join(db, "features_synth_palette.npz"),
            ids=ids,
            moments=np.stack([np.full(9, float(i), dtype=np.float32) for i in ids]),
        )
        np.savez_compressed(
            os.path.join(db, "features_synth_style.npz"),
            ids=ids,
            edge_histogram=np.stack([np.full(256, float(i), dtype=np.float32) for i in ids]),
            gram=np.stack([np.full(512, float(i), dtype=np.float32) for i in ids]),
            gram_pca_components=np.ones((512, 41152 // 64), dtype=np.float32),  # shared basis
            gram_pca_mean=np.ones(41152 // 64, dtype=np.float32),
        )
    return db, lat, root, np.stack(vecs), paths


def _convert(tmp, **kw):
    db, lat, root, vecs, paths = _write_legacy(tmp, **kw)
    found = legacy.discover(db_dir=db, latents_dir=lat)
    assert len(found) == 1, f"expected one dataset, got {found}"
    out_dir = os.path.join(tmp, "bundles")
    report = legacy.convert(found[0], out_dir, verbose=False)
    return report, root, vecs, paths


# ─────────────────────────── fidelity ───────────────────────────
def test_discover_finds_the_dataset(tmp):
    db, lat, *_ = _write_legacy(tmp)
    found = legacy.discover(db_dir=db, latents_dir=lat)
    assert found[0].name == "synth"
    assert found[0].modality == "image"
    assert 2 in found[0].latent_paths
    assert found[0].palette_path and found[0].style_path


def test_migration_is_lossless_relative_to_the_legacy_index(tmp):
    """
    Migration must not lose anything the legacy index still held.

    It cannot recover what the index itself threw away: db.py builds
    Index(ndim=..., metric="cos") with no dtype, and usearch defaults to bf16,
    so the stored vectors were already quantised before we ever saw them.
    """
    from usearch.index import Index

    db, lat, _root, orig, _paths = _write_legacy(tmp)
    found = legacy.discover(db_dir=db, latents_dir=lat)
    report = legacy.convert(found[0], os.path.join(tmp, "bundles"), verbose=False)

    with open(os.path.join(db, "index_synth_image.pkl"), "rb") as f:
        blob, idx2path = pickle.load(f)
    idx = Index.restore(blob)
    from_index = np.asarray([idx.get(k) for k in sorted(int(k) for k in idx2path)],
                            dtype=np.float32)
    if from_index.ndim == 3:
        from_index = from_index[:, 0, :]

    with Bundle.open(report["out"]) as b:
        np.testing.assert_array_equal(np.asarray(b.vectors), from_index)
        # ...and the loss against the true embeddings is bf16-sized, not larger
        assert np.abs(np.asarray(b.vectors) - orig).max() < 0.05


def test_migrated_bundles_declare_bf16_provenance(tmp):
    """A migrated bundle must not claim a precision it does not have."""
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        assert b.vector_precision == "bf16"
        assert "bf16" in repr(b)


def test_layout_labels_and_clusters_survive(tmp):
    report, root, _vecs, paths = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        assert b.layout is not None and b.layout.shape == (N, 2)
        for i, it in enumerate(b.items):
            n = int(it.name.split("_")[1].split(".")[0])
            assert b.layout[i][0] == float(n)
            assert b.layout[i][1] == -float(n)
            assert it.cluster_id == n % 4
            assert it.label == f"cluster{n % 4}"


def test_model_identity_is_recorded(tmp):
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        assert b.model.id == legacy.LEGACY_MODELS["image"].id
        assert b.model.dim == DIM
        assert b.model.modality == "image"


def test_paths_become_relative_and_portable(tmp):
    report, root, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        for it in b.items:
            assert not os.path.isabs(it.rel_path)
            assert "\\" not in it.rel_path
        # recorded root plus rel_path must reach the real file again
        found = b.resolve(b.manifest["source"]["root"])
        assert len(found) == N


# ─────────────────────────── feature re-keying ───────────────────────────
def test_feature_ids_are_remapped_to_row_order(tmp):
    """Legacy ids are usearch keys; a bundle's rows are positional."""
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        for block in ("palette", "style"):
            d = b.feature(block)
            assert d is not None, block
            ids = d["ids"]
            assert ids.min() >= 0 and ids.max() < len(b.items)
            assert len(set(ids.tolist())) == len(ids), "remapped ids must stay unique"


def test_feature_rows_stay_attached_to_the_right_item(tmp):
    """
    The synthetic features encode their legacy id as the array value, so we can
    prove row r of the feature block still describes item ids[r].
    """
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        d = b.feature("palette")
        for pos, new_row in enumerate(d["ids"]):
            legacy_id = int(d["moments"][pos][0])       # value == original legacy id
            item = b.items[int(new_row)]
            n = int(item.name.split("_")[1].split(".")[0])
            assert n == legacy_id, (
                f"feature row {pos} points at item {n} but carries data for {legacy_id}"
            )


def test_shared_pca_basis_is_not_filtered(tmp):
    """Per-item arrays get filtered; a shared basis must be copied whole."""
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        d = b.feature("style")
        assert d["gram_pca_components"].shape == (512, 41152 // 64)
        assert d["gram_pca_mean"].shape == (41152 // 64,)
        assert d["gram"].shape[0] == len(d["ids"])      # this one IS per-item


def test_feature_row_without_a_matching_item_is_dropped_and_reported(tmp):
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        d = b.feature("palette")
        assert len(d["ids"]) == N - 1, "id 3 was absent from the feature file"


# ─────────────────────────── awkward inputs ───────────────────────────
def test_missing_source_files_are_carried_not_dropped(tmp):
    report, *_ = _convert(tmp, missing=3)
    assert report["missing_files"] == 3
    with Bundle.open(report["out"]) as b:
        assert len(b) == N, "vectors must be kept even when the pixels are gone"
        assert b.verify() == []
        assert len({it.id for it in b.items}) == N, "ids must stay unique"


def test_duplicate_content_is_collapsed_with_a_warning(tmp):
    report, *_ = _convert(tmp, dup=True)
    with Bundle.open(report["out"]) as b:
        assert len(b) == N - 1
        assert b.verify() == []
        assert b.vectors.shape[0] == len(b.items), "vectors must be filtered alongside items"
    assert any("duplicate" in w for w in report["warnings"])


def test_dataset_without_features_still_converts(tmp):
    report, *_ = _convert(tmp, with_features=False)
    with Bundle.open(report["out"]) as b:
        assert b.feature_blocks() == []
        assert b.verify() == []
        assert len(b) == N


def test_converted_bundle_passes_verify(tmp):
    report, *_ = _convert(tmp)
    with Bundle.open(report["out"]) as b:
        assert b.verify() == []


def test_output_filename_is_predictable(tmp):
    report, *_ = _convert(tmp)
    assert os.path.basename(report["out"]) == "synth_image" + SUFFIX



# ─────────────────────────── deleting a dataset ───────────────────────────
def _fake_dataset(tmp, name="trip", modality="image", dims=(2,), extras=True):
    """Lay out the files Arcana creates for one dataset, plus decoys."""
    import numpy as np
    db = os.path.join(tmp, "databases")
    lat = os.path.join(tmp, "latents")
    bun = os.path.join(tmp, "bundles")
    for d in (db, lat, bun):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(db, f"index_{name}_{modality}.pkl"), "wb") as f:
        pickle.dump((b"blob", {0: os.path.join(tmp, "media", "a.jpg")}), f)
    for d in dims:
        with open(os.path.join(lat, f"latent_space_{name}_{modality}_{d}d.pkl"), "wb") as f:
            pickle.dump({"x": 1}, f)
    if extras:
        np.savez(os.path.join(db, f"features_{name}_palette.npz"), ids=np.arange(1))
        np.savez(os.path.join(db, f"features_{name}_style.npz"), ids=np.arange(1))
        open(os.path.join(bun, f"{name}_{modality}.arcana"), "wb").close()

    # decoys that must survive: another dataset, and the user's media
    with open(os.path.join(db, f"index_other_{modality}.pkl"), "wb") as f:
        pickle.dump((b"blob", {0: "x"}), f)
    with open(os.path.join(lat, f"latent_space_other_{modality}_2d.pkl"), "wb") as f:
        pickle.dump({"x": 1}, f)
    media = os.path.join(tmp, "media")
    os.makedirs(media, exist_ok=True)
    open(os.path.join(media, "a.jpg"), "wb").close()
    return db, lat, bun


def _dirs(tmp):
    return {"db_dir": os.path.join(tmp, "databases"),
            "latents_dir": os.path.join(tmp, "latents"),
            "bundles_dir": os.path.join(tmp, "bundles")}


def test_dataset_files_lists_everything_it_owns(tmp):
    from arcana.legacy import dataset_files
    _fake_dataset(tmp)
    got = [os.path.basename(p) for p in dataset_files("trip", "image", **_dirs(tmp))]
    assert "index_trip_image.pkl" in got
    assert "latent_space_trip_image_2d.pkl" in got
    assert "features_trip_palette.npz" in got
    assert "features_trip_style.npz" in got
    assert "trip_image.arcana" in got


def test_dataset_files_never_lists_media(tmp):
    from arcana.legacy import dataset_files
    _fake_dataset(tmp)
    for p in dataset_files("trip", "image", **_dirs(tmp)):
        assert "media" not in p.replace(os.sep, "/").split("/"), p


def test_delete_removes_only_that_dataset(tmp):
    from arcana.legacy import delete_dataset
    db, lat, bun = _fake_dataset(tmp)
    res = delete_dataset("trip", "image", **_dirs(tmp))

    assert res["failed"] == [], res["failed"]
    assert len(res["removed"]) == 5, res["removed"]
    assert not os.path.exists(os.path.join(db, "index_trip_image.pkl"))
    assert not os.path.exists(os.path.join(lat, "latent_space_trip_image_2d.pkl"))
    assert not os.path.exists(os.path.join(bun, "trip_image.arcana"))

    # the other dataset and the photographs are untouched
    assert os.path.exists(os.path.join(db, "index_other_image.pkl"))
    assert os.path.exists(os.path.join(lat, "latent_space_other_image_2d.pkl"))
    assert os.path.exists(os.path.join(tmp, "media", "a.jpg"))


def test_delete_dry_run_changes_nothing(tmp):
    from arcana.legacy import delete_dataset
    db, _lat, _bun = _fake_dataset(tmp)
    res = delete_dataset("trip", "image", dry_run=True, **_dirs(tmp))
    assert len(res["removed"]) == 5
    assert res["bytes"] > 0
    assert os.path.exists(os.path.join(db, "index_trip_image.pkl"))


def test_delete_reports_reclaimed_bytes(tmp):
    from arcana.legacy import delete_dataset
    db, _lat, _bun = _fake_dataset(tmp)
    with open(os.path.join(db, "features_trip_palette.npz"), "wb") as f:
        f.write(b"x" * 50_000)
    res = delete_dataset("trip", "image", dry_run=True, **_dirs(tmp))
    assert res["bytes"] >= 50_000, res["bytes"]


def test_delete_handles_several_latent_dimensions(tmp):
    from arcana.legacy import delete_dataset
    _db, lat, _bun = _fake_dataset(tmp, dims=(2, 3))
    res = delete_dataset("trip", "image", **_dirs(tmp))
    assert not os.path.exists(os.path.join(lat, "latent_space_trip_image_2d.pkl"))
    assert not os.path.exists(os.path.join(lat, "latent_space_trip_image_3d.pkl"))
    assert len(res["removed"]) == 6, res["removed"]


def test_delete_of_an_unknown_dataset_is_a_no_op(tmp):
    from arcana.legacy import delete_dataset
    _fake_dataset(tmp)
    res = delete_dataset("nope", "image", **_dirs(tmp))
    assert res["removed"] == [] and res["failed"] == []


def test_delete_refuses_paths_outside_the_data_directories(tmp):
    """
    The name reaches os.path.join from a filename on disk, so a crafted one is
    the only route out of Arcana's own folders -- and the cost of being wrong
    is somebody's photographs.
    """
    from arcana.legacy import delete_dataset
    _fake_dataset(tmp)
    outside = os.path.join(tmp, "precious.txt")
    with open(outside, "wb") as f:
        f.write(b"do not delete me")

    escape = os.path.join("..", "precious")
    res = delete_dataset(escape, "image", **_dirs(tmp))
    assert res["removed"] == [], res["removed"]
    assert os.path.exists(outside)


def test_migration_resolves_the_encoder_from_vector_width(tmp):
    """
    Legacy indexes record no model, so migration assumed whatever db.py
    hardcoded -- ViT-H/14. For a dataset built with ViT-B/32 that produced a
    bundle claiming CLIP-ViT-H-14 with dim=512, a combination that does not
    exist, and require_model() then compared against a model that never built
    it. The width identifies the encoder unambiguously.
    """
    from arcana.legacy import model_for_vectors
    from arcana import models as _models

    for m in _models.MODELS:
        got = model_for_vectors(m.modality, m.dim)
        assert got.id == m.id, (m.modality, m.dim, got.id)
        assert got.dim == m.dim


def test_migration_keeps_the_real_width_for_an_unknown_encoder(tmp):
    # No catalogue match: the id is a guess, but the dimension must stay true
    # or the bundle misdescribes its own vectors.
    from arcana.legacy import model_for_vectors
    got = model_for_vectors("image", 333)
    assert got.dim == 333

# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_legacy_")
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
