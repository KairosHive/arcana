"""
Tests for re-pointing a dataset after its media folder moved.

Run with:  python tests/test_relocate.py
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana.bundle import Bundle, BundleWriter, Item, ModelSpec, SUFFIX  # noqa: E402
from arcana import legacy, relocate  # noqa: E402

DIM = 1024
N = 10


def _make_media(root, n=N, subdirs=True):
    paths = []
    for i in range(n):
        d = os.path.join(root, f"sub{i % 3}") if subdirs else root
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"IMG_{i:04d}.jpg")
        with open(p, "wb") as f:
            f.write(b"JPEG" + bytes([i % 251]) * (400 + i))
        paths.append(p)
    return paths


def _build_bundle(tmp, media_root, paths, out_name="set_image"):
    items = [Item.for_file(p, media_root) for p in paths]
    vecs = np.arange(len(paths) * DIM, dtype=np.float32).reshape(len(paths), DIM) / 1000.0
    out = os.path.join(tmp, out_name + SUFFIX)
    with BundleWriter(out, name="set", root=media_root,
                      model=ModelSpec(id="test/model", dim=DIM, modality="image")) as w:
        w.set_items(items)
        w.set_vectors(vecs)
        w.set_layout(np.zeros((len(paths), 2), dtype=np.float32), algo="tsne")
        for it in items:
            w.add_thumbnail(it.id, b"WEBP" + it.id.encode()[:6])
    return out, vecs


def _move_media(src_root, dst_root, restructure=True, rename=False):
    """Copy the media somewhere else, optionally reorganising and renaming."""
    os.makedirs(dst_root, exist_ok=True)
    i = 0
    for dirpath, _d, files in os.walk(src_root):
        for fn in sorted(files):
            if restructure:
                d = os.path.join(dst_root, f"batch{i // 4:02d}")
            else:
                d = os.path.join(dst_root, os.path.relpath(dirpath, src_root))
            os.makedirs(d, exist_ok=True)
            name = f"renamed_{i:04d}.jpg" if rename else fn
            shutil.copyfile(os.path.join(dirpath, fn), os.path.join(d, name))
            i += 1


# ─────────────────────────── bundles ───────────────────────────
def test_bundle_relocates_when_the_folder_simply_moved(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, vecs = _build_bundle(tmp, media, paths)

    moved = os.path.join(tmp, "elsewhere", "photos")
    _move_media(media, moved, restructure=False)
    shutil.rmtree(media)

    r = relocate.relocate_bundle(bp, moved)
    assert r["found"] == N and r["missing"] == 0, r
    with Bundle.open(bp) as b:
        assert len(b.resolve(moved)) == N
        np.testing.assert_array_equal(np.asarray(b.vectors), vecs)


def test_bundle_relocates_through_a_reorganised_folder(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, _ = _build_bundle(tmp, media, paths)

    moved = os.path.join(tmp, "reorganised")
    _move_media(media, moved, restructure=True)
    shutil.rmtree(media)

    r = relocate.relocate_bundle(bp, moved)
    assert r["found"] == N, r
    with Bundle.open(bp) as b:
        assert len(b.resolve(moved)) == N
        assert all(it.rel_path.startswith("batch") for it in b.items), \
            "rel_paths must be rewritten to the new structure"


def test_bundle_relocates_through_renames(tmp):
    """Content fingerprints survive files being renamed entirely."""
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, _ = _build_bundle(tmp, media, paths)

    moved = os.path.join(tmp, "renamed")
    _move_media(media, moved, restructure=True, rename=True)
    shutil.rmtree(media)

    r = relocate.relocate_bundle(bp, moved)
    assert r["found"] == N, r
    with Bundle.open(bp) as b:
        assert len(b.resolve(moved)) == N


def test_bundle_relocation_preserves_everything_else(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, vecs = _build_bundle(tmp, media, paths)
    with Bundle.open(bp) as b:
        ids_before = [it.id for it in b.items]
        labels_before = [it.label for it in b.items]
        thumb_before = b.thumbnail(b.items[0].id)
        layout_before = np.array(b.layout)

    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)
    relocate.relocate_bundle(bp, moved)

    with Bundle.open(bp) as b:
        assert [it.id for it in b.items] == ids_before, "ids must not change"
        assert [it.label for it in b.items] == labels_before
        assert b.thumbnail(b.items[0].id) == thumb_before, "thumbnails must survive"
        np.testing.assert_array_equal(np.asarray(b.vectors), vecs)
        np.testing.assert_array_equal(np.asarray(b.layout), layout_before)
        assert b.verify() == []


def test_bundle_dry_run_changes_nothing(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, _ = _build_bundle(tmp, media, paths)
    with Bundle.open(bp) as b:
        root_before = b.manifest["source"]["root"]

    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)

    r = relocate.relocate_bundle(bp, moved, dry_run=True)
    assert r["found"] == N
    assert r["changed"] is False
    with Bundle.open(bp) as b:
        assert b.manifest["source"]["root"] == root_before, "dry run must not rewrite"


def test_bundle_reports_items_it_cannot_find(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    bp, _ = _build_bundle(tmp, media, paths)

    moved = os.path.join(tmp, "partial")
    _move_media(media, moved, restructure=True)
    # delete three of them at the destination
    got = [os.path.join(dp, f) for dp, _d, fs in os.walk(moved) for f in fs]
    for p in sorted(got)[:3]:
        os.remove(p)
    shutil.rmtree(media)

    r = relocate.relocate_bundle(bp, moved)
    assert r["found"] == N - 3
    assert r["missing"] == 3, "missing items must be reported, not silently dropped"
    with Bundle.open(bp) as b:
        assert len(b) == N, "the dataset keeps all its vectors even with pixels missing"


# ─────────────────────────── legacy pickles ───────────────────────────
def _write_legacy(tmp, media_root, paths):
    from usearch.index import Index
    db = os.path.join(tmp, "databases"); os.makedirs(db, exist_ok=True)
    lat = os.path.join(tmp, "latents"); os.makedirs(lat, exist_ok=True)
    index = Index(ndim=DIM, metric="cos", dtype="f32")
    idx2path = {}
    rng = np.random.default_rng(0)
    for i, p in enumerate(paths):
        index.add(i, rng.standard_normal(DIM).astype(np.float32))
        idx2path[i] = os.path.abspath(p)
    with open(os.path.join(db, "index_set_image.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)
    pd.DataFrame({
        "x": np.arange(len(paths), dtype=np.float32),
        "y": np.arange(len(paths), dtype=np.float32),
        "path": [os.path.abspath(p) for p in paths],
        "cluster_id": np.zeros(len(paths), dtype=int),
        "label": ["c0"] * len(paths),
    }).to_pickle(os.path.join(lat, "latent_space_set_image_2d.pkl"))
    return db, lat


def test_legacy_pickles_get_their_paths_rewritten(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)

    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)
    shutil.rmtree(media)

    ds = legacy.discover(db_dir=db, latents_dir=lat)[0]
    r = relocate.relocate_legacy(ds, moved)
    assert r["found"] == N, r
    assert r["changed"] is True

    with open(ds.index_path, "rb") as f:
        _blob, idx2path = pickle.load(f)
    for p in idx2path.values():
        assert os.path.exists(p), f"index still points at a missing file: {p}"
        assert moved.lower() in p.lower()

    df = pd.read_pickle(ds.latent_paths[2])
    assert all(os.path.exists(p) for p in df["path"]), "latent frame must be rewritten too"


def test_legacy_relocation_writes_a_backup(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)
    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)

    ds = legacy.discover(db_dir=db, latents_dir=lat)[0]
    relocate.relocate_legacy(ds, moved)
    assert os.path.exists(ds.index_path + ".bak")
    assert os.path.exists(ds.latent_paths[2] + ".bak")


def test_legacy_dry_run_changes_nothing(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)
    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)

    ds = legacy.discover(db_dir=db, latents_dir=lat)[0]
    with open(ds.index_path, "rb") as f:
        before = pickle.load(f)[1]
    r = relocate.relocate_legacy(ds, moved, dry_run=True)
    assert r["found"] == N and r["changed"] is False
    with open(ds.index_path, "rb") as f:
        assert pickle.load(f)[1] == before
    assert not os.path.exists(ds.index_path + ".bak")


def test_legacy_index_vectors_are_untouched_by_relocation(tmp):
    """Relocation must never re-encode anything."""
    from usearch.index import Index
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)
    with open(os.path.join(db, "index_set_image.pkl"), "rb") as f:
        blob_before, _ = pickle.load(f)
    before = np.asarray([Index.restore(blob_before).get(k) for k in range(N)], dtype=np.float32)

    moved = os.path.join(tmp, "moved")
    _move_media(media, moved, restructure=True)
    ds = legacy.discover(db_dir=db, latents_dir=lat)[0]
    relocate.relocate_legacy(ds, moved)

    with open(ds.index_path, "rb") as f:
        blob_after, _ = pickle.load(f)
    after = np.asarray([Index.restore(blob_after).get(k) for k in range(N)], dtype=np.float32)
    np.testing.assert_array_equal(before, after)


# ─────────────────────────── detection ───────────────────────────
def test_dataset_health_reports_healthy_when_files_are_present(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)
    try:
        h = relocate.dataset_health("set", "image", db_dir=db, latents_dir=lat)
        assert h["ok"] is True
        assert h["missing"] == 0
        assert h["total"] == N
        assert os.path.normcase(h["root"]) == os.path.normcase(media)
    finally:
        pass


def test_dataset_health_detects_a_moved_folder(tmp):
    media = os.path.join(tmp, "photos")
    paths = _make_media(media)
    db, lat = _write_legacy(tmp, media, paths)
    shutil.rmtree(media)                      # the folder went away
    try:
        h = relocate.dataset_health("set", "image", db_dir=db, latents_dir=lat)
        assert h["ok"] is False
        assert h["missing"] == h["checked"] == N
        assert h["root"], "the recorded root must be reported so the user knows where to look"
    finally:
        pass


def test_dataset_health_samples_rather_than_stating_everything(tmp):
    """82k stat calls on every dropdown change would be far too slow."""
    media = os.path.join(tmp, "photos")
    paths = _make_media(media, n=60)
    db, lat = _write_legacy(tmp, media, paths)
    try:
        h = relocate.dataset_health("set", "image", sample=10, db_dir=db, latents_dir=lat)
        assert h["total"] == 60
        assert h["checked"] <= 10, "must sample, not check every path"
        h_all = relocate.dataset_health("set", "image", sample=0, db_dir=db, latents_dir=lat)
        assert h_all["checked"] == 60, "sample=0 must check everything"
    finally:
        pass


def test_dataset_health_on_an_unknown_dataset_is_not_fatal(tmp):
    h = relocate.dataset_health("does-not-exist", "image")
    assert h["ok"] is False and h["error"]


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_reloc_")
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
