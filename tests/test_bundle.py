"""
Tests for the portable dataset bundle.

Run with:  python -m pytest tests/test_bundle.py -q
       or: python tests/test_bundle.py     (no pytest needed)
"""

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana.bundle import (  # noqa: E402
    Bundle, BundleWriter, BundleError, ModelMismatch, ModelSpec, Item,
    fingerprint, list_bundles, SUFFIX,
)

MODEL = ModelSpec(id="laion/CLIP-ViT-B-32", dim=8, modality="image", normalized=True)


def _make_files(root, n=6):
    """n small files with distinct contents."""
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    paths = []
    for i in range(n):
        d = root if i % 2 == 0 else os.path.join(root, "sub")
        p = os.path.join(d, f"img{i:03d}.jpg")
        with open(p, "wb") as f:
            f.write(b"JPEGDATA" + bytes([i]) * (512 + i))
        paths.append(p)
    return paths


def _build(tmp, *, n=6, thumbs=True, layout=True, features=True, name="testset"):
    root = os.path.join(tmp, "images")
    os.makedirs(root, exist_ok=True)
    paths = _make_files(root, n)
    items = [Item.for_file(p, root, cluster_id=i % 3, label=f"c{i % 3}") for i, p in enumerate(paths)]
    vecs = np.arange(n * MODEL.dim, dtype=np.float32).reshape(n, MODEL.dim) / 100.0

    out = os.path.join(tmp, name + SUFFIX)
    with BundleWriter(out, name=name, model=MODEL, root=root, created_at="2026-09-01T00:00:00Z") as w:
        w.set_items(items)
        w.set_vectors(vecs)
        if layout:
            w.set_layout(np.linspace(-1, 1, n * 2).reshape(n, 2).astype(np.float32), algo="tsne")
        if thumbs:
            for it in items:
                w.add_thumbnail(it.id, b"WEBPFAKE" + it.id.encode()[:4])
        if features:
            w.add_feature_block("palette", {
                "ids": np.arange(n, dtype=np.int32),
                "moments": np.ones((n, 9), dtype=np.float32),
            })
    return out, root, items, vecs


# ─────────────────────────── round trip ───────────────────────────
def test_roundtrip_preserves_everything(tmp):
    out, root, items, vecs = _build(tmp)
    with Bundle.open(out) as b:
        assert b.name == "testset"
        assert len(b) == len(items)
        assert b.model.id == MODEL.id and b.model.dim == MODEL.dim
        assert b.model.normalized is True
        np.testing.assert_array_equal(b.vectors, vecs)
        assert b.vectors.dtype == np.dtype("<f4")
        assert b.layout is not None and b.layout.shape == (len(items), 2)
        assert [it.id for it in b.items] == [it.id for it in items]
        assert b.items[0].name == items[0].name


def test_vectors_come_from_the_memory_map(tmp):
    """The float block must be STORED, so it can be mapped instead of copied."""
    out, *_ = _build(tmp)
    with zipfile.ZipFile(out) as z:
        assert z.getinfo("vectors.f32").compress_type == zipfile.ZIP_STORED
    with Bundle.open(out) as b:
        _ = b.vectors
        assert b._mm is not None, "vectors should have been memory-mapped"


def test_mapped_and_copied_reads_agree(tmp):
    """The mmap fast path and the plain-read fallback must return identical data."""
    out, _root, _items, vecs = _build(tmp)
    with Bundle.open(out) as b:
        mapped = np.array(b.vectors)
    with Bundle.open(out) as b:
        b._data_offset = lambda info: None      # force the fallback
        copied = np.array(b.vectors)
    np.testing.assert_array_equal(mapped, copied)
    np.testing.assert_array_equal(mapped, vecs)


def test_thumbnails_and_features(tmp):
    out, _root, items, _ = _build(tmp)
    with Bundle.open(out) as b:
        assert b.has_thumbnails()
        assert b.thumbnail(items[0].id).startswith(b"WEBPFAKE")
        assert b.thumbnail("nope") is None
        assert b.feature_blocks() == ["palette"]
        pal = b.feature("palette")
        assert pal["moments"].shape == (len(items), 9)
        assert b.feature("style") is None


# ─────────────────────────── portability ───────────────────────────
def test_no_absolute_paths_anywhere_in_the_bundle(tmp):
    """The whole point: nothing machine-specific in the payload."""
    out, root, _items, _ = _build(tmp)
    with zipfile.ZipFile(out) as z:
        items_raw = z.read("items.jsonl").decode()
        manifest = json.loads(z.read("manifest.json"))
    for line in items_raw.splitlines():
        rec = json.loads(line)
        assert not os.path.isabs(rec["rel_path"]), rec["rel_path"]
        assert "\\" not in rec["rel_path"], "paths must be POSIX-style"
        assert root.replace("\\", "/") not in rec["rel_path"]
    # The recorded root is a hint for rebinding, and the only place a local path appears.
    assert manifest["source"]["root"] == root


def test_resolve_finds_files_under_a_new_root(tmp):
    out, root, items, _ = _build(tmp)
    moved = os.path.join(tmp, "relocated")
    shutil.move(root, moved)
    with Bundle.open(out) as b:
        found = b.resolve(moved)
        assert len(found) == len(items)
        assert all(os.path.exists(p) for p in found.values())


def test_rebind_recovers_renamed_and_reorganised_files(tmp):
    """Files renamed and moved between subfolders are recovered by fingerprint."""
    out, root, items, _ = _build(tmp)
    moved = os.path.join(tmp, "relocated")
    shutil.move(root, moved)
    flat = os.path.join(moved, "all")
    os.makedirs(flat)
    for i, fn in enumerate(sorted(os.listdir(moved))):
        src = os.path.join(moved, fn)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(flat, f"renamed_{i}.jpg"))
    for fn in sorted(os.listdir(os.path.join(moved, "sub"))):
        shutil.move(os.path.join(moved, "sub", fn),
                    os.path.join(flat, f"sub_renamed_{fn}"))

    with Bundle.open(out) as b:
        assert len(b.resolve(moved)) < len(items), "recorded paths should now be stale"
        found = b.rebind(moved)
        assert len(found) == len(items), "every item should be recovered by fingerprint"
        for it in items:
            assert fingerprint(found[it.id]) == it.id


def test_fingerprint_is_content_not_name(tmp):
    a = os.path.join(tmp, "a.bin")
    b = os.path.join(tmp, "b.bin")
    with open(a, "wb") as f:
        f.write(b"same content here")
    shutil.copyfile(a, b)
    assert fingerprint(a) == fingerprint(b)
    with open(b, "ab") as f:
        f.write(b"x")
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_distinguishes_large_files_differing_only_in_the_middle(tmp):
    """A same-size file with one changed byte in the middle must not collide."""
    big = 3 << 20
    a = os.path.join(tmp, "a.bin")
    b = os.path.join(tmp, "b.bin")
    data = bytearray(os.urandom(big))
    with open(a, "wb") as f:
        f.write(data)
    data2 = bytearray(data)
    data2[big // 2] ^= 0xFF
    with open(b, "wb") as f:
        f.write(data2)
    assert os.path.getsize(a) == os.path.getsize(b)
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_handles_empty_and_multichunk_files(tmp):
    empty = os.path.join(tmp, "empty.bin")
    open(empty, "wb").close()
    assert len(fingerprint(empty)) == 32

    big = os.path.join(tmp, "big.bin")
    payload = os.urandom((1 << 20) * 2 + 12345)   # spans several read chunks
    with open(big, "wb") as f:
        f.write(payload)
    import hashlib as _h
    expect = _h.blake2b(payload, digest_size=16).hexdigest()
    assert fingerprint(big) == expect, "chunked hashing must equal one-shot hashing"


# ─────────────────────────── guards ───────────────────────────
def test_model_mismatch_is_refused(tmp):
    out, *_ = _build(tmp)
    with Bundle.open(out) as b:
        b.require_model(ModelSpec(id=MODEL.id, dim=8, modality="image"))
        for bad in (
            ModelSpec(id="openai/clip-vit-large-patch14", dim=8, modality="image"),
            ModelSpec(id=MODEL.id, dim=512, modality="image"),
            ModelSpec(id=MODEL.id, dim=8, modality="audio"),
        ):
            try:
                b.require_model(bad)
            except ModelMismatch as e:
                assert MODEL.id in str(e)
            else:
                raise AssertionError(f"should have refused {bad}")


def test_writer_rejects_dimension_mismatch(tmp):
    root = os.path.join(tmp, "images")
    os.makedirs(root)
    items = [Item.for_file(p, root) for p in _make_files(root, 3)]
    out = os.path.join(tmp, "bad" + SUFFIX)
    w = BundleWriter(out, name="bad", model=MODEL, root=root)
    w.set_items(items)
    w.set_vectors(np.zeros((3, 99), dtype=np.float32))   # model.dim is 8
    try:
        w.close()
    except BundleError as e:
        assert "99" in str(e) and "8" in str(e)
    else:
        raise AssertionError("expected a dimension mismatch error")
    assert not os.path.exists(out)


def test_writer_rejects_row_count_mismatch(tmp):
    root = os.path.join(tmp, "images")
    os.makedirs(root)
    items = [Item.for_file(p, root) for p in _make_files(root, 3)]
    w = BundleWriter(os.path.join(tmp, "x" + SUFFIX), name="x", model=MODEL, root=root)
    w.set_items(items)
    w.set_vectors(np.zeros((5, MODEL.dim), dtype=np.float32))
    try:
        w.close()
    except BundleError as e:
        assert "5" in str(e) and "3" in str(e)
    else:
        raise AssertionError("expected a row count mismatch error")


def test_writer_rejects_duplicate_ids(tmp):
    root = os.path.join(tmp, "images")
    os.makedirs(root)
    paths = _make_files(root, 2)
    dup = os.path.join(root, "copy.jpg")
    shutil.copyfile(paths[0], dup)                       # identical content -> identical id
    items = [Item.for_file(p, root) for p in paths + [dup]]
    w = BundleWriter(os.path.join(tmp, "d" + SUFFIX), name="d", model=MODEL, root=root)
    w.set_items(items)
    w.set_vectors(np.zeros((3, MODEL.dim), dtype=np.float32))
    try:
        w.close()
    except BundleError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected duplicate ids to be refused")


def test_writer_refuses_empty(tmp):
    w = BundleWriter(os.path.join(tmp, "e" + SUFFIX), name="e", model=MODEL)
    try:
        w.close()
    except BundleError as e:
        assert "no items" in str(e)
    else:
        raise AssertionError("expected an empty bundle to be refused")


def test_failed_write_leaves_no_partial_file(tmp):
    out = os.path.join(tmp, "boom" + SUFFIX)
    try:
        with BundleWriter(out, name="boom", model=MODEL) as w:
            w.set_items([Item(id="a", rel_path="a.jpg")])
            raise RuntimeError("simulated failure mid-build")
    except RuntimeError:
        pass
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".partial")


# ─────────────────────────── integrity ───────────────────────────
def test_verify_passes_on_a_good_bundle(tmp):
    out, *_ = _build(tmp)
    with Bundle.open(out) as b:
        assert b.verify() == []


def test_verify_catches_non_finite_vectors(tmp):
    root = os.path.join(tmp, "images")
    os.makedirs(root)
    items = [Item.for_file(p, root) for p in _make_files(root, 3)]
    v = np.zeros((3, MODEL.dim), dtype=np.float32)
    v[1, 2] = np.nan
    out = os.path.join(tmp, "nan" + SUFFIX)
    with BundleWriter(out, name="nan", model=MODEL, root=root) as w:
        w.set_items(items)
        w.set_vectors(v)
    with Bundle.open(out) as b:
        assert any("NaN" in p for p in b.verify())


def test_open_rejects_a_non_bundle_zip(tmp):
    p = os.path.join(tmp, "not" + SUFFIX)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "hi")
    try:
        Bundle.open(p)
    except BundleError as e:
        assert "manifest" in str(e).lower() or "not an Arcana bundle" in str(e)
    else:
        raise AssertionError("expected a non-bundle zip to be refused")


def test_open_rejects_a_future_format_version(tmp):
    out, *_ = _build(tmp)
    future = os.path.join(tmp, "future" + SUFFIX)
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(future, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "manifest.json":
                m = json.loads(data)
                m["format_version"] = 999
                data = json.dumps(m).encode()
            dst.writestr(info, data)
    try:
        Bundle.open(future)
    except BundleError as e:
        assert "999" in str(e)
    else:
        raise AssertionError("expected a future format version to be refused")


def test_manifest_item_count_disagreement_is_caught(tmp):
    out, *_ = _build(tmp)
    lying = os.path.join(tmp, "lying" + SUFFIX)
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(lying, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "manifest.json":
                m = json.loads(data)
                m["n_items"] = 999
                data = json.dumps(m).encode()
            dst.writestr(info, data)
    try:
        Bundle.open(lying)
    except BundleError as e:
        assert "999" in str(e)
    else:
        raise AssertionError("expected an item count disagreement to be caught")


# ─────────────────────────── discovery ───────────────────────────
def test_list_bundles_reads_only_metadata(tmp):
    d = os.path.join(tmp, "datasets")
    os.makedirs(d)
    for nm in ("alpha", "beta"):
        out, *_ = _build(tmp, name=nm)
        shutil.move(out, os.path.join(d, nm + SUFFIX))
    with open(os.path.join(d, "junk.arcana"), "wb") as f:
        f.write(b"not a zip")
    with open(os.path.join(d, "ignored.txt"), "w") as f:
        f.write("x")

    listed = list_bundles(d)
    assert [e["name"] for e in listed] == ["alpha", "beta"], "junk must be skipped, not fatal"
    assert listed[0]["model_id"] == MODEL.id
    assert listed[0]["dim"] == MODEL.dim
    assert listed[0]["n_components"] == 2
    assert list_bundles(os.path.join(tmp, "nope")) == []


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_bundle_")
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
