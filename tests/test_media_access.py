"""
Tests for the media-serving boundary.

The six media endpoints take a client-supplied path. Before the allowlist they
would read anything on the machine. The allowlist itself then reintroduced the
same hole in a subtler way: dataset directories were collapsed with
os.path.commonpath() and the ancestor registered as a PREFIX root, so a dataset
whose files sat in two top-level folders registered 'C:\\' and made the whole
drive readable again.

These tests pin both the boundary and the shape of the grant.

Run with:  python tests/test_media_access.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcana.arcana as A  # noqa: E402  (heavy import, done once)

CLIENT = A.app.server.test_client()


def _reset():
    with A._ALLOWED_ROOTS_LOCK:
        A._ALLOWED_ROOTS.clear()
        A._ALLOWED_FILES.clear()
        A._REGISTERED_DATASETS.clear()


def _write(path, data=b"x" * 64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _png(path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 24), (10, 120, 200)).save(path, "PNG")
    return path


# ─────────────────────────── the regression ───────────────────────────
def test_a_dataset_spanning_two_folders_does_not_expose_the_drive(tmp):
    """
    The exact shape that used to collapse to 'C:\\'. Two files with no common
    ancestor except the drive root.
    """
    _reset()
    a = _png(os.path.join(tmp, "media", "img", "a.png"))
    b = _png(os.path.join(tmp, "b.png"))
    secret = _write(os.path.join(tmp, "secrets.env"), b"API_KEY=hunter2")

    A.register_dataset_files({0: a, 1: b})

    assert A.resolve_media_request(secret) is None, \
        "a sibling of an indexed file must not become readable"
    r = CLIENT.get("/audio", query_string={"p": secret})
    assert r.status_code == 404, f"/audio leaked {secret}: {r.data[:40]!r}"
    # and the indexed files themselves still work
    assert A.resolve_media_request(a) is not None
    assert A.resolve_media_request(b) is not None


def test_no_ancestor_is_ever_registered_as_a_prefix_root(tmp):
    _reset()
    before = set(A._ALLOWED_ROOTS)
    A.register_dataset_files({
        0: os.path.join(tmp, "one", "deep", "a.png"),
        1: os.path.join(tmp, "two", "b.png"),
    })
    assert set(A._ALLOWED_ROOTS) == before, \
        "dataset registration must not add prefix roots at all"
    assert len(A._ALLOWED_FILES) == 2


def test_siblings_of_an_indexed_file_are_not_served(tmp):
    _reset()
    indexed = _png(os.path.join(tmp, "album", "indexed.png"))
    sibling = _png(os.path.join(tmp, "album", "private.png"))
    A.register_dataset_files({0: indexed})

    assert A.resolve_media_request(indexed) is not None
    assert A.resolve_media_request(sibling) is None, \
        "the grant is the file, not its folder"
    assert CLIENT.get("/thumb", query_string={"p": indexed}).status_code == 200
    assert CLIENT.get("/thumb", query_string={"p": sibling}).status_code == 404


# ─────────────────────────── too-broad prefix roots ───────────────────────────
def test_register_media_root_refuses_drive_and_home_roots(tmp):
    _reset()
    home = os.path.expanduser("~")
    for bad in (os.path.abspath(os.sep), home, os.path.dirname(home)):
        before = set(A._ALLOWED_ROOTS)
        A.register_media_root(bad)
        assert set(A._ALLOWED_ROOTS) == before, f"{bad!r} must be refused as a media root"


def test_register_media_root_still_accepts_an_ordinary_folder(tmp):
    _reset()
    d = os.path.join(tmp, "Pictures", "trip")
    os.makedirs(d)
    A.register_media_root(d)
    assert any(os.path.normcase(r) == os.path.normcase(os.path.realpath(d))
               for r in A._ALLOWED_ROOTS)


def test_a_prefix_root_does_grant_its_subtree(tmp):
    """Prefix semantics are still what an explicitly-named folder means."""
    _reset()
    root = os.path.join(tmp, "media")
    deep = _png(os.path.join(root, "a", "b", "c.png"))
    A.register_media_root(root)
    assert A.resolve_media_request(deep) is not None


# ─────────────────────────── the original hole stays shut ───────────────────────────
def test_arbitrary_absolute_paths_are_refused(tmp):
    _reset()
    A._seed_allowed_roots()
    outside = _write(os.path.join(tmp, "outside.txt"), b"not yours")
    for ep in ("audio", "thumb", "preview", "palette", "awave", "aspec"):
        r = CLIENT.get(f"/{ep}", query_string={"p": outside})
        assert r.status_code == 404, f"/{ep} served a file outside every root"


def test_traversal_out_of_a_granted_root_is_refused(tmp):
    _reset()
    root = os.path.join(tmp, "media")
    os.makedirs(root)
    secret = _write(os.path.join(tmp, "secret.env"), b"nope")
    A.register_media_root(root)
    escape = os.path.join(root, "..", "secret.env")
    assert A.resolve_media_request(escape) is None
    assert CLIENT.get("/audio", query_string={"p": escape}).status_code == 404


def test_a_missing_file_inside_a_root_is_refused_not_crashed(tmp):
    _reset()
    root = os.path.join(tmp, "media")
    os.makedirs(root)
    A.register_media_root(root)
    assert A.resolve_media_request(os.path.join(root, "nope.png")) is None
    assert A.resolve_media_request("") is None


def test_registration_is_memoised_per_dataset(tmp):
    """load_data() runs on every scatter update; 82k paths must not be rewalked."""
    _reset()
    A.register_dataset_files({0: os.path.join(tmp, "a.png")}, cache_key="ds:1")
    n1 = len(A._ALLOWED_FILES)
    A.register_dataset_files({0: os.path.join(tmp, "b.png")}, cache_key="ds:1")
    assert len(A._ALLOWED_FILES) == n1, "a repeat call with the same key must be a no-op"
    A.register_dataset_files({0: os.path.join(tmp, "b.png")}, cache_key="ds:2")
    assert len(A._ALLOWED_FILES) == n1 + 1


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_media_")
        try:
            fn(tmp)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    _reset()
    A._seed_allowed_roots()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        import traceback
        for name, e in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(e), e, e.__traceback__)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
