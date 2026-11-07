"""
Tests for core.py: search, the marks filter, dedupe and dataset loading.

These used to live inside arcana.py, which builds the Dash app on import, so
nothing could search without starting a web application. The first test pins
the reason the module exists; the rest pin that moving the code changed nothing.

No test here downloads a model: the text encoder is replaced by a fixed vector.

Run with:  python tests/test_core.py
"""

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arcana import core  # noqa: E402


def _index(vectors):
    from usearch.index import Index
    index = Index(ndim=len(vectors[0]), metric="cos")
    idx2path = {}
    for k, v in enumerate(vectors):
        index.add(k, np.asarray(v, dtype=np.float32))
        idx2path[k] = f"/photos/{k}.jpg"
    return index, idx2path


def _write_dataset(tmp, name="synth", *, with_latent=True):
    """databases/ and latents/ laid out as db.py writes them."""
    db = os.path.join(tmp, "databases")
    lat = os.path.join(tmp, "latents")
    os.makedirs(db, exist_ok=True)
    os.makedirs(lat, exist_ok=True)
    index, idx2path = _index([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open(os.path.join(db, f"index_{name}_image.pkl"), "wb") as f:
        pickle.dump((index.save(), idx2path), f)
    if with_latent:
        pd.DataFrame({
            "x": [0.0, 1.0, 2.0], "y": [0.0, -1.0, -2.0],
            "path": list(idx2path.values()), "label": [1, 2, 3],
        }).to_pickle(os.path.join(lat, f"latent_space_{name}_image_2d.pkl"))
    return db, lat


# ─────────────────────────── the reason it exists ───────────────────────────

def test_importing_core_builds_no_app_and_loads_no_model(tmp):
    """
    The MCP server imports this to answer a query. If it pulled in Dash it
    would be constructing a web app to do so, and if it pulled in torch every
    agent session would pay seconds of start-up before its first tool call.

    Run in a fresh interpreter: by the time this test runs, other tests in the
    same process have usually imported both.
    """
    code = ("import sys, arcana.core; "
            "bad = [m for m in ('dash', 'torch', 'transformers') if m in sys.modules]; "
            "assert not bad, bad")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_app_uses_core_rather_than_a_copy(tmp):
    """Two copies of the filter could drift; the app must hold the same objects."""
    import arcana.arcana as app
    assert app.search is core.search
    assert app.marks_mask is core.marks_mask
    assert app.has_marks is core.has_marks
    assert app._cosine_group is core.cosine_group
    assert app.text_model_for_dim is core.text_model_for_dim
    assert app.UNRATED == core.UNRATED and app.NO_COLOUR == core.NO_COLOUR


# ─────────────────────────── search ───────────────────────────

def test_search_ranks_by_distance(tmp):
    index, idx2path = _index([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0]])
    with _query_vector([1, 0, 0, 0]):
        hits = core.search(index, idx2path, "anything", 2)
    assert [k for k, _, _ in hits] == [0, 1]
    assert hits[0][1] == idx2path[0]
    assert hits[0][2] <= hits[1][2]


def test_search_with_a_filter_fills_the_page_from_what_survives(tmp):
    """
    Asking usearch for `n` and then filtering is what used to return an empty
    page under a selective star filter. The filter applies to the whole ranking.
    """
    vecs = [[1, 0, 0, 0]] * 3 + [[0, 1, 0, 0]] * 3
    index, idx2path = _index(vecs)
    with _query_vector([1, 0, 0, 0]):
        hits = core.search(index, idx2path, "anything", 2, allowed={4, 5})
    assert sorted(k for k, _, _ in hits) == [4, 5]


class _query_vector:
    """Stand in for the text encoder so no model is downloaded."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def __enter__(self):
        self.saved = core.encode_query
        core.encode_query = lambda query, index, modality="image": self.vec
        return self

    def __exit__(self, *exc):
        core.encode_query = self.saved


# ─────────────────────────── dedupe ───────────────────────────

def test_cosine_group_puts_near_copies_together_in_rank_order(tmp):
    index, idx2path = _index([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0.01, 0, 0]])
    keys = [0, 1, 2]
    groups = core.cosine_group(keys, [idx2path[k] for k in keys], index, thresh=0.08)
    assert [g["keys"] for g in groups] == [[0, 2], [1]]
    assert [g["gid"] for g in groups] == ["g0", "g1"]


def test_cosine_group_of_one_is_one_group(tmp):
    index, _ = _index([[1, 0, 0, 0]])
    assert core.cosine_group([0], ["a"], index) == [{"gid": "g0", "keys": [0], "paths": ["a"]}]


# ─────────────────────────── marks ───────────────────────────

def test_marks_mask_combines_stars_colours_and_dates(tmp):
    frame = pd.DataFrame({
        "rating": [5, 3, None, 1],
        "colour": ["Red", "", "Red", ""],
        "captured": pd.to_datetime(["2025-06-05", "2025-06-05", None, "2024-01-01"]),
    })
    assert list(core.marks_mask(frame, stars=3)) == [True, True, False, False]
    assert list(core.marks_mask(frame, stars=core.UNRATED)) == [False, False, True, False]
    assert list(core.marks_mask(frame, colours=[core.NO_COLOUR])) == [False, True, False, True]
    assert list(core.marks_mask(frame, stars=3, colours=["Red"],
                                date_from="2025-06-05", date_to="2025-06-05")) == [True, False, False, False]


def test_a_dataset_without_marks_is_left_alone(tmp):
    frame = pd.DataFrame({"path": ["a", "b"]})
    assert not core.has_marks(frame)
    assert core.marks_mask(frame, stars=4, colours=["Red"]).all()


# ─────────────────────────── loading ───────────────────────────

def test_load_data_types_the_frame_for_plotting(tmp):
    _, lat = _write_dataset(tmp)
    df = core.load_data("synth", latent_dir=lat)
    assert df["x"].dtype == np.float32
    assert df["label"].tolist() == ["1", "2", "3"]


def test_open_dataset_is_cached_until_its_files_change(tmp):
    """An agent searches a dozen times in a row; restoring the index each time would dominate."""
    db, lat = _write_dataset(tmp)
    first = core.open_dataset("synth", db_dir=db, latent_dir=lat)
    assert core.open_dataset("synth", db_dir=db, latent_dir=lat) is first
    assert len(first.idx2path) == 3 and first.frame is not None

    # Another process reworks the dataset: the next call must see it.
    later = time.time() + 5
    os.utime(os.path.join(lat, "latent_space_synth_image_2d.pkl"), (later, later))
    assert core.open_dataset("synth", db_dir=db, latent_dir=lat) is not first


def test_an_index_without_a_layout_still_opens(tmp):
    """Indexing that stopped after encoding leaves a searchable index and no map."""
    db, lat = _write_dataset(tmp, name="stranded", with_latent=False)
    ds = core.open_dataset("stranded", db_dir=db, latent_dir=lat)
    assert ds.frame is None
    assert len(ds.idx2path) == 3


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_core_")
        try:
            fn(tmp)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
