"""
The dataset lifecycle ops and the old console scripts as aliases.

Indexing itself (db.index_dataset and friends) has its own tests. Here it is
stubbed, so these check what the ops add: refusals in the caller's terms,
arguments mapped onto db.py, jobs, relocation and deletion on a real
synthetic dataset, and each old script's flags translated into one op line.

Run with:  python -m pytest tests/test_lifecycle.py -q
"""

import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import aliases, core, db, models, ops  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_NO_SERVER", "1")
    return synth.use(tmp, monkeypatch)


def run(*lines, actor="cli"):
    return ops.exec_lines(list(lines), actor)


@pytest.fixture
def stub_db(monkeypatch):
    calls = []

    def recorder(name):
        def fn(*a, **k):
            calls.append((name, a, k))
            if k.get("progress"):
                k["progress"](0.5, "halfway", 1, 2)
            return {"n_items": 7, "added": 3}
        return fn
    for name in ("index_dataset", "extend_dataset", "rework_dataset"):
        monkeypatch.setattr(db, name, recorder(name))
    monkeypatch.setattr(db, "warm_label_cache", lambda *a, **k: None)
    monkeypatch.setattr(models, "ensure_model", lambda *a, **k: None)
    return calls


def test_create_maps_its_flags_onto_index_dataset(tmp, ds, stub_db):
    folder = os.path.join(tmp, "new")
    os.makedirs(os.path.join(folder, "Temples"))
    open(os.path.join(folder, "Temples", "a.jpg"), "wb").close()
    b = run(f'dataset create holiday --folder "{folder}" --feature style --feature palette '
            f"--k 5 --labels folders --dims 3 --no-background")
    assert b.ok, b.refused and b.refused.refusal
    name, a, k = stub_db[0]
    assert name == "index_dataset" and a == (os.path.abspath(folder), "holiday")
    assert k["features"] == "clip,palette,style" and k["k"] == 5 and k["n_components"] == 3
    assert k["labels"] == "Temples"
    assert "indexed 7 items" in b.steps[0].result.text()


def test_create_refuses_what_it_cannot_do(tmp, ds, stub_db):
    assert "Not a folder" in run("dataset create x --folder nope").refused.refusal["reason"]
    assert "exists" in run(f'dataset create synth --folder "{tmp}"').refused.refusal["reason"]
    assert "No encoder" in run(f'dataset create y --folder "{tmp}" --model nope').refused.refusal["reason"]
    assert "letters" in run(f'dataset create "a/b" --folder "{tmp}"').refused.refusal["reason"]


def test_create_runs_as_a_job_by_default(tmp, ds, stub_db):
    b = run(f'dataset create bg --folder "{tmp}"')
    jid = b.steps[0].result.job
    waited = run(f"job wait {jid} --timeout 30").steps[0].result.job
    assert waited.status == "done" and "indexed 7 items" in waited.result_text


def test_extend_finds_the_folder_from_the_index(tmp, ds, stub_db):
    assert run("dataset extend synth --no-background").ok
    name, a, k = stub_db[0]
    assert name == "extend_dataset" and a[0] == os.path.join(tmp, "photos")


def test_finish_reworks_without_encoding(tmp, ds, stub_db):
    assert run("dataset finish synth --k 4 --labels none --no-background").ok
    name, a, k = stub_db[0]
    assert name == "rework_dataset" and k["k"] == 4 and k["labels"] == ""


def test_relocate_follows_files_that_moved(tmp, ds):
    moved = os.path.join(tmp, "moved")
    shutil.move(os.path.join(tmp, "photos"), moved)
    dry = run(f'dataset relocate synth --root "{moved}" --dry-run').steps[0].result
    assert dry.found == synth.N and not dry.changed
    assert not os.path.exists(core.open_dataset("synth").idx2path[0])
    done = run(f'dataset relocate synth --root "{moved}"').steps[0].result
    assert done.changed and done.found == synth.N
    assert os.path.exists(core.open_dataset("synth").idx2path[0])


def test_delete_is_the_persons_and_dry_runs_first(tmp, ds):
    dry = run("dataset delete synth --dry-run").steps[0].result
    assert dry.removed and os.path.exists(dry.removed[0])
    assert "for the person" in run("dataset delete synth", actor="agent").refused.refusal["reason"]
    gone = run("dataset delete synth").steps[0].result
    assert not os.path.exists(gone.removed[0])
    assert all(os.path.exists(p) for p in ds.values())        # the pictures stay


def test_migrate_converts_to_a_bundle(tmp, ds):
    r = run("dataset migrate synth --no-background")
    assert r.ok, r.refused and r.refused.refusal
    assert "converted" in r.steps[0].result.text()
    assert any(f.endswith(".arcana") for f in os.listdir(os.path.join(tmp, "bundles")))


# ─────────────────────────── the old scripts ───────────────────────────

@pytest.fixture
def captured(monkeypatch):
    lines = []
    from arcana import cli
    monkeypatch.setattr(cli, "main", lambda tokens: lines.append(tokens) or 0)
    return lines


@pytest.mark.parametrize("fn,argv,want", [
    (aliases.build_latent, ["--imgs_path", "D:/p", "--name", "n", "--features", "clip,palette",
                            "--k", "6"],
     ["dataset", "create", "n", "--folder", "D:/p", "--modality", "image", "--dims", "2",
      "--k", "6", "--no-background", "--feature", "palette"]),
    (aliases.extend, ["--path", "D:/p", "--name", "n", "--modality", "audio"],
     ["dataset", "extend", "n@audio", "--folder", "D:/p", "--no-background"]),
    (aliases.finish, ["--name", "n", "--n_components", "3", "--features", "clip,style"],
     ["dataset", "finish", "n", "--dims", "3", "--k", "0", "--no-background", "--feature", "style"]),
    (aliases.marks, ["--name", "n"], ["dataset", "marks", "n"]),
    (aliases.migrate, ["--name", "n"], ["dataset", "migrate", "n", "--no-background"]),
    (aliases.relocate, ["--name", "n", "--new-root", "D:/p", "--dry-run"],
     ["dataset", "relocate", "n", "--root", "D:/p", "--dry-run"]),
    (aliases.lut, ["--content", "a.jpg", "--reference", "b.jpg", "--out", "x.cube"],
     ["colour", "lut", "a.jpg", "--reference", "b.jpg", "--out", "x.cube", "--n", "33",
      "--steps", "8", "--strength", "1.0"]),
])
def test_each_old_script_is_one_op_line(fn, argv, want, captured, capsys):
    assert fn(argv) == 0
    assert captured == [want]
    assert "is now: arcana " + " ".join(want[:2]) in capsys.readouterr().err
    ops.parse_tokens(want)                       # and the line is one the grammar takes


def test_a_flag_without_an_op_runs_the_old_script(monkeypatch, captured, capsys):
    ran = []
    import arcana.db as dbmod
    monkeypatch.setattr(dbmod, "main", lambda: ran.append(sys.argv[1:]) or 0)
    assert aliases.build_latent(["--imgs_path", "p", "--name", "n", "--k_metric", "davies"]) == 0
    assert ran == [["--imgs_path", "p", "--name", "n", "--k_metric", "davies"]] and not captured
    assert "runs as before" in capsys.readouterr().err


def test_the_scripts_point_at_the_aliases():
    text = open(os.path.join(os.path.dirname(HERE), "pyproject.toml"), encoding="utf-8").read()
    for name, fn in [("arcana-build-latent", "build_latent"), ("arcana-extend", "extend"),
                     ("arcana-finish", "finish"), ("arcana-marks", "marks"),
                     ("arcana-migrate", "migrate"), ("arcana-relocate", "relocate"),
                     ("arcana-lut", "lut")]:
        assert f'{name} = "arcana.aliases:{fn}"' in text
    assert 'arcana-mcp = "arcana.mcp_server:main"' in text
