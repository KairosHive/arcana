"""
The colour, style, generate and job ops, on the synthetic dataset.

Results are new files with out: handles in the ledger, marked derived or
generated wherever they go, and originals are never written to. Harmonizing
respects the board's hero and the one-third ceiling on derived pictures.
Generation is stubbed with a flat picture, so no model is loaded.

Run with:  python -m pytest tests/test_colour_ops.py -q
"""

import os
import sys
import time

import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import boards, ledger, ops  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def run(*lines, actor="cli"):
    b = ops.exec_lines(list(lines), actor)
    assert b.ok, b.refused and b.refused.refusal["reason"]
    return [s.result for s in b.steps]


def refused(*lines, actor="cli"):
    b = ops.exec_lines(list(lines), actor)
    assert not b.ok
    return b.refused.refusal["reason"]


def test_a_transfer_is_a_new_file_with_a_handle_and_the_original_is_untouched(tmp, ds):
    mtime = os.path.getmtime(ds[3])
    r, = run("colour transfer synth:3 --reference synth:20 --method mkl")
    assert r.handle.startswith("out:") and os.path.isfile(r.path)
    assert os.path.getmtime(ds[3]) == mtime
    e = ledger.get(r.handle)
    assert e["kind"] == "derived" and e["derived_from"] == "synth:3"
    assert e["sources"] == ["synth:3", "synth:20"]
    # The handle works wherever a picture is taken.
    shown, = run(f"item show {r.handle} --no-describe")
    assert "kind: derived" in shown.text()


def test_a_palette_can_stand_in_for_a_reference(tmp, ds):
    r, = run("colour transfer synth:3 --palette '#c8743f:0.7' --palette '#1c1a17:0.3' --method lab")
    assert r.reference.startswith("palette #c8743f")
    assert "not both" in refused("colour transfer synth:3 --reference synth:2 --palette '#ffffff'")


def test_a_transfer_is_undoable_from_the_ledger(tmp, ds):
    r, = run("colour transfer synth:3 --reference synth:20 --method lab")
    assert ledger.get(r.handle)
    run("undo")
    assert ledger.get(r.handle) is None


def test_background_runs_as_a_job_and_records_when_done(tmp, ds):
    started, = run("colour transfer synth:3 --reference synth:20 --method idt --background")
    assert started.job
    waited, = run(f"job wait {started.job} --timeout 60")
    assert waited.job.status == "done" and "out:" in waited.job.result_text
    handle = waited.job.result_text.split()[0]
    assert ledger.get(handle) is not None
    assert any("colour transfer" in l for l in run("history")[0].undo)


def _hero_board(n_found=6):
    keys = [12, 7, 3, 1, 20, 21][:n_found]
    lines = ['board create "H" --id h', f"board add h synth:{keys[0]} --role hero"]
    lines += [f"board add h {' '.join(f'synth:{k}' for k in keys[1:])} --role support"]
    run(*lines)


def test_harmonize_puts_each_result_beside_its_original(tmp, ds):
    _hero_board()
    res, = run("colour harmonize h synth:3 --reference synth:12 --method mkl")
    rec = boards.load("h")
    i = rec["items"].index(ds[3])
    made = rec["items"][i + 1]
    assert made == res.made[0].path
    d = rec["details"][made]
    assert d["kind"] == "derived" and d["role"] == "support" and "harmonized toward" in d["note"]
    assert ds[3] in rec["items"]                      # the original stays


def test_harmonize_never_touches_the_hero_and_keeps_a_third(tmp, ds):
    _hero_board()
    assert "hero" in refused("colour harmonize h synth:12 --reference synth:3 --method mkl")
    assert "more than a third" in refused(
        "colour harmonize h synth:3 synth:1 synth:20 synth:21 --reference synth:12 --method lab")


def test_harmonize_defaults_to_the_board_palette(tmp, ds):
    _hero_board()
    run("board palette set h --colour '#334455:0.7' --colour '#aa8866:0.3'")
    res, = run("colour harmonize h synth:3 --method lab")
    assert res.made[0].reference.startswith("palette #334455")


def test_an_agent_harmonizes_only_its_own_boards(tmp, ds):
    _hero_board()
    assert "board copy h" in refused("colour harmonize h synth:3 --reference synth:12 "
                                     "--method lab", actor="agent")


def test_texture_style_transfer_and_a_luminance_map(tmp, ds):
    r, map_ = run("style transfer synth:3 --source synth:20 --method texture --strength 0.5",
                  "style control-map synth:3 --kind luminance")
    assert r.handle.startswith("out:") and ledger.get(r.handle)["op"] == "style transfer"
    assert map_.attachments and map_.kind == "luminance"


def test_diffusion_methods_say_what_they_need(tmp, ds, monkeypatch):
    monkeypatch.setitem(registry.FLAG_PROBES, "diffusers", lambda: False)
    assert "diffusers" in refused("style transfer synth:3 --source synth:2 --method ip_adapter")


def _stub_generation(monkeypatch):
    from arcana import generate
    flat = Image.new("RGB", (64, 64), (120, 110, 100))
    monkeypatch.setattr(generate, "txt2img", lambda *a, **k: flat)
    monkeypatch.setattr(generate, "img2img", lambda *a, **k: flat)
    monkeypatch.setitem(registry.FLAG_PROBES, "diffusers", lambda: True)


def test_generate_marks_its_pictures_and_keeps_to_two_per_board(tmp, ds, monkeypatch):
    _stub_generation(monkeypatch)
    _hero_board()
    a, = run('generate --prompt "rain on tin" --for-board h --note "the archive has no rain"')
    assert ledger.get(a.handle)["kind"] == "generated"
    assert boards.load("h")["details"][a.path]["kind"] == "generated"
    run('generate --prompt "more rain" --init synth:3 --for-board h')
    assert "at most 2" in refused('generate --prompt "a third" --for-board h')
    assert "--role" in refused('generate --prompt "x" --for-board h --role hero')


def test_generate_writes_only_under_output(tmp, ds, monkeypatch):
    _stub_generation(monkeypatch)
    assert "not a folder under output" in refused('generate --prompt "x" --folder ../../escape')
    r, = run('generate --prompt "x" --folder stories/s1/poetry_injected')
    assert os.path.join("output", "stories", "s1", "poetry_injected") in r.path


def test_jobs_can_be_listed_and_an_unknown_one_is_named(tmp, ds):
    assert "No job" in refused("job show nope")
    assert run("job list")[0].text()


def test_an_unknown_model_names_the_known_ones(tmp, ds):
    assert "modflows" in refused("model download nope --no-background")
