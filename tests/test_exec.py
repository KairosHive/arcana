"""
Batches, transactions and undo.

A batch of op lines is one transaction and one undo step: a refusal anywhere
rolls back everything before it, and undo takes back the whole batch. Each
actor (person, cli, agent) has its own history, and an agent changes only
boards an agent made (A7).

Run with:  python -m pytest tests/test_exec.py -q
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import boards, ops, state  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def run(*lines, actor="cli"):
    return ops.exec_lines(list(lines), actor)


def board(slug):
    return boards.load(slug)


def test_a_refused_line_rolls_back_the_whole_batch(tmp, ds):
    b = run('board create "Rain" --id rain', "board add rain synth:0 synth:1",
            "board add rain synth:999", "board add rain synth:2")
    assert not b.ok and b.rolled_back
    assert [s.result is not None for s in b.steps] == [True, True, False, False]
    assert board("rain") is None
    assert state.STORE.history("cli")["undo"] == []


def test_a_typo_anywhere_stops_the_batch_before_anything_runs(tmp, ds):
    b = run('board create "Rain" --id rain', "board ad rain synth:0")
    assert not b.ok and b.steps[0].result is None
    assert board("rain") is None


def test_a_batch_is_one_undo_step(tmp, ds):
    run('board create "Rain" --id rain', "board add rain synth:0", "board add rain synth:1")
    assert len(board("rain")["items"]) == 2
    assert len(state.STORE.history("cli")["undo"]) == 1
    assert run("undo").steps[0].result.label.startswith('board create "Rain"')
    assert board("rain") is None
    run("redo")
    assert len(board("rain")["items"]) == 2


def test_reads_in_a_batch_see_the_writes_before_them(tmp, ds):
    b = run('board create "Rain" --id rain', "board add rain synth:0 --role hero", "board show rain")
    shown = b.steps[2].result
    assert shown.metrics.count == 1 and shown.metrics.heroes == [1]


def test_each_actor_undoes_only_its_own_steps(tmp, ds):
    run('board create "Theirs" --id theirs', "board add theirs synth:0", actor="person")
    run('board create "Agents" --id agents', "board add agents synth:1", actor="agent")
    run("board add theirs synth:2", actor="cli")

    assert run("undo", actor="agent").steps[0].result.label.startswith('board create "Agents"')
    assert board("agents") is None
    assert len(board("theirs")["items"]) == 2          # the person's and the terminal's

    assert run("undo", actor="cli").steps[0].result.label == "board add theirs synth:2"
    assert len(board("theirs")["items"]) == 1
    assert run("undo", actor="agent").steps[0].result.label is None


def test_undo_refuses_to_discard_a_later_change(tmp, ds):
    run('board create "Rain" --id rain', actor="agent")
    run("board add rain synth:0", actor="person")
    b = run("undo", actor="agent")
    assert not b.ok and "has changed since" in b.steps[0].refusal["reason"]
    assert board("rain") is not None


def test_an_agent_changes_only_boards_an_agent_made(tmp, ds):
    run('board create "Mine" --id mine', "board add mine synth:0", actor="person")
    b = run("board note mine synth:0 --text hi", actor="agent")
    assert "board copy mine --id mine-agent" in b.steps[0].refusal["reason"]
    b = run("board copy mine --id mine-agent", "board note mine-agent synth:0 --text hi", actor="agent")
    assert b.ok
    assert board("mine-agent")["author"] == "agent" and board("mine")["author"] == "person"
    assert "note" not in (board("mine")["details"].get(synth_path(0)) or {})


def synth_path(k):
    from arcana import core
    return core.open_dataset("synth").idx2path[k]


def test_the_person_keeps_the_agents_notes_when_saving_the_collection(tmp, ds):
    """The Moodboard tab saves paths only. The agent's sections and notes survive it."""
    run('board create "Agents" --id agents', "board section add agents --id s1 --title Water",
        'board add agents synth:0 synth:1 --section s1 --note "why"', actor="agent")
    p0, p1, p2 = synth_path(0), synth_path(1), synth_path(2)
    b = run(f'board save Agents "{p1}" "{p0}" "{p2}" --id agents', actor="person")
    assert b.ok, b.steps[0].refusal
    rec = board("agents")
    assert rec["items"] == [p1, p0, p2]
    assert rec["details"][p0]["note"] == "why" and rec["sections"][0]["id"] == "s1"
    assert rec["details"][p2]["handle"] == "synth:2"


def test_board_metrics_flag_what_the_design_skill_checks(tmp, ds, monkeypatch):
    # The water pictures sit at cosine ~0.86, so a lower line makes them a pair.
    from arcana import core
    monkeypatch.setattr(core, "NEAR_DUPLICATE", 0.8)
    run('board create "Dup" --id dup', "board add dup synth:0 synth:1 --role hero")
    m = run("board show dup").steps[0].result.metrics
    assert m.near_duplicates == [(1, 2)]
    assert any("2 heroes" in w for w in m.warnings)
    assert any("near-duplicates" in w for w in m.warnings)


def test_derived_pictures_are_marked_on_the_board(tmp, ds):
    from arcana import ledger
    out = os.path.join(tmp, "output", "made.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").close()
    tx = state.STORE.begin("agent", "record")
    entry = ledger.record(tx, out, op="colour transfer", args={}, sources=["synth:3"])
    state.STORE.commit(tx)
    run('board create "D" --id d', f"board add d {entry['handle']}", actor="agent")
    shown = run("board show d").steps[0].result
    item = shown.items[0]
    assert item.kind == "derived" and item.derived_from == "synth:3" and item.handle == entry["handle"]
    assert "DERIVED from synth:3" in shown.text()


def test_settings_set_is_undoable_and_for_the_person(tmp, ds):
    run("settings set --pixels on")
    assert json.load(open(os.path.join(tmp, "agent.json")))["pixels"] is True
    run("undo")
    assert not os.path.exists(os.path.join(tmp, "agent.json"))
    assert not run("settings set --pixels on", actor="agent").ok


def test_undo_runs_on_its_own(tmp, ds):
    b = run('board create "Rain" --id rain', "undo")
    assert not b.ok and "runs on its own" in b.steps[1].refusal["reason"]


def test_every_commit_bumps_the_version_once(tmp, ds):
    seen = []
    state.STORE.subscribe(lambda v, labels: seen.append((v, labels)))
    v0 = state.STORE.version
    run('board create "Rain" --id rain', "board add rain synth:0", "board add rain synth:1")
    assert state.STORE.version == v0 + 1
    run("board show rain")
    assert state.STORE.version == v0 + 1
    assert len([s for s in seen if s[0] > v0]) == 1


def test_the_signature_moves_when_a_board_is_written_by_hand(tmp, ds):
    import time
    before = state.STORE.signature()
    time.sleep(0.01)
    with open(os.path.join(boards.boards_dir(), "hand.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "Hand", "items": []}, fh)
    assert state.STORE.signature() != before


def test_export_copies_and_leaves_the_originals(tmp, ds):
    run('board create "Rain" --id rain', "board add rain synth:0 synth:1")
    mtime = os.path.getmtime(synth_path(0))
    r = run("board export rain").steps[0].result
    assert r.copied == 2 and os.path.isdir(r.folder)
    assert os.path.getmtime(synth_path(0)) == mtime
    assert not run("board export rain --folder ../escape").ok
