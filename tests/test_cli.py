"""
The `arcana` command: one script for the app and every op.

main() is called with an argv list against the synthetic dataset, and its
output read from pytest's capture. Starting the app is replaced by a stub, so
nothing here imports Dash.

Run with:  python -m pytest tests/test_cli.py -q
"""

import io
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from arcana import cli  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    return synth.use(tmp, monkeypatch)


def run(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_no_arguments_starts_the_app(monkeypatch):
    started = []
    monkeypatch.setattr(cli, "_start_app", lambda: started.append(True) or 0)
    assert cli.main([]) == 0 and started == [True]


def test_the_arcana_script_is_the_cli():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert re.search(r'^arcana = "arcana\.cli:main"$', text, re.M)


def test_an_op_line_runs_and_prints_its_result(tmp, ds, capsys):
    code, out, err = run(capsys, "search", "text", "synth", "--phrase", "water", "--k", "3")
    assert code == 0 and not err
    assert "## water — strong" in out
    assert {int(k) for k in re.findall(r"synth:(\d+)", out)} <= set(range(synth.N_WATER))
    assert len(re.findall(r"^#\d+  synth:", out, re.M)) == 3


def test_a_refusal_goes_to_stderr_with_the_ops_help_and_exits_1(tmp, ds, capsys):
    code, out, err = run(capsys, "search", "text", "synth", "--phrase", "water", "--k", "500")
    assert code == 1 and not out
    assert err.startswith("Refused: --k must be between 1 and 48, got 500.")
    assert "search text <dataset>" in err


def test_json_output_is_the_result_model(tmp, ds, capsys):
    code, out, _ = run(capsys, "--json", "search", "text", "synth", "--phrase", "water", "--k", "2")
    data = json.loads(out)
    assert code == 0 and data["op"] == "search text"
    assert [h["handle"] for h in data["result"]["facets"][0]["hits"]][0].startswith("synth:")

    code, out, _ = run(capsys, "--json", "search", "text", "nope", "--phrase", "water")
    assert code == 1 and "Known datasets" in json.loads(out)["refused"]["reason"]


def test_sheets_are_written_to_output_and_named(tmp, ds, capsys):
    code, out, _ = run(capsys, "--sheets", "search", "text", "synth", "--phrase", "water", "--k", "4")
    paths = re.findall(r"^Sheet: (.+)$", out, re.M)
    assert code == 0 and len(paths) == 1
    assert os.path.dirname(paths[0]) == os.path.join(tmp, "output", "sheets")
    assert open(paths[0], "rb").read(2) == b"\xff\xd8"


def test_item_sheet_writes_its_sheet_without_being_asked(tmp, ds, capsys):
    code, out, _ = run(capsys, "item", "sheet", "synth:1", "synth:2", "synth:3")
    assert code == 0 and len(re.findall(r"^Sheet: ", out, re.M)) == 1


def test_a_search_without_sheets_writes_nothing(tmp, ds, capsys):
    before = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    run(capsys, "search", "text", "synth", "--phrase", "water")
    after = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    assert after == before


def test_help_at_every_level(tmp, ds, capsys):
    code, out, _ = run(capsys, "--help")
    assert code == 0 and "Learn it in three steps" in out and "One line is one op" in out
    code, out, _ = run(capsys, "search", "like", "--help")
    assert code == 0 and out.startswith("search like <handles>...")
    code, out, _ = run(capsys, "op", "list")
    assert code == 0 and "search text" in out and "item describe" in out


def test_a_batch_runs_in_order_and_stops_at_a_refusal(tmp, ds, capsys, monkeypatch):
    lines = os.path.join(tmp, "lines.txt")
    with open(lines, "w", encoding="utf-8") as fh:
        fh.write("dataset list\n# a comment\nsearch text nope --phrase water\ndataset survey synth\n")
    code, out, _ = run(capsys, "--batch", lines)
    assert code == 1
    assert "[1] dataset list" in out and "Line 2 was refused" in out and "(not run)" in out

    monkeypatch.setattr(sys, "stdin", io.StringIO("dataset list\nsearch text synth --phrase water\n"))
    code, out, _ = run(capsys, "--batch", "-")
    assert code == 0 and "## water" in out


def test_completion_from_the_command_line(tmp, ds, capsys):
    code, out, _ = run(capsys, "complete", "dataset survey ")
    assert code == 0 and out.split() == ["synth"]
    code, out, _ = run(capsys, "complete", "search like synth:1 --mode ")
    assert out.split() == ["semantic", "palette", "style"]
    code, out, _ = run(capsys, "completion", "bash")
    assert code == 0 and "_arcana_complete" in out


def test_an_unknown_global_option_shows_usage(capsys):
    code, _, err = run(capsys, "--frobnicate")
    assert code == 2 and "Usage" in err


def test_every_line_in_the_readmes_command_block_is_an_op():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    block = text.split("## Command line", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    lines = [l.split("  #")[0].strip() for l in block.splitlines()]
    lines = [l[len("arcana "):] for l in lines if l.startswith("arcana ")]
    assert len(lines) >= 10
    from arcana import ops
    for line in lines:
        ops.parse_line(line)
