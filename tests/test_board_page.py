"""
A board's own designed page, and boards as PDF.

`board page set` copies a self-contained HTML page beside the rendered boards
and records it on the board, under the same rules as every board write.
`board render` never overwrites it, `board page clear` detaches it and keeps
the file, and `board pdf` prints it, or the rendered board, with a headless
Chromium.

Run with:  python -m pytest tests/test_board_page.py -q
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import boards, ops, paths, workspace  # noqa: E402
from arcana.ops import board as board_ops  # noqa: E402

import synth  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Disintegration</title>
<style>@page { size: A4 landscape; margin: 0 } body { margin: 0; background: #eee }
.spread { height: 100vh; break-inside: avoid; break-after: page; display: grid; place-items: center }
</style></head><body>
<section class="spread"><h1>one</h1></section><section class="spread"><h1>two</h1></section>
<script>document.body.dataset.ready = '1'</script></body></html>"""


@pytest.fixture
def ds(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_NO_BROWSER", "1")
    return synth.use(tmp, monkeypatch)


def run(*lines, actor="cli"):
    return ops.exec_lines(list(lines), actor)


def write_page(where):
    os.makedirs(os.path.dirname(where), exist_ok=True)
    with open(where, "w", encoding="utf-8") as fh:
        fh.write(PAGE)
    return where


def test_a_board_carries_its_page_and_render_leaves_it(tmp, ds):
    src = write_page(os.path.join(workspace.workspace_dir(), "drafts", "d.html"))
    b = run('board create "Disintegration" --id disintegration', "board add disintegration synth:0",
            f'board page set disintegration "{src}"', actor="agent")
    assert b.ok, b.refused and b.refused.refusal
    dest = board_ops.page_path("disintegration")
    assert open(dest, encoding="utf-8").read() == PAGE
    assert boards.load("disintegration")["page"]["source"] == "d.html"
    assert run("board show disintegration").steps[0].result.page == dest
    run("board render disintegration")
    assert open(dest, encoding="utf-8").read() == PAGE
    assert os.path.isfile(os.path.join(os.path.dirname(dest), "disintegration.html"))


def test_a_relative_path_is_read_beside_output(tmp, ds):
    write_page(board_ops.page_path("rel"))
    b = run('board create "Rel" --id rel', "board page set rel output/boards/rel.page.html")
    assert b.ok, b.refused and b.refused.refusal


def test_an_agent_hands_in_pages_from_its_workspace_or_output_only(tmp, ds):
    outside = write_page(os.path.join(tmp, "elsewhere", "p.html"))
    run('board create "A" --id a', actor="agent")
    reason = run(f'board page set a "{outside}"', actor="agent").refused.refusal["reason"]
    assert "workspace" in reason
    assert run(f'board page set a "{outside}"', actor="cli").ok      # the terminal may


def test_an_agent_cannot_put_a_page_on_the_persons_board(tmp, ds):
    src = write_page(os.path.join(workspace.workspace_dir(), "p.html"))
    run('board create "Mine" --id mine', actor="person")
    reason = run(f'board page set mine "{src}"', actor="agent").refused.refusal["reason"]
    assert "board copy mine" in reason
    assert not os.path.exists(board_ops.page_path("mine"))


def test_only_html_and_existing_files(tmp, ds):
    run('board create "B" --id b')
    assert "No file" in run('board page set b "nope.html"').refused.refusal["reason"]
    txt = os.path.join(tmp, "x.txt")
    open(txt, "w").close()
    assert ".html" in run(f'board page set b "{txt}"').refused.refusal["reason"]


def test_clear_detaches_and_keeps_the_file_and_undo_restores(tmp, ds):
    src = write_page(os.path.join(tmp, "p.html"))
    run('board create "C" --id c', f'board page set c "{src}"')
    assert run("board page clear c").ok
    assert "page" not in boards.load("c") and os.path.isfile(board_ops.page_path("c"))
    assert board_ops.page_file("c") is None
    run("undo")
    assert board_ops.page_file("c") == board_ops.page_path("c")


def test_pdf_prints_the_page_when_there_is_one(tmp, ds, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = next(c.split("=", 1)[1] for c in cmd if c.startswith("--print-to-pdf="))
        with open(out, "wb") as fh:
            fh.write(b"%PDF-1.4 /Type /Pages /Type /Page /Type /Page")
    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(board_ops, "chromium", lambda: "chrome.exe")
    src = write_page(os.path.join(tmp, "p.html"))
    run('board create "P" --id p', "board add p synth:0", f'board page set p "{src}"')
    r = run("board pdf p").steps[0].result
    assert r.source == "designed page" and r.pages == 2 and r.path.endswith("p.pdf")
    assert calls[0][-1].endswith("p.page.html")
    run("board page clear p")
    r = run("board pdf p --layout grid").steps[0].result
    assert r.source == "grid layout" and calls[1][-1].endswith("/p.html")


def test_pdf_says_when_there_is_no_browser(tmp, ds, monkeypatch):
    monkeypatch.setattr(board_ops, "chromium", lambda: None)
    run('board create "Q" --id q')
    assert "No Chrome or Edge" in run("board pdf q").refused.refusal["reason"]


@pytest.mark.skipif(board_ops.chromium() is None, reason="no Chrome or Edge on this machine")
def test_a_real_print_follows_the_pages_own_print_css(tmp, ds):
    src = write_page(os.path.join(tmp, "p.html"))
    run('board create "Real" --id real', f'board page set real "{src}"')
    r = run("board pdf real --wait 1").steps[0].result
    assert r.pages == 2
    assert open(r.path, "rb").read(5) == b"%PDF-"
