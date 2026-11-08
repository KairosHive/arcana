"""
Boards as HTML and as one picture.

An HTML board is one file that references every picture on the board, marks
derived ones, carries the palette and a reason for each section, and opens
where the original paths are gone. Its PNG is as wide as asked.

Run with:  python -m pytest tests/test_compose.py -q
"""

import os
import sys
from html.parser import HTMLParser

import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import compose, ops, state  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def board(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_NO_BROWSER", "1")
    idx = synth.use(tmp, monkeypatch)
    from arcana import ledger
    made = os.path.join(tmp, "output", "made.png")
    os.makedirs(os.path.dirname(made), exist_ok=True)
    Image.new("RGB", (40, 30), (200, 120, 60)).save(made)
    tx = state.STORE.begin("cli", "seed")
    entry = ledger.record(tx, made, op="colour transfer", args={}, sources=["synth:3"],
                          derived_from="synth:3")
    state.STORE.commit(tx)
    b = ops.exec_lines([
        'board create "Water & <wood>" --id b --concept "Water against wood."',
        'board section add b --id s1 --title Water --rationale "Where it starts."',
        'board section add b --id s2 --title Wood --rationale "What it meets."',
        'board add b synth:0 --section s1 --role hero --note "the one"',
        'board add b synth:1 synth:2 --section s1',
        'board add b synth:5 synth:6 --section s2 --role texture',
        f"board add b {entry['handle']} --section s2",
        "board add b synth:30",
        "board palette set b --colour '#223344:0.6:ground' --colour '#aa7733:0.4:accent'",
    ], "cli")
    assert b.ok, b.refused and b.refused.refusal
    return idx, entry


class _Parsed(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figures, self.imgs, self.classes, self.text = 0, [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "figure":
            self.figures += 1
            self.classes.append(a.get("class", ""))
        if tag == "img" and "data-full" in a:      # the lightbox's own img has none
            self.imgs.append(a)

    def handle_data(self, data):
        self.text.append(data)


def _render(layout="editorial"):
    shown = ops.call("board show", {"board": "b"})
    return shown, compose.to_html(shown, layout)


def test_the_page_carries_every_picture_and_marks_the_derived_one(tmp, board):
    shown, page = _render()
    p = _Parsed()
    p.feed(page)
    assert p.figures == len(shown.items) == 7
    assert all(i["src"].startswith("data:image/jpeg;base64,") for i in p.imgs)
    assert all(i["data-full"].startswith("file:///") for i in p.imgs)
    assert sum("kind-derived" in c for c in p.classes) == 1
    text = " ".join(p.text)
    assert "DERIVED" in text and "from synth:3" in text
    assert "Where it starts." in text and "What it meets." in text
    assert "#223344" in text and "Water against wood." in text


def test_names_are_escaped(tmp, board):
    _, page = _render()
    assert "Water &amp; &lt;wood&gt;" in page and "<wood>" not in page


def test_each_layout_is_a_class_on_the_body(tmp, board):
    for layout in compose.LAYOUTS:
        _, page = _render(layout)
        assert f'class="layout-{layout}"' in page
    with pytest.raises(ValueError):
        _render("collage")


def test_the_hero_leads_the_editorial_page_once(tmp, board):
    _, page = _render("editorial")
    assert page.count('class="item role-hero hero"') == 1
    assert page.index("role-hero") < page.index('id="s1"')
    _, page = _render("grid")
    assert page.count("role-hero") == 1


def test_a_missing_file_leaves_a_placeholder(tmp, board):
    idx, _ = board
    os.remove(idx[30])
    _, page = _render()
    assert "file missing" in page


def test_the_png_is_as_wide_as_asked_and_render_writes_both(tmp, board):
    shown = ops.call("board show", {"board": "b"})
    img = compose.to_png(shown, 1200)
    assert img.width == 1200 and img.height > 400
    html_res, png_res = [s.result for s in ops.exec_lines(
        ["board render b --layout masonry", "board render b --format png --width 900"]).steps]
    assert html_res.path.endswith(os.path.join("boards", "b.html")) and os.path.isfile(html_res.path)
    assert Image.open(png_res.path).width == 900
    assert png_res.attachments and png_res.attachments[0].mime == "image/png"


def test_render_writes_nothing_but_the_board_files(tmp, board):
    before = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    ops.call("board render", {"board": "b"})
    after = {os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs}
    assert {os.path.relpath(p, tmp) for p in after - before} == {os.path.join("output", "boards", "b.html")}


def test_open_is_for_the_person(tmp, board):
    assert ops.call("board open", {"board": "b"}, actor="cli").path.endswith("b.html")
    with pytest.raises(ops.Refusal, match="for the person"):
        ops.call("board open", {"board": "b"}, actor="agent")
