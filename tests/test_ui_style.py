"""
One palette, declared once.

arcana/ui_style.py is where the Python side gets its colours, and
frontend/src/styles/tokens.css carries the same values as CSS variables for the
Svelte page because a stylesheet cannot read Python. These tests keep the two from drifting apart,
and keep the hues out of the component code, where they used to be: sixteen
copies of #00bcd4 and five of a magenta that ui_style.py never defined.
"""

import os
import re

from arcana import ui_style as ui

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "arcana")
CSS = os.path.join(os.path.dirname(HERE), "frontend", "src", "styles", "tokens.css")

# CSS variable -> token. Both are the palette; this is the mapping.
MIRROR = {
    "--arc-bg": ui.BG,
    "--arc-surface": ui.SURFACE,
    "--arc-surface-2": ui.SURFACE_2,
    "--arc-line": ui.LINE,
    "--arc-ink": ui.INK,
    "--arc-ink-dim": ui.INK_DIM,
    "--arc-ink-faint": ui.INK_FAINT,
    "--arc-accent": ui.ACCENT,
    "--arc-target": ui.ROLE_TARGET,
    "--arc-ok": ui.OK,
    "--arc-warn": ui.WARN,
    "--arc-bad": ui.BAD,
}

# The hues. Neutral greys may still sit in component code; these may not.
HUES = {ui.ACCENT, ui.ROLE_TARGET, ui.OK, ui.WARN, ui.BAD, ui.ROLE_REF}


def _css_vars():
    text = open(CSS, encoding="utf-8").read()
    root = re.search(r":root\s*\{(.*?)\}", text, re.S).group(1)
    return dict(re.findall(r"(--arc-[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", root))


def test_the_stylesheet_mirrors_the_tokens():
    found = _css_vars()
    for var, want in MIRROR.items():
        assert found.get(var, "").lower() == want.lower(), (var, found.get(var), want)


def test_the_page_components_use_tokens_not_hues():
    """A Svelte component names a colour through var(--arc-*), never by its hex."""
    src = os.path.join(os.path.dirname(HERE), "frontend", "src")
    hex_re = re.compile(r"#[0-9a-fA-F]{6}")
    hues = {h.lower() for h in HUES}
    leaked = {}
    for root, _dirs, files in os.walk(src):
        for name in files:
            if not name.endswith(".svelte") or root.endswith("styles"):
                continue
            text = open(os.path.join(root, name), encoding="utf-8").read()
            found = {h.lower() for h in hex_re.findall(text)} & hues
            if found:
                leaked[name] = sorted(found)
    assert not leaked, f"components spell out palette hues: {leaked}; use the tokens"


def test_no_hue_is_spelled_out_in_component_code():
    hex_re = re.compile(r"#[0-9a-fA-F]{6}\b")
    for name in ("arcana.py", "ui_datasets.py"):
        text = open(os.path.join(PKG, name), encoding="utf-8").read()
        # The ON1 colour-tag swatches are the tags' own colours, not ours.
        text = re.sub(r"COLOUR_SWATCH = \{.*?\}", "", text, flags=re.S)
        found = {h.lower() for h in hex_re.findall(text)}
        leaked = found & {h.lower() for h in HUES}
        assert not leaked, f"{name} spells out palette hues {sorted(leaked)}; use ui_style tokens"


def test_the_roles_are_distinct_and_distinct_from_feedback_colours():
    assert ui.ROLE_REF.lower() != ui.ROLE_TARGET.lower()
    assert ui.ROLE_TARGET.lower() not in {ui.OK.lower(), ui.BAD.lower(), ui.WARN.lower()}
