# ui_style.py — one set of design tokens for the whole app
#
# Before this, buttons across the moodboard used six unrelated colours
# (#00bcd4, #e040fb, #444, #2a6a4f, #333, #00796b) with no rule behind them, and
# two were full width. Full-width buttons read as banners: they make a panel
# look like a form to work through top-to-bottom rather than a set of choices,
# and they give a minor action the same weight as the main one.
#
# The rule here is that colour means something:
#
#   ACCENT  the single primary action in a card
#   OK      done, healthy, or a save that has somewhere to go
#   WARN    needs attention but is not broken
#   BAD     broken, or destructive
#   neutral everything else -- bordered, not filled
#
# Buttons size to their label. If two buttons sit together, at most one is
# filled.

from __future__ import annotations

# ── palette ──────────────────────────────────────────────────────────────────
INK = "#ececf0"          # primary text
INK_DIM = "#9a9aa2"      # secondary text
INK_FAINT = "#6c6c74"    # captions, disabled

BG = "#121212"           # page
SURFACE = "#1b1b1f"      # card
SURFACE_2 = "#232329"    # input, well
LINE = "#33333b"         # hairline

ACCENT = "#00bcd4"
ACCENT_INK = "#08272c"   # text on a filled accent button
OK = "#43b581"
WARN = "#d99a34"
BAD = "#e05252"
MUTED = "#7f7f8a"

# Per-modality accents, so image and audio datasets are distinguishable at a
# glance without reading the word.
MODALITY = {"image": ACCENT, "audio": "#a982d9"}

# The moodboard's two picture roles. Genuine colour coding: the R and T badges
# are the only place two things must be told apart at a glance, so they get
# their own hues and nothing else in the app reuses them.
ROLE_REF = ACCENT          # the palette source
ROLE_TARGET = "#e05fd8"    # the picture that receives the palette
ROLE_OFF = "#2a3a4a"       # neither role selected


# ── building blocks ──────────────────────────────────────────────────────────
def button(kind: str = "primary", **extra) -> dict:
    """
    A button that sizes to its label.

    kind: primary | secondary | success | danger | ghost
    """
    base = {
        "padding": "8px 15px",
        "borderRadius": "6px",
        "fontSize": "12.5px",
        "fontWeight": "600",
        "cursor": "pointer",
        "border": "1px solid transparent",
        "whiteSpace": "nowrap",
        "width": "auto",
        "flex": "0 0 auto",
        "lineHeight": "1.2",
    }
    if kind == "primary":
        base |= {"backgroundColor": ACCENT, "color": ACCENT_INK}
    elif kind == "success":
        base |= {"backgroundColor": OK, "color": "#06251a"}
    elif kind == "danger":
        base |= {"backgroundColor": "transparent", "color": BAD,
                 "border": f"1px solid {BAD}55"}
    elif kind == "ghost":
        base |= {"backgroundColor": "transparent", "color": INK_DIM,
                 "border": "1px solid transparent"}
    else:                                            # secondary
        base |= {"backgroundColor": "transparent", "color": INK,
                 "border": f"1px solid {LINE}"}
    return base | extra


def input_box(**extra) -> dict:
    return {
        "backgroundColor": SURFACE_2, "color": INK,
        "border": f"1px solid {LINE}", "borderRadius": "6px",
        "padding": "8px 10px", "fontSize": "13px",
    } | extra


def card(**extra) -> dict:
    return {
        "backgroundColor": SURFACE, "border": f"1px solid {LINE}",
        "borderRadius": "10px", "padding": "18px 20px", "marginBottom": "14px",
    } | extra


def section_title(**extra) -> dict:
    return {"fontSize": "13px", "fontWeight": "700", "color": INK,
            "letterSpacing": "0.3px", "marginBottom": "2px"} | extra


def hint(**extra) -> dict:
    return {"fontSize": "12px", "color": INK_DIM, "lineHeight": "1.5",
            "maxWidth": "62ch"} | extra


def field_label(**extra) -> dict:
    return {"fontSize": "11px", "color": INK_FAINT, "marginBottom": "5px",
            "textTransform": "uppercase", "letterSpacing": "0.6px"} | extra


def row(gap: str = "10px", **extra) -> dict:
    """A horizontal group. Buttons in one of these keep their natural width."""
    return {"display": "flex", "alignItems": "center", "gap": gap,
            "flexWrap": "wrap"} | extra
