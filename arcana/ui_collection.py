# ui_collection.py -- the active collection, shared by every tab
#
# A collection is a board. The person picks the active one in the header, and
# that choice is a setting (`settings set --collection`), so every tab, every
# port the app is served on, the terminal and agents see the same pictures.
# "+ Collection" anywhere adds to it with `board add`; removing and clearing are
# `board remove`. Nothing about the collection lives in the browser.
#
# One owner: `_sync` writes collection-items (the paths the Moodboard rail and
# Alchemy show) and board-details (their badges) from the active board. Whatever
# changes the board bumps collection-rev, so the sync runs at once; a change
# from elsewhere (a terminal, an agent, the app on another port) arrives with
# the state signature the app already polls every two seconds.
#
# The viewer steps through the active collection one picture at a time, large,
# with its role, section and note. A picture anywhere with the id
# {"type": "collection-open", "index": <path>, "from": <where>} opens it there,
# `from` keeping the ids of two tabs' tiles apart; the arrow keys step and
# Escape closes (assets/collection.js).
#
# layout() gives the header controls, the stores and the viewer; register(app)
# the callbacks. add(), remove() and clear() are what other modules call.

from __future__ import annotations

import os
import time
import urllib.parse

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

try:
    from . import ops as _ops
    from . import ui_style as _ui
except ImportError:                                   # pragma: no cover
    import ops as _ops
    import ui_style as _ui

DEFAULT_NAME = "Collection"


# ─────────────────────────── the active collection ───────────────────────────

def active() -> str | None:
    """The active board's id, or None when none is chosen or it has been deleted."""
    slug = _ops.archive.settings()["collection"]
    if not slug:
        return None
    try:
        _ops.call("board show", {"board": slug}, actor="person")
    except _ops.Refusal:
        return None
    return slug


def use(slug: str | None) -> None:
    _ops.call("settings set", {"collection": slug or ""}, actor="person")


def ensure_active() -> str:
    """The active board, made and chosen first when there is none."""
    slug = active()
    if slug:
        return slug
    boards = {b.slug: b for b in _ops.call("board list", actor="person").boards}
    slug = "collection"
    if slug not in boards:
        slug = _ops.call("board create", {"name": DEFAULT_NAME, "id": slug}, actor="person").board.slug
    use(slug)
    return slug


def contents(slug: str | None) -> tuple[list[str], dict, str]:
    """(paths of the pictures present, their badges, the board's name) for a board."""
    if not slug:
        return [], {}, ""
    try:
        rec = _ops.call("board show", {"board": slug}, actor="person")
    except _ops.Refusal:
        return [], {}, ""
    here = [i.path for i in rec.items if not i.missing]
    details = {i.path: {"role": i.role, "kind": i.kind, "section": i.section, "note": i.note,
                        "derived_from": i.derived_from} for i in rec.items}
    return here, details, rec.name


def add(refs: list[str]) -> str:
    """Put pictures (paths or handles) on the active collection. Returns its id."""
    slug = ensure_active()
    refs = [r for r in refs if r]
    if refs:
        _ops.call("board add", {"board": slug, "items": refs}, actor="person")
    return slug


def remove(refs: list[str]) -> None:
    slug = active()
    refs = [r for r in refs if r]
    if slug and refs:
        _ops.call("board remove", {"board": slug, "items": refs}, actor="person")


def clear() -> None:
    slug = active()
    if slug:
        remove(contents(slug)[0])


def rev() -> float:
    """A new value for collection-rev: whoever changed the collection returns one."""
    return time.time()


# ─────────────────────────── layout ───────────────────────────

def layout() -> html.Div:
    """The header controls: which collection is active, a new one, and the import of pictures an
    older version of the app kept in this browser."""
    return html.Div(
        [
            html.Span("Collection", style=_ui.field_label(marginBottom="0", marginRight="10px",
                                                         color=_ui.INK_DIM)),
            dcc.Dropdown(id="collection-picker", options=[], value=None, clearable=False,
                         placeholder="Choose a collection...",
                         style={"width": "260px", "color": "#000"}),
            dcc.Input(id="collection-new", type="text", debounce=True, placeholder="+ new collection",
                      style=_ui.input_box(width="150px", padding="6px 9px", fontSize="12px", marginLeft="8px")),
            html.Button(id="collection-import", n_clicks=0, style={"display": "none"}),
            html.Span(id="collection-status", style=_ui.hint(fontSize="11.5px", marginLeft="10px",
                                                             maxWidth="40ch")),
            dcc.Store(id="collection-items", storage_type="memory"),
            dcc.Store(id="collection-rev", storage_type="memory"),
            # What an older version of the app kept in this browser, under its old key. Read once,
            # to offer the import; emptied when imported.
            dcc.Store(id="moodboard-store", storage_type="local"),
            dcc.Store(id="collection-view", storage_type="memory"),
            _viewer(),
        ],
        style={"display": "flex", "alignItems": "center"},
    )


ARROW = {"border": "none", "background": "rgba(20, 19, 17, 0.55)", "color": _ui.INK, "fontSize": "28px",
         "width": "52px", "height": "52px", "borderRadius": "50%", "cursor": "pointer", "flex": "0 0 auto"}


def _viewer() -> html.Div:
    """The collection, one picture at a time over the page."""
    return html.Div(
        [
            html.Div([
                html.Span(id="collection-view-where", style={"fontSize": "13px", "color": _ui.INK}),
                html.Div(style={"flex": "1"}),
                html.Button("×", id="collection-view-close", n_clicks=0, title="Close (Esc)",
                            style={**ARROW, "width": "40px", "height": "40px", "fontSize": "24px"}),
            ], style={"display": "flex", "alignItems": "center", "padding": "14px 20px"}),
            html.Div([
                html.Button("‹", id="collection-view-prev", n_clicks=0, title="Previous (←)", style=ARROW),
                html.Img(id="collection-view-img", alt="",
                         style={"maxWidth": "calc(100vw - 180px)", "maxHeight": "calc(100vh - 190px)",
                                "objectFit": "contain", "borderRadius": "6px", "display": "block"}),
                html.Button("›", id="collection-view-next", n_clicks=0, title="Next (→)", style=ARROW),
            ], style={"flex": "1", "minHeight": "0", "display": "flex", "alignItems": "center",
                      "justifyContent": "center", "gap": "24px"}),
            html.Div(id="collection-view-caption",
                     style=_ui.hint(fontSize="13px", color=_ui.INK, textAlign="center", maxWidth="80ch",
                                    margin="10px auto 20px", lineHeight="1.5")),
        ],
        id="collection-viewer", className="collection-viewer",
        style={"display": "none"},
    )


# ─────────────────────────── callbacks ───────────────────────────

def register(app) -> None:
    register_viewer(app)

    @app.callback(Output("collection-picker", "options"), Output("collection-picker", "value"),
                  Input("state-signature", "data"), Input("collection-rev", "data"))
    def _picker(_sig, _rev):
        opts = []
        for b in _ops.call("board list", actor="person").boards:
            bits = f"{b.count} picture" + ("s" if b.count != 1 else "")
            opts.append({"label": f"{b.name}  ·  {bits}", "value": b.slug})
        return opts, active()

    @app.callback(Output("collection-rev", "data", allow_duplicate=True),
                  Output("collection-status", "children"),
                  Input("collection-picker", "value"), prevent_initial_call=True)
    def _pick(slug):
        if not slug or slug == active():
            return no_update, no_update
        try:
            use(slug)
        except _ops.Refusal as e:
            return no_update, e.reason
        return rev(), ""

    @app.callback(Output("collection-rev", "data", allow_duplicate=True),
                  Output("collection-status", "children", allow_duplicate=True),
                  Output("collection-new", "value"),
                  Input("collection-new", "value"), prevent_initial_call=True)
    def _new(name):
        name = (name or "").strip()
        if not name:
            return no_update, no_update, no_update
        try:
            slug = _ops.call("board create", {"name": name}, actor="person").board.slug
            use(slug)
        except _ops.Refusal as e:
            return no_update, e.reason, no_update
        return rev(), f"“{name}” is the active collection.", ""

    @app.callback(Output("collection-items", "data"), Output("board-details", "data"),
                  Input("collection-rev", "data"), Input("state-signature", "data"),
                  State("collection-items", "data"))
    def _sync(_rev, _sig, shown):
        items, details, _ = contents(active())
        if items == shown and ctx.triggered_id == "state-signature":
            return no_update, no_update
        return items, details

    @app.callback(Output("collection-import", "children"), Output("collection-import", "style"),
                  Input("moodboard-store", "data"))
    def _offer_import(old):
        old = [p for p in (old or []) if isinstance(p, str)]
        if not old:
            return "", {"display": "none"}
        return (f"Import {len(old)} picture{'s' if len(old) != 1 else ''} this browser kept",
                _ui.button("secondary", padding="4px 10px", fontSize="11.5px", marginLeft="8px"))

    @app.callback(Output("collection-rev", "data", allow_duplicate=True),
                  Output("collection-status", "children", allow_duplicate=True),
                  Output("moodboard-store", "data"),
                  Input("collection-import", "n_clicks"), State("moodboard-store", "data"),
                  prevent_initial_call=True)
    def _import(n, old):
        old = [p for p in (old or []) if isinstance(p, str)]
        if not n or not old:
            return no_update, no_update, no_update
        try:
            slug = add(old)
        except _ops.Refusal as e:
            return no_update, f"Could not import: {e.reason}", no_update
        return rev(), f"Imported {len(old)} into {slug}.", []


# ─────────────────────────── the viewer ───────────────────────────

VIEWER_OPEN = {"display": "flex", "flexDirection": "column", "position": "fixed", "inset": "0", "zIndex": 1000,
               "backgroundColor": "rgba(14, 13, 12, 0.94)"}


def _caption(path: str, d: dict) -> list:
    bits = []
    if d.get("kind") in ("derived", "generated"):
        bits.append(html.Span(d["kind"].upper(), style={"color": _ui.WARN, "fontWeight": "700",
                                                         "letterSpacing": "0.6px", "fontSize": "11px"}))
    for key in ("role", "section"):
        if d.get(key):
            bits.append(html.Span(d[key], style={"color": _ui.INK_DIM}))
    joined = []
    for b in bits:
        if joined:
            joined.append(html.Span("  ·  ", style={"color": _ui.INK_FAINT}))
        joined.append(b)
    head = html.Div(joined, style={"marginBottom": "4px"}) if joined else None
    return [head, html.Div(d.get("note") or ""),
            html.Div(os.path.basename(path), style={"color": _ui.INK_FAINT, "fontSize": "11.5px", "marginTop": "4px"})]


def register_viewer(app) -> None:

    @app.callback(Output("collection-view", "data"),
                  Input({"type": "collection-open", "index": ALL, "from": ALL}, "n_clicks"),
                  Input("collection-view-prev", "n_clicks"), Input("collection-view-next", "n_clicks"),
                  Input("collection-view-close", "n_clicks"),
                  State("collection-view", "data"), State("collection-items", "data"),
                  prevent_initial_call=True)
    def _step(_opens, _prev, _next, _close, at, items):
        hit = ctx.triggered_id
        items = items or []
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update                     # tiles redrawn with n_clicks 0, not a click
        if hit == "collection-view-close" or not items:
            return None
        if isinstance(hit, dict):
            return items.index(hit["index"]) if hit["index"] in items else no_update
        if at is None:
            return no_update
        step = -1 if hit == "collection-view-prev" else 1
        return (int(at) + step) % len(items)

    @app.callback(Output("collection-viewer", "style"), Output("collection-view-img", "src"),
                  Output("collection-view-where", "children"), Output("collection-view-caption", "children"),
                  Input("collection-view", "data"), Input("collection-items", "data"),
                  State("board-details", "data"))
    def _show(at, items, details):
        items = items or []
        if at is None or not items:
            return {"display": "none"}, no_update, "", ""
        at = min(int(at), len(items) - 1)
        path = items[at]
        slug = active()
        where = f"{at + 1} of {len(items)}" + (f"  ·  {slug}" if slug else "")
        return (VIEWER_OPEN, "/preview?w=2048&q=88&p=" + urllib.parse.quote(path), where,
                _caption(path, (details or {}).get(path) or {}))
