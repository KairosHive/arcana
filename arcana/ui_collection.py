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
# layout() gives the header controls and the stores; register(app) the
# callbacks. add(), remove() and clear() are what other modules call.

from __future__ import annotations

import time

from dash import Input, Output, State, ctx, dcc, html, no_update

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
            html.Label("Collection:", style={"marginLeft": "28px", "marginRight": "6px"}),
            dcc.Dropdown(id="collection-picker", options=[], value=None, clearable=False,
                         placeholder="Choose a collection...",
                         style={"width": "220px", "display": "inline-block", "verticalAlign": "middle",
                                "color": "#000"}),
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
        ],
        style={"display": "flex", "alignItems": "center"},
    )


# ─────────────────────────── callbacks ───────────────────────────

def register(app) -> None:

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
