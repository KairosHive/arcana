# ui_archive.py -- the Archive tab
#
# Everything the person has kept, as one tree (arcana/catalogue.py): collections, films, videos,
# stories, pictures made, search saves, imports and exports. The tree is on the left; a row is
# dragged onto a folder (or a collection onto another collection) to move it there (assets/
# archive.js). The middle plays or shows the selection: a video plays, a story or a saved search
# runs as a slideshow, a collection shows its pictures and opens in the collection viewer, a film
# opens in Alchemy's Video subtab. The right side renames, moves, merges, makes folders, deletes
# into the trash and restores.
#
# Every change is an `archive` op, so the terminal and agents organise the same tree.

from __future__ import annotations

import os
import time
import urllib.parse

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

try:
    from . import catalogue as _cat
    from . import ops as _ops
    from . import ui_style as _ui
except ImportError:                                   # pragma: no cover
    import catalogue as _cat
    import ops as _ops
    import ui_style as _ui

SMALL = {"fontSize": "12px"}
LABEL = _ui.field_label(marginBottom="4px", marginTop="10px")
PANEL_VISIBLE = {"display": "flex", "gap": "14px", "flex": "1 1 auto", "minWidth": "0", "minHeight": "0",
                 "flexWrap": "wrap", "alignContent": "flex-start", "overflowY": "auto"}
ICON = {"collections": "▦", "films": "▶", "videos": "▶", "stories": "≡",
        "pictures": "▣", "searches": "⌕", "imports": "↓", "exports": "↑"}


def _media(path, route="preview", w=None):
    q = "p=" + urllib.parse.quote(path)
    return f"/{route}?{q}" + (f"&w={w}" if w else "")


def _call(name, args):
    try:
        return _ops.call(name, args, actor="person"), None
    except _ops.Refusal as e:
        return None, e.reason


# ─────────────────────────── layout ───────────────────────────

def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="arc-sel", storage_type="memory"),
            dcc.Store(id="arc-rev", storage_type="memory"),
            dcc.Store(id="arc-drop", storage_type="memory"),
            dcc.Store(id="arc-frame", storage_type="memory", data=0),
            dcc.Store(id="arc-empty-armed", storage_type="memory"),
            dcc.Interval(id="arc-play", interval=2500, disabled=True),
            html.Div([
                html.Div("Archive", style=_ui.section_title()),
                html.Div("Everything kept, by kind. Drag a row onto a folder, or a collection onto another "
                         "collection, to move it.", style=_ui.hint(fontSize="11.5px")),
                html.Div(id="arc-tree", className="arc-tree"),
            ], style={"flex": "0 0 330px", "minWidth": "0", "minHeight": "0", "maxHeight": "100%", "overflowY": "auto",
                      **_ui.card(padding="12px", marginBottom="0")}),
            html.Div(id="arc-preview", style={"flex": "1 1 460px", "minWidth": "min(460px, 100%)", "minHeight": "0",
                                              "overflowY": "auto", **_ui.card(padding="12px", marginBottom="0")}),
            html.Div([
                html.Div(id="arc-title", style=_ui.section_title()),
                html.Div(id="arc-where", style=_ui.hint(fontSize="11px", maxWidth="none", wordBreak="break-all")),
                html.Div("NAME", style=LABEL),
                html.Div([
                    dcc.Input(id="arc-name", type="text", debounce=False,
                              style=_ui.input_box(flex="1", minWidth="0", padding="5px 8px", fontSize="12px")),
                    html.Button("Rename", id="arc-rename", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div("MOVE INTO", style=LABEL),
                html.Div([
                    dcc.Dropdown(id="arc-move-to", placeholder="a folder...", style={**SMALL, "flex": "1", "minWidth": "0"}),
                    html.Button("Move", id="arc-move", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div(id="arc-merge-wrap", children=[
                    html.Div("MERGE INTO", style=LABEL),
                    html.Div([
                        dcc.Dropdown(id="arc-merge-to", placeholder="another of the same kind...",
                                     style={**SMALL, "flex": "1", "minWidth": "0"}),
                        html.Button("Merge", id="arc-merge", n_clicks=0,
                                    style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                    ], style={"display": "flex", "gap": "6px"}),
                ]),
                html.Div("NEW FOLDER HERE", style=LABEL),
                html.Div([
                    dcc.Input(id="arc-folder-name", type="text", placeholder="folder name...",
                              style=_ui.input_box(flex="1", minWidth="0", padding="5px 8px", fontSize="12px")),
                    html.Button("Make", id="arc-folder", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div([
                    html.Button("Delete", id="arc-delete", n_clicks=0,
                                style=_ui.button("danger", padding="5px 11px", fontSize="12px")),
                    html.Span("Deleting moves it to the trash, where Restore brings it back.",
                              style=_ui.hint(fontSize="11px")),
                ], style=_ui.row("8px", marginTop="14px")),
                html.Div(id="arc-status", style=_ui.hint(fontSize="12px", marginTop="10px", color=_ui.INK)),
                html.Div([
                    html.Div("TRASH", style=LABEL),
                    html.Div(id="arc-trash"),
                    html.Button("Empty the trash", id="arc-empty", n_clicks=0,
                                style=_ui.button("ghost", padding="4px 8px", fontSize="11.5px", marginTop="6px")),
                ], style={"marginTop": "14px", "borderTop": f"1px solid {_ui.LINE}", "paddingTop": "6px"}),
            ], style={"flex": "0 0 320px", "maxWidth": "100%", "minWidth": "0", **_ui.card(padding="12px", marginBottom="0")}),
        ],
        id="archive-panel", style={"display": "none"},
    )


# ─────────────────────────── the tree ───────────────────────────

def _row(node: dict, sel: str | None, depth: int, drop: bool = False) -> html.Div:
    kind = node["kind"]
    count = node.get("count") or 0
    label = [html.Span(ICON.get(kind, "▪"), className="arc-icon"), html.Span(node["shown"], className="arc-name")]
    if count and node["id"].split(":", 1)[0] in ("collection", "film", "dir"):
        label.append(html.Span(str(count), className="arc-count"))
    return html.Div(label, id={"type": "arc-node", "id": node["id"]}, n_clicks=0, draggable="true",
                    className="arc-row" + (" sel" if node["id"] == sel else "") + (" arc-drop-target" if drop else ""),
                    **{"data-id": node["id"], "data-kind": kind},
                    style={"paddingLeft": f"{8 + 14 * depth}px"})


def _branch(node: dict, sel: str | None, depth: int, open_: bool) -> list:
    """A kind, a folder or a collection with what is inside it."""
    out = []
    kids = []
    for f in sorted(node.get("folders") or [], key=lambda f: f["shown"].lower()):
        kids += _branch(f, sel, depth + 1, open_=False)
    items = node.get("items") or []
    groups: dict = {}
    if node["id"] == "kind:pictures":
        for it in items:
            groups.setdefault(it.get("group") or "", []).append(it)
    else:
        groups[""] = items
    for g, its in sorted(groups.items()):
        rows = []
        for it in its:
            if it.get("folders") or it.get("items"):
                rows += _branch(it, sel, depth + 1 + (1 if g else 0), open_=False)
            else:
                rows.append(_row(it, sel, depth + 1 + (1 if g else 0), drop=it["kind"] == "collections"))
        if g:
            kids.append(html.Details([html.Summary(g, className="arc-group", style={"paddingLeft": f"{8 + 14 * (depth + 1)}px"}),
                                      *rows], className="arc-details"))
        else:
            kids += rows
    head = _row(node, sel, depth, drop=node["id"].split(":", 1)[0] in ("kind", "folder", "collection"))
    out.append(html.Details([html.Summary(head, className="arc-summary"), *kids], open=open_, className="arc-details"))
    return out


def _flat(roots: dict) -> dict:
    """Every node of the tree by id."""
    found = {}

    def walk(n):
        found[n["id"]] = n
        for f in n.get("folders") or []:
            walk(f)
        for i in n.get("items") or []:
            walk(i)
    for r in roots.values():
        walk(r)
    return found


# ─────────────────────────── callbacks ───────────────────────────

def register(app) -> None:

    @app.callback(Output("archive-panel", "style"), Input("mode-select", "value"))
    def _show(mode):
        return PANEL_VISIBLE if mode == "archive" else {"display": "none"}

    @app.callback(Output("arc-tree", "children"), Output("arc-trash", "children"),
                  Input("arc-rev", "data"), Input("arc-sel", "data"), Input("mode-select", "value"),
                  Input("state-signature", "data"))
    def _tree(_rev, sel, mode, _sig):
        if mode != "archive":
            return no_update, no_update
        data = _cat.read()
        roots = _cat.tree(data)
        rows = []
        for k in _cat.KINDS:
            rows += _branch(roots[k], sel, 0, open_=k in ("collections", "films", "videos"))
        trash = [html.Div([html.Span(e["name"], className="arc-name"),
                           html.Button("Restore", id={"type": "arc-restore", "id": i}, n_clicks=0,
                                       style=_ui.button("ghost", padding="2px 6px", fontSize="11px"))],
                          className="arc-trash-row") for i, e in data["trash"].items()]
        return rows, trash or html.Div("Empty.", style=_ui.hint(fontSize="11.5px"))

    @app.callback(Output("arc-sel", "data"), Output("arc-frame", "data", allow_duplicate=True),
                  Input({"type": "arc-node", "id": ALL}, "n_clicks"), prevent_initial_call=True)
    def _select(_n):
        hit = ctx.triggered_id
        if not isinstance(hit, dict) or not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update
        return hit["id"], 0

    # What is selected: its name, where it lives, and where it can go.
    @app.callback(Output("arc-title", "children"), Output("arc-where", "children"), Output("arc-name", "value"),
                  Output("arc-move-to", "options"), Output("arc-merge-to", "options"),
                  Output("arc-merge-wrap", "style"),
                  Input("arc-sel", "data"), Input("arc-rev", "data"))
    def _fields(sel, _rev):
        if not sel:
            return "Nothing selected", "Click something in the tree.", "", [], [], {"display": "none"}
        roots = _cat.tree()
        nodes = _flat(roots)
        n = nodes.get(sel)
        if n is None:
            return "Gone", "It is no longer in the archive.", "", [], [], {"display": "none"}
        kind = n["kind"]
        targets = [{"label": f"top of {_cat.KINDS[kind]}", "value": f"kind:{kind}"}]
        merge = []
        for other in nodes.values():
            if other["id"] == sel or other["kind"] != kind:
                continue
            if other["id"].startswith("folder:"):
                targets.append({"label": f"folder · {other['shown']}", "value": other["id"]})
                if sel.startswith("folder:"):
                    merge.append({"label": other["shown"], "value": other["id"]})
            elif other["id"].startswith("collection:") and sel.startswith("collection:"):
                targets.append({"label": f"collection · {other['shown']}", "value": other["id"]})
                merge.append({"label": other["shown"], "value": other["id"]})
        mergeable = sel.startswith(("collection:", "folder:"))
        where = n.get("path") or ("a folder of the archive" if sel.startswith("folder:") else "")
        return (n["shown"], where, n["shown"], targets if not sel.startswith("kind:") else [], merge,
                {} if mergeable else {"display": "none"})

    # The selection, shown or played.
    @app.callback(Output("arc-preview", "children"), Output("arc-play", "disabled"),
                  Input("arc-sel", "data"), Input("arc-frame", "data"), Input("arc-rev", "data"),
                  State("arc-play", "disabled"))
    def _preview(sel, frame, _rev, paused):
        if not sel or sel.startswith(("kind:", "folder:")):
            return html.Div("Select a collection, a film, a video, a story or a picture to see it here.",
                            style=_ui.hint(fontSize="12.5px", padding="40px 10px", textAlign="center", maxWidth="none")), True
        res, err = _call("archive show", {"id": sel})
        if err:
            return html.Div(err, style=_ui.hint(fontSize="12px")), True
        pics = []
        from .alchemy.inputs import resolve
        for p in res.contents:                      # a film names pictures by handle; show their files
            try:
                pics.append(resolve(p))
            except ValueError:
                pics.append(str(p))
        head = html.Div([html.Span(res.name, style=_ui.section_title()),
                         html.Span(f"  {len(pics)} picture{'s' if len(pics) != 1 else ''}" if res.kind not in ("videos",) else "",
                                   style=_ui.hint(fontSize="11.5px"))], style={"marginBottom": "8px"})
        if res.kind == "videos":
            return [head, html.Video(src=_media(res.path, "film-video"), controls=True, autoPlay=True, loop=True,
                                     style={"width": "100%", "maxHeight": "70vh", "borderRadius": "6px",
                                            "background": "#0e0d0c"})], True
        if res.kind in ("pictures", "imports"):
            return [head, html.Img(src=_media(res.path, "preview", 2048), alt="",
                                   style={"maxWidth": "100%", "maxHeight": "72vh", "display": "block", "margin": "0 auto",
                                          "borderRadius": "6px"})], True
        actions = []
        if res.kind == "collections":
            actions = [html.Button("Use as the active collection", id="arc-use", n_clicks=0,
                                   style=_ui.button("secondary", padding="5px 10px", fontSize="12px"))]
        elif res.kind == "films":
            actions = [html.Button("Open in Alchemy's Video", id="arc-open-film", n_clicks=0,
                                   style=_ui.button("secondary", padding="5px 10px", fontSize="12px"))]
        if not pics:
            return [head, *actions, html.Div("Nothing in it yet.", style=_ui.hint(fontSize="12px"))], True
        i = int(frame or 0) % len(pics)
        big = pics[i]
        stage = html.Div([
            html.Img(src=_media(big, "preview", 1600), alt="", className="arc-slide"),
            html.Div([
                html.Button("‹", id={"type": "arc-step", "d": -1}, n_clicks=0, className="arc-nav"),
                html.Button("❚❚" if not paused else "▶", id="arc-toggle", n_clicks=0, className="arc-nav",
                            title="Play a slideshow"),
                html.Button("›", id={"type": "arc-step", "d": 1}, n_clicks=0, className="arc-nav"),
                html.Span(f"{i + 1} / {len(pics)}  ·  {os.path.basename(str(pics[i]))}",
                          style=_ui.hint(fontSize="11.5px", maxWidth="none")),
            ], style=_ui.row("8px", marginTop="8px", justifyContent="center")),
        ], className="arc-stage")
        strip = html.Div([html.Img(src="/thumb?p=" + urllib.parse.quote(str(p)), alt="", n_clicks=0,
                                   id={"type": "arc-pick", "i": k}, className="arc-strip" + (" sel" if k == i else ""))
                          for k, p in enumerate(pics[:300])], className="arc-strip-row")
        return [head, html.Div(actions, style=_ui.row("6px", marginBottom="8px")) if actions else None, stage, strip], no_update

    @app.callback(Output("arc-frame", "data", allow_duplicate=True),
                  Input({"type": "arc-step", "d": ALL}, "n_clicks"), Input({"type": "arc-pick", "i": ALL}, "n_clicks"),
                  Input("arc-play", "n_intervals"), State("arc-frame", "data"), prevent_initial_call=True)
    def _step(_s, _p, _tick, frame):
        hit = ctx.triggered_id
        if hit == "arc-play":
            return int(frame or 0) + 1
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        if isinstance(hit, dict) and hit.get("type") == "arc-pick":
            return int(hit["i"])
        if isinstance(hit, dict) and hit.get("type") == "arc-step":
            return int(frame or 0) + int(hit["d"])
        return no_update

    @app.callback(Output("arc-play", "disabled", allow_duplicate=True), Input("arc-toggle", "n_clicks"),
                  State("arc-play", "disabled"), prevent_initial_call=True)
    def _toggle(n, disabled):
        return no_update if not n else not disabled

    # Organising, through the archive ops.
    @app.callback(Output("arc-rev", "data"), Output("arc-status", "children"), Output("arc-sel", "data", allow_duplicate=True),
                  Output("arc-empty-armed", "data"), Output("arc-empty", "children"),
                  Input("arc-rename", "n_clicks"), Input("arc-move", "n_clicks"), Input("arc-merge", "n_clicks"),
                  Input("arc-folder", "n_clicks"), Input("arc-delete", "n_clicks"), Input("arc-drop", "data"),
                  Input({"type": "arc-restore", "id": ALL}, "n_clicks"), Input("arc-empty", "n_clicks"),
                  State("arc-sel", "data"), State("arc-name", "value"), State("arc-move-to", "value"),
                  State("arc-merge-to", "value"), State("arc-folder-name", "value"), State("arc-empty-armed", "data"),
                  prevent_initial_call=True)
    def _organise(_r, _m, _g, _f, _d, drop, _restore, _empty, sel, name, into, merge_into, folder, armed):
        hit = ctx.triggered_id
        if not ctx.triggered or (hit != "arc-drop" and not ctx.triggered[0].get("value")):
            return (no_update,) * 5
        now = time.time()
        disarm = (None, "Empty the trash")
        if hit == "arc-empty":
            if not armed or now - armed > 6:
                return no_update, "Click Empty the trash again within a few seconds to delete it all for good.", \
                    no_update, now, "Click again to empty"
            res, err = _call("archive empty", {"yes": True})
            return now, err or res.body, no_update, *disarm
        if isinstance(hit, dict) and hit.get("type") == "arc-restore":
            res, err = _call("archive restore", {"id": hit["id"]})
            return now, err or res.body, (hit["id"] if not err else no_update), *disarm
        if hit == "arc-drop":
            d = drop or {}
            if not d.get("id") or not d.get("into") or d["id"] == d["into"]:
                return (no_update,) * 5
            res, err = _call("archive move", {"id": d["id"], "into": d["into"]})
            return now, err or res.body, d["id"], *disarm
        if hit == "arc-folder":
            nodes = _flat(_cat.tree())
            node = nodes.get(sel or "")
            if node is None:
                return no_update, "Select a kind or a folder to make the new folder in.", no_update, *disarm
            args = {"name": folder or "", "kind": node["kind"]}
            if (sel or "").startswith("folder:"):
                args["into"] = sel
            res, err = _call("archive folder add", args)
            return now, err or res.text(), (res.id if res else no_update), *disarm
        if not sel or sel.startswith("kind:"):
            return no_update, "Select something in the tree first.", no_update, *disarm
        if hit == "arc-rename":
            res, err = _call("archive rename", {"id": sel, "name": name or ""})
            new_sel = sel
            if res and sel.startswith("film:"):
                from .alchemy.film import slug
                new_sel = "film:" + slug(name)
            return now, err or res.body, new_sel, *disarm
        if hit == "arc-move":
            res, err = _call("archive move", {"id": sel, "into": into or f"kind:{sel.split(':', 1)[0]}"})
            return now, err or res.body, no_update, *disarm
        if hit == "arc-merge":
            if not merge_into:
                return no_update, "Choose what to merge it into.", no_update, *disarm
            res, err = _call("archive merge", {"id": sel, "into": merge_into})
            return now, err or res.body, (merge_into if not err else no_update), *disarm
        if hit == "arc-delete":
            res, err = _call("archive delete", {"id": sel})
            return now, err or res.body, (None if not err else no_update), *disarm
        return (no_update,) * 5

    # A collection becomes the active one; a film opens in the Video subtab.
    @app.callback(Output("collection-rev", "data", allow_duplicate=True), Output("arc-status", "children", allow_duplicate=True),
                  Input("arc-use", "n_clicks"), State("arc-sel", "data"), prevent_initial_call=True)
    def _use(n, sel):
        if not n or not (sel or "").startswith("collection:"):
            return no_update, no_update
        res, err = _call("settings set", {"collection": sel.split(":", 1)[1]})
        return (time.time(), f"{sel.split(':', 1)[1]} is the active collection.") if not err else (no_update, err)

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("film-loaded", "data", allow_duplicate=True),
                  Output("alc-subtab", "value", allow_duplicate=True), Output("mode-select", "value"),
                  Input("arc-open-film", "n_clicks"), State("arc-sel", "data"), prevent_initial_call=True)
    def _open_film(n, sel):
        if not n or not (sel or "").startswith("film:"):
            return (no_update,) * 4
        res, err = _call("alchemy film show", {"film": sel.split(":", 1)[1]})
        if err:
            return (no_update,) * 4
        return res.spec, time.time(), "video", "alchemy"

