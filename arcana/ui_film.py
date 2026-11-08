# ui_film.py -- the Video subtab of Alchemy
#
# A film is pictures morphed through a token space on a timeline (arcana/alchemy/film.py). This tab
# edits one: the collection on the left, where a click puts a picture on the timeline at the
# playhead; the player in the middle, with the timeline under it; the film's settings, the selected
# key, the lanes and the render on the right.
#
# The timeline is drawn and edited in the browser (assets/film.js): picture keys, look keys and the
# keys of every lane are dragged there, and the playhead scrubs. The film itself lives in one store,
# film-spec, kept in the browser between visits; every edit, from the timeline or from the panel,
# writes it, and every view reads it. Looks come from the Image subtab ("Add look to film").
#
# Frames are drawn by `alchemy film preview`, one request at a time as in the Image subtab, and a
# render is `alchemy film make` as a job whose progress fills the render bar.

from __future__ import annotations

import base64
import json
import os
import urllib.parse

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

try:
    from . import ops as _ops
    from . import paths as _paths
    from . import ui_style as _ui
    from .alchemy import finish as _finish
    from .alchemy import methods as _methods
except ImportError:                                   # pragma: no cover
    import ops as _ops
    import paths as _paths
    import ui_style as _ui
    from alchemy import finish as _finish
    from alchemy import methods as _methods

SMALL = {"fontSize": "12px"}
LABEL = _ui.field_label(marginBottom="4px", marginTop="10px")
EASE_OPTIONS = [{"label": e, "value": e} for e in ("linear", "smooth", "in", "out", "hold")]
GAP = 4.0                                             # seconds between pictures added one after another
REFINE_RANGE = {"feather": (0, 40, 1), "grow": (-40, 40, 1), "follow": (0, 1, 0.05),
                "ragged": (0, 1, 0.05), "amount": (0, 1, 0.05)}
NUMBER_RANGE = {"x": (0, 1, 0.01), "y": (0, 1, 0.01), "r": (0.02, 1, 0.01), "angle": (0, 360, 1),
                "height": (0.02, 1, 0.01), "count": (1, 40, 1), "near": (0, 1, 0.01), "far": (0, 1, 0.01),
                "size": (4, 200, 1), "coverage": (0.02, 1, 0.01)}
EMPTY_FILM = {"name": "untitled film", "fps": 30, "size": 640, "aspect": "auto", "space": "DINOv2",
              "loop": True, "even": True, "pictures": [], "looks": [], "lanes": {}}


def _film():
    try:
        from .alchemy import film
    except ImportError:                               # pragma: no cover
        from alchemy import film
    return film


def _thumb(path):
    return "/thumb?p=" + urllib.parse.quote(path)


def _section(title, body, open_=True, note=None):
    head = [html.Span(title)]
    if note:
        head.append(html.Span(note, className="alc-sec-note"))
    return html.Details([html.Summary(head), html.Div(body, className="alc-sec-body")], open=open_,
                        className="alc-sec")


def _input(id_, value=None, type_="number", width="100%", **kw):
    return dcc.Input(id=id_, type=type_, value=value, debounce=True,
                     style=_ui.input_box(width=width, boxSizing="border-box", padding="5px 8px", fontSize="12px"), **kw)


# ─────────────────────────── layout ───────────────────────────

def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="film-spec", storage_type="local"),
            dcc.Store(id="film-time", storage_type="memory", data=0.0),
            dcc.Store(id="film-selected", storage_type="memory"),
            dcc.Store(id="film-meta", storage_type="memory"),
            dcc.Store(id="film-request", storage_type="memory"),
            dcc.Store(id="film-loaded", storage_type="memory"),
            dcc.Store(id="film-job", storage_type="memory"),
            dcc.Store(id="film-sink", storage_type="memory"),
            dcc.Interval(id="film-poll", interval=600, disabled=True),
            _cast(),
            _stage(),
            _settings(),
        ],
        id="alc-video-view", style={"display": "none"},
    )


def _cast():
    return html.Div(
        [
            html.Div("Pictures", style=_ui.section_title()),
            html.Div("Click a picture to put it on the timeline at the playhead. A picture on the "
                     "timeline is dragged along it to change when the film passes it.",
                     style=_ui.hint(fontSize="11.5px")),
            html.Div(id="film-cast", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(70px, 1fr))",
                                            "gap": "6px", "alignContent": "start", "flex": "1 1 0", "minHeight": "0",
                                            "overflowY": "auto", "padding": "8px 2px"}),
        ],
        style={"flex": "0 0 220px", "minWidth": "0", "minHeight": "0", "display": "flex", "flexDirection": "column",
               **_ui.card(padding="12px", marginBottom="0")},
    )


def _stage():
    return html.Div(
        [
            html.Div([
                html.Img(id="film-frame", alt="", className="film-frame"),
                html.Div(id="film-empty", className="film-empty"),
            ], className="film-player"),
            html.Div([
                html.Button("▶", id="film-play", n_clicks=0, title="Play a draft (space)",
                            className="film-play"),
                html.Span(id="film-clock", className="film-clock"),
                html.Div(style={"flex": "1"}),
                html.Span(id="film-info", style=_ui.hint(fontSize="11.5px", maxWidth="none")),
            ], style=_ui.row("10px", margin="8px 0 6px")),
            html.Div(id="film-timeline", className="film-timeline", tabIndex="0"),
            html.Div(id="film-status", style=_ui.hint(fontSize="12px", marginTop="6px", maxWidth="none", color=_ui.INK)),
        ],
        style={"flex": "1 1 auto", "minWidth": "0", "minHeight": "0", "overflowY": "auto",
               **_ui.card(padding="12px", marginBottom="0")},
    )


def _settings():
    return html.Div(
        [
            _section("Film", [
                html.Div("NAME", style=LABEL), _input("film-name", type_="text"),
                html.Div([
                    html.Div([html.Div("LENGTH (S)", style=LABEL), _input("film-length", min=1, step=0.5)],
                             style={"flex": "1"}),
                    html.Div([html.Div("FPS", style=LABEL), _input("film-fps", min=6, max=60, step=1)],
                             style={"flex": "1"}),
                    html.Div([html.Div("SIZE", style=LABEL), _input("film-size", min=256, max=2048, step=16)],
                             style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px"}),
                html.Div("SHAPE", style=LABEL),
                dcc.Dropdown(id="film-aspect", clearable=False, style=SMALL,
                             options=[{"label": "the first picture's", "value": "auto"}]
                             + [{"label": a, "value": a} for a in ("3:2", "2:3", "4:3", "16:9", "9:16", "1:1")]),
                html.Div("MORPH THROUGH", style=LABEL),
                dcc.Dropdown(id="film-space", clearable=False, style=SMALL,
                             options=[{"label": f"{k}: {v}", "value": k} for k, v in _film().spaces.SPACES.items()]),
                dcc.Checklist(id="film-flags", style={**SMALL, "marginTop": "8px"},
                              options=[{"label": " loop back to the first picture", "value": "loop"},
                                       {"label": " even motion between pictures", "value": "even"}]),
            ], note="length, shape, model"),
            _section("Timeline", [
                html.Div("SPACE THE PICTURES", style={**LABEL, "marginTop": "0"}),
                html.Div([html.Button(lab, id={"type": "film-space-pics", "shape": key}, n_clicks=0,
                                      title=doc, style=_ui.button("secondary", padding="4px 8px", fontSize="11.5px"))
                          for key, lab, doc in (("even", "even", "every passage as long"),
                                                ("faster", "speeding up", "each passage shorter than the one before"),
                                                ("slower", "slowing down", "each passage longer than the one before"),
                                                ("smooth", "slow at both ends", "slow to start and to end"))],
                         style=_ui.row("6px")),
                html.Div("ADD A LANE", style=LABEL),
                html.Div([
                    dcc.Dropdown(id="film-lane-pick", placeholder="a setting to key in time...", style={**SMALL, "flex": "1",
                                                                                                         "minWidth": "0"}),
                    html.Button("Add", id="film-lane-add", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div("Looks come from the Image subtab: set a step there and press Add look to film. "
                         "Between two looks every number moves from one to the other.",
                         style=_ui.hint(fontSize="11px", marginTop="8px")),
            ], note="spacing, lanes"),
            _section("Selected key", [
                html.Div(id="film-key-title", style=_ui.hint(fontSize="12px", color=_ui.INK)),
                html.Div([
                    html.Div([html.Div("TIME (S)", style=LABEL), _input("film-key-t", step=0.05)], style={"flex": "1"}),
                    html.Div([html.Div("VALUE", style=LABEL), _input("film-key-v", step="any")], id="film-key-v-wrap",
                             style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px"}),
                html.Div("EASE TO THE NEXT KEY", style=LABEL),
                dcc.Dropdown(id="film-key-ease", options=EASE_OPTIONS, clearable=False, style=SMALL),
                html.Button("Remove key", id="film-key-remove", n_clicks=0,
                            style=_ui.button("ghost", padding="4px 8px", fontSize="11.5px", marginTop="8px")),
            ], note="time, value, ease"),
            _section("Keep and render", [
                html.Div([
                    dcc.Dropdown(id="film-pick", placeholder="Kept films...", style={**SMALL, "flex": "1", "minWidth": "0"}),
                    html.Button("Open", id="film-open", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 10px", fontSize="12px")),
                    html.Button("New", id="film-new", n_clicks=0,
                                style=_ui.button("ghost", padding="5px 10px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Button("Keep this film", id="film-save", n_clicks=0,
                            style=_ui.button("secondary", padding="5px 11px", fontSize="12px", marginTop="8px")),
                html.Div([
                    html.Div([html.Div("DRAW AT", style=LABEL),
                              dcc.Dropdown(id="film-draw", clearable=False, value=768, style=SMALL,
                                           options=[{"label": f"{s} px", "value": s} for s in (512, 640, 768, 1024, 1536)])],
                             style={"flex": "1"}),
                    html.Div([html.Div("VIDEO", style=LABEL),
                              dcc.Dropdown(id="film-target", clearable=False, value=0, style=SMALL,
                                           options=[{"label": "as drawn", "value": 0}]
                                           + [{"label": f"{s} px (Real-ESRGAN)", "value": s} for s in (1920, 2560, 3840)])],
                             style={"flex": "1.3"}),
                ], style={"display": "flex", "gap": "8px"}),
                html.Button("Render", id="film-render", n_clicks=0,
                            style=_ui.button("primary", padding="6px 14px", fontSize="12.5px", marginTop="10px")),
                html.Div(html.Div(id="film-render-fill", className="film-render-fill"), id="film-render-bar",
                         className="film-render-bar", style={"display": "none"}),
                html.Div(id="film-render-status", style=_ui.hint(fontSize="11.5px", marginTop="6px")),
                html.Video(id="film-video", controls=True, loop=True,
                           style={"display": "none", "width": "100%", "marginTop": "8px", "borderRadius": "6px"}),
            ], note="save, open, video"),
        ],
        style={"flex": "0 0 330px", "minWidth": "0", "minHeight": "0", "overflowY": "auto", "paddingRight": "4px"},
    )


# ─────────────────────────── the film's lanes ───────────────────────────

def lane_meta(spec: dict) -> tuple[list, dict]:
    """The settings a lane can key, and each one's label and range: (options, {path: meta})."""
    film = _film()
    meta: dict = {}
    methods = {k.get("settings", {}).get("method") for k in spec.get("looks") or []}
    for mid in sorted(m for m in methods if m and m != "none"):
        try:
            method = _methods.get(mid)
        except ValueError:
            continue
        for k in method.knobs:
            if k.kind in ("float", "int", "angle") and k.lo is not None:
                meta[f"knobs.{k.key}"] = {"label": f"{method.label} · {k.label}", "lo": k.lo, "hi": k.hi,
                                          "step": k.step or 0.01, "group": "method"}
    for key, group, label, lo, hi, step, doc in _finish.SPEC:
        meta[f"finish.{key}"] = {"label": f"finish · {label}", "lo": lo, "hi": hi, "step": step, "group": "finish"}
    for key, (lo, hi, step) in REFINE_RANGE.items():
        meta[f"mask.{key}"] = {"label": f"mask · {key}", "lo": lo, "hi": hi, "step": step, "group": "mask"}
    most_layers, most_materials = [], 0
    for k in spec.get("looks") or []:
        s = k.get("settings") or {}
        m = s.get("mask") or {}
        layers = m.get("layers") if isinstance(m, dict) else ([m] if isinstance(m, str) else m) or []
        if len(layers or []) > len(most_layers):
            most_layers = list(layers)
        most_materials = max(most_materials, len([x for x in s.get("materials") or [] if x]))
    for i, line in enumerate(most_layers, 1):
        meta[f"mask.{i}.coverage"] = {"label": f"mask {i} · coverage", "lo": 0.02, "hi": 1, "step": 0.01, "group": "mask"}
        got = film.layer_numbers(line)
        if got:
            for name in got[1]:
                lo, hi, step = NUMBER_RANGE[name]
                meta[f"mask.{i}.{name}"] = {"label": f"mask {i} · {got[0]} {name}", "lo": lo, "hi": hi, "step": step,
                                            "group": "mask"}
    for i in range(1, most_materials + 1):
        meta[f"weights.{i}"] = {"label": f"share of [{i}]", "lo": 0, "hi": 1, "step": 0.05, "group": "pictures"}
    meta["zoom"] = {"label": "material zoom", "lo": 0.5, "hi": 4, "step": 0.25, "group": "pictures"}
    for path, keys in (spec.get("lanes") or {}).items():
        if path not in meta:
            vs = [float(k["v"]) for k in keys] or [0.0]
            meta[path] = {"label": path, "lo": min(vs + [0.0]), "hi": max(vs + [1.0]), "step": 0.01, "group": "other"}
    options = [{"label": v["label"], "value": p} for p, v in meta.items() if p not in (spec.get("lanes") or {})]
    return options, meta


def _value_at(spec: dict, path: str, t: float, lo: float):
    """A setting's value at time t, from the looks, to start a new lane where the film already is."""
    film = _film()
    s = film.look_at(spec, t)
    head, _, rest = path.partition(".")
    try:
        if path == "zoom":
            return float(s.get("zoom", 1.0))
        if head in ("knobs", "finish"):
            v = (s.get(head) or {}).get(rest)
            if v is None and head == "finish":
                return float(getattr(_finish.Finish(), rest))
            if v is None:
                for k in s.get("method") and _methods.get(s["method"]).knobs or ():
                    if k.key == rest:
                        return float(k.default)
            return float(v) if v is not None else lo
        if head == "weights":
            w = s.get("weights") or []
            i = int(rest) - 1
            return float(w[i]) if i < len(w) else 1.0
        if head == "mask":
            m = s.get("mask") or {}
            if rest in REFINE_RANGE:
                return float(m.get(rest, {"feather": 4.0, "amount": 1.0}.get(rest, 0.0)))
            idx, _, num = rest.partition(".")
            layers = m.get("layers") or []
            line = layers[int(idx) - 1] if int(idx) - 1 < len(layers) else ""
            if num == "coverage":
                cov = (m.get("coverage") or [None] * len(layers))
                return float(cov[int(idx) - 1]) if int(idx) - 1 < len(cov) and cov[int(idx) - 1] is not None else 0.4
            got = film.layer_numbers(line)
            if got and num in got[1]:
                return float(got[2][got[1].index(num)])
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    return lo


def _normal(spec):
    """The film with its defaults, or (None, the reason) when it is not a film yet."""
    try:
        return _film().normalised(spec), None
    except (ValueError, KeyError, TypeError) as e:
        return None, str(e)


# ─────────────────────────── callbacks ───────────────────────────

def register(app) -> None:

    @app.server.route("/film-video")
    def _film_video():
        """A rendered film, from output/ only."""
        from flask import Response, request, send_file

        p = request.args.get("p") or ""
        real = os.path.realpath(p)
        if not (p.endswith(".mp4") and os.path.isfile(real)
                and _paths.is_within(real, [os.path.realpath(_paths.subdir("output"))])):
            return Response(status=404)
        return send_file(real, mimetype="video/mp4", conditional=True)

    @app.callback(Output("film-cast", "children"), Input("collection-items", "data"))
    def _cast(items):
        items = items or []
        if not items:
            return html.Div("The active collection is empty. Pictures added with + Collection on any tab "
                            "appear here.", style=_ui.hint(gridColumn="1 / -1", fontSize="11.5px"))
        return [html.Img(src=_thumb(p), alt="", title=os.path.basename(p), n_clicks=0,
                         id={"type": "film-add", "index": p},
                         style={"width": "100%", "aspectRatio": "1", "objectFit": "cover", "borderRadius": "5px",
                                "border": f"1px solid {_ui.LINE}", "cursor": "copy", "display": "block",
                                "backgroundColor": _ui.SURFACE_2})
                for p in items]

    # The film's own settings in the panel, from the film whenever one is opened or made.
    @app.callback(Output("film-name", "value"), Output("film-length", "value"), Output("film-fps", "value"),
                  Output("film-size", "value"), Output("film-aspect", "value"), Output("film-space", "value"),
                  Output("film-flags", "value"),
                  Input("film-loaded", "data"), State("film-spec", "data"))
    def _fields(_loaded, spec):
        s = {**EMPTY_FILM, **(spec or {})}
        f, _ = _normal(s)
        length = round(f["length"], 2) if f else s.get("length")
        return (s["name"], length, s["fps"], s["size"], s["aspect"], s["space"],
                [k for k in ("loop", "even") if s.get(k, True)])

    @app.callback(Output("film-spec", "data", allow_duplicate=True),
                  Input("film-name", "value"), Input("film-length", "value"), Input("film-fps", "value"),
                  Input("film-size", "value"), Input("film-aspect", "value"), Input("film-space", "value"),
                  Input("film-flags", "value"), State("film-spec", "data"), prevent_initial_call=True)
    def _from_fields(name, length, fps, size, aspect, space, flags, spec):
        s = {**EMPTY_FILM, **(spec or {})}
        new = {**s, "name": (name or s["name"]).strip() or s["name"], "fps": int(fps or s["fps"]),
               "size": int(size or s["size"]), "aspect": aspect or s["aspect"], "space": space or s["space"],
               "loop": "loop" in (flags or []), "even": "even" in (flags or [])}
        if length:
            new["length"] = float(length)
        return no_update if new == s else new

    @app.callback(Output("film-meta", "data"), Output("film-lane-pick", "options"), Output("film-status", "children"),
                  Output("film-empty", "children"),
                  Input("film-spec", "data"))
    def _meta(spec):
        s = {**EMPTY_FILM, **(spec or {})}
        options, meta = lane_meta(s)
        f, why = _normal(s)
        pictures = []
        from .alchemy.inputs import resolve
        for p in sorted(s.get("pictures") or [], key=lambda p: float(p["t"])):
            try:
                pictures.append({"thumb": _thumb(resolve(p["ref"])), "name": os.path.basename(str(p["ref"]))})
            except ValueError:
                pictures.append({"thumb": "", "name": str(p["ref"])})
        looks = []
        for k in s.get("looks") or []:
            st = k.get("settings") or {}
            m = st.get("method") or "none"
            looks.append({"label": "morph only" if m == "none" else _methods.get(m).label if m in _methods.METHODS else m})
        length = f["length"] if f else max([float(p["t"]) for p in s.get("pictures") or []] + [GAP])
        status = "" if f or len(s.get("pictures") or []) < 2 else why
        empty = "" if len(s.get("pictures") or []) >= 2 else \
            "Click two or more pictures on the left to start a film. Each lands at the playhead."
        return ({"length": length, "lanes": meta, "pictures": pictures, "looks": looks, "ok": f is not None},
                options, status, empty)

    # The timeline is drawn in the browser (assets/film.js) from the film and its meta.
    app.clientside_callback(
        "function (spec, meta, t, sel) { if (window.filmTimeline) window.filmTimeline.draw(spec, meta, t, sel);"
        " return window.dash_clientside.no_update; }",
        Output("film-sink", "data"),
        Input("film-spec", "data"), Input("film-meta", "data"), Input("film-time", "data"),
        Input("film-selected", "data"),
    )
    app.clientside_callback(
        "function (t, meta) { var L = (meta && meta.length) || 0; var f = function (x) { var m = Math.floor(x / 60);"
        " var s = (x - 60 * m).toFixed(2); return (m ? m + ':' : '') + (m && x - 60 * m < 10 ? '0' : '') + s; };"
        " return f(t || 0) + ' / ' + f(L); }",
        Output("film-clock", "children"), Input("film-time", "data"), Input("film-meta", "data"),
    )

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("film-time", "data", allow_duplicate=True),
                  Input({"type": "film-add", "index": ALL}, "n_clicks"),
                  State("film-spec", "data"), State("film-time", "data"), prevent_initial_call=True)
    def _add_picture(_n, spec, t):
        hit = ctx.triggered_id
        if not isinstance(hit, dict) or not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update
        s = {**EMPTY_FILM, **(spec or {})}
        pics = sorted(s.get("pictures") or [], key=lambda p: float(p["t"]))
        t = float(t or 0.0)
        taken = {round(float(p["t"]), 3) for p in pics}
        if not pics:
            t = 0.0
        elif round(t, 3) in taken:
            t = float(pics[-1]["t"]) + GAP          # the playhead is on a picture: add after the last
        s["pictures"] = pics + [{"t": round(t, 3), "ref": hit["index"], "ease": "linear"}]
        if t >= float(s.get("length") or 0):
            s.pop("length", None)                  # a picture past the end: the length follows the pictures
        return s, t

    @app.callback(Output("film-spec", "data", allow_duplicate=True),
                  Input({"type": "film-space-pics", "shape": ALL}, "n_clicks"),
                  State("film-spec", "data"), prevent_initial_call=True)
    def _space(_n, spec):
        hit = ctx.triggered_id
        if not isinstance(hit, dict) or not ctx.triggered[0].get("value"):
            return no_update
        try:
            return _film().spaced(spec, hit["shape"])
        except (ValueError, KeyError, TypeError):
            return no_update

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("film-selected", "data", allow_duplicate=True),
                  Input("film-lane-add", "n_clicks"),
                  State("film-lane-pick", "value"), State("film-spec", "data"), State("film-time", "data"),
                  State("film-meta", "data"), prevent_initial_call=True)
    def _add_lane(n, path, spec, t, meta):
        if not n or not path:
            return no_update, no_update
        s = {**EMPTY_FILM, **(spec or {})}
        m = ((meta or {}).get("lanes") or {}).get(path) or {"lo": 0, "hi": 1}
        f, _ = _normal(s)
        t = float(t or 0.0)
        v = _value_at(f, path, t, m["lo"]) if f else m["lo"]
        lanes = dict(s.get("lanes") or {})
        lanes[path] = [{"t": round(t, 3), "v": v, "ease": "linear"}]
        return {**s, "lanes": lanes}, {"lane": path, "i": 0}

    # The selected key, in the panel.
    @app.callback(Output("film-key-title", "children"), Output("film-key-t", "value"), Output("film-key-v", "value"),
                  Output("film-key-v-wrap", "style"), Output("film-key-ease", "value"),
                  Input("film-selected", "data"), Input("film-spec", "data"), State("film-meta", "data"))
    def _key_fields(sel, spec, meta):
        key, label = _selected(spec, sel, meta)
        if key is None:
            return "Click a key on the timeline to edit it here.", None, None, {"flex": "1"}, None
        has_v = "v" in key
        return (label, round(float(key["t"]), 3), key.get("v") if has_v else None,
                {"flex": "1"} if has_v else {"display": "none"}, key.get("ease", "linear"))

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("film-selected", "data", allow_duplicate=True),
                  Input("film-key-t", "value"), Input("film-key-v", "value"), Input("film-key-ease", "value"),
                  Input("film-key-remove", "n_clicks"),
                  State("film-selected", "data"), State("film-spec", "data"), prevent_initial_call=True)
    def _key_edit(t, v, ease, _remove, sel, spec):
        if not sel or not spec:
            return no_update, no_update
        s = json.loads(json.dumps(spec))
        keys = _keys_of(s, sel)
        i = int(sel.get("i", -1))
        if keys is None or not 0 <= i < len(keys):
            return no_update, no_update
        if ctx.triggered_id == "film-key-remove":
            keys.pop(i)
            if sel["lane"] not in ("pictures", "looks") and not keys:
                s["lanes"].pop(sel["lane"], None)
            return s, None
        key = keys[i]
        if t is not None:
            key["t"] = round(max(0.0, float(t)), 3)
        if v is not None and "v" in key:
            key["v"] = float(v)
        if ease:
            key["ease"] = ease
        return (no_update, no_update) if s == spec else (s, no_update)

    # Opening, making and keeping films.
    @app.callback(Output("film-pick", "options"), Input("film-loaded", "data"), Input("film-render-status", "children"),
                  Input("state-signature", "data"))
    def _films(_a, _b, _c):
        try:
            return [{"label": f.name, "value": f.name} for f in _ops.call("alchemy film list", actor="person").films]
        except _ops.Refusal:
            return []

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("film-loaded", "data"),
                  Output("film-time", "data", allow_duplicate=True), Output("film-selected", "data", allow_duplicate=True),
                  Output("film-render-status", "children", allow_duplicate=True),
                  Input("film-open", "n_clicks"), Input("film-new", "n_clicks"), State("film-pick", "value"),
                  prevent_initial_call=True)
    def _open(_o, _n, name):
        import time as _t

        if ctx.triggered_id == "film-new":
            return dict(EMPTY_FILM), _t.time(), 0.0, None, "A new film."
        if not name:
            return no_update, no_update, no_update, no_update, "Choose a kept film first."
        try:
            spec = _ops.call("alchemy film show", {"film": name}, actor="person").spec
        except _ops.Refusal as e:
            return no_update, no_update, no_update, no_update, e.reason
        return spec, _t.time(), 0.0, None, f"Opened {name}."

    @app.callback(Output("film-render-status", "children", allow_duplicate=True),
                  Input("film-save", "n_clicks"), State("film-spec", "data"), prevent_initial_call=True)
    def _save(n, spec):
        if not n:
            return no_update
        try:
            res = _ops.call("alchemy film save", {"spec": spec}, actor="person")
        except _ops.Refusal as e:
            return e.reason
        return f"Kept as {res.film.name}."

    # The render, as a job whose progress fills the bar.
    @app.callback(Output("film-job", "data"), Output("film-poll", "disabled"),
                  Output("film-render-status", "children", allow_duplicate=True),
                  Output("film-render-bar", "style"), Output("film-video", "style", allow_duplicate=True),
                  Input("film-render", "n_clicks"), State("film-spec", "data"), State("film-draw", "value"),
                  State("film-target", "value"), prevent_initial_call=True)
    def _render(n, spec, draw, target):
        if not n:
            return (no_update,) * 5
        args = {"spec": spec, "size": int(draw or 768), "background": True}
        if target:
            args["target"] = int(target)
        try:
            started = _ops.call("alchemy film make", args, actor="person")
        except _ops.Refusal as e:
            return no_update, True, e.reason, {"display": "none"}, no_update
        return started.job, False, "Starting the render...", {"display": "block"}, {"display": "none"}

    @app.callback(Output("film-render-fill", "style"), Output("film-render-status", "children", allow_duplicate=True),
                  Output("film-poll", "disabled", allow_duplicate=True), Output("film-video", "src"),
                  Output("film-video", "style", allow_duplicate=True), Output("film-render-bar", "style", allow_duplicate=True),
                  Input("film-poll", "n_intervals"), State("film-job", "data"), prevent_initial_call=True)
    def _progress(_n, job):
        if not job:
            return no_update, no_update, True, no_update, no_update, no_update
        try:
            v = _ops.call("job show", {"job": job}, actor="person").job
        except _ops.Refusal as e:
            return no_update, e.reason, True, no_update, no_update, {"display": "none"}
        width = {"width": f"{100 * (v.fraction or 0):.1f}%"}
        if v.status in ("queued", "running"):
            return width, v.message or "Rendering...", False, no_update, no_update, no_update
        if v.status == "done" and v.result:
            r = v.result
            where = os.path.relpath(r["path"], _paths.subdir("output")).replace(os.sep, "/")
            return ({"width": "100%"}, f"Rendered {r['frames']} frames, {r['seconds']:.1f} s at {r['size'][0]} x "
                    f"{r['size'][1]}, as {r['id']} in output/{where}.", True,
                    "/film-video?p=" + urllib.parse.quote(r["path"]),
                    {"display": "block", "width": "100%", "marginTop": "8px", "borderRadius": "6px"}, {"display": "none"})
        return width, v.error or f"The render ended: {v.status}.", True, no_update, no_update, {"display": "none"}

    # A look from the Image subtab, at the playhead.
    try:
        from . import ui_alchemy as _ia
    except ImportError:                               # pragma: no cover
        import ui_alchemy as _ia

    @app.callback(Output("film-spec", "data", allow_duplicate=True), Output("alc-keep-status", "children", allow_duplicate=True),
                  inputs={"n": Input("alc-snap", "n_clicks")},
                  state={**_ia._controls("state"), **_ia.ID_STATES, "spec": State("film-spec", "data"),
                         "t": State("film-time", "data")},
                  prevent_initial_call=True)
    def _snapshot(n, **v):
        if not n:
            return no_update, no_update
        args = _ia.settings(v)
        from .ops.alchemy import AlchemyPreviewArgs, settings_of
        try:
            look = settings_of(AlchemyPreviewArgs(**{k: val for k, val in args.items()
                                                     if k not in ("picture", "material", "weight")}))
        except (ValueError, _ops.Refusal) as e:
            return no_update, f"Could not take this look: {e}"
        look["materials"] = list(args.get("material") or [])
        if args.get("weight"):
            look["weights"] = list(args["weight"])
        s = {**EMPTY_FILM, **(v.get("spec") or {})}
        t = round(float(v.get("t") or 0.0), 3)
        looks = [k for k in s.get("looks") or [] if round(float(k["t"]), 3) != t]
        s["looks"] = sorted(looks + [{"t": t, "ease": "smooth", "settings": look}], key=lambda k: float(k["t"]))
        return s, f"Added this look to the film at {t:.2f} s. The Video subtab shows it on the look lane."

    # Frames: one request at a time (the gate in film.js), the newest film and moment next.
    app.clientside_callback(
        "function () { return window.filmGate ? window.filmGate.ask() : {seq: Date.now()}; }",
        Output("film-request", "data"),
        Input("film-spec", "data"), Input("film-time", "data"), Input("alc-subtab", "value"),
    )

    @app.callback(Output("film-frame", "src"), Output("film-info", "children"),
                  Input("film-request", "data"),
                  State("film-spec", "data"), State("film-time", "data"), State("alc-subtab", "value"),
                  prevent_initial_call=True)
    def _frame(_req, spec, t, sub):
        if sub != "video":
            return no_update, no_update
        f, why = _normal({**EMPTY_FILM, **(spec or {})})
        if f is None:
            return no_update, ""
        try:
            res = _ops.call("alchemy film preview", {"spec": f, "at": float(t or 0.0), "size": 512}, actor="person")
        except _ops.Refusal as e:
            return no_update, e.reason
        except Exception as e:                        # a crash reaches the page as words, and the gate reopens
            return no_update, f"{type(e).__name__}: {e}"
        src = "data:image/jpeg;base64," + base64.b64encode(res.attachments[0].data).decode() if res.attachments else no_update
        return src, f"{res.size[0]} x {res.size[1]} · {res.ms} ms"


def _keys_of(spec: dict, sel: dict):
    lane = (sel or {}).get("lane")
    if lane in ("pictures", "looks"):
        spec[lane] = sorted(spec.get(lane) or [], key=lambda k: float(k["t"]))
        return spec[lane]
    keys = (spec.get("lanes") or {}).get(lane)
    if keys is not None:
        keys.sort(key=lambda k: float(k["t"]))
    return keys


def _selected(spec, sel, meta):
    if not spec or not sel:
        return None, ""
    s = json.loads(json.dumps(spec))
    keys = _keys_of(s, sel)
    i = int(sel.get("i", -1))
    if keys is None or not 0 <= i < len(keys):
        return None, ""
    lane = sel["lane"]
    if lane == "pictures":
        label = f"Picture {i + 1}: {os.path.basename(str(keys[i]['ref']))}"
    elif lane == "looks":
        m = (keys[i].get("settings") or {}).get("method") or "none"
        label = f"Look {i + 1}: " + ("morph only" if m == "none" else m)
    else:
        label = (((meta or {}).get("lanes") or {}).get(lane) or {}).get("label", lane) + f", key {i + 1}"
    return keys[i], label
