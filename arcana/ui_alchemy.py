# ui_alchemy.py -- the Alchemy tab
#
# Three columns. On the left, the collection, where pictures take roles: [T]
# the picture, [1] [2] [3] the materials. In the middle, the stage: the result
# over the picture as it was, cut by a line dragged across it. On the right, the
# settings in folding sections: recipe, method, pictures, mask, finish. Each
# method shows its main knobs first, led by the one that decides how much the
# picture or the materials dominate. Its other knobs and the model's looks fold
# away under them.
#
# Sliders preview while they move. One preview is out at a time; when it lands,
# the newest settings go out next, so a drag redraws as fast as the step can be
# drawn. The image stays until the next one replaces it. A step slower than
# real time fills a bar over the stage, which completes as the picture lands.
#
# Every action goes through an op (`alchemy preview`, `alchemy make`, `alchemy
# recipe save`), so the tab, the CLI and agents do the same thing. layout() and
# register(app) are the whole interface.

from __future__ import annotations

import base64
import hashlib
import os
import urllib.parse

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

try:
    from . import ops as _ops
    from . import paths as _paths
    from . import ui_collection as _collection
    from . import ui_style as _ui
    from .alchemy import finish as _finish
    from .alchemy import mask as _mask
    from .alchemy import methods as _methods
    from .alchemy import recipes as _recipes
except ImportError:                                   # pragma: no cover
    import ops as _ops
    import paths as _paths
    import ui_collection as _collection
    import ui_style as _ui
    from alchemy import finish as _finish
    from alchemy import mask as _mask
    from alchemy import methods as _methods
    from alchemy import recipes as _recipes

ROLES = ("T", "1", "2", "3")
ROLE_COLOUR = {"T": _ui.ROLE_TARGET, "1": _ui.ACCENT, "2": "#b08d57", "3": "#7c8f8a"}
LAYERS = 3
WORDS, CUSTOM = "__words", "__custom"
FAMILY_LABEL = {"pixels": "Pixels", "tokens": "Tokens", "colour": "Colour", "generative": "Generative"}
REFINE = [  # (key, label, lo, hi, step, default, doc)
    ("feather", "feather", 0, 40, 1, 4.0, "how soft the edge is, in pixels at 768"),
    ("grow", "grow", -40, 40, 1, 0.0, "the edge moved out (more) or in (less), in pixels at 768"),
    ("follow", "follow edges", 0, 1, 0.05, 0.0, "how closely the edge snaps to the picture's contours"),
    ("ragged", "ragged", 0, 1, 0.05, 0.0, "the edge broken by grown noise"),
    ("amount", "amount", 0, 1, 0.05, 1.0, "how strongly the step shows through the mask"),
]
FIN_DEFAULT = {k: getattr(_finish.Finish(), k) for k, *_ in _finish.SPEC}

LABEL = _ui.field_label(marginBottom="4px", marginTop="10px")
KNOB_LABEL = _ui.hint(fontSize="11.5px", marginTop="6px", color=_ui.INK)
SLIDER_TIP = {"placement": "bottom", "always_visible": False}
SMALL = {"fontSize": "12px"}
PANEL_VISIBLE = {"display": "flex", "gap": "14px", "flex": "1 1 auto", "minWidth": "0", "minHeight": "0"}


# ─────────────────────────── building blocks ───────────────────────────

def _slider(id_, lo, hi, step, value):
    """A slider that previews while it moves."""
    return dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks={lo: f"{lo:g}", hi: f"{hi:g}"},
                      tooltip=SLIDER_TIP, updatemode="drag")


def _section(title, body, open_=True, note=None, id_=None):
    head = [html.Span(title)]
    if note:
        head.append(html.Span(note, className="alc-sec-note"))
    kw = {"id": id_} if id_ else {}
    return html.Details([html.Summary(head), html.Div(body, className="alc-sec-body")],
                        open=open_, className="alc-sec", **kw)


def _sub(title, body, id_=None, open_=False):
    kw = {"id": id_} if id_ else {}
    return html.Details([html.Summary(title), html.Div(body)], open=open_, className="alc-sub", **kw)


def _knob_control(k: _methods.Knob, value):
    kid = {"type": "alc-knob", "key": k.key}
    if k.kind in ("float", "angle", "int"):
        step = k.step or (1 if k.kind == "int" else 0.01)
        return _slider(kid, k.lo, k.hi, step, value)
    if k.kind == "choice":
        return dcc.Dropdown(id=kid, options=[{"label": str(o), "value": o} for o in k.options],
                            value=value, clearable=False, style=SMALL)
    if k.kind == "bool":
        return dcc.Checklist(id=kid, options=[{"label": " " + k.label, "value": "on"}],
                             value=["on"] if value else [], style=SMALL)
    return dcc.Input(id=kid, type="text", value=value or "", debounce=True,
                     placeholder=k.doc, style=_ui.input_box(width="100%", boxSizing="border-box",
                                                            padding="6px 9px", fontSize="12px"))


def _knob_row(k, value, lead=False):
    row = html.Div([html.Div(k.label, title=k.doc, style=KNOB_LABEL),
                    _knob_control(k, bool(value) if k.kind == "bool" else value)],
                   id={"type": "alc-knob-wrap", "key": k.key})
    return html.Div([row, html.Div(k.doc, style=_ui.hint(fontSize="11px"))], className="alc-lead-knob") \
        if lead else row


def _method_options(family):
    return [{"label": m.label + ("  (seconds)" if m.cost != "live" else ""), "value": m.id}
            for m in _methods.METHODS.values() if m.family == family]


def _layer_options():
    opts = [{"label": "none", "value": "none"},
            {"label": "words...", "value": WORDS},
            {"label": "custom source...", "value": CUSTOM}]
    for g in _mask.GROUPS:
        opts.append({"label": f"── {g} ──", "value": f"__group {g}", "disabled": True})
        opts += [{"label": p.name, "value": p.name} for p in _mask.PRESETS if p.group == g]
    return opts


def _recipe_options():
    return [{"label": f"{r.group} · {r.name}", "value": r.name} for r in _recipes.every()]


def _thumb(path):
    return "/thumb?p=" + urllib.parse.quote(path)


# ─────────────────────────── layout ───────────────────────────

def _cast_column():
    return html.Div(
        [
            html.Div("Pictures", style=_ui.section_title()),
            html.Div("T is the picture. 1, 2 and 3 are the materials mixed into it. They come from the "
                     "active collection, chosen in the header.", style=_ui.hint(fontSize="11.5px")),
            html.Div(id="alc-slots", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                                            "gap": "6px", "margin": "10px 0"}),
            html.Div(id="alc-cast", style={"display": "grid",
                                           "gridTemplateColumns": "repeat(auto-fill, minmax(76px, 1fr))",
                                           "gap": "6px", "alignContent": "start", "flex": "1 1 0",
                                           "minHeight": "0", "overflowY": "auto", "padding": "2px"}),
        ],
        style={"flex": "0 0 250px", "minWidth": "0", "minHeight": "0", "display": "flex",
               "flexDirection": "column", **_ui.card(padding="12px", marginBottom="0")},
    )


def _stage_column():
    stage = html.Div(
        [
            # No style prop: alchemy.js owns the bar, and React must not reset what it sets.
            html.Div(html.Div(className="alc-loadbar-fill"), id="alc-loadbar", className="alc-loadbar"),
            html.Div(id="alc-stage-meta", style={"display": "none"}),
            html.Div(id="alc-keep-meta", style={"display": "none"}),
            html.Div(
                [
                    html.Img(id="alc-before", className="alc-before"),
                    html.Div(html.Img(id="alc-image"), className="alc-after"),
                    html.Div(className="alc-handle"),
                    html.Span("before", className="alc-tag alc-tag-l"),
                    html.Span(id="alc-after-tag", children="after", className="alc-tag alc-tag-r"),
                ],
                id="alc-compare", className="alc-compare"),
            html.Div(id="alc-empty", style=_ui.hint(fontSize="12.5px", padding="40px 10px", textAlign="center",
                                                    maxWidth="none")),
        ],
        style={"position": "relative", "textAlign": "center", "paddingTop": "8px"},
    )
    return html.Div(
        [
            html.Div([
                dcc.RadioItems(id="alc-view", options=[{"label": " before and after", "value": "compare"},
                                                       {"label": " mask", "value": "mask"}],
                               value="compare", inline=True, labelStyle={"marginRight": "12px", **SMALL}),
                dcc.Dropdown(id="alc-size", options=[{"label": f"preview {s}", "value": s} for s in (512, 640, 768, 1024)],
                             value=640, clearable=False, style={**SMALL, "width": "128px"}),
                dcc.Checklist(id="alc-live", options=[{"label": " live", "value": "on"}], value=["on"], style=SMALL),
                html.Button("Draw", id="alc-draw", n_clicks=0,
                            style=_ui.button("secondary", padding="5px 11px", fontSize="12px")),
                html.Div(style={"flex": "1"}),
                dcc.Dropdown(id="alc-keep-size",
                             options=[{"label": "keep at full size", "value": 0}]
                             + [{"label": f"keep at {s}", "value": s} for s in (2048, 4096)],
                             value=0, clearable=False, style={**SMALL, "width": "150px"}),
                html.Button("Keep", id="alc-keep", n_clicks=0,
                            style=_ui.button("primary", padding="6px 14px", fontSize="12.5px")),
            ], style=_ui.row("10px", marginBottom="6px")),
            stage,
            html.Div(id="alc-info", style=_ui.hint(fontSize="11.5px", marginTop="8px", maxWidth="none",
                                                   textAlign="center")),
            html.Div(id="alc-status", style=_ui.hint(fontSize="12px", marginTop="4px", maxWidth="none",
                                                     color=_ui.INK, textAlign="center")),
            html.Div(id="alc-keep-status", style=_ui.hint(fontSize="12px", marginTop="4px", maxWidth="none",
                                                          color=_ui.OK, textAlign="center")),
        ],
        style={"flex": "1 1 auto", "minWidth": "0", "minHeight": "0", "overflowY": "auto",
               **_ui.card(padding="12px", marginBottom="0")},
    )


def _layer_row(i):
    kids = []
    if i > 1:
        kids.append(dcc.Dropdown(id=f"alc-comb-{i}", options=[{"label": c, "value": c} for c in ("and", "or", "but not")],
                                 value="and", clearable=False, style={**SMALL, "width": "92px"}))
    kids.append(dcc.Dropdown(id=f"alc-layer-{i}", options=_layer_options(), value="none", clearable=False,
                             style={**SMALL, "flex": "1", "minWidth": "0"}))
    return html.Div([
        html.Div(kids, style={"display": "flex", "gap": "6px", "alignItems": "center"}),
        dcc.Input(id=f"alc-words-{i}", type="text", debounce=True, placeholder="what to find, such as woman, sky",
                  style={"display": "none"}),
        html.Div([html.Div("coverage", style=_ui.hint(fontSize="11px")),
                  _slider(f"alc-cov-{i}", 0.05, 1, 0.05, 0.5)], id=f"alc-cov-wrap-{i}", style={"display": "none"}),
    ], style={"marginBottom": "8px"})


def _settings_column():
    weights = [html.Div([html.Div(f"share of [{i}]", style=KNOB_LABEL), _slider(f"alc-w{i}", 0, 1, 0.05, 1.0)],
                        id=f"alc-w-wrap-{i}", style={"display": "none"}) for i in (1, 2, 3)]
    finish_groups = []
    for group in dict.fromkeys(g for _, g, *_ in _finish.SPEC):
        rows = [html.Div([html.Div(label, title=doc, style=KNOB_LABEL),
                          _slider({"type": "alc-fin", "key": key}, lo, hi, step, FIN_DEFAULT[key])])
                for key, g, label, lo, hi, step, doc in _finish.SPEC if g == group]
        finish_groups.append(_sub(group, rows, open_=group == "light"))
    return html.Div(
        [
            _section("Recipe", [
                html.Div([
                    dcc.Dropdown(id="alc-recipe", options=_recipe_options(), placeholder="Choose a recipe...",
                                 clearable=True, style={**SMALL, "flex": "1", "minWidth": "0"}),
                    html.Button("Apply", id="alc-recipe-apply", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 11px", fontSize="12px")),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div(id="alc-recipe-doc", style=_ui.hint(fontSize="11.5px", marginTop="6px")),
                _sub("keep these settings as a recipe", [
                    dcc.Input(id="alc-recipe-name", type="text", placeholder="Recipe name...",
                              style=_ui.input_box(width="100%", boxSizing="border-box", padding="6px 9px",
                                                  fontSize="12px", marginTop="6px")),
                    html.Button("Save recipe", id="alc-recipe-save", n_clicks=0,
                                style=_ui.button("secondary", padding="5px 11px", fontSize="12px", marginTop="6px")),
                    html.Div(id="alc-recipe-status", style=_ui.hint(fontSize="11.5px", marginTop="6px")),
                ]),
            ], note="sets only what it holds"),
            _section("Method", [
                dcc.RadioItems(id="alc-family", options=[{"label": " " + FAMILY_LABEL[f], "value": f}
                                                         for f in _methods.FAMILIES],
                               value="tokens", inline=True, labelStyle={"marginRight": "10px", **SMALL}),
                html.Div(id="alc-family-doc", style=_ui.hint(fontSize="11px", margin="4px 0 8px")),
                dcc.Dropdown(id="alc-method", options=_method_options("tokens"), value="tokens.mix",
                             clearable=False, style=SMALL),
                html.Div(id="alc-method-doc", style=_ui.hint(fontSize="11.5px", margin="6px 0 8px")),
                html.Div(id="alc-knobs-main"),
                _sub("more", html.Div(id="alc-knobs-more"), id_="alc-more-sec"),
                _sub("looks", html.Div(id="alc-knobs-looks"), id_="alc-looks-sec"),
            ]),
            _section("Pictures", [
                html.Div("WHO LEADS", style=LABEL),
                dcc.RadioItems(id="alc-lead", options=[{"label": " the picture", "value": "picture"},
                                                       {"label": " the material", "value": "material"}],
                               value="picture", inline=True, labelStyle={"marginRight": "12px", **SMALL}),
                html.Div("With the material leading, the picture is hidden in material [1].",
                         style=_ui.hint(fontSize="11px")),
                *weights,
                html.Div("MATERIAL ZOOM", style=LABEL),
                _slider("alc-zoom", 0.5, 4, 0.25, 1.0),
                html.Div("CROP AND SEED", style=LABEL),
                dcc.Input(id="alc-crop", type="number", min=0, max=99999, step=1, value=0, debounce=True,
                          style=_ui.input_box(width="90px", padding="5px 8px", fontSize="12px")),
            ], open_=False, note="lead, shares, zoom"),
            _section("Mask", [
                *[_layer_row(i) for i in range(1, LAYERS + 1)],
                dcc.Checklist(id="alc-invert", options=[{"label": " invert", "value": "on"}], value=[],
                              style={**SMALL, "margin": "4px 0"}),
                _sub("refine the edge", [html.Div([html.Div(label, title=doc, style=KNOB_LABEL),
                                                   _slider(f"alc-{key}", lo, hi, step, default)])
                                         for key, label, lo, hi, step, default, doc in REFINE]),
            ], open_=False, note="where the step shows"),
            _section("Finish", [
                dcc.RadioItems(id="alc-fin-scope", options=[{"label": " whole picture", "value": "whole"},
                                                            {"label": " inside the mask", "value": "mask"}],
                               value="whole", inline=True, labelStyle={"marginRight": "10px", **SMALL}),
                *finish_groups,
                html.Button("Reset finish", id="alc-fin-reset", n_clicks=0,
                            style=_ui.button("ghost", padding="4px 8px", fontSize="11.5px", marginTop="8px")),
            ], open_=False, note="light and colour"),
        ],
        style={"flex": "0 0 370px", "minWidth": "0", "minHeight": "0", "overflowY": "auto", "paddingRight": "4px"},
    )


def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="alc-roles", storage_type="local"),
            dcc.Store(id="alc-knob-seed", storage_type="memory"),
            dcc.Store(id="alc-before-key", storage_type="memory"),
            dcc.Store(id="alc-request", storage_type="memory"),
            _cast_column(),
            _stage_column(),
            _settings_column(),
        ],
        id="alchemy-panel",
        style={"display": "none"},
    )


# ─────────────────────────── from controls to op arguments ───────────────────────────

def _controls(kind):
    """Every control a step reads, as Inputs (the preview) or States (Keep, Save recipe)."""
    C = Input if kind == "input" else State
    d = dict(roles=C("alc-roles", "data"), w1=C("alc-w1", "value"), w2=C("alc-w2", "value"),
             w3=C("alc-w3", "value"), lead=C("alc-lead", "value"), zoom=C("alc-zoom", "value"),
             crop=C("alc-crop", "value"), method=C("alc-method", "value"),
             knobs=C({"type": "alc-knob", "key": ALL}, "value"),
             invert=C("alc-invert", "value"), fins=C({"type": "alc-fin", "key": ALL}, "value"),
             scope=C("alc-fin-scope", "value"))
    for i in range(1, LAYERS + 1):
        d[f"layer{i}"] = C(f"alc-layer-{i}", "value")
        d[f"words{i}"] = C(f"alc-words-{i}", "value")
        d[f"cov{i}"] = C(f"alc-cov-{i}", "value")
        if i > 1:
            d[f"comb{i}"] = C(f"alc-comb-{i}", "value")
    for key, *_ in REFINE:
        d[key] = C(f"alc-{key}", "value")
    return d


ID_STATES = dict(knob_ids=State({"type": "alc-knob", "key": ALL}, "id"),
                 fin_ids=State({"type": "alc-fin", "key": ALL}, "id"))


def _source(choice, words):
    if choice in (None, "none") or str(choice).startswith("__group"):
        return None
    if choice == WORDS:
        return f"prompt:{words.strip()}" if (words or "").strip() else None
    if choice == CUSTOM:
        return (words or "").strip() or None
    return choice


def settings(v: dict, with_pictures=True) -> dict:
    """The op arguments the controls describe."""
    method = _methods.get(v["method"])
    names = {k.key: k for k in method.knobs}
    knobs = {}
    for kid, val in zip(v.get("knob_ids") or [], v.get("knobs") or []):
        k = names.get(kid["key"])
        if k is not None:
            knobs[k.key] = bool(val) if k.kind == "bool" else val
    layers, combine, coverage = [], [], []
    for i in range(1, LAYERS + 1):
        src = _source(v[f"layer{i}"], v[f"words{i}"])
        if src is None:
            continue
        if layers:
            combine.append(v.get(f"comb{i}") or "and")
        layers.append(src)
        coverage.append(float(v[f"cov{i}"] if v[f"cov{i}"] is not None else 0.5))
    sets = []
    if layers:
        sets += [f"mask.{key}={v[key]}" for key, *_ in REFINE if v.get(key) is not None]
        if v.get("invert"):
            sets.append("mask.invert=true")
    finish = {fid["key"]: val for fid, val in zip(v.get("fin_ids") or [], v.get("fins") or [])
              if val is not None and val != FIN_DEFAULT.get(fid["key"])}
    if finish and v.get("scope") == "mask":
        finish["scope"] = "mask"
    out = {"method": method.id, "knobs": knobs, "lead": v.get("lead") or "picture",
           "zoom": float(v.get("zoom") or 1.0), "crop": int(v.get("crop") or 0),
           "mask": layers, "combine": combine, "coverage": coverage, "set": sets, "finish": finish}
    if with_pictures:
        roles = v.get("roles") or {}
        mats, weights = [], []
        for i, w in (("1", v["w1"]), ("2", v["w2"]), ("3", v["w3"])):
            if roles.get(i):
                mats.append(roles[i])
                weights.append(float(w if w is not None else 1.0))
        out.update(picture=roles.get("T"), material=mats, weight=weights)
    return out


def _call(name, args):
    try:
        return _ops.call(name, args, actor="person"), None
    except _ops.Refusal as e:
        return None, e.reason
    except Exception as e:                       # a crash in an engine reaches the page as words
        return None, f"{type(e).__name__}: {e}"


def _src(att):
    return "data:image/jpeg;base64," + base64.b64encode(att.data).decode()


# ─────────────────────────── callbacks ───────────────────────────

def register(app) -> None:
    try:
        from .alchemy.memory import MODELS
    except ImportError:                               # pragma: no cover
        from alchemy.memory import MODELS
    MODELS.start()

    @app.callback(Output("alchemy-panel", "style"), Input("mode-select", "value"))
    def _show(mode):
        return PANEL_VISIBLE if mode == "alchemy" else {"display": "none"}

    @app.callback(Output("alc-cast", "children"), Output("alc-slots", "children"),
                  *[Output(f"alc-w-wrap-{i}", "style") for i in (1, 2, 3)],
                  Input("collection-items", "data"), Input("alc-roles", "data"))
    def _cast(collection, roles):
        roles = roles or {}
        slots = []
        for r in ROLES:
            p = roles.get(r)
            label = "picture" if r == "T" else f"material {r}"
            body = html.Img(src=_thumb(p), style={"width": "100%", "aspectRatio": "1", "objectFit": "cover",
                                                  "borderRadius": "5px", "display": "block"}) if p else \
                html.Div(label, style={"aspectRatio": "1", "display": "flex", "alignItems": "center",
                                       "justifyContent": "center", "fontSize": "10.5px", "color": _ui.INK_FAINT,
                                       "border": f"1px dashed {_ui.LINE}", "borderRadius": "5px",
                                       "textAlign": "center"})
            slots.append(html.Div([html.Div(f"[{r}]", style={"fontSize": "10px", "fontWeight": "700",
                                                             "color": ROLE_COLOUR[r], "marginBottom": "2px"}),
                                   body]))
        mats = [r for r in ("1", "2", "3") if roles.get(r)]
        shares = [{} if len(mats) > 1 and r in mats else {"display": "none"} for r in ("1", "2", "3")]
        collection = collection or []
        if not collection:
            return [html.Div("The active collection is empty. Pictures added with + Collection on any tab "
                             "appear here.", style=_ui.hint(gridColumn="1 / -1", fontSize="11.5px"))], \
                slots, *shares
        tiles = []
        for path in collection:
            held = [r for r in ROLES if roles.get(r) == path]
            border = f"2px solid {ROLE_COLOUR[held[0]]}" if held else f"1px solid {_ui.LINE}"
            chips = [html.Button(r, id={"type": "alc-role", "role": r, "path": path}, n_clicks=0,
                                 title="the picture" if r == "T" else f"material {r}",
                                 style={"flex": "1", "padding": "0", "height": "18px", "fontSize": "10px",
                                        "fontWeight": "700", "cursor": "pointer", "borderRadius": "3px",
                                        "border": f"1px solid {_ui.LINE}",
                                        "backgroundColor": ROLE_COLOUR[r] if r in held else _ui.ROLE_OFF,
                                        "color": "#fff" if r in held else _ui.INK_DIM})
                     for r in ROLES]
            tiles.append(html.Div([
                html.Img(src=_thumb(path), style={"width": "100%", "aspectRatio": "1", "objectFit": "cover",
                                                  "borderRadius": "5px", "border": border, "display": "block"}),
                html.Div(chips, style={"display": "flex", "gap": "2px", "marginTop": "3px"}),
            ]))
        return tiles, slots, *shares

    @app.callback(Output("alc-roles", "data"),
                  Input({"type": "alc-role", "role": ALL, "path": ALL}, "n_clicks"),
                  State("alc-roles", "data"), prevent_initial_call=True)
    def _role(_clicks, roles):
        hit = ctx.triggered_id
        if not isinstance(hit, dict) or not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update                     # tiles redrawn with n_clicks 0, not a click
        roles = dict(roles or {})
        r, path = hit["role"], hit["path"]
        roles[r] = None if roles.get(r) == path else path
        return roles

    @app.callback(Output("alc-family", "value"), Output("alc-method", "options"), Output("alc-method", "value"),
                  Output("alc-knob-seed", "data"), Output("alc-family-doc", "children"),
                  Input("alc-family", "value"), Input("alc-recipe-apply", "n_clicks"),
                  State("alc-method", "value"), State("alc-recipe", "value"))
    def _choose_method(family, _apply, current, recipe):
        if ctx.triggered_id == "alc-recipe-apply":
            if not recipe:
                return (no_update,) * 5
            r = _recipes.get(recipe)
            m = r.settings.get("method")
            if not m:
                return (no_update,) * 5
            fam = _methods.get(m).family
            return (fam, _method_options(fam), m, {"method": m, "knobs": r.settings.get("knobs") or {}},
                    _methods.FAMILIES[fam])
        opts = _method_options(family)
        value = current if current in {o["value"] for o in opts} else opts[0]["value"]
        return no_update, opts, value, no_update, _methods.FAMILIES[family]

    @app.callback(Output("alc-knobs-main", "children"), Output("alc-knobs-more", "children"),
                  Output("alc-more-sec", "style"), Output("alc-knobs-looks", "children"),
                  Output("alc-looks-sec", "style"), Output("alc-method-doc", "children"),
                  Input("alc-method", "value"), Input("alc-knob-seed", "data"))
    def _knobs(method_id, seed):
        m = _methods.get(method_id)
        given = (seed or {}).get("knobs") if (seed or {}).get("method") == method_id else {}
        values = _methods.settle_knobs(m, given or {})
        parts = {s: [] for s in _methods.SECTIONS}
        for k in m.knobs:
            lead = k.section == "main" and not parts["main"]
            parts[k.section].append(_knob_row(k, values[k.key], lead=lead))
        lo, hi = m.materials
        mats = f"{lo}" if lo == hi else f"{lo} to {hi}"
        doc = f"{m.doc}. Takes {mats} material{'s' if hi != 1 else ''}."
        if m.cost != "live":
            doc += " Takes several seconds, so it draws on Draw."
        shown = lambda rows: {} if rows else {"display": "none"}  # noqa: E731
        return (parts["main"], parts["more"], shown(parts["more"]), parts["looks"], shown(parts["looks"]), doc)

    @app.callback(Output({"type": "alc-knob-wrap", "key": ALL}, "style"),
                  Input({"type": "alc-knob", "key": ALL}, "value"),
                  State({"type": "alc-knob", "key": ALL}, "id"), State("alc-method", "value"))
    def _knob_shown(values, ids, method_id):
        m = _methods.get(method_id)
        names = {k.key: k for k in m.knobs}
        vals = {i["key"]: v for i, v in zip(ids, values)}
        return [{} if names.get(i["key"]) is None or _methods.applies(names[i["key"]], vals) else {"display": "none"}
                for i in ids]

    for i in range(1, LAYERS + 1):
        @app.callback(Output(f"alc-words-{i}", "style"), Output(f"alc-words-{i}", "placeholder"),
                      Output(f"alc-cov-wrap-{i}", "style"), Output(f"alc-cov-{i}", "value"),
                      Input(f"alc-layer-{i}", "value"))
        def _layer(choice):
            box = _ui.input_box(width="100%", boxSizing="border-box", padding="5px 8px", fontSize="12px",
                                marginTop="4px")
            if choice == WORDS:
                return box, "what to find, such as woman, sky, rust", {"display": "none"}, no_update
            if choice == CUSTOM:
                return box, "a source line, such as colour:#3a6ea5 or spot:0.5,0.4,0.2", {}, no_update
            p = _mask.PRESET.get(choice)
            if p is not None and p.coverage is not None:
                return {"display": "none"}, no_update, {}, p.coverage
            return {"display": "none"}, no_update, {"display": "none"}, no_update

    recipe_outputs = [Output("alc-recipe-doc", "children"), Output("alc-lead", "value"), Output("alc-zoom", "value"),
                      Output({"type": "alc-fin", "key": ALL}, "value"), Output("alc-fin-scope", "value"),
                      Output("alc-invert", "value")]
    for i in range(1, LAYERS + 1):
        recipe_outputs += [Output(f"alc-layer-{i}", "value"), Output(f"alc-words-{i}", "value")]
        if i > 1:
            recipe_outputs.append(Output(f"alc-comb-{i}", "value"))
    recipe_outputs += [Output(f"alc-{key}", "value") for key, *_ in REFINE]
    rest_n = len(recipe_outputs) - 5            # invert, the layer rows and the refinements

    @app.callback(*recipe_outputs, Input("alc-recipe-apply", "n_clicks"), Input("alc-fin-reset", "n_clicks"),
                  Input("alc-recipe", "value"), State({"type": "alc-fin", "key": ALL}, "id"),
                  prevent_initial_call=True)
    def _apply_recipe(_apply, _reset, recipe, fin_ids):
        keep = [no_update] * rest_n
        same = [no_update] * len(fin_ids)
        if ctx.triggered_id == "alc-fin-reset":
            return [no_update, no_update, no_update, [FIN_DEFAULT[f["key"]] for f in fin_ids], "whole", *keep]
        r = _recipes.get(recipe) if recipe else None
        doc = r.doc if r else ""
        if ctx.triggered_id == "alc-recipe" or r is None:
            return [doc, no_update, no_update, same, no_update, *keep]
        s = r.settings
        fin = s.get("finish")
        fins = [fin.get(f["key"], no_update) for f in fin_ids] if fin else same
        scope = fin.get("scope", "whole") if fin else no_update
        mask = s.get("mask")
        if mask is None:
            rest = keep
        else:
            spec = _mask.MaskSpec.of(mask)
            rest = [["on"] if spec.invert else []]
            for i in range(1, LAYERS + 1):
                line = spec.layers[i - 1] if i - 1 < len(spec.layers) else "none"
                if line in _mask.PRESET or line == "none":
                    rest += [line, ""]
                elif str(line).startswith("prompt:"):
                    rest += [WORDS, line[7:]]
                else:
                    rest += [CUSTOM, line]
                if i > 1:
                    rest.append(spec.combine[i - 2] if i - 2 < len(spec.combine) else "and")
            rest += [getattr(spec, key) for key, *_ in REFINE]
        return [doc, s.get("lead", no_update), s.get("zoom", no_update), fins, scope, *rest]

    # What the stage is drawing, for alchemy.js to time the load bar by: method, model, view, size.
    app.clientside_callback(
        """function (m, v, s, k, knobs, ids) {
            var model = '';
            (ids || []).forEach(function (id, i) { if (id.key === 'model') model = '|' + knobs[i]; });
            return [m + model + '|' + v + '|' + s, 'keep|' + m + model + '|' + k];
        }""",
        Output("alc-stage-meta", "title"), Output("alc-keep-meta", "title"),
        Input("alc-method", "value"), Input("alc-view", "value"), Input("alc-size", "value"),
        Input("alc-keep-size", "value"), Input({"type": "alc-knob", "key": ALL}, "value"),
        State({"type": "alc-knob", "key": ALL}, "id"),
    )

    # The gate: one preview out at a time. A change while one is out is held, and when the preview
    # lands, alchemy.js sends the newest state at once (window.alcGate.release). A drag therefore
    # redraws as fast as the step can be drawn, instead of queueing a request per slider tick.
    gate_inputs = list(_controls("input").values()) + [
        Input("alc-view", "value"), Input("alc-size", "value"), Input("alc-draw", "n_clicks")]
    app.clientside_callback(
        """function () {
            var fired = (dash_clientside.callback_context.triggered || []).map(function (t) { return t.prop_id; });
            var draw = fired.some(function (p) { return p.indexOf('alc-draw.') === 0; });
            return window.alcGate ? window.alcGate.ask(draw) : {seq: Date.now(), draw: draw};
        }""",
        Output("alc-request", "data"), *gate_inputs,
    )

    preview_state = {k: State(dep.component_id, dep.component_property)
                     for k, dep in _controls("input").items()}
    preview_state.update(view=State("alc-view", "value"), size=State("alc-size", "value"),
                         live=State("alc-live", "value"), before_key=State("alc-before-key", "data"), **ID_STATES)

    @app.callback(Output("alc-image", "src"), Output("alc-before", "src"), Output("alc-after-tag", "children"),
                  Output("alc-info", "children"), Output("alc-status", "children"),
                  Output("alc-compare", "style"), Output("alc-empty", "children"), Output("alc-before-key", "data"),
                  inputs={"req": Input("alc-request", "data")}, state=preview_state, prevent_initial_call=True)
    def _preview(req, **v):
        # Every path answers, so the stage image's loading mark clears and the gate opens again.
        try:
            return _drawn(req or {}, v)
        except Exception as e:                   # the page shows it, and the gate stays open
            return no_update, no_update, no_update, "", f"{type(e).__name__}: {e}", no_update, "", no_update

    def _drawn(req, v):
        roles = v.get("roles") or {}
        method = _methods.get(v["method"])
        hidden = {"display": "none"}
        if not roles.get("T") and not any(roles.get(r) for r in ("1", "2", "3")):
            return (no_update, no_update, no_update, "", "", hidden,
                    "Give a picture [T] and a material [1] from the collection on the left.", no_update)
        if not req.get("draw") and (not v.get("live") or (method.cost != "live" and v.get("view") == "compare")):
            return (no_update,) * 4 + ("Draw shows this step.", no_update, "", no_update)
        args = settings(v)
        mask_view = v.get("view") == "mask"
        args.update(view="mask" if mask_view else "compare", size=int(v.get("size") or 640))
        res, err = _call("alchemy preview", args)
        before = None
        if res is not None and mask_view:
            b, err = _call("alchemy preview", {**args, "view": "before"})
            before = b.attachments[0] if b and b.attachments else None
        if err:
            return no_update, no_update, no_update, "", err, no_update, "", no_update
        after = res.attachments[0]
        before = before or (res.attachments[1] if len(res.attachments) > 1 else after)
        # The picture as it was rarely changes between previews, so it is sent only when it does.
        key = hashlib.blake2b(before.data, digest_size=8).hexdigest()
        before_src = no_update if key == v.get("before_key") else _src(before)
        bits = [res.method, f"{res.size[0]} x {res.size[1]}", f"{res.ms} ms"]
        if res.mask_share is not None:
            bits.append(f"mask {res.mask_share:.0%}")
        return (_src(after), before_src, "mask" if mask_view else "after", "  ·  ".join(bits), "", {}, "", key)

    keep_states = _controls("state")

    @app.callback(Output("alc-keep-status", "children"),
                  Output("collection-rev", "data", allow_duplicate=True),
                  inputs={"n": Input("alc-keep", "n_clicks")},
                  state={**keep_states, **ID_STATES, "keep_size": State("alc-keep-size", "value")},
                  running=[(Output("alc-keep", "disabled"), True, False)],
                  prevent_initial_call=True)
    def _keep(n, **v):
        if not n:
            return no_update, no_update
        args = settings(v)
        args.update(size=int(v.get("keep_size") or 0))
        res, err = _call("alchemy make", args)
        if err:
            return err, no_update
        where = os.path.relpath(res.path, _paths.subdir("output")).replace(os.sep, "/")
        try:
            slug = _collection.add([res.id])
        except _ops.Refusal as e:
            return f"Kept {res.id} as {where} in output, but could not add it to the collection: {e.reason}", no_update
        return (f"Kept {res.size[0]} x {res.size[1]} as {res.id}, in output/{where}, and added it to {slug}.",
                _collection.rev())

    @app.callback(Output("alc-recipe-status", "children"), Output("alc-recipe", "options"),
                  inputs={"n": Input("alc-recipe-save", "n_clicks")},
                  state={**keep_states, **ID_STATES, "name": State("alc-recipe-name", "value")},
                  prevent_initial_call=True)
    def _save_recipe(n, **v):
        if not n:
            return no_update, no_update
        args = settings(v, with_pictures=False)
        args = {k: val for k, val in args.items() if val not in ({}, [], None)}
        args["name"] = (v.get("name") or "").strip()
        if not args["name"]:
            return "Give the recipe a name.", no_update
        res, err = _call("alchemy recipe save", args)
        if err:
            return err, no_update
        return f"Saved {res.recipe.name}.", _recipe_options()
