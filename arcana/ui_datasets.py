# ui_datasets.py — the dataset manager panel
#
# Building a dataset used to mean finding a terminal and getting
# `arcana-build-latent --imgs_path ... --name ... --n_components 2` right, which
# rules out anyone who does not already know the tool. This is that command as a
# panel: pick a folder, pick an encoder, watch a progress bar.
#
# Kept out of arcana.py because that file is already 3,700 lines. layout() and
# register(app) are the whole interface.

from __future__ import annotations

import os

from dash import ALL, dcc, html, Input, Output, State, ctx, no_update

try:
    from . import models as _models
    from . import paths as _paths
    from . import db as _db
    from .jobs import MANAGER
    from . import gpu as _gpu
except ImportError:                      # loose-script fallback
    import models as _models
    import paths as _paths
    import db as _db
    from jobs import MANAGER
    import gpu as _gpu

# ── a small design system, shared by this panel and the moodboard ─────────────
#
# Colour carries meaning here rather than decoration:
#   cyan   the one primary action in a section
#   green  something finished or healthy
#   amber  needs attention but is not broken
#   red    broken, or destructive
#   grey   everything else
#
# Buttons are sized to their label. Full-width buttons read as banners and make
# a panel look like a form to fill top-to-bottom rather than a set of choices.

INK = "#ececf0"
INK_DIM = "#9a9aa2"
INK_FAINT = "#6c6c74"
SURFACE = "#1b1b1f"
SURFACE_2 = "#232329"
LINE = "#33333b"

ACCENT = "#00bcd4"
OK = "#43b581"
WARN = "#d99a34"
BAD = "#e05252"

CARD = {"backgroundColor": SURFACE, "border": f"1px solid {LINE}",
        "borderRadius": "10px", "padding": "18px 20px", "marginBottom": "14px"}

SECTION_TITLE = {"fontSize": "13px", "fontWeight": "700", "color": INK,
                 "letterSpacing": "0.3px", "marginBottom": "2px"}
SECTION_HINT = {"fontSize": "12px", "color": INK_DIM, "marginBottom": "16px",
                "lineHeight": "1.5", "maxWidth": "62ch"}
FIELD_LABEL = {"fontSize": "11px", "color": INK_FAINT, "marginBottom": "5px",
               "textTransform": "uppercase", "letterSpacing": "0.6px"}

INPUT = {"backgroundColor": SURFACE_2, "color": INK,
         "border": f"1px solid {LINE}", "borderRadius": "6px",
         "padding": "8px 10px", "fontSize": "13px", "width": "100%"}


def _btn(kind="primary", **extra):
    base = {"padding": "8px 15px", "borderRadius": "6px", "fontSize": "12.5px",
            "fontWeight": "600", "cursor": "pointer", "border": "1px solid transparent",
            "whiteSpace": "nowrap", "width": "auto", "flex": "0 0 auto"}
    if kind == "primary":
        base |= {"backgroundColor": ACCENT, "color": "#08272c"}
    elif kind == "secondary":
        base |= {"backgroundColor": "transparent", "color": INK,
                 "border": f"1px solid {LINE}"}
    elif kind == "danger":
        base |= {"backgroundColor": "transparent", "color": BAD,
                 "border": f"1px solid {BAD}44"}
    return base | extra


BTN = _btn("primary")
BTN_QUIET = _btn("secondary")
BTN_WARN = _btn("danger")


def _pill(text, tone=INK_FAINT):
    return html.Span(text, style={
        "fontSize": "10px", "fontWeight": "600", "letterSpacing": "0.4px",
        "padding": "3px 8px", "borderRadius": "100px",
        "color": tone, "border": f"1px solid {tone}55",
        "backgroundColor": f"{tone}14", "whiteSpace": "nowrap"})


def _field(label, control, grow="1", min_width="200px"):
    return html.Div([html.Div(label, style=FIELD_LABEL), control],
                    style={"flex": grow, "minWidth": min_width})


def _model_options(modality: str, n_items: int = 0, decode_ms: float | None = None):
    """Dropdown options that say what each choice costs."""
    rows = _models.catalogue(modality, n_items=n_items, decode_ms=decode_ms)
    opts = []
    for r in rows:
        bits = [f"{r['dim']}-d"]
        bits.append("ready" if r["downloaded"] else f"{r['download_mb']:,} MB download")
        if r["estimate"]:
            bits.append(r["estimate"])
        opts.append({"label": f"{r['label']}  ·  {' · '.join(bits)}", "value": r["id"]})
    return opts


# ──────────────────────────────────────────────────────────────────────────────
# layout
# ──────────────────────────────────────────────────────────────────────────────
def _step(n, title, hint, body, **extra):
    """
    One numbered step.

    Indexing is a sequence -- point at a folder, say what it is, press go -- but
    the panel used to present it as one undifferentiated form with eight
    controls and no indication of what depended on what. Numbering turns "which
    of these do I have to fill in?" into "do this, then this".
    """
    return html.Div(
        style={"display": "flex", "gap": "14px", "padding": "16px 0",
               "borderTop": "1px solid " + LINE} | extra,
        children=[
            html.Div(str(n), style={
                "flex": "0 0 24px", "height": "24px", "borderRadius": "100px",
                "backgroundColor": SURFACE_2, "color": INK_DIM,
                "border": "1px solid " + LINE, "fontSize": "12px",
                "fontWeight": "700", "display": "flex", "alignItems": "center",
                "justifyContent": "center", "marginTop": "1px"}),
            html.Div(style={"flex": "1", "minWidth": "0"}, children=[
                html.Div(title, style={"fontSize": "13px", "fontWeight": "600",
                                       "color": INK, "marginBottom": "2px"}),
                html.Div(hint, style={"fontSize": "12px", "color": INK_DIM,
                                      "lineHeight": "1.5", "marginBottom": "12px",
                                      "maxWidth": "68ch"}),
                body,
            ]),
        ])


def layout() -> html.Div:
    default_model = _models.default_for("image")

    return html.Div(
        id="datasets-panel",
        style={"display": "none"},
        children=[
            dcc.Interval(id="dm-poll", interval=700, disabled=True),
            dcc.Store(id="dm-job", storage_type="memory"),
            # Bumped when a job finishes; the lists listen to this instead of
            # the progress timer.
            dcc.Store(id="dm-refresh", data=0, storage_type="memory"),

            # -- running job -------------------------------------------------
            html.Div(id="dm-job-card", style={**CARD, "display": "none"}, children=[
                html.Div(style={"display": "flex", "alignItems": "baseline",
                                "gap": "10px", "marginBottom": "10px"}, children=[
                    html.Div("Working", id="dm-job-title", style=SECTION_TITLE),
                    html.Div(id="dm-job-message",
                             style={"fontSize": "12.5px", "color": INK_DIM, "flex": "1"}),
                    html.Div(id="dm-job-detail",
                             style={"fontSize": "11px", "color": INK_FAINT,
                                    "fontVariantNumeric": "tabular-nums"}),
                ]),
                html.Div(style={"backgroundColor": SURFACE_2, "borderRadius": "100px",
                                "height": "6px", "overflow": "hidden"},
                         children=html.Div(id="dm-job-bar",
                                           style={"width": "0%", "height": "100%",
                                                  "backgroundColor": ACCENT,
                                                  "transition": "width .3s"})),
                html.Div(html.Button("Cancel", id="dm-cancel", n_clicks=0,
                                     style=BTN_QUIET),
                         style={"marginTop": "12px"}),
            ]),

            # -- add a dataset, as numbered steps ----------------------------
            html.Div(style=CARD, children=[
                html.Div("Add a dataset", style={**SECTION_TITLE, "fontSize": "15px"}),
                html.Div("Arcana reads your files where they are. Nothing is copied, "
                         "moved or uploaded \u2014 it builds a searchable index beside "
                         "them and leaves the originals alone.",
                         style={**SECTION_HINT, "marginBottom": "4px"}),

                _step(1, "Point at a folder",
                      "Everything inside it, including subfolders, gets indexed.",
                      html.Div([
                          html.Div(style={"display": "flex", "gap": "8px",
                                          "flexWrap": "wrap"}, children=[
                              dcc.Input(id="dm-folder", type="text", debounce=True,
                                        placeholder="paste a path, or use Browse",
                                        style={**INPUT, "flex": "1",
                                               "minWidth": "260px"}),
                              html.Button("Browse\u2026", id="dm-browse", n_clicks=0,
                                          style=BTN_QUIET),
                          ]),
                          html.Div(id="dm-folder-scan",
                                   style={"fontSize": "12.5px", "minHeight": "20px",
                                          "marginTop": "10px"}),
                      ]),
                      borderTop="none", paddingTop="10px"),

                _step(2, "Check what it found",
                      "The media type is detected from the folder -- change it if the "
                      "guess is wrong. The name is what you will pick from the Dataset "
                      "menu later.",
                      html.Div(style={"display": "flex", "gap": "18px",
                                      "flexWrap": "wrap", "alignItems": "flex-end"},
                               children=[
                          _field("Media type",
                                 dcc.RadioItems(
                                     id="dm-modality", value="image",
                                     options=[{"label": "Images", "value": "image"},
                                              {"label": "Audio", "value": "audio"}],
                                     inline=True,
                                     inputStyle={"marginRight": "6px",
                                                 "accentColor": ACCENT},
                                     labelStyle={"marginRight": "18px", "color": INK,
                                                 "fontSize": "13px",
                                                 "cursor": "pointer"}),
                                 grow="0", min_width="190px"),
                          _field("Name",
                                 dcc.Input(id="dm-name", type="text", debounce=True,
                                           placeholder="taken from the folder",
                                           style=INPUT),
                                 grow="1", min_width="200px"),
                      ])),

                _step(3, "Choose the quality",
                      "A stronger encoder understands your prompts better but takes "
                      "longer to index. You can change your mind later by re-indexing.",
                      html.Div([
                          dcc.Dropdown(id="dm-model", clearable=False,
                                       value=default_model.id,
                                       options=_model_options("image"),
                                       style={"color": "#111", "fontSize": "12.5px"}),
                          html.Div(id="dm-model-note",
                                   style={"fontSize": "12px", "color": INK_DIM,
                                          "lineHeight": "1.5", "marginTop": "10px",
                                          "maxWidth": "70ch", "minHeight": "18px"}),
                          # What the app can actually see. The estimates above
                          # are computed from this, so when someone with a good
                          # card is told "on the CPU" they deserve to know why
                          # rather than assume the app is simply slow.
                          html.Div(id="dm-hardware",
                                   style={"fontSize": "11.5px", "color": INK_FAINT,
                                          "marginTop": "8px", "maxWidth": "70ch",
                                          "lineHeight": "1.5"}),
                      ])),

                _step(4, "How to name the groups",
                      "Arcana clusters the collection and gives each group a name. "
                      "Leave this alone and it picks both for you.",
                      html.Div([
                          html.Div(style={"display": "flex", "gap": "18px",
                                          "flexWrap": "wrap",
                                          "alignItems": "flex-end"}, children=[
                              _field("How many groups",
                                     dcc.Dropdown(
                                         id="dm-k", clearable=False, value=0,
                                         options=[{"label": "Automatic", "value": 0},
                                                  {"label": "6 — broad strokes", "value": 6},
                                                  {"label": "12", "value": 12},
                                                  {"label": "18", "value": 18},
                                                  {"label": "24 — fine grained", "value": 24}],
                                         style={"color": "#111", "fontSize": "12.5px"}),
                                     grow="0", min_width="210px"),
                              _field("Names come from",
                                     dcc.Dropdown(
                                         id="dm-vocab", clearable=False, value="builtin",
                                         options=[{"label": "Arcana's word list", "value": "builtin"},
                                                  {"label": "This folder's subfolder names", "value": "folders"},
                                                  {"label": "Don't name them", "value": "none"}],
                                         style={"color": "#111", "fontSize": "12.5px"}),
                                     grow="1", min_width="240px"),
                          ]),
                          html.Div(id="dm-label-note",
                                   style={"fontSize": "11.5px", "color": INK_DIM,
                                          "marginTop": "10px", "lineHeight": "1.5",
                                          "maxWidth": "70ch", "minHeight": "17px"}),
                      ])),

                _step(5, "Extras, then go",
                      "Palette and style power the moodboard's similarity search. They "
                      "cost time now and cannot be added later without re-reading every "
                      "file, so tick them if you think you might want them.",
                      html.Div([
                          dcc.Checklist(id="dm-features", inline=True,
                                        inputStyle={"marginRight": "5px",
                                                    "accentColor": ACCENT},
                                        labelStyle={"marginRight": "18px", "color": INK,
                                                    "fontSize": "12.5px"},
                                        options=[
                                            {"label": "Colour palette", "value": "palette"},
                                            {"label": "Style / texture", "value": "style"},
                                            {"label": "Thumbnails", "value": "thumbnails"},
                                        ], value=[]),
                          html.Div(style={"display": "flex", "gap": "14px",
                                          "alignItems": "center", "flexWrap": "wrap",
                                          "marginTop": "18px"}, children=[
                              html.Button("Start indexing", id="dm-start", n_clicks=0,
                                          style=BTN),
                              html.Div(id="dm-start-status",
                                       style={"fontSize": "12.5px", "flex": "1",
                                              "minWidth": "200px"}),
                          ]),
                      ])),
            ]),

            # -- existing datasets -------------------------------------------
            html.Div(style=CARD, children=[
                html.Div("Your datasets", style=SECTION_TITLE),
                html.Div("Green means the dataset is finished and every file is still "
                         "where the index expects it \u2014 only those appear in the "
                         "Dataset menu. The tags show what each one can do.",
                         style=SECTION_HINT),
                dcc.Store(id="dm-pending-delete", storage_type="memory"),
                # Rendered once, hidden until armed. Building it inside a
                # callback would mean dm-del-yes/dm-del-cancel do not exist at
                # page load, and a Dash callback whose Inputs are missing never
                # fires -- which is precisely how the first version of this
                # broke.
                html.Div(id="dm-delete-confirm",
                         style={"display": "none"},
                         children=[
                             html.Div(id="dm-delete-text"),
                             html.Div(style={"display": "flex", "gap": "8px",
                                             "marginTop": "10px"}, children=[
                                 html.Button("Delete", id="dm-del-yes", n_clicks=0,
                                             style=_btn("danger", padding="6px 12px",
                                                        fontSize="12px", color="#fff",
                                                        backgroundColor=BAD)),
                                 html.Button("Cancel", id="dm-del-cancel", n_clicks=0,
                                             style=_btn("secondary", padding="6px 12px",
                                                        fontSize="12px")),
                             ]),
                         ]),
                html.Div(id="dm-list"),
            ]),

            # -- encoders, folded away ---------------------------------------
            # Secondary: most people never open this. It matters when a model
            # needs downloading or labels preparing, and step 3's note says so
            # at the moment that is relevant.
            html.Details(style={**CARD, "marginBottom": "4px"}, children=[
                html.Summary("Encoders and labels", style={
                    **SECTION_TITLE, "cursor": "pointer", "marginBottom": "0",
                    "userSelect": "none"}),
                html.Div("Downloaded once each. A dataset can only be searched with "
                         "the encoder that built it, which is why changing encoder "
                         "means re-indexing.",
                         style={**SECTION_HINT, "marginTop": "10px"}),
                html.Div(id="dm-model-job",
                         style={"fontSize": "12px", "color": ACCENT,
                                "minHeight": "17px", "marginTop": "4px"}),
                html.Div(id="dm-models"),
            ]),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# helpers used by callbacks
# ──────────────────────────────────────────────────────────────────────────────
# Remove buttons use pattern-matching ids. The rest of this panel uses plain
# sequential ones, because pattern ids serialise to JSON and produce CSS
# selectors that break some tooling -- but that only works where the number of
# components is fixed. Declaring dm-del-0..59 for a list of ten datasets means
# fifty of the callback's Inputs do not exist, and Dash then never fires it at
# all: the buttons rendered and did nothing. A varying number of components is
# exactly what ALL is for.

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def next_name(folder: str, current_name: str | None, last_suggested: str | None) -> str | None:
    """
    What the Name field should become when the folder changes, or None to leave
    it alone.

    Three cases, and the third is the one that is easy to get wrong:

      empty field                 -> suggest from the folder
      field holds our last guess  -> replace it; the user never chose it, and
                                     leaving the previous folder's name behind
                                     is worse than an empty box because it
                                     looks deliberate
      field holds anything else   -> the user typed it; never touch it
    """
    suggested = suggest_name(folder)
    if not suggested:
        return None
    current = (current_name or "").strip()
    if not current or current == (last_suggested or ""):
        return suggested
    return None


def folder_vocabulary(root: str, limit: int = 60) -> list[str]:
    """
    Subfolder names, cleaned up, for use as a label vocabulary.

    Arcana names a cluster by finding the nearest word in a fixed 100-word list
    (Portrait, Street, Sunset...). That list knows nothing about the pictures:
    a library of circuit boards gets named out of a vocabulary with no word for
    circuit board. Most people have already written a better vocabulary without
    realising -- it is their folder tree.

    Only directories that actually contain media are counted, so an "originals"
    or "export" wrapper folder does not become a label.
    """
    if not root or not os.path.isdir(root):
        return []
    words: list[str] = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        has_media = False
        for _r, _d, files in os.walk(sub):
            if any(os.path.splitext(f)[1].lower() in IMAGE_EXTS | AUDIO_EXTS
                   for f in files):
                has_media = True
                break
        if not has_media:
            continue
        cleaned = " ".join(w for w in entry.replace("_", " ").replace("-", " ").split())
        if cleaned and cleaned.lower() not in {w.lower() for w in words}:
            words.append(cleaned)
        if len(words) >= limit:
            break
    return words


def scan_both(folder: str, cap: int = 200_000, keep: int = 40) -> dict:
    """
    Count images and audio in one walk, keeping a sample of each.

    Two separate scan_media() calls would walk the tree twice, and on a network
    drive or a 80k-file library that is the slowest thing the panel does. The
    counts are what let the media type be detected instead of asked for: a
    folder of .wav files is not ambiguous, and making someone tell the app what
    they just pointed it at is a question with a knowable answer.
    """
    n_img = n_aud = 0
    s_img: list[str] = []
    s_aud: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                n_img += 1
                if len(s_img) < keep:
                    s_img.append(os.path.join(root, f))
            elif ext in AUDIO_EXTS:
                n_aud += 1
                if len(s_aud) < keep:
                    s_aud.append(os.path.join(root, f))
            if n_img + n_aud >= cap:
                return {"image": n_img, "audio": n_aud,
                        "sample_image": s_img, "sample_audio": s_aud}
    return {"image": n_img, "audio": n_aud,
            "sample_image": s_img, "sample_audio": s_aud}


def suggest_name(folder: str) -> str:
    """
    A dataset name from the folder, so the field does not start empty.

    Lowercased, spaces and punctuation to hyphens, because the name ends up in
    filenames (index_<name>_<modality>.pkl).
    """
    base = os.path.basename(os.path.normpath(folder or ""))
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in base).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.lower()[:48]


def scan_media(folder: str, modality: str, cap: int = 200_000,
               keep: int = 40) -> tuple[int, list[str]]:
    """
    Count the media in a folder, and keep a handful of paths.

    The sample is what lets the time estimate be measured rather than guessed:
    decode cost dominates on a GPU and varies by an order of magnitude between
    a 2 MP PNG and a 24 MP JPEG.
    """
    exts = IMAGE_EXTS if modality == "image" else AUDIO_EXTS
    n = 0
    sample: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                n += 1
                if len(sample) < keep:
                    sample.append(os.path.join(root, f))
                if n >= cap:
                    return n, sample
    return n, sample


def count_media(folder: str, modality: str, cap: int = 200_000) -> int:
    return scan_media(folder, modality, cap, keep=0)[0]


def _pill(text: str, colour: str) -> html.Span:
    return html.Span(text, style={
        "fontSize": "10px", "padding": "2px 7px", "borderRadius": "3px",
        "backgroundColor": colour, "color": "#fff", "marginLeft": "8px"})


_HEALTH_CACHE: dict = {}


def invalidate_dataset_cache() -> None:
    _HEALTH_CACHE.clear()
    _EXTRAS_CACHE.clear()


_EXTRAS_CACHE: dict = {}


def _dataset_extras(d) -> dict:
    """
    What this dataset carries besides its vectors: named cluster labels, a
    colour-palette feature block, a style feature block.

    Palette and style are file-existence checks and effectively free. Labels
    need the 2-D latent frame, whose `label` column holds strings once the
    clusters have been named and plain cluster integers before that -- so the
    dtype is the test. The frames are small (3 KB - 9 MB) but this still runs
    for every dataset on every list rebuild, so results are cached against the
    file's mtime.
    """
    latent = (d.latent_paths or {}).get(2) or next(iter((d.latent_paths or {}).values()), None)
    key = (d.name, d.modality, latent,
           os.path.getmtime(latent) if latent and os.path.exists(latent) else 0)
    hit = _EXTRAS_CACHE.get(key)
    if hit is not None:
        return hit

    out = {
        "palette": bool(d.palette_path and os.path.exists(d.palette_path)),
        "style": bool(d.style_path and os.path.exists(d.style_path)),
        "labels": False,
    }
    if latent and os.path.exists(latent):
        try:
            import pickle
            with open(latent, "rb") as fh:
                df = pickle.load(fh)
            col = getattr(df, "columns", ())
            if "label" in col:
                out["labels"] = df["label"].dtype == object
        except Exception:
            pass
    _EXTRAS_CACHE[key] = out
    return out


def dataset_rows() -> list:
    """
    One row per dataset.

    dataset_health() unpickles an entire index to count its paths -- 188 MB for
    the largest one here -- so results are cached and only recomputed when
    something actually changes. Rebuilding this on a 700 ms timer made the app
    unresponsive.
    """
    try:
        from .relocate import dataset_health
        from .legacy import discover
    except ImportError:
        from relocate import dataset_health
        from legacy import discover

    found = discover()
    if not found:
        return [html.Div(style={"padding": "10px 0"}, children=[
            html.Div("No datasets yet.",
                     style={"fontSize": "13px", "color": INK, "fontWeight": "600",
                            "marginBottom": "4px"}),
            html.Div("Use the steps above to point Arcana at a folder of images or "
                     "audio. Once it has indexed one, it appears here and in the "
                     "Dataset menu at the top of every tab.",
                     style={"fontSize": "12px", "color": INK_DIM,
                            "lineHeight": "1.5", "maxWidth": "62ch"}),
        ])]

    # One job at a time, so a single lookup covers every row.
    try:
        active = MANAGER.active()
    except Exception:
        active = None

    rows = []
    for i, d in enumerate(found):
        key = (d.name, d.modality, d.index_path,
               os.path.getmtime(d.index_path) if os.path.exists(d.index_path) else 0)
        h = _HEALTH_CACHE.get(key)
        if h is None:
            try:
                h = dataset_health(d.name, d.modality, sample=40)
            except Exception:
                h = {"ok": True, "total": 0, "missing": 0, "root": "", "error": ""}
            _HEALTH_CACHE[key] = h

        ex = _dataset_extras(d)

        # A dataset is only usable once it has a 2-D map: get_matching_datasets()
        # requires BOTH an index and a latent file before it will offer the
        # dataset in the Dataset menu, while this list is built from discover(),
        # which needs only an index. Anything half-built therefore used to show
        # as "ready" here and be absent from the menu, with nothing on screen
        # explaining the difference -- the single most confusing thing about
        # this panel.
        has_map = bool(d.latent_paths)
        building = (active or {}).get("label") == "Indexing " + d.name

        if h.get("error"):
            pill, tone = _pill("unreadable", BAD), BAD
        elif building:
            pill, tone = _pill("building\u2026", ACCENT), ACCENT
        elif not has_map:
            pill, tone = _pill("unfinished", WARN), WARN
        elif h["missing"]:
            pill, tone = _pill("files missing", WARN), WARN
        else:
            pill, tone = _pill("ready", OK), OK

        rows.append(html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px",
                   "padding": "10px 0", "flexWrap": "wrap",
                   "borderTop": (f"1px solid {LINE}" if i else "none")},
            children=[
                # A thin status stripe reads faster than a word at the far right.
                html.Div(style={"width": "3px", "alignSelf": "stretch",
                                "minHeight": "22px", "borderRadius": "2px",
                                "backgroundColor": tone}),
                html.Span(d.name, style={"fontWeight": "600", "fontSize": "13px",
                                         "color": INK, "minWidth": "150px"}),
                _pill(d.modality, ACCENT if d.modality == "image" else "#a982d9"),
                html.Span(f"{h['total']:,} items" if h["total"] else "\u2014",
                          style={"fontSize": "11.5px", "color": INK_DIM,
                                 "minWidth": "80px",
                                 "fontVariantNumeric": "tabular-nums"}),
                html.Span(h.get("root", ""), title=h.get("root", ""),
                          style={"fontSize": "11px", "color": INK_FAINT, "flex": "1",
                                 "minWidth": "160px", "overflow": "hidden",
                                 "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                # What the dataset can actually do. Palette and style gate the
                # moodboard's similarity search; without them "Find Similar"
                # just refuses, and until now the only way to find out was to
                # try it.
                html.Div(
                    [
                        _pill("labels", OK if ex["labels"] else INK_FAINT),
                        _pill("palette", OK if ex["palette"] else INK_FAINT),
                        _pill("style", OK if ex["style"] else INK_FAINT),
                    ],
                    title=("Named clusters / colour-palette features / style features. "
                           "Grey means the dataset was indexed without it; palette and "
                           "style can only be added by re-indexing."),
                    style={"display": "flex", "gap": "5px"},
                ),
                pill,
                html.Button("Remove",
                            id={"type": "dm-del", "index": f"{d.name}::{d.modality}"},
                            n_clicks=0,
                            title=f"Delete the index for {d.name}. Your files stay "
                                  f"where they are.",
                            style=_btn("danger", padding="4px 10px",
                                       fontSize="11px")),
            ]))

        if not has_map and not building:
            rows.append(html.Div(
                "Indexed, but the 2-D map was never built \u2014 probably an "
                "interrupted run. It cannot be opened until you index that "
                "folder again.",
                style={"fontSize": "11.5px", "color": WARN, "lineHeight": "1.5",
                       "padding": "0 0 10px 17px", "maxWidth": "70ch"}))
    return rows


def model_rows() -> list:
    # Ask about every modality, not just images. This used to call
    # label_cache_status("image") only, so CLAP -- the audio encoder -- was
    # never looked up: it showed no labels pill at all and counted as complete
    # the moment its weights were on disk, whether or not it could name a
    # cluster. Audio labels are a real thing (assets/labels_audio.txt) and CLAP
    # needs them exactly as much as CLIP does.
    status = {}
    for modality in ("image", "audio"):
        try:
            for s in _db.label_cache_status(modality):
                status[s["model_id"]] = s
        except Exception:
            pass
    rows = []
    for i, m in enumerate(_models.MODELS):
        downloaded = _models.is_downloaded(m.id)
        labels_ready = status.get(m.id, {}).get("ready", False)
        complete = downloaded and labels_ready

        pills = [_pill("downloaded", OK) if downloaded
                 else _pill(f"{m.download_mb:,} MB", INK_FAINT)]
        pills.append(_pill("labels ready", OK) if labels_ready
                     else _pill("labels pending", WARN))

        rows.append(html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px",
                   "padding": "11px 0", "flexWrap": "wrap",
                   "borderTop": (f"1px solid {LINE}" if i else "none")},
            children=[
                html.Div(style={"width": "3px", "alignSelf": "stretch",
                                "minHeight": "26px", "borderRadius": "2px",
                                "backgroundColor": OK if complete else INK_FAINT}),
                html.Div(style={"flex": "1", "minWidth": "260px"}, children=[
                    html.Div(m.label, style={"fontWeight": "600", "fontSize": "13px",
                                             "color": INK}),
                    html.Div(m.blurb, style={"fontSize": "11.5px", "color": INK_DIM,
                                             "lineHeight": "1.45", "marginTop": "2px"}),
                ]),
                html.Div(pills, style={"display": "flex", "gap": "6px",
                                       "flexWrap": "wrap"}),
                html.Button("Ready" if complete else
                            ("Prepare labels" if downloaded else "Download"),
                            id=f"dm-get-model-{i}", n_clicks=0,
                            disabled=complete,
                            style={**(BTN_QUIET if not complete else
                                      _btn("secondary", color=INK_FAINT,
                                           cursor="default")),
                                   "minWidth": "112px"}),
            ]))
    return rows


# ── module state ─────────────────────────────────────────────────────────────
# Derived from the filesystem, read only on the server, and needed by two
# callbacks at once. A dcc.Store round trip would make the time estimate lag a
# keystroke behind the folder.
_picker_busy: dict = {"open": False}
_scanned_count: dict = {"n": 0, "decode_ms": None}

# Which finished job the lists were last refreshed for, so the refresh token is
# bumped exactly once per job rather than on every poll tick.
_last_seen: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# callbacks
# ──────────────────────────────────────────────────────────────────────────────
def register(app) -> None:
    """Wire the panel up. Called once from arcana.py after the layout is built."""

    @app.callback(
        Output("dm-pending-delete", "data"),
        [Input({"type": "dm-del", "index": ALL}, "n_clicks"),
         Input("dm-del-cancel", "n_clicks")],
        prevent_initial_call=True,
    )
    def _arm_delete(_row_clicks, _cancel):
        """Remember which dataset the user asked to remove; do not delete yet."""
        trig = ctx.triggered_id
        if trig == "dm-del-cancel":
            return None
        if not isinstance(trig, dict) or trig.get("type") != "dm-del":
            return no_update
        # Rebuilding the list recreates every button with n_clicks=0, which
        # fires this callback; only a real click carries a value.
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        # The id carries the dataset, so this never depends on row order.
        name, _, modality = str(trig.get("index", "")).partition("::")
        if not name or not modality:
            return no_update
        return {"name": name, "modality": modality}


    @app.callback(
        [Output("dm-delete-text", "children"),
         Output("dm-delete-confirm", "style")],
        Input("dm-pending-delete", "data"),
    )
    def _confirm_delete(pending):
        """
        Ask before deleting, and say exactly what will and will not go.

        The distinction that matters is that Arcana indexes media in place, so
        removing a dataset removes an index, not a library. Saying so on the
        confirmation is the difference between a safe button and a frightening
        one.
        """
        hidden = {"display": "none"}
        shown = {"display": "block", "marginBottom": "12px",
                 "border": f"1px solid {BAD}55", "borderRadius": "8px",
                 "padding": "12px 14px", "backgroundColor": f"{BAD}0f"}
        if not pending:
            return "", hidden
        try:
            from .legacy import delete_dataset
        except ImportError:
            from legacy import delete_dataset
        try:
            preview = delete_dataset(pending["name"], pending["modality"],
                                     dry_run=True)
        except Exception as e:
            return (html.Div(f"Could not inspect that dataset: {e}",
                             style={"fontSize": "12px", "color": BAD}), shown)

        mb = preview["bytes"] / (1024 * 1024)
        return ([
            html.Div(f"Remove \u201c{pending['name']}\u201d?",
                     style={"fontSize": "13px", "fontWeight": "600",
                            "color": INK, "marginBottom": "4px"}),
            html.Div(f"Deletes {len(preview['removed'])} index and feature files, "
                     f"freeing {mb:,.1f} MB. Your {pending['modality']} files are "
                     f"not touched \u2014 Arcana only ever read them where they "
                     f"are. Re-indexing the folder rebuilds this.",
                     style={"fontSize": "12px", "color": INK_DIM,
                            "lineHeight": "1.5", "maxWidth": "70ch"}),
        ], shown)


    @app.callback(
        [Output("dm-refresh", "data", allow_duplicate=True),
         Output("dm-pending-delete", "data", allow_duplicate=True),
         Output("dm-start-status", "children", allow_duplicate=True)],
        Input("dm-del-yes", "n_clicks"),
        [State("dm-pending-delete", "data"), State("dm-refresh", "data")],
        prevent_initial_call=True,
    )
    def _do_delete(n, pending, token):
        if not n or not pending:
            return no_update, no_update, no_update
        try:
            from .legacy import delete_dataset
        except ImportError:
            from legacy import delete_dataset
        res = delete_dataset(pending["name"], pending["modality"])
        invalidate_dataset_cache()
        if res["failed"]:
            msg = html.Span(
                f"Removed {len(res['removed'])} files; {len(res['failed'])} could "
                f"not be deleted (in use?).", style={"color": WARN})
        else:
            mb = res["bytes"] / (1024 * 1024)
            msg = html.Span(
                f"Removed \u201c{pending['name']}\u201d and freed {mb:,.1f} MB.",
                style={"color": OK})
        return (token or 0) + 1, None, msg


    @app.callback(
        [Output("dm-model", "options"), Output("dm-model", "value")],
        [Input("dm-modality", "value"), Input("dm-folder-scan", "children")],
        State("dm-model", "value"),
    )
    def _models_for_modality(modality, _scan, current):
        modality = modality or "image"
        opts = _model_options(modality, _scanned_count.get("n", 0),
                              _scanned_count.get("decode_ms"))
        ids = [o["value"] for o in opts]
        value = current if current in ids else _models.default_for(modality).id
        return opts, value

    @app.callback(
        [Output("dm-folder", "value"), Output("dm-start-status", "children",
                                              allow_duplicate=True)],
        Input("dm-browse", "n_clicks"),
        State("dm-folder", "value"),
        prevent_initial_call=True,
    )
    def _browse(n, current):
        """
        Open a native folder chooser on the machine running Arcana.

        A browser cannot give a server a filesystem path, and this is a local
        app, so the dialog is opened server-side. folderpicker runs it in a
        subprocess with a timeout -- a GUI dialog on a request thread is a
        reliable way to wedge a server.
        """
        if not n:
            return no_update, no_update
        # One dialog at a time. Each click spawns a blocking subprocess, so
        # clicking again while one is pending used to stack invisible dialogs --
        # five were left running on one machine before this guard existed.
        if _picker_busy.get("open"):
            return no_update, html.Span(
                "A folder chooser is already open — look for it behind this "
                "window.", style={"color": WARN})
        try:
            from . import folderpicker
        except ImportError:
            import folderpicker
        _picker_busy["open"] = True
        try:
            picked = folderpicker.pick_folder(current or None)
        except folderpicker.PickerUnavailable as e:
            return no_update, html.Span(
                f"No folder dialog available here ({e}). Paste the path instead.",
                style={"color": WARN})
        except TimeoutError:
            return no_update, html.Span("The folder dialog timed out.",
                                        style={"color": WARN})
        except Exception as e:
            return no_update, html.Span(
                f"Could not open the folder dialog: {type(e).__name__}: {e}",
                style={"color": BAD})
        finally:
            _picker_busy["open"] = False
        if not picked:
            return no_update, ""            # cancelled
        return picked, ""


    @app.callback(
        [Output("dm-modality", "value"), Output("dm-name", "value")],
        Input("dm-folder", "value"),
        State("dm-name", "value"),
        prevent_initial_call=True,
    )
    def _detect(folder, current_name):
        """
        Answer step 2 on the user's behalf where the answer is knowable.

        The media type and the name were both required fields with an obvious
        default sitting right there in the folder they had just chosen. Asking
        for them made a four-field form out of a one-field decision.

        The name is only filled while the user has not typed one, so this never
        overwrites a deliberate choice.
        """
        if not folder:
            return no_update, no_update
        path = os.path.expanduser(str(folder).strip().strip('"'))
        if not os.path.isdir(path):
            return no_update, no_update

        counts = scan_both(path)
        _last_seen["counts"] = counts
        modality = "audio" if counts["audio"] > counts["image"] else "image"

        # Replace the name when it is empty, or when it is still the suggestion
        # we made for the previous folder -- otherwise pointing at a second
        # folder leaves the first folder's name behind, which is worse than an
        # empty field because it looks deliberate. A name the user actually
        # typed is never touched.
        chosen = next_name(path, current_name, _last_seen.get("suggested"))
        name = no_update
        if chosen is not None:
            name = chosen
            _last_seen["suggested"] = chosen
        return modality, name


    @app.callback(
        Output("dm-folder-scan", "children"),
        [Input("dm-folder", "value"), Input("dm-modality", "value")],
    )
    def _scan(folder, modality):
        _scanned_count["n"] = 0
        if not folder:
            return ""
        folder = os.path.expanduser(str(folder).strip().strip('"'))
        if not os.path.isdir(folder):
            return html.Span("Not a folder: " + folder, style={"color": "#e74c3c"})
        n, sample = scan_media(folder, modality or "image")
        _scanned_count["n"] = n
        # Measure this folder rather than assuming a constant.
        _scanned_count["decode_ms"] = (
            _models.measure_decode_ms(sample) if (modality or "image") == "image" else None)
        counts = _last_seen.get("counts") or {}
        other = "audio" if (modality or "image") == "image" else "image"
        n_other = counts.get(other, 0)

        if not n:
            kind = "images" if (modality or "image") == "image" else "audio files"
            msg = "No " + kind + " in that folder"
            if n_other:
                msg += f", but {n_other:,} {other} files are. Switch the media type above."
            else:
                msg += "."
            return html.Span(msg, style={"color": WARN})

        noun = "images" if (modality or "image") == "image" else "audio files"
        msg = f"Found {n:,} {noun}."
        if n_other:
            msg += f" ({n_other:,} {other} files here too \u2014 index those separately.)"
        return html.Span(msg, style={"color": OK})

    @app.callback(
        Output("dm-label-note", "children"),
        [Input("dm-k", "value"), Input("dm-vocab", "value"),
         Input("dm-folder", "value")],
    )
    def _label_note(k, vocab, folder):
        """Say what the choices will actually produce, before the long job runs."""
        if vocab == "none":
            return ("Groups still get found, they just stay numbered. You can name "
                    "them later only by re-indexing.")

        bits = []
        n = _scanned_count.get("n", 0)
        if not k:
            try:
                auto = _db.auto_k(n) if n else None
            except Exception:
                auto = None
            bits.append(f"About {auto} groups for {n:,} files."
                        if auto else "The number of groups scales with the collection.")
        else:
            bits.append(f"Exactly {k} groups.")

        if vocab == "folders":
            path = os.path.expanduser(str(folder or "").strip().strip('"'))
            words = folder_vocabulary(path) if path else []
            if not words:
                bits.append("No subfolders with media in them, so Arcana's own word "
                            "list will be used instead.")
            else:
                shown = ", ".join(words[:6]) + ("..." if len(words) > 6 else "")
                bits.append(f"Names chosen from your {len(words)} subfolders: {shown}")
        else:
            bits.append("Names chosen from Arcana's 100-word list.")
        return " ".join(bits)


    @app.callback(
        Output("dm-hardware", "children"),
        Input("mode-select", "value"),
    )
    def _hardware(mode):
        """
        Say what the app can see, and what to do about it.

        check_cuda_installation() in color_transfer.py has always produced
        exactly this advice and nothing ever called it, so a user with a good
        card sat through CPU-speed indexing with no hint that anything was
        wrong -- and a user whose card has no compiled kernels got a crash
        part-way through instead of a sentence.
        """
        if mode != "datasets":
            return no_update
        try:
            line = _gpu.describe()
        except Exception:
            return ""
        if _gpu.available():
            return line
        # No usable GPU. Only mention the pip route to someone who has hardware
        # that would actually benefit; telling a laptop owner to rebuild from
        # source is noise.
        extra = (" The packaged build ships CPU-only PyTorch to keep the "
                 "download under a gigabyte. Installing Arcana from source into "
                 "a CUDA environment uses the card instead.")
        v = _gpu.verdict()
        return line + (extra if v.get("name") else "")


    @app.callback(
        Output("dm-model-note", "children"),
        [Input("dm-model", "value"), Input("dm-folder-scan", "children")],
    )
    def _model_note(model_id, _scan):
        m = _models.get(model_id or "")
        if not m:
            return ""
        n = _scanned_count.get("n", 0)
        bits = [m.blurb]
        if n:
            where = "using your GPU" if _models.gpu_available() else "on the CPU"
            est = _models.estimate_text(m, n, decode_ms=_scanned_count.get("decode_ms"))
            bits.append(f"Indexing {n:,} files with this encoder takes {est} {where}.")
        if not _models.is_downloaded(m.id):
            bits.append(f"It will be downloaded first ({m.download_mb:,} MB).")
        return " ".join(bits)

    @app.callback(
        [Output("dm-job", "data", allow_duplicate=True),
         Output("dm-poll", "disabled", allow_duplicate=True),
         Output("dm-start-status", "children")],
        Input("dm-start", "n_clicks"),
        [State("dm-folder", "value"), State("dm-name", "value"),
         State("dm-modality", "value"), State("dm-model", "value"),
         State("dm-features", "value"),
         State("dm-k", "value"), State("dm-vocab", "value")],
        prevent_initial_call=True,
    )
    def _start(n, folder, name, modality, model_id, features, k, vocab):
        def err(text):
            return no_update, no_update, html.Span(text, style={"color": "#e74c3c"})

        if not n:
            return no_update, no_update, no_update
        if not folder:
            return err("Choose a folder first.")
        folder = os.path.expanduser(str(folder).strip().strip('"'))
        if not os.path.isdir(folder):
            return err("Not a folder: " + folder)
        name = (name or "").strip()
        if not name:
            return err("Give the dataset a name.")
        if not all(c.isalnum() or c in "-_ " for c in name):
            return err("Names may only contain letters, numbers, spaces, - and _.")
        if MANAGER.active():
            return err("Something is already running. Wait for it, or cancel it.")

        modality = modality or "image"
        feats = set(features or [])
        feature_arg = ",".join(["clip"] + sorted(feats & {"palette", "style"}))

        # _read_label_list takes either a path or an inline comma list, so a
        # vocabulary built from folder names needs no temporary file.
        label_src = None                       # None -> the built-in list
        if vocab == "folders":
            words = folder_vocabulary(folder)
            label_src = ",".join(words) if words else None
        elif vocab == "none":
            label_src = ""                     # empty -> clusters stay numbered

        def job(handle):
            handle.update(fraction=0.0, message="Preparing")
            _models.ensure_model(model_id, handle)
            # Cluster names come from label embeddings in this encoder's space,
            # so build them before the long part rather than failing after it.
            if vocab != "none":
                _db.warm_label_cache(model_id, modality)
            handle.update(fraction=0.02, message="Indexing")
            return _db.index_dataset(
                folder, name, modality=modality, model_id=model_id,
                features=feature_arg, thumbnails=("thumbnails" in feats),
                k=int(k or 0), labels=label_src,
                progress=lambda f, m, d, t: handle.update(
                    fraction=f, message=(m or None),
                    detail=(f"{d:,} of {t:,}" if t else ""), done=d, total=t),
                should_cancel=lambda: handle.cancelled,
            )

        jid = MANAGER.submit(job, kind="index", label="Indexing " + name)
        return jid, False, html.Span("Started.", style={"color": "#4caf50"})

    @app.callback(
        [Output("dm-job", "data", allow_duplicate=True),
         Output("dm-poll", "disabled", allow_duplicate=True),
         Output("dm-model-job", "children", allow_duplicate=True)],
        [Input(f"dm-get-model-{i}", "n_clicks") for i in range(len(_models.MODELS))],
        prevent_initial_call=True,
    )
    def _get_model(*clicks):
        if not any(c for c in clicks if c):
            return no_update, no_update, no_update
        trig = ctx.triggered_id
        if not isinstance(trig, str) or not trig.startswith("dm-get-model-"):
            return no_update, no_update, no_update
        try:
            m = _models.MODELS[int(trig.rsplit("-", 1)[1])]
        except (ValueError, IndexError):
            return no_update, no_update, no_update
        model_id = m.id
        # One job at a time. This used to return silently, so pressing the
        # button during an index run looked like the button was broken.
        if MANAGER.active():
            return no_update, no_update, html.Span(
                "Something else is running -- wait for it to finish.",
                style={"color": WARN})

        def job(handle):
            _models.ensure_model(model_id, handle)
            # Every encoder needs its own label matrix, audio included. This was
            # gated on modality == "image", so CLAP's labels were never built
            # from here even though assets/labels_audio.txt exists.
            handle.update(fraction=0.9, message="Preparing labels")
            _db.warm_label_cache(model_id, m.modality)
            handle.update(fraction=1.0, message=m.label + " is ready")
            return {"model_id": model_id}

        jid = MANAGER.submit(job, kind="download", label="Getting " + m.label)
        return jid, False, html.Span(f"Preparing {m.label}...")

    @app.callback(
        [Output("dm-job-card", "style"), Output("dm-job-title", "children"),
         Output("dm-job-message", "children"), Output("dm-job-detail", "children"),
         Output("dm-job-bar", "style"), Output("dm-poll", "disabled"),
         Output("dm-refresh", "data"), Output("dm-model-job", "children")],
        [Input("dm-poll", "n_intervals"), Input("dm-job", "data")],
        State("dm-refresh", "data"),
    )
    def _poll(_ticks, job_id, refresh):
        hidden = {**CARD, "display": "none"}

        def bar(pct, colour="#00bcd4"):
            return {"width": f"{pct}%", "height": "100%",
                    "backgroundColor": colour, "transition": "width .3s"}

        snap = MANAGER.snapshot(job_id) if job_id else None
        if snap is None:
            return (hidden, "", "", "", bar(0), True, refresh or 0, "")

        frac = snap.get("fraction")
        pct = int(round((frac if frac is not None else 0) * 100))
        status = snap["status"]
        if status == "failed":
            colour, title = "#e74c3c", "Failed"
            msg = snap.get("error") or "Something went wrong."
        elif status == "cancelled":
            colour, title = "#8a6a2b", "Cancelled"
            msg = snap.get("message") or ""
        elif status == "done":
            colour, title, pct = "#4caf50", "Finished", 100
            msg = snap.get("message") or "Done."
        else:
            colour = "#00bcd4"
            title = snap.get("label") or "Working"
            msg = snap.get("message") or ""

        finished = status in ("done", "failed", "cancelled")
        # Bump the refresh token exactly once, when the job stops. That is the
        # only moment the dataset and encoder lists can have changed.
        token = refresh or 0
        if finished and _last_seen.get("job") != job_id:
            _last_seen["job"] = job_id
            invalidate_dataset_cache()
            token = token + 1
        # The encoder list is at the bottom of a scrolling panel while the job
        # card is at the top, so a download started from "Prepare labels" gave
        # no visible sign. Echo the same progress next to the button.
        if snap.get("kind") == "download":
            if status == "done":
                inline = html.Span(msg, style={"color": OK})
            elif status in ("failed", "cancelled"):
                inline = html.Span(msg, style={"color": BAD})
            else:
                inline = html.Span(f"{title} - {msg} ({pct}%)" if msg
                                   else f"{title} ({pct}%)")
        else:
            inline = ""

        return ({**CARD, "display": "block"}, title, msg, snap.get("detail") or "",
                bar(pct, colour), finished, token, inline)


    @app.callback(
        [Output("dm-list", "children"), Output("dm-models", "children")],
        [Input("mode-select", "value"), Input("dm-refresh", "data")],
    )
    def _lists(mode, _token):
        """
        Rebuilt only when the panel is opened or a job finishes -- never on the
        progress timer.
        """
        if mode != "datasets":
            return no_update, no_update
        return dataset_rows(), model_rows()

    @app.callback(
        Output("dm-job", "data", allow_duplicate=True),
        Input("dm-cancel", "n_clicks"),
        State("dm-job", "data"),
        prevent_initial_call=True,
    )
    def _cancel(n, job_id):
        if n and job_id:
            MANAGER.cancel(job_id)
        return job_id
