# arcana.py -- the app's server
#
# One Flask application serving four things:
#
#   /                 the page, a Svelte bundle built into arcana/web (frontend/)
#   /api/*            the page's JSON (arcana/api), as actor "person"
#   /ops/*            the ops for the terminal and agents (server.py), by token
#   /agent/*          the Agent tab's terminals (agent_term.py)
#   /thumb /preview /palette /audio /awave /aspec /board-page /film-video /download
#                     media (media.py)
#
# The page is static. It reaches everything through /api, with a token that is
# written into index.html as it is served and nowhere else. The search,
# filtering and grouping it shows come from core.py, the same functions the
# ops use, which this module re-exports under the names the tests pin.

from __future__ import annotations

import os

from flask import Flask, Response, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider

try:
    from . import paths as _paths
    from . import core as _core
    from . import media as _media
    from . import server as _ops_server
    from . import agent_term as _agent_term
    from . import api as _api
    from . import ops as _ops
    from .api import search as _api_search
except ImportError:                                   # pragma: no cover
    import paths as _paths
    import core as _core
    import media as _media
    import server as _ops_server
    import agent_term as _agent_term
    import api as _api
    import ops as _ops
    from api import search as _api_search

WEB_DIR = os.path.join(_paths.APP_ROOT, "web")

# ─────────────────────────── re-exports the tests and older callers use ───────────────────────────

APP_ROOT = _paths.APP_ROOT
LATENTS_DIR = _media.LATENTS_DIR
DB_DIR = _media.DB_DIR
OUTPUT_DIR = _media.OUTPUT_DIR
IMAGES_ROOT = _media.IMAGES_ROOT
AUDIO_EXTS = _media.AUDIO_EXTS

_ALLOWED_ROOTS = _media._ALLOWED_ROOTS
_ALLOWED_FILES = _media._ALLOWED_FILES
_ALLOWED_ROOTS_LOCK = _media._ALLOWED_ROOTS_LOCK
_REGISTERED_DATASETS = _media._REGISTERED_DATASETS
_seed_allowed_roots = _media._seed_allowed_roots
register_media_root = _media.register_media_root
register_dataset_files = _media.register_dataset_files
register_dataset_roots = _media.register_dataset_files
register_dataset_dirs = _media.register_dataset_files
resolve_media_request = _media.resolve_media_request
resolve_path = _media.resolve_path
make_thumbnail_bytes = _media.make_thumbnail_bytes
make_resized_bytes = _media.make_resized_bytes
make_waveform_png = _media.make_waveform_png
make_melspec_png = _media.make_melspec_png

MARK_COLUMNS = _core.MARK_COLUMNS
NO_COLOUR = _core.NO_COLOUR
UNRATED = _core.UNRATED
has_marks = _core.has_marks
marks_mask = _core.marks_mask
_allowed_keys = _core.keys_where
_cosine_group = _core.cosine_group
CLIP_MODEL_ID = _core.CLIP_MODEL_ID
text_model_for_dim = _core.text_model_for_dim
load_clip_text = _core.load_clip_text
load_clap = _core.load_clap
search = _core.search

load_data = _api_search.load_data
load_index = _api_search.load_index
star_options = _api_search._star_options
colour_options = _api_search._colour_options
_matrix_counts = _api_search._matrix_counts
COLOUR_SWATCH = _api_search.COLOUR_SWATCH
get_matching_datasets = None     # set below, once the api package has imported ui_datasets


# ─────────────────────────── the app ───────────────────────────

def _index_html() -> bytes | None:
    try:
        with open(os.path.join(WEB_DIR, "index.html"), "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _boot_script() -> bytes:
    """The page's tokens and version, as the one script the static page cannot ship."""
    import json
    t = _api.tokens()
    data = json.dumps({"ui": t["ui"], "agent": t["agent"], "version": t["version"]})
    return b"<script>window.__ARCANA__=" + data.encode("utf-8") + b"</script>"


class _JSON(DefaultJSONProvider):
    """numpy scalars and arrays, which the search and the map hand over, as plain JSON."""

    @staticmethod
    def default(o):
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


def create_app() -> Flask:
    app = Flask("arcana", static_folder=None)
    app.json = _JSON(app)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    _ops_server.register(app)
    _agent_term.register(app)
    _media.register(app)
    _api.register(app)

    @app.get("/")
    def _page():
        html = _index_html()
        if html is None:
            return Response(
                "<!doctype html><title>Arcana</title><body style='font:15px system-ui;padding:40px;"
                "background:#141311;color:#ede8df'><h1>Arcana</h1><p>The page has not been built. "
                "Run <code>npm install &amp;&amp; npm run build</code> in <code>frontend/</code>, "
                "or <code>npm run dev</code> there for a development page.</p></body>",
                mimetype="text/html")
        at = html.rfind(b"</head>")
        boot = _boot_script()
        html = html[:at] + boot + html[at:] if at >= 0 else boot + html
        return Response(html, mimetype="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/assets/<path:name>")
    def _assets(name):
        # Hashed Vite output: immutable. The logo files beside the package stay
        # reachable too (arcana/assets), for the header mark.
        web_assets = os.path.join(WEB_DIR, "assets")
        if os.path.isfile(os.path.join(web_assets, name)):
            return send_from_directory(web_assets, name, max_age=31536000)
        return send_from_directory(_paths.ASSETS_DIR, name, max_age=86400)

    @app.get("/<path:name>")
    def _static(name):
        # Anything else Vite put at the root of the bundle (favicon, fonts).
        if os.path.isfile(os.path.join(WEB_DIR, name)):
            return send_from_directory(WEB_DIR, name, max_age=86400)
        return jsonify({"error": "Not found."}), 404

    # The old Dash app object had `.server`; the launcher and the tests only
    # need `.run` and `.test_client`, so the Flask app is its own server.
    app.server = app
    return app


app = create_app()

try:
    from . import ui_datasets as _ui_datasets
except ImportError:                                   # pragma: no cover
    import ui_datasets as _ui_datasets
get_matching_datasets = _ui_datasets.picker_options


def main():
    _paths.use_utf8_console()
    # 8051, not 8050: the packaged app (installer/launcher.py) listens on 8050,
    # and localStorage is keyed by origin, so a development server on the same
    # port would share and confuse its state. ARCANA_DEV_PORT overrides.
    port = int(os.environ.get("ARCANA_DEV_PORT", "8051"))
    _ops_server.announce(port)
    print(f"Arcana is running at http://127.0.0.1:{port}/", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
