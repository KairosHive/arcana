"""
Arcana on a scratch data folder, for looking at the page without touching the
real one.

A development server started the ordinary way shares the person's data folder
with the app they have open, and announces itself in server.json there, which
takes the agent's connection away from that app. This starts the same server
against a temporary folder holding the synthetic photo dataset the tests use
and a small audio dataset of the same name, on its own port.

    uv run python tests/sandbox_app.py          # http://127.0.0.1:8061/

The vectors are synthetic, so typing a search fails for want of a matching
encoder. Everything that draws (tabs, tags, the map, cards, the collection)
works.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

PORT = int(os.environ.get("ARCANA_SANDBOX_PORT", "8061"))


def _remembered_videos() -> str:
    """The folder named in tests/.sandbox_videos (ignored by git), for a launcher that cannot set a variable."""
    try:
        with open(os.path.join(HERE, ".sandbox_videos"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def main():
    root = tempfile.mkdtemp(prefix="arcana-sandbox-")
    os.environ["ARCANA_DATA_DIR"] = root
    os.environ["ARCANA_DEV_PORT"] = str(PORT)
    os.environ["ARCANA_NO_BROWSER"] = "1"

    import synth
    from test_modality import write_audio_dataset
    synth.write_dataset(root, palette_features=False)
    write_audio_dataset(root)
    print(f"Sandbox data folder: {root}", flush=True)

    # A real video dataset beside the synthetic ones, when a folder is given:
    #   ARCANA_SANDBOX_VIDEOS=<folder> uv run python tests/sandbox_app.py
    # It is indexed with the small CLIP, so its searches are real ones.
    videos = os.environ.get("ARCANA_SANDBOX_VIDEOS") or _remembered_videos()
    if videos and os.path.isdir(videos):
        from arcana import db
        print(f"Indexing the videos in {videos} ...", flush=True)
        db.index_dataset(videos, "clips", modality="video",
                         model_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

    from arcana import arcana as _app
    _app.main()


if __name__ == "__main__":
    main()
