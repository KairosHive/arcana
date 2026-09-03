"""
Entry point for the frozen desktop build.

Console scripts are a packaging-time concept, so a frozen app needs a real
module to start from. This also does the things a double-clicked application
has to do that `arcana-build-latent`-style CLI use does not:

  * put writable state somewhere the user can actually write to, because the
    install directory of a packaged app is read-only
  * open a browser at the app once the server is listening
  * keep a log, since there is no console to print to in a windowed build
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import webbrowser

HOST = "127.0.0.1"
PORT = int(os.environ.get("ARCANA_PORT", "8050"))


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def _default_data_dir() -> str:
    """Per-user, writable. Mirrors arcana.paths so both agree."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "Arcana")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Arcana")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "arcana")


def _setup_environment() -> str:
    """
    Point Arcana at a writable data directory before anything imports it.

    A frozen app lives somewhere like C:\\Program Files, so the source-checkout
    fallback in arcana.paths (keep using arcana/databases if it has content)
    must not be allowed to win.
    """
    data_dir = os.environ.get("ARCANA_DATA_DIR") or _default_data_dir()
    os.environ["ARCANA_DATA_DIR"] = data_dir
    os.makedirs(data_dir, exist_ok=True)

    if not os.environ.get("ARCANA_MEDIA_ROOTS"):
        media = os.path.join(data_dir, "media")
        os.makedirs(media, exist_ok=True)
        os.environ["ARCANA_MEDIA_ROOTS"] = media

    # HuggingFace would otherwise cache models next to the executable.
    os.environ.setdefault("HF_HOME", os.path.join(data_dir, "models"))
    return data_dir


def _tee_log(data_dir: str):
    """A windowed build has no console; keep the output anyway."""
    if not _frozen():
        return
    log_path = os.path.join(data_dir, "arcana.log")
    try:
        f = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        return
    sys.stdout = f
    sys.stderr = f
    print(f"\n--- Arcana starting, data dir {data_dir} ---")


def _open_browser_when_ready() -> None:
    """Poll the port rather than guessing at a delay; startup is model-bound."""
    import socket
    import time

    deadline = time.time() + 180
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect((HOST, PORT))
            except OSError:
                time.sleep(0.5)
                continue
        try:
            webbrowser.open(f"http://{HOST}:{PORT}/")
        except Exception:
            pass
        return


def main() -> int:
    data_dir = _setup_environment()
    _tee_log(data_dir)

    # Imported only now, so the environment above is already in place.
    from arcana.arcana import app

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    print(f"Arcana is running at http://{HOST}:{PORT}/")
    app.run(host=HOST, port=PORT, debug=False)
    return 0


if __name__ == "__main__":
    # MUST come before anything else. On Windows multiprocessing uses "spawn",
    # which re-launches the executable for every worker -- and in a frozen app
    # the executable is this application. Without freeze_support() each worker
    # starts a whole second copy of Arcana (Dash server and all) instead of
    # running the job it was handed, and the pool then kills it: indexing with
    # palette or style features failed on every single image with "A process in
    # the process pool was terminated abruptly", after ~36 minutes per image.
    multiprocessing.freeze_support()
    sys.exit(main())
