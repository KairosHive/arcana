# paths.py — where Arcana keeps its data
#
# Everything writable used to live inside the package directory (arcana/databases,
# arcana/latents, arcana/output, arcana/cache_specs), and several of those were
# created with os.makedirs at import time. That works from a source checkout and
# fails on the first line of a packaged app, where the install directory is
# read-only.
#
# This module is the single place that answers "where does data go", with three
# rules, in order:
#
#   1. ARCANA_DATA_DIR, if set, wins. One env var to relocate everything.
#   2. Otherwise, if a source checkout already has data in arcana/<dir>, keep
#      using it -- an existing install must not appear to lose its datasets.
#   3. Otherwise, the per-user data directory for the platform.
#
# Nothing here creates a directory on import. Call ensure_writable() (or
# ensure_dir) at the point of first write, so read-only installs and fresh
# machines both survive being imported.

from __future__ import annotations

import hashlib
import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_ROOT, "assets")

ENV_DATA_DIR = "ARCANA_DATA_DIR"
ENV_MEDIA_ROOTS = "ARCANA_MEDIA_ROOTS"

# Subdirectories of the data directory.
# "boards" holds saved moodboards as JSON. They live here rather than in
# browser localStorage so they survive clearing site data, and so two
# machines sharing ARCANA_DATA_DIR share their boards along with datasets.
_SUBDIRS = ("databases", "latents", "bundles", "output", "cache", "models",
            "boards")

# Legacy in-package locations, kept working for existing checkouts.
_LEGACY = {
    "databases": os.path.join(APP_ROOT, "databases"),
    "latents": os.path.join(APP_ROOT, "latents"),
    "bundles": os.path.join(APP_ROOT, "bundles"),
    "output": os.path.join(APP_ROOT, "output"),
    "cache": os.path.join(APP_ROOT, "cache_specs"),
}


def _platform_data_dir() -> str:
    """Per-user, writable, conventional for the platform."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "Arcana")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Arcana")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "arcana")


def _has_content(path: str) -> bool:
    try:
        with os.scandir(path) as it:
            for entry in it:
                if not entry.name.startswith("."):
                    return True
    except OSError:
        pass
    return False


def data_dir() -> str:
    """Root of Arcana's writable state."""
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return _platform_data_dir()


def subdir(name: str) -> str:
    """
    Absolute path for one of Arcana's data subdirectories.

    Does not create it -- read it freely, and call ensure_dir() before writing.
    """
    if name not in _SUBDIRS:
        raise ValueError(f"unknown data subdirectory {name!r}; expected one of {_SUBDIRS}")

    # An explicit ARCANA_DATA_DIR overrides everything, including legacy layouts.
    if not os.environ.get(ENV_DATA_DIR):
        legacy = _LEGACY.get(name)
        if legacy and _has_content(legacy):
            return legacy

    return os.path.join(data_dir(), name)


def ensure_dir(path: str) -> str:
    """
    Create a directory, raising something a human can act on if we cannot.

    Call this immediately before writing, never at import time.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Arcana could not create {path!r} ({e.strerror or e}). "
            f"If Arcana is installed in a read-only location, set {ENV_DATA_DIR} "
            f"to a folder you can write to and restart."
        ) from e
    return path


def listdir_safe(path: str) -> list[str]:
    """os.listdir that returns [] for a directory that does not exist yet."""
    try:
        return os.listdir(path)
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError:
        return []


# --------------------------------------------------------------------------------------
# media roots -- the only directories media may ever be served from
# --------------------------------------------------------------------------------------
def _default_media_roots() -> list[str]:
    roots = []
    # The classic sibling images/ folder of a source checkout.
    sibling = os.path.abspath(os.path.join(APP_ROOT, "..", "images"))
    if os.path.isdir(sibling):
        roots.append(sibling)
    media = os.path.join(data_dir(), "media")
    if os.path.isdir(media):
        roots.append(media)
    return roots


def media_roots() -> list[str]:
    """
    Directories that media requests are allowed to read from.

    ARCANA_MEDIA_ROOTS, if set, is an os.pathsep-separated list and replaces the
    defaults. Anything outside these roots is not servable -- see is_within().
    """
    env = os.environ.get(ENV_MEDIA_ROOTS)
    if env:
        out = []
        for part in env.split(os.pathsep):
            part = part.strip()
            if part:
                out.append(os.path.abspath(os.path.expanduser(part)))
        return out
    return _default_media_roots()


def is_within(path: str, roots: list[str] | None = None) -> bool:
    """
    True when `path` resolves to somewhere inside one of `roots`.

    Resolves symlinks on both sides before comparing, so a link planted inside a
    media root cannot be used to escape it.
    """
    if roots is None:
        roots = media_roots()
    if not roots:
        return False
    try:
        target = os.path.realpath(os.path.abspath(path))
    except OSError:
        return False
    for root in roots:
        try:
            real_root = os.path.realpath(os.path.abspath(root))
        except OSError:
            continue
        try:
            if os.path.commonpath([target, real_root]) == real_root:
                return True
        except ValueError:
            continue        # different drives on Windows
    return False


def safe_join(root: str, *parts: str) -> str | None:
    """
    Join user-supplied path components onto `root`, or None if they escape it.

    Use for anything a client names -- an output folder, a dataset name -- so
    '../../..' and absolute paths cannot redirect a write.
    """
    cleaned = []
    for p in parts:
        p = str(p).replace("\\", "/").strip()
        if not p:
            continue
        if os.path.isabs(p) or (len(p) > 1 and p[1] == ":"):
            return None
        cleaned.extend(seg for seg in p.split("/") if seg not in ("", "."))
    if any(seg == ".." for seg in cleaned):
        return None
    if not cleaned:
        return None
    candidate = os.path.abspath(os.path.join(root, *cleaned))
    root_abs = os.path.abspath(root)
    try:
        if os.path.commonpath([candidate, root_abs]) != root_abs:
            return None
    except ValueError:
        return None
    return candidate


def fit_filename(out_dir: str, left: str, right: str, suffix: str, ext: str) -> str:
    """
    Build "<left>_from_<right>_<suffix><ext>" that actually fits on disk.

    Midjourney-style filenames run past 200 characters; two of them joined, under
    an already-deep output directory, sail past Windows' 260-character path limit
    and Pillow fails with a bare "No such file or directory". Budget from the real
    directory length rather than trusting a fixed truncation, and keep a hash of
    the full names so two long names that share a prefix stay distinguishable.
    """
    MAX_PATH = 250                      # a little headroom under Windows' 260
    fixed = len("_from_") + 1 + len(suffix) + len(ext)
    budget = MAX_PATH - len(os.path.abspath(out_dir)) - 1 - fixed

    if budget < 16:
        # The output directory alone is near the limit; fall back to a name that
        # is short no matter what.
        digest = hashlib.blake2b(f"{left}|{right}".encode("utf-8", "replace"),
                                 digest_size=6).hexdigest()
        return f"ct_{digest}_{suffix}{ext}"

    per_side = max(8, budget // 2)
    if len(left) + len(right) <= budget:
        l, r = left, right
    else:
        digest = hashlib.blake2b(f"{left}|{right}".encode("utf-8", "replace"),
                                 digest_size=4).hexdigest()
        per_side = max(8, (budget - len(digest) - 1) // 2)
        l, r = left[:per_side], right[:per_side]
        suffix = f"{digest}_{suffix}"
    return f"{l}_from_{r}_{suffix}{ext}"


def describe() -> str:
    """One-line summary for logs and bug reports."""
    lines = [f"data dir: {data_dir()}"]
    for name in _SUBDIRS:
        try:
            p = subdir(name)
        except ValueError:
            continue
        tag = "  (legacy in-package)" if p == _LEGACY.get(name) else ""
        lines.append(f"  {name:10s} {p}{tag}")
    lines.append(f"media roots: {media_roots() or '(none configured)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
