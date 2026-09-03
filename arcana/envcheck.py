# envcheck.py — does the installed environment still match the lock?
#
# This exists because the environment silently drifted off requirements-lock.txt
# by nine packages, and one of them (transformers 4 -> 5) changed
# CLIPModel.get_image_features to return a BaseModelOutputWithPooling instead of
# a tensor. Indexing then failed at the first image, several layers deep, with an
# error that pointed at Arcana rather than at the environment.
#
# The cause was ordinary: `uv pip install <anything>` re-resolves the whole
# environment, so installing a tool upgrades your pinned libraries as a side
# effect. Constraints in pyproject.toml prevent most of it; this catches the
# rest, loudly, before it wastes an afternoon.
#
#   python -m arcana.envcheck          # report, exit 1 on drift
#   python -m arcana.envcheck --quiet  # exit code only

from __future__ import annotations

import os
import sys

LOCK_NAME = "requirements-lock.txt"


def _repo_lock() -> str | None:
    """Find requirements-lock.txt next to the package, or above it."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.dirname(here), here):
        p = os.path.join(d, LOCK_NAME)
        if os.path.exists(p):
            return p
    return None


def parse_lock(path: str) -> dict[str, str]:
    """
    name -> version for the pins that apply to this interpreter.

    Environment markers are honoured, so a pin that only applies to another
    Python version is not reported as missing.
    """
    pins: dict[str, str] = {}
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None

    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        if Requirement is not None:
            try:
                req = Requirement(line)
            except Exception:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            for spec in req.specifier:
                if spec.operator == "==":
                    pins[req.name.lower().replace("_", "-")] = spec.version
                    break
        else:                                    # crude fallback
            line = line.split(";")[0].strip()
            if "==" in line:
                n, v = line.split("==", 1)
                pins[n.strip().lower().replace("_", "-")] = v.strip()
    return pins


def _installed(name: str) -> str | None:
    import importlib.metadata as md
    for candidate in (name, name.replace("-", "_")):
        try:
            return md.version(candidate)
        except md.PackageNotFoundError:
            continue
    return None


def _base(version: str) -> str:
    """Compare 2.9.1+cu128 against 2.9.1: the CUDA local tag is not drift."""
    return version.split("+", 1)[0]


def drift(lock_path: str | None = None) -> list[tuple[str, str, str]]:
    """
    Returns (package, locked, installed) for every mismatch.

    "installed" is "MISSING" when the package is not present at all.
    """
    lock_path = lock_path or _repo_lock()
    if not lock_path:
        return []
    out = []
    for name, want in sorted(parse_lock(lock_path).items()):
        got = _installed(name)
        if got is None:
            out.append((name, want, "MISSING"))
        elif _base(got) != _base(want):
            out.append((name, want, got))
    return out


def report(lock_path: str | None = None) -> str:
    lock_path = lock_path or _repo_lock()
    if not lock_path:
        return f"No {LOCK_NAME} found; cannot check the environment."
    bad = drift(lock_path)
    if not bad:
        n = len(parse_lock(lock_path))
        return f"Environment matches the lock ({n} packages)."
    lines = [f"{len(bad)} package(s) do not match {os.path.basename(lock_path)}:", ""]
    for name, want, got in bad:
        lines.append(f"  {name:24s} locked {want:14s} installed {got}")
    lines += [
        "",
        "Restore with:",
        "  uv sync",
        "",
        "This usually happens because `uv pip install <tool>` re-resolves the whole",
        "environment. Install tools elsewhere, or re-run `uv sync` afterwards.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quiet = "--quiet" in argv or "-q" in argv
    bad = drift()
    if not quiet:
        print(report())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
