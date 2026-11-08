# aliases.py -- the old console scripts, as spellings of `arcana <op>`
#
# Arcana had nine console scripts, each with its own flags. Their work is ops
# now, under the one `arcana` command. For one release each old name stays:
# it translates its flags into an op line, says the new spelling on stderr,
# and runs it. A flag the op has no equivalent for (a k-means metric, a Gram
# matrix size, a bundle path) sends the call to the old script unchanged, so
# nothing that worked before stops working. arcana-mcp is not here: MCP
# clients launch it by name, so it stays for good.

from __future__ import annotations

import argparse
import sys

try:
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths


def _ds(name: str, modality: str | None) -> str:
    return name if (modality or "image") == "image" else f"{name}@{modality}"


def _run(old: str, tokens: list[str] | None, legacy_main, argv: list[str]) -> int:
    from . import cli
    from .ops import grammar
    if tokens is None:
        print(f"[{old}] runs as before. Most of what it does is `arcana dataset ...` now; "
              f"`arcana op list dataset` shows the ops.", file=sys.stderr)
        sys.argv = [old] + argv
        result = legacy_main()
        return result if isinstance(result, int) else 0
    print(f"[{old}] is now: arcana {grammar.join(tokens)}", file=sys.stderr)
    return cli.main(tokens)


def _parse(parser: argparse.ArgumentParser, argv: list[str]):
    """Known flags, or None when any flag has no op equivalent."""
    try:
        args, extra = parser.parse_known_args(argv)
    except SystemExit:
        return None
    return None if extra else args


def build_latent(argv=None) -> int:
    """arcana-build-latent -> arcana dataset create"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--imgs_path", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--modality", default="image")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--features", default="clip")
    p.add_argument("--model", default=None)
    p.add_argument("--thumbnails", action="store_true")
    a = _parse(p, argv)
    tokens = None
    if a is not None:
        tokens = ["dataset", "create", a.name, "--folder", a.imgs_path, "--modality", a.modality,
                  "--dims", str(a.n_components), "--k", str(a.k), "--no-background"]
        tokens += [x for f in a.features.split(",") if f.strip() in ("palette", "style")
                   for x in ("--feature", f.strip())]
        if a.model:
            tokens += ["--model", a.model]
        if a.thumbnails:
            tokens.append("--thumbnails")
    from .db import main as legacy
    return _run("arcana-build-latent", tokens, legacy, argv)


def extend(argv=None) -> int:
    """arcana-extend -> arcana dataset extend"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--path", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--modality", default="image")
    p.add_argument("--prune", action="store_true")
    a = _parse(p, argv)
    tokens = None
    if a is not None:
        tokens = ["dataset", "extend", _ds(a.name, a.modality), "--folder", a.path,
                  "--no-background"] + (["--prune"] if a.prune else [])
    from .db import extend_main as legacy
    return _run("arcana-extend", tokens, legacy, argv)


def finish(argv=None) -> int:
    """arcana-finish -> arcana dataset finish"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--name", required=True)
    p.add_argument("--modality", default="image")
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--features", default="clip")
    p.add_argument("--thumbnails", action="store_true")
    a = _parse(p, argv)
    tokens = None
    if a is not None:
        tokens = ["dataset", "finish", _ds(a.name, a.modality), "--dims", str(a.n_components),
                  "--k", str(a.k), "--no-background"]
        tokens += [x for f in a.features.split(",") if f.strip() in ("palette", "style")
                   for x in ("--feature", f.strip())]
        if a.thumbnails:
            tokens.append("--thumbnails")
    from .db import finish_main as legacy
    return _run("arcana-finish", tokens, legacy, argv)


def marks(argv=None) -> int:
    """arcana-marks -> arcana dataset marks"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--name", required=True)
    p.add_argument("--modality", default="image")
    p.add_argument("--n_components", type=int, default=2)    # marks refreshes every layout
    a = _parse(p, argv)
    tokens = None if a is None else ["dataset", "marks", _ds(a.name, a.modality)]
    from .db import marks_main as legacy
    return _run("arcana-marks", tokens, legacy, argv)


def migrate(argv=None) -> int:
    """arcana-migrate -> arcana dataset migrate"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--all", action="store_true")
    p.add_argument("--name")
    p.add_argument("--modality")
    p.add_argument("--thumbnails", action="store_true")
    a = _parse(p, argv)
    tokens = None
    if a is not None and (a.all or a.name):
        tokens = ["dataset", "migrate"] + ([_ds(a.name, a.modality)] if a.name else [])
        tokens += ["--no-background"] + (["--thumbnails"] if a.thumbnails else [])
    from .legacy import main as legacy_main
    return _run("arcana-migrate", tokens, lambda: legacy_main(sys.argv[1:]), argv)


def relocate(argv=None) -> int:
    """arcana-relocate -> arcana dataset relocate"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--name", required=True)
    p.add_argument("--modality", default="image")
    p.add_argument("--new-root", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = _parse(p, argv)
    tokens = None
    if a is not None:
        tokens = ["dataset", "relocate", _ds(a.name, a.modality), "--root", a.new_root]
        if a.dry_run:
            tokens.append("--dry-run")
    from .relocate import main as legacy_main
    return _run("arcana-relocate", tokens, lambda: legacy_main(sys.argv[1:]), argv)


def lut(argv=None) -> int:
    """arcana-lut -> arcana colour lut"""
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--content", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=33)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--strength", type=float, default=1.0)
    a = _parse(p, argv)
    tokens = None
    if a is not None:
        tokens = ["colour", "lut", a.content, "--reference", a.reference, "--out", a.out,
                  "--n", str(a.size), "--steps", str(a.steps), "--strength", str(a.strength)]
    from .lut import main as legacy
    return _run("arcana-lut", tokens, legacy, argv)
