# cli.py -- the `arcana` command
#
#   arcana                              start the app, as it always has
#   arcana <op line>                    run one op: arcana search text japan --phrase rain
#   arcana --help                       orient, then how a line is written
#   arcana <op> --help                  that op's help, generated from its declaration
#   arcana --batch lines.txt            run a file of op lines as one batch (- for stdin)
#   arcana complete "<partial line>"    completion candidates, for the shell scripts
#   arcana completion powershell        a completion script (also bash, zsh)
#
# Options before the op: --json prints the result as JSON, --sheets draws
# contact sheets for searches and writes every picture a result carries to
# output/sheets.
#
# The client knows nothing about any op. Help, validation and completion all
# come from arcana.ops, so a new op is on the command line the moment it is
# registered. Today ops run in this process. From P1 the client posts the line
# to the running app when one is open, so the app stays the one owner of the
# models and the GPU (OPS_PLAN.md, "Transports").

from __future__ import annotations

import json
import sys

try:
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths


USAGE = """\
Usage
  arcana                          start the app
  arcana [--json] [--sheets] <op> [arguments]
  arcana <op> --help              help for one op
  arcana op list                  every op
  arcana --batch FILE             run op lines from a file, - for stdin
  arcana completion powershell    shell completion (also bash, zsh)
"""


def _start_app() -> int:
    from .arcana import main as app_main
    app_main()
    return 0


def _print(text: str, stream=None) -> None:
    stream = stream or sys.stdout
    stream.write(text if text.endswith("\n") else text + "\n")
    stream.flush()


def _report(result, op_name: str, as_json: bool, write_files: bool) -> None:
    from . import ops
    written = ops.render.write_attachments(result, op_name) if write_files else []
    if as_json:
        _print(json.dumps({"op": op_name, "result": ops.render.result_json(result)},
                          ensure_ascii=False, indent=2))
        return
    _print(result.text())
    for path in written:
        _print(f"Sheet: {path}")


def main(argv=None) -> int:
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _start_app()

    as_json = sheets = False
    batch_file = None
    while argv and argv[0].startswith("-"):
        flag = argv.pop(0)
        if flag == "--json":
            as_json = True
        elif flag == "--sheets":
            sheets = True
        elif flag == "--batch":
            if not argv:
                _print("--batch needs a file of op lines, or - for stdin.", sys.stderr)
                return 2
            batch_file = argv.pop(0)
        elif flag in ("-h", "--help"):
            argv = ["help-page"]
            break
        elif flag == "--version":
            from . import __version__
            _print(f"Arcana {__version__}")
            return 0
        else:
            _print(f"Unknown option {flag}. Options before the op are --json, --sheets "
                   f"and --batch.\n\n" + USAGE, sys.stderr)
            return 2

    from . import ops
    if argv == ["help-page"]:
        ctx = ops.context("cli")
        _print(ops.call("orient", ctx=ctx).text() + "\n" + ops.grammar.GRAMMAR + "\n" + USAGE)
        return 0
    if argv and argv[0] == "complete":
        for cand in ops.grammar.complete(" ".join(argv[1:]) if len(argv) > 2 else
                                         (argv[1] if len(argv) > 1 else "")):
            _print(cand)
        return 0
    if argv and argv[0] == "completion":
        try:
            _print(ops.grammar.completion_script(argv[1] if len(argv) > 1 else "powershell"))
        except ops.Refusal as e:
            _print(e.reason, sys.stderr)
            return 2
        return 0

    ctx = ops.context("cli", sheets=sheets)

    if batch_file is not None:
        try:
            if batch_file == "-":
                lines = sys.stdin.read().splitlines()
            else:
                with open(batch_file, encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
        except OSError as e:
            _print(f"Could not read {batch_file}: {e.strerror}.", sys.stderr)
            return 2
        batch = ops.exec_lines(lines, ctx=ctx)
        for step in batch.steps:
            if step.result is not None and (sheets or step.op == "item sheet"):
                ops.render.write_attachments(step.result, step.op or "sheet")
        if as_json:
            _print(json.dumps(ops.render.batch_json(batch), ensure_ascii=False, indent=2))
        else:
            _print(ops.render.batch_text(batch))
        return 0 if batch.ok else 1

    try:
        parsed = ops.parse_tokens(argv)
        if parsed.help:
            _print(ops.help_text(parsed.op))
            return 0
        result = ops.call(parsed.op.name, parsed.args, ctx=ctx)
    except ops.Refusal as e:
        if as_json:
            _print(json.dumps({"refused": e.as_dict()}, ensure_ascii=False, indent=2))
        else:
            _print(ops.render.refusal_text(e.as_dict()), sys.stderr)
        return 1
    # A sheet is the whole point of `item sheet`, so it is written without --sheets.
    _report(result, parsed.op.name, as_json, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
