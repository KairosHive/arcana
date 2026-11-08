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
#   arcana --headless [--port N]       serve the ops without the app's pages
#
# Options before the op: --json prints the result as JSON, --sheets draws
# contact sheets for searches and writes every picture a result carries to
# output/sheets, --local runs the op in this process even when the app is open.
#
# The client knows nothing about any op. Help, validation and completion all
# come from arcana.ops, so a new op is on the command line the moment it is
# registered. When the app (or `arcana --headless`) is running, the line is
# posted to it, so the app stays the one owner of the models, the jobs and the
# GPU, and its Moodboard tab sees the change at once. Otherwise the op runs
# here. See OPS_PLAN.md, "Transports".

from __future__ import annotations

import json
import os
import sys
import time

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
  arcana --headless [--port N]    serve the ops for agents, without the app's pages
  arcana completion powershell    shell completion (also bash, zsh)

Options before the op: --json, --sheets, --local (run here even if the app is open).
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


def _write_remote_attachments(res: dict) -> None:
    """Pictures a server sent back, written here and named in its steps."""
    import base64
    from . import ops
    folder = None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, step in enumerate(res.get("steps", []), 1):
        for j, att in enumerate(step.get("attachments") or [], 1):
            folder = folder or ops.render.sheets_dir()
            slug = (step.get("op") or "sheet").replace(" ", "-")
            ext = ".png" if att.get("mime") == "image/png" else ".jpg"
            path = os.path.join(folder, f"{stamp}-{slug}-{i}-{j}{ext}")
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(att["data"]))
            att["path"] = path
            att.pop("data", None)


def _print_remote(res: dict, as_json: bool, single: bool) -> int:
    _write_remote_attachments(res)
    if as_json:
        if single and res.get("steps"):
            step = res["steps"][0]
            body = ({"refused": step["refusal"]} if step.get("refusal")
                    else {"op": step.get("op"), "result": step.get("result")})
        else:
            body = {k: v for k, v in res.items() if k != "text"}
        _print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 1
    if single and res.get("steps") and res["steps"][0].get("refusal"):
        from . import ops
        _print(ops.render.refusal_text(res["steps"][0]["refusal"]), sys.stderr)
        return 1
    _print(res.get("text", ""))
    for step in res.get("steps", []):
        for att in step.get("attachments") or []:
            if att.get("path"):
                _print(f"Sheet: {att['path']}")
    return 0 if res.get("ok") else 1


def main(argv=None) -> int:
    _paths.use_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _start_app()

    as_json = sheets = local = False
    batch_file = None
    while argv and argv[0].startswith("-"):
        flag = argv.pop(0)
        if flag == "--json":
            as_json = True
        elif flag == "--sheets":
            sheets = True
        elif flag == "--local":
            local = True
        elif flag == "--headless":
            port = 0
            if argv[:1] == ["--port"] and len(argv) > 1 and argv[1].isdigit():
                port = int(argv[1])
            from . import server
            server.serve_headless(port)
            return 0
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
            _print(f"Unknown option {flag}. Options before the op are --json, --sheets, "
                   f"--local and --batch.\n\n" + USAGE, sys.stderr)
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

    from . import server
    remote = None if local else server.find()
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
        if remote:
            return _print_remote(server.remote_exec(remote, lines, "cli", sheets), as_json, False)
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
        if remote:
            line = ops.grammar.join(argv)
            return _print_remote(server.remote_exec(remote, [line], "cli", sheets), as_json, True)
        result = ops.call(parsed.op.name, parsed.args, ctx=ctx, label=ops.grammar.join(argv))
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
