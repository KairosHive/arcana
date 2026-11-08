# workspace.py -- where an agent started from the app works
#
# The Agent tab runs an agent's command line (Claude Code) in a terminal. Its
# working folder is the Arcana workspace, beside the folder that holds what
# Arcana makes (boards, colour transfers, moodboard exports), so the agent's
# drafts and notes sit with the work and the outputs are one step away.
#
# Each launch refreshes what the agent needs to find there, and leaves
# everything else in the folder alone:
#
#   CLAUDE.md, AGENTS.md     what Arcana is and how to drive it, for Claude Code
#                            and for agents that read AGENTS.md
#   .mcp.json                Arcana's MCP server for this install, so the agent
#                            gets the `exec` tool without any setup
#   .claude/settings.json    approves that server, so the first launch asks nothing
#   .claude/skills/          the moodboard, board-design and harmonize skills
#
# A person's own notes for the agent belong in CLAUDE.local.md, which is never
# rewritten.

from __future__ import annotations

import json
import os
import shutil
import sys

try:
    from . import paths as _paths
except ImportError:                                   # pragma: no cover
    import paths as _paths

WORKSPACE_NAME = "workspace"
SKILLS = ("moodboard", "board-design", "harmonize")

GUIDE = """\
# Arcana workspace

This folder is your working folder inside Arcana, the person's picture
archive app. Keep drafts and notes here. Arcana's own outputs are beside it in
`{output}`: rendered boards in `boards/`, colour transfers in
`color_transfer/`, restyles in `transfers/`, exported moodboards in
`moodboards/`.

## How to work with Arcana

Use the `exec` tool of the `arcana` MCP server. It takes op lines, the same
lines a person types after `arcana` in a terminal, and runs a list of them as
one batch and one undo step. Begin with:

    exec(["orient"])                     what is on this machine
    exec(["op list"])                    every op
    exec(["search text --help"])         any op's help

The `arcana` command works in this terminal too, as the same agent.

- Name pictures by handle (`japan:1423`), and derived pictures by their
  `out:` handle. Paths are for the person.
- You change only boards an agent made. To work on one of the person's
  boards, `board copy` it first.
- Whether you see thumbnails is the person's choice, shown in every result.
  Blind, you work from palettes, cluster names and descriptions.
- Boards you build appear in the app's Moodboard tab within two seconds.

## Skills

`.claude/skills/` holds three skills: `moodboard` (building a board from a
brief), `board-design` (judging and improving one) and `harmonize` (when and
how to recolour). Use them.

The person may leave notes for you in `CLAUDE.local.md`.
"""


def workspace_dir() -> str:
    """The workspace: beside Arcana's output folder."""
    out = _paths.subdir("output")
    return os.path.join(os.path.dirname(os.path.abspath(out)), WORKSPACE_NAME)


def mcp_command() -> dict:
    """How to start Arcana's MCP server from this install."""
    if getattr(sys, "frozen", False):
        return {"command": sys.executable, "args": ["--mcp"]}
    return {"command": sys.executable, "args": ["-m", "arcana.mcp_server"]}


def _skill_sources() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "skills"), os.path.join(os.path.dirname(here), "plugin", "skills")]


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def prepare() -> str:
    """Create or refresh the workspace. Returns its path."""
    ws = workspace_dir()
    _paths.ensure_dir(ws)
    out = os.path.abspath(_paths.subdir("output"))
    guide = GUIDE.format(output=out.replace("\\", "/"))
    _write(os.path.join(ws, "CLAUDE.md"), guide)
    _write(os.path.join(ws, "AGENTS.md"), guide)

    server = {"mcpServers": {"arcana": {**mcp_command(),
                                        "env": {"ARCANA_DATA_DIR": _paths.data_dir()}}}}
    _write(os.path.join(ws, ".mcp.json"), json.dumps(server, indent=2) + "\n")

    settings_path = os.path.join(ws, ".claude", "settings.json")
    try:
        with open(settings_path, encoding="utf-8") as fh:
            settings = json.load(fh)
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        settings = {}
    settings["enabledMcpjsonServers"] = sorted(set(settings.get("enabledMcpjsonServers") or [])
                                               | {"arcana"})
    allow = set((settings.get("permissions") or {}).get("allow") or [])
    allow.add("mcp__arcana__exec")
    settings.setdefault("permissions", {})["allow"] = sorted(allow)
    _write(settings_path, json.dumps(settings, indent=2) + "\n")

    for name in SKILLS:
        for src in _skill_sources():
            f = os.path.join(src, name, "SKILL.md")
            if os.path.isfile(f):
                dest = os.path.join(ws, ".claude", "skills", name)
                os.makedirs(dest, exist_ok=True)
                shutil.copyfile(f, os.path.join(dest, "SKILL.md"))
                break
    return ws
