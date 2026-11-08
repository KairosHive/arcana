# mcp_server.py -- Arcana offered to an agent over MCP, as one tool
#
# An MCP client (Claude Code, Claude Desktop, or any client for a local model)
# starts this as its own process and talks to it over stdin/stdout. The agent
# brings the judgment -- breaking a theme into angles, choosing, cutting -- and
# Arcana brings the archive and the tools: search over the person's own
# pictures, boards, colour and style transfer, run on this machine.
#
# There is one tool, `exec`, and it takes op lines, the same lines a person
# types after `arcana` in a terminal:
#
#     exec(["dataset survey japan",
#           "search text japan --phrase \"rain on a dark street\" --k 8"])
#
# A list runs as one batch: one undo step, rolled back whole if any line is
# refused. A refusal comes back as error content carrying the op's help, so the
# model reads the right shape beside the complaint and corrects itself. The
# tool's description is the orient page, so the model's first sight of Arcana
# is the page a person reads. See OPS_PLAN.md, "MCP".
#
# Rules that hold whatever the model asks:
#
#   * The actor is always `agent`. A model cannot claim to be the person, so it
#     changes only boards an agent made, and never deletes a board or changes
#     settings.
#   * Pictures are named by handle (japan:1423), never by path.
#   * Blind unless the person says otherwise, in agent.json. While blind, no
#     picture leaves the machine and every result says so.
#
# When the app is open (server.json in the data dir answers), each call is
# forwarded to it, so the app's encoders, jobs and GPU serve the agent and its
# Moodboard tab shows the agent's boards as they change. Otherwise the ops run
# in this process.
#
# stdout carries the protocol, and Arcana's modules print progress freely. The
# MCP SDK's stdio transport (2.x) keeps file descriptor 1 off the wire while it
# serves. tests/test_mcp_server.py checks it over a real pipe.

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field

try:
    from . import ops, server
    from .ops import archive as _archive
except ImportError:                                   # pragma: no cover
    import ops
    import server
    from ops import archive as _archive

ACTOR = "agent"
MAX_COMMANDS = 40

TOOL_GUIDE = """\
Run Arcana ops. Each command is one op line, the same grammar as the `arcana`
command line: the op's name, bare arguments, then --flags. A list runs in
order as one batch and one undo step, and if any line is refused every change
in the batch is rolled back.

Start with ["orient"], then ["op list"], and ["<op> --help"] for any op.
"""


def orient_page() -> str:
    try:
        return ops.call("orient", ctx=ops.context(ACTOR)).text()
    except Exception as e:                            # the page must never stop the server
        return f"(orient could not run: {e})\n"


def _content(res: dict) -> list:
    out = [TextContent(type="text", text=res.get("text", "").rstrip())]
    for step in res.get("steps", []):
        for att in step.get("attachments") or []:
            out.append(ImageContent(type="image", data=att["data"],
                                    mime_type=att.get("mime", "image/jpeg")))
    out.append(TextContent(type="text", text=_archive.mode_line()))
    return out


def run(commands: list[str]) -> list:
    """
    One batch, forwarded to the running app or run here. Returns MCP content,
    or raises ToolError with the refusal and the op's help.
    """
    remote = server.find()
    if remote:
        res = server.remote_exec(remote, list(commands), ACTOR)
    else:
        res = server.run_lines(list(commands), ACTOR)
    if not res.get("ok"):
        raise ToolError(res.get("text", "Refused.").rstrip() + "\n\n" + _archive.mode_line())
    return _content(res)


mcp = MCPServer("arcana", instructions=TOOL_GUIDE + "\n" + orient_page())


@mcp.tool(name="exec", description=TOOL_GUIDE,
          annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                      idempotent_hint=False, open_world_hint=False),
          structured_output=False)
def exec_ops(commands: Annotated[list[str], Field(
        min_length=1, max_length=MAX_COMMANDS,
        description="Op lines, e.g. [\"search text japan --phrase \\\"wet asphalt\\\" --k 8\"].")]) -> list:
    return run(commands)


@mcp.resource("arcana://orient", name="orient", mime_type="text/plain",
              description="What Arcana is, what is on this machine, and how to search.")
def _orient_resource() -> str:
    return orient_page()


@mcp.resource("arcana://ops", name="ops", mime_type="text/plain",
              description="Every op, one line each, with its kind.")
def _ops_resource() -> str:
    return ops.call("op list", ctx=ops.context(ACTOR)).text()


def skill_text(name: str) -> str | None:
    """A skill's body from arcana/skills, without its front matter."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", name, "SKILL.md")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


@mcp.prompt(name="moodboard", description="Build a moodboard from a theme and a dataset.")
def moodboard(theme: str, dataset: str) -> str:
    guide = skill_text("moodboard") or ""
    return (f"Build a moodboard for \"{theme}\" from the dataset `{dataset}` with Arcana's "
            f"exec tool. Begin with exec([\"orient\", \"dataset survey {dataset}\"]).\n\n{guide}")


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
