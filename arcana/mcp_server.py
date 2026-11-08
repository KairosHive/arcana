# mcp_server.py — Arcana's archive, offered to an agent over MCP
#
# An MCP client (Claude Code, Claude Desktop, or any client for a local model)
# starts this as its own process and talks to it over stdin/stdout. The agent
# brings the judgment -- breaking a theme into angles, choosing, cutting -- and
# this brings the archive: search over the person's own pictures, run here, on
# this machine, against indexes the app already built. See AGENT_PLAN.md.
#
# Four rules hold everywhere in this file:
#
#   * Read only. Phase 1 tools look; nothing writes, and no tool can ever touch
#     an original picture.
#   * Pictures are named by handle, "dataset:key" (japan:1423), never by path.
#     Shorter for the model to repeat, and stable within an index.
#   * Blind unless the person says otherwise. Whether thumbnails may leave the
#     machine is decided in agent.json in the data dir, by the person. No tool
#     can change it, so a model cannot decide for itself to start sending
#     pictures. While blind, every result says so.
#
# stdout carries the protocol, and Arcana's modules print progress freely. The
# MCP SDK's stdio transport (2.x) takes care of that: while it serves, file
# descriptor 1 points at stderr and the protocol runs on a private duplicate,
# so a stray print() -- or a native library writing to fd 1 -- misses the wire.
# tests/test_mcp_server.py checks it over a real pipe.
#
# The work itself lives in arcana.ops, where each tool's body became an op
# (dataset list, dataset survey, search facets, search like, search palette,
# item show). The tools below hand their arguments to ops.call() with the
# agent actor and turn the result into MCP content. P1 of OPS_PLAN.md replaces
# the six tools with one, exec, which takes op lines directly.

from __future__ import annotations

import base64
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

try:
    from . import ops
    from .ops import archive as _archive
    from .ops.search import Facet, swatch as _swatch
except ImportError:                                   # pragma: no cover
    import ops
    from ops import archive as _archive
    from ops.search import Facet, swatch as _swatch


# The helpers tests and older callers reach for here, now living in ops.archive.
CONFIG_NAME = _archive.CONFIG_NAME
settings = _archive.settings
_mode_line = _archive.mode_line
dataset_id = _archive.dataset_id
split_dataset_id = _archive.split_dataset_id
handle = _archive.handle
parse_handle = _archive.parse_handle
open_dataset = _archive.open_dataset


class Filters(BaseModel):
    """The person's own decisions about their pictures, from ON1 / Lightroom sidecars."""
    min_stars: int = Field(default=0, ge=-1, le=5,
                           description="Only pictures rated at least this. -1 means never rated.")
    colours: list[str] = Field(default_factory=list,
                               description="Colour tags to keep, e.g. ['Red']. '(none)' means untagged.")
    date_from: str | None = Field(default=None, description="YYYY-MM-DD, inclusive.")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD, inclusive.")

    def as_args(self) -> dict:
        return {"min_stars": self.min_stars, "tag": list(self.colours),
                "date_from": self.date_from, "date_to": self.date_to}


# ─────────────────────────── the tools' work ───────────────────────────
# Plain functions, so tests can call them without an MCP client.

def _content(name: str, args: dict) -> list:
    """Run one op as the agent and return its result as MCP content."""
    ctx = ops.context("agent")
    result = ops.call(name, args, ctx=ctx)
    out = [TextContent(type="text", text=result.text().rstrip())]
    for att in ops.render.visible_attachments(result, ctx):
        out.append(ImageContent(type="image", data=base64.b64encode(att.data).decode("ascii"),
                                mime_type=att.mime))
    out.append(TextContent(type="text", text=_archive.mode_line()))
    return out


def do_list_datasets() -> list:
    return _content("dataset list", {})


def do_survey(ds_id: str, per_cluster: int = 4) -> list:
    return _content("dataset survey", {"dataset": ds_id, "per_cluster": per_cluster})


def do_search(ds_id: str, facets: list[Facet], k: int = 12, diversity: float = 0.3,
              filters: Filters | None = None) -> list:
    return _content("search facets", {"dataset": ds_id,
                                      "facet": [f.model_dump() for f in facets],
                                      "k": k, "diversity": diversity,
                                      **(filters or Filters()).as_args()})


def do_more_like(handles: list[str], k: int = 12, mode: str = "semantic",
                 diversity: float = 0.3, exclude: list[str] | None = None) -> list:
    return _content("search like", {"handles": handles, "k": k, "mode": mode,
                                    "diversity": diversity, "exclude": exclude or []})


def do_find_by_palette(ds_id: str, colours: list[str], weights: list[float] | None = None,
                       k: int = 12) -> list:
    if weights is not None and len(weights) != len(colours):
        raise ValueError("weights, when given, are one positive number per colour.")
    entries = [f"{c}:{w}" for c, w in zip(colours, weights)] if weights else list(colours)
    return _content("search palette", {"dataset": ds_id, "colour": entries, "k": k})


def do_inspect(handles: list[str]) -> list:
    return _content("item show", {"handles": handles})


# ─────────────────────────── the server ───────────────────────────

INSTRUCTIONS = """\
Arcana searches the person's own image archive, on their machine, by what the
pictures show. Nothing here edits or moves their files.

Work like this: list_datasets, then survey a dataset to see what it actually
holds before planning, then search. Refer to pictures by handle (japan:1423).

CLIP, which ranks the pictures, sees concrete visual things: objects, places,
light, weather, colour, texture, composition, photographic technique. It cannot
see abstractions, so translate them ('melancholy' -> 'empty bench in rain,
overcast, muted colours'). Give each idea several phrasings, and use negatives
for things that keep intruding.

Every search facet comes back marked strong, fair or weak. Weak means the
archive has little for that idea: say so plainly rather than presenting the
results as a match.

Every result states whether you can see thumbnails (pixels) or not (blind).
Only the person can change that.
"""

def _call(fn, *args):
    """
    Run a tool's work, turning the failures we expect into messages for the model.

    The SDK shows the model a ToolError's text. Any other exception counts as a
    crash and reaches it only as "Error executing tool search", which is how
    "No dataset called 'nope'. Known datasets: ..." would otherwise be lost.
    """
    try:
        return fn(*args)
    except ops.Refusal as e:
        raise ToolError(e.reason) from e
    except (ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e


mcp = MCPServer("arcana", instructions=INSTRUCTIONS)
_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

# Tools are plain functions: the SDK runs them on a worker thread, so a search
# that spends seconds loading an encoder the first time does not stall the
# protocol loop. Each one hands straight to its do_* counterpart above.

PerFacet = Annotated[int, Field(ge=1, le=24, description="Pictures per facet.")]
Diversity = Annotated[float, Field(
    ge=0.0, le=1.0,
    description="0 = closest matches only; higher spreads results across different "
                "takes on the idea.")]


@mcp.tool(annotations=_READ, structured_output=False)
def list_datasets() -> list:
    """The person's datasets: size, encoder, clusters, which features and marks exist."""
    return _call(do_list_datasets)


@mcp.tool(annotations=_READ, structured_output=False)
def survey(dataset: str,
           per_cluster: Annotated[int, Field(ge=1, le=8)] = 4) -> list:
    """
    What a dataset holds: its clusters, largest first, each with its most typical
    pictures. Do this before planning a search, so the plan fits the archive.
    """
    return _call(do_survey, dataset, per_cluster)


@mcp.tool(annotations=_READ, structured_output=False)
def search(dataset: str,
           facets: Annotated[list[Facet], Field(min_length=1, max_length=8)],
           k: PerFacet = 12,
           diversity: Diversity = 0.3,
           filters: Filters | None = None) -> list:
    """
    Search a dataset for several visual ideas at once, one facet per idea.

    Each facet returns up to k pictures, near-duplicates removed, never repeating
    a picture an earlier facet already showed. Each picture has p: how sure CLIP
    is that it shows the facet's phrases rather than something generic. Each
    facet is marked strong, fair or weak.
    """
    return _call(do_search, dataset, facets, k, diversity, filters)


@mcp.tool(annotations=_READ, structured_output=False)
def more_like(handles: Annotated[list[str], Field(min_length=1, max_length=8)],
              k: PerFacet = 12,
              mode: Annotated[str, Field(
                  description="semantic (what the pictures show), palette (their "
                              "colours) or style (texture and grain).")] = "semantic",
              diversity: Diversity = 0.3,
              exclude: Annotated[list[str], Field(
                  description="Handles not to return again.")] = []) -> list:
    """
    Pictures like these: search by example. Semantic mode blends several
    examples; palette and style modes use the first one.
    """
    return _call(do_more_like, handles, k, mode, diversity, exclude)


@mcp.tool(annotations=_READ, structured_output=False)
def find_by_palette(dataset: str,
                    colours: Annotated[list[str], Field(
                        min_length=1, max_length=8,
                        description="Hex colours, e.g. ['#1f2a36', '#b5651d'].")],
                    weights: Annotated[list[float] | None, Field(
                        description="Optional share of each colour.")] = None,
                    k: PerFacet = 12) -> list:
    """Pictures whose colours match a palette you choose."""
    return _call(do_find_by_palette, dataset, colours, weights, k)


@mcp.tool(annotations=_READ, structured_output=False)
def inspect(handles: Annotated[list[str], Field(min_length=1, max_length=4)]) -> list:
    """
    A close look at up to four pictures: size, palette, marks, cluster, and the
    picture itself in pixels mode (or a CLIP description of it in blind mode).
    """
    return _call(do_inspect, handles)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
